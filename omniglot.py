import os
import torch
import functools
import numpy as np

from torchvision import transforms, datasets

from .abstract_dataset import AbstractLoader
from .utils import temp_seed


class OmniglotDatasetWithFixedRandomTestShuffle(datasets.Omniglot):
    """Do a fixed random shuffle of the test set."""

    def __init__(self, root, background=True, transform=None, target_transform=None, download=False):
        super(OmniglotDatasetWithFixedRandomTestShuffle, self).__init__(root=root,
                                                                        background=background,
                                                                        transform=transform,
                                                                        target_transform=target_transform,
                                                                        download=download)

        # For the test set we are going to do a fixed random shuffle
        if background is False:
            with temp_seed(1234):  # deterministic shuffle of test set
                idx = np.random.permutation(np.arange(len(self._flat_character_images)))
                first = np.array([i[0] for i in self._flat_character_images])[idx]
                second = np.array([i[1] for i in self._flat_character_images])[idx].astype(np.int32)
                self._flat_character_images = [(fi, si) for fi, si in zip(first, second)]


class BinarizedOmniglotDataset(OmniglotDatasetWithFixedRandomTestShuffle):
    """Standard binary omniglot pytorch dataset with PIL binarization."""

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image, character_class = super(BinarizedOmniglotDataset, self).__getitem__(index)

        # workaround to go back to B/W for grayscale
        image = transforms.ToTensor()(transforms.ToPILImage()(image).convert('1'))
        # image[image > 0.2] = 1.0 # XXX: workaround to upsample / downsample

        return image, character_class


class BinarizedOmniglotBurdaDataset(torch.utils.data.Dataset):
    def __init__(self, path, split='train', download=True,
                 transform=None, target_transform=None, **kwargs):
        self.split = split
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform

        # load the images-paths and labels
        self.train_dataset, self.test_dataset = self.read_dataset(path)

        # determine train-test split
        if split == 'train':
            self.imgs = self.train_dataset
        else:
            with temp_seed(1234):
                perm = np.random.permutation(np.arange(len(self.test_dataset)))
                self.imgs = self.test_dataset[perm]

        print("[{}] {} samples".format(split, len(self.imgs)))

    def read_dataset(self, path):
        """Helper to read the matlab files."""
        import scipy.io as sio
        data_file = os.path.join(path, 'chardata.mat')
        if not os.path.isfile(data_file):
            import requests
            dataset_url = 'https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat'
            os.makedirs(path, exist_ok=True)
            open(os.path.join(path, 'chardata.mat'), 'wb').write(requests.get(dataset_url, allow_redirects=True).content)

        def reshape_data(data):
            return data.reshape((-1, 1, 28, 28))  # .reshape((-1, 28*28), order='fortran')

        # read full dataset and return the train and test data
        omni_raw = sio.loadmat(data_file)
        train_data = reshape_data(omni_raw['data'].T.astype('float32'))
        test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))
        return train_data, test_data

    def __getitem__(self, index):
        img = self.imgs[index]
        img = transforms.ToPILImage()(torch.from_numpy(img))

        # handle transforms if requested
        if self.transform is not None:
            img = self.transform(img)

        target = 0
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class OmniglotLoader(AbstractLoader):
    """Simple Omniglor loader using pytorch loader, there is no validation set."""

    def __init__(self, path, batch_size, num_replicas=0,
                 train_sampler=None, test_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 cuda=True, **kwargs):
        # Curry the train and test dataset generators.
        train_generator = functools.partial(OmniglotDatasetWithFixedRandomTestShuffle,
                                            root=path, background=True, download=True)
        test_generator = functools.partial(OmniglotDatasetWithFixedRandomTestShuffle,
                                           root=path, background=False, download=True)

        super(OmniglotLoader, self).__init__(batch_size=batch_size,
                                             train_dataset_generator=train_generator,
                                             test_dataset_generator=test_generator,
                                             train_sampler=train_sampler,
                                             test_sampler=test_sampler,
                                             train_transform=train_transform,
                                             train_target_transform=train_target_transform,
                                             test_transform=test_transform,
                                             test_target_transform=test_target_transform,
                                             num_replicas=num_replicas, cuda=cuda, **kwargs)
        self.output_size = 1623  # fixed
        self.loss_type = 'ce'    # fixed

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.input_shape = list(test_img.size()[1:])
        print("derived image shape = ", self.input_shape)


class BinarizedOmniglotLoader(AbstractLoader):
    """Binarized omniglot loader using pytorch omniglot w/ PIL binarization; no validation set."""

    def __init__(self, path, batch_size, num_replicas=0,
                 train_sampler=None, test_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 cuda=True, **kwargs):
        # Curry the train and test dataset generators.
        train_generator = functools.partial(BinarizedOmniglotDataset, root=path, background=True, download=True)
        test_generator = functools.partial(BinarizedOmniglotDataset, root=path, background=False, download=True)

        super(BinarizedOmniglotLoader, self).__init__(batch_size=batch_size,
                                                      train_dataset_generator=train_generator,
                                                      test_dataset_generator=test_generator,
                                                      train_sampler=train_sampler,
                                                      test_sampler=test_sampler,
                                                      train_transform=train_transform,
                                                      train_target_transform=train_target_transform,
                                                      test_transform=test_transform,
                                                      test_target_transform=test_target_transform,
                                                      num_replicas=num_replicas, cuda=cuda, **kwargs)
        self.output_size = 1623  # fixed
        self.loss_type = 'ce'    # fixed

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.input_shape = list(test_img.size()[1:])
        print("derived image shape = ", self.input_shape)


class BinarizedOmniglotBurdaLoader(AbstractLoader):
    """Simple BinarizedMNIST-Burda loader, there is no validation set."""

    def __init__(self, path, batch_size, num_replicas=0,
                 train_sampler=None, test_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 cuda=True, **kwargs):

        assert train_target_transform is None and test_target_transform is None, "No labels for Burda-Omniglot."

        # Curry the train and test dataset generators.
        train_generator = functools.partial(BinarizedOmniglotBurdaDataset, path=path, split='train', download=True)
        test_generator = functools.partial(BinarizedOmniglotBurdaDataset, path=path, split='test', download=True)

        super(BinarizedOmniglotBurdaLoader, self).__init__(batch_size=batch_size,
                                                           train_dataset_generator=train_generator,
                                                           test_dataset_generator=test_generator,
                                                           train_sampler=train_sampler,
                                                           test_sampler=test_sampler,
                                                           train_transform=train_transform,
                                                           train_target_transform=train_target_transform,
                                                           test_transform=test_transform,
                                                           test_target_transform=test_target_transform,
                                                           num_replicas=num_replicas, cuda=cuda, **kwargs)
        self.output_size = 0   # fixed (Burda version has no labels)
        self.loss_type = 'ce'  # fixed

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.input_shape = list(test_img.size()[1:])
        print("derived image shape = ", self.input_shape)
