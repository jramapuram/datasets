import os
import torch
import numpy as np
import torch.utils.data as data

from PIL import Image
from copy import deepcopy
from torchvision import transforms, datasets

from .abstract_dataset import AbstractLoader
from .utils import create_loader, temp_seed


class OmniglotLoader(object):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1, **kwargs):
        assert train_sampler is None and test_sampler is None, "omniglot loader does not support samplers"

        # first get the datasets
        train_dataset, test_dataset = self.get_datasets(path, transform, target_transform)

        # use the non-standard solution used in Kanerva machine & Beta-VAE
        #train_dataset, test_dataset, train_sampler, test_sampler = self.get_samplers(train_dataset, test_dataset)
        train_dataset, test_dataset = self.get_samplers(train_dataset, test_dataset)

        # build the loaders
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        self.train_loader = create_loader(train_dataset,
                                          train_sampler,
                                          batch_size,
                                          shuffle=True if train_sampler is None else False,
                                          **kwargs)

        self.test_loader = create_loader(test_dataset,
                                         test_sampler,
                                         batch_size,
                                         shuffle=False,
                                         **kwargs)

        # self.output_size = 964 # np.max(labels) gives this (+0)
        # self.batch_size = batch_size
        # self.img_shp = [1, 105, 105]

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.img_shp = list(test_img.size()[1:])
        print("derived image shape = ", self.img_shp)

        # determine number of samples
        self.output_size = 0
        self.determine_output_size(self.train_loader, **kwargs)

        # print dataset sizes
        print("[train]\t {} samples".format(len(train_dataset)))
        print("[test]\t {} samples".format(len(test_dataset)))

    def get_samplers(self, train_dataset, test_dataset):
        # make the train set larger as per Wu et. al (Kanerva Machine) & Burda et. al (Beta-VAE)
        num_train_samples = 24345
        # num_extra_train = 24345 - 17500
        # num_test = 11900 - num_extra_train

        # create one larger dataset
        #new_train_dataset = data.ConcatDataset([train_dataset, test_dataset])
        new_train_dataset = train_dataset + test_dataset
        new_test_dataset = deepcopy(new_train_dataset)
        num_test_samples = len(new_train_dataset) - num_train_samples

        # build our subset samples and return both datasets
        train_sampler, test_sampler = None, None
        with temp_seed(1234):
            full_dataset_range = np.random.permutation(np.arange(num_train_samples + num_test_samples))
            train_range = full_dataset_range[0:num_train_samples]
            test_range = full_dataset_range[num_train_samples:]
            assert len(train_range) == num_train_samples
            assert len(test_range) == num_test_samples
            # train_sampler = data.SubsetRandomSampler(train_range)
            # test_sampler = data.SubsetRandomSampler(test_range)
            new_train_dataset = data.Subset(new_train_dataset, train_range)
            new_test_dataset = data.Subset(new_test_dataset, test_range)

        # return new_train_dataset, new_test_dataset, train_sampler, test_sampler

        return new_train_dataset, new_test_dataset


    def determine_output_size(self, train_loader, **kwargs):
        # determine output size
        if 'output_size' not in kwargs or kwargs['output_size'] is None:
            for _, label in self.train_loader:
                if not isinstance(label, (float, int)) and len(label) > 1:
                    l = np.array(label).max()
                    if l > self.output_size:
                        self.output_size = l
                else:
                    l = label.max().item()
                    if l > self.output_size:
                        self.output_size = l

            self.output_size = self.output_size + 1
        else:
            self.output_size = kwargs['output_size']

        print("determined output_size: ", self.output_size)

    @staticmethod
    def get_datasets(path, transform=None, target_transform=None):
        if transform:
            assert isinstance(transform, list)

        transform_list = []
        if transform:
            transform_list.extend(transform)

        # add ToTensor if it isn't there
        transform_names = [str(tt) for tt in transform_list]
        if 'ToTensor()' not in transform_names:
            transform_list.append(transforms.ToTensor())

        train_dataset = datasets.Omniglot(path, background=True, download=True,
                                          transform=transforms.Compose(transform_list),
                                          target_transform=target_transform)
        test_dataset = datasets.Omniglot(path, background=False, download=True,
                                         transform=transforms.Compose(transform_list),
                                         target_transform=target_transform)

        with temp_seed(1234): # deterministic shuffle of test set
            idx = np.random.permutation(np.arange(len(test_dataset._flat_character_images)))
            first = np.array([i[0] for i in test_dataset._flat_character_images])[idx]
            second = np.array([i[1] for i in test_dataset._flat_character_images])[idx].astype(np.int32)
            test_dataset._flat_character_images = [(fi, si) for fi, si in zip(first, second)]

        return train_dataset, test_dataset


class BinarizedOmniglotLoader(OmniglotLoader):
    @staticmethod
    def get_datasets(path, transform=None, target_transform=None):
        if transform:
            assert isinstance(transform, list)

        transform_list = []
        if transform:
            transform_list.extend(transform)

        # add ToTensor if it isn't there
        transform_names = [str(tt) for tt in transform_list]
        if 'ToTensor()' not in transform_names:
            transform_list.append(transforms.ToTensor())

        train_dataset = BinarizedOmniglotDataset(path, background=True, download=True,
                                                 transform=transforms.Compose(transform_list),
                                                 target_transform=target_transform)
        test_dataset = BinarizedOmniglotDataset(path, background=False, download=True,
                                                transform=transforms.Compose(transform_list),
                                                target_transform=target_transform)

        with temp_seed(1234): # deterministic shuffle of test set
            idx = np.random.permutation(np.arange(len(test_dataset._flat_character_images)))
            first = np.array([i[0] for i in test_dataset._flat_character_images])[idx]
            second = np.array([i[1] for i in test_dataset._flat_character_images])[idx].astype(np.int32)
            test_dataset._flat_character_images = [(fi, si) for fi, si in zip(first, second)]

        return train_dataset, test_dataset


class BinarizedOmniglotDataset(datasets.Omniglot):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image, character_class = super(BinarizedOmniglotDataset, self).__getitem__(index)

        # XXX: workaround to go back to B/W for grayscale
        image = transforms.ToTensor()(transforms.ToPILImage()(image).convert('1'))
        # image[image > 0.2] = 1.0 # XXX: workaround to upsample / downsample

        return image, character_class


class BinarizedOmniglotBurdaDataset(torch.utils.data.Dataset):
    def __init__(self, path, split='train', train=True, download=True,
                 transform=None, target_transform=None, **kwargs):
        self.split = split
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform

        # hard-coded
        self.output_size = 0

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
        import scipy.io as sio
        data_file = os.path.join(path, 'chardata.mat')
        if not os.path.isfile(data_file):
            import requests
            dataset_url = 'https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat'
            open(os.path.join(path, 'chardata.mat'), 'wb').write(requests.get(dataset_url, allow_redirects=True).content)

        def reshape_data(data):
            return data.reshape((-1, 1, 28, 28))#.reshape((-1, 28*28), order='fortran')

        # read full dataset and return the train and test data
        omni_raw = sio.loadmat(data_file)
        train_data = reshape_data(omni_raw['data'].T.astype('float32'))
        test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))
        return train_data, test_data

    def __getitem__(self, index):
        img = self.imgs[index]

        # handle transforms if requested
        if self.transform is not None and len(self.transform.transforms) > 1:
            img = transforms.ToTensor()(
                self.transform(transforms.ToPILImage()(torch.from_numpy(img))))
        else:
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img)

        target = 0
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class EmptyToTensor(object):
    ''' hack of ToTensor: since it is checked in superclass '''
    def __repr__(self):
        return 'ToTensor()'

    def __call__(self, x):
        return x


class BinarizedOmniglotBurdaLoader(AbstractLoader):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1, **kwargs):
        if isinstance(transform, list): # hack to do ToTensor()
            transform.extend([EmptyToTensor()])
        else:
            transform = [EmptyToTensor()]

        # use the abstract class to build the loader
        super(BinarizedOmniglotBurdaLoader, self).__init__(BinarizedOmniglotBurdaDataset, path=path,
                                                           batch_size=batch_size,
                                                           train_sampler=train_sampler,
                                                           test_sampler=test_sampler,
                                                           transform=transform,
                                                           target_transform=target_transform,
                                                           use_cuda=use_cuda, **kwargs)
        self.output_size = 0    # burda has no labels
        self.loss_type = 'none' # fixed
        print("derived output size = ", self.output_size)

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.img_shp = list(test_img.size()[1:])
        print("derived image shape = ", self.img_shp)
