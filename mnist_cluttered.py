import os
import functools
import torch
import numpy as np

from PIL import Image

from .abstract_dataset import AbstractLoader


def load_cluttered_mnist(path, segment='train'):
    """Load the required npz file and return images + labels."""
    loaded = np.load(os.path.join(path, "{}.npz".format(segment)))
    return [loaded['data'], loaded['labels']]


class ClutteredMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, path, segment='train', transform=None, target_transform=None):

        self.path = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform
        self.segment = segment.lower().strip()  # train or test or val

        # load the images and labels
        self.imgs, self.labels = self._load_from_path()

    def _load_from_path(self):
        # load the tensor dataset from npz
        imgs, labels = load_cluttered_mnist(self.path, segment=self.segment)
        print("imgs_{} = {} | lbl_{} = {} ".format(self.segment, imgs.shape,
                                                   self.segment, labels.shape))
        return imgs, labels

    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index]
        img = Image.fromarray(np.uint8(img.squeeze()*255))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ClutteredMNISTLoader(AbstractLoader):
    """Simple ClutteredMNIST loader, there is no validation set."""

    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 num_replicas=1, cuda=True, **kwargs):

        # Curry the train and test dataset generators.
        train_generator = functools.partial(ClutteredMNISTDataset, path=path, segment='train')
        test_generator = functools.partial(ClutteredMNISTDataset, path=path, segment='test')

        super(ClutteredMNISTLoader, self).__init__(batch_size=batch_size,
                                                   train_dataset_generator=train_generator,
                                                   test_dataset_generator=test_generator,
                                                   train_sampler=train_sampler,
                                                   test_sampler=test_sampler,
                                                   train_transform=train_transform,
                                                   train_target_transform=train_target_transform,
                                                   test_transform=test_transform,
                                                   test_target_transform=test_target_transform,
                                                   num_replicas=num_replicas, cuda=cuda, **kwargs)
        self.output_size = self.determine_output_size()
        self.loss_type = 'ce'  # fixed

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.input_shape = list(test_img.size()[1:])
        print("derived image shape = ", self.input_shape)
