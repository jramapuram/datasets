import torch
import numpy as np
from scipy.misc import imread, imresize
from torchvision import datasets, transforms

from .utils import binarize, create_loader


class FashionMNISTLoader(object):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, binarize_images=False,
                 use_cuda=1, **kwargs):
        # first get the datasets, binarize fashionmnist though
        if binarize_images:
            transform = [binarize] if transform is None else [*transform, binarize]

        train_dataset, test_dataset = self.get_datasets(path, transform,
                                                        target_transform)

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

        self.output_size = 10
        self.batch_size = batch_size

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.img_shp = list(test_img.size()[1:])

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

        train_dataset = datasets.FashionMNIST(path, train=True, download=True,
                                              transform=transforms.Compose(transform_list),
                                              target_transform=target_transform)
        test_dataset = datasets.FashionMNIST(path, train=False,
                                             transform=transforms.Compose(transform_list),
                                             target_transform=target_transform)
        return train_dataset, test_dataset
