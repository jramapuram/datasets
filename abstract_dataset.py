import torch
import numpy as np
from scipy.misc import imread, imresize
from torchvision import datasets, transforms

from .utils import binarize, create_loader


class AbstractLoader(object):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, binarize_images=True,
                 use_cuda=1, **kwargs):
        ''' base class, set self.dataset to be the DataSet, eg: datasets.MNIST '''
        raise NotImplementedError("__init__ not implemented")

    @staticmethod
    def get_datasets(path, transform=None, target_transform=None):
        if transform:
            assert isinstance(transform, list)

        transform_list = []
        if transform:
            transform_list.extend(transform)

        transform_list.append(transforms.ToTensor())
        train_dataset = self.dataset(path, train=True, download=True,
                                              transform=transforms.Compose(transform_list),
                                              target_transform=target_transform)
        test_dataset = self.dataset(path, train=False,
                                    transform=transforms.Compose(transform_list),
                                    target_transform=target_transform)
        return train_dataset, test_dataset
