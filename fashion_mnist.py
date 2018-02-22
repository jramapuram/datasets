import torch
import numpy as np
from scipy.misc import imread, imresize
from torchvision import datasets, transforms

from datasets.utils import binarize

class FashionMNISTLoader(object):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, binarize_images=True, use_cuda=1):
        # first get the datasets, binarize fashionmnist though
        if binarize_images:
            transform = [binarize] if transform is None else [transform, binarize]

        train_dataset, test_dataset = self.get_datasets(path, transform,
                                                        target_transform)

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_sampler = train_sampler(train_dataset) if train_sampler else None
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True if train_sampler is None else False,
            sampler=train_sampler,
            **kwargs)


        test_sampler = test_sampler(test_dataset) if test_sampler else None
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=False,
            sampler=test_sampler,
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

        transform_list.append(transforms.ToTensor())
        train_dataset = datasets.FashionMNIST(path, train=True, download=True,
                                              transform=transforms.Compose(transform_list),
                                              target_transform=target_transform)
        test_dataset = datasets.FashionMNIST(path, train=False,
                                             transform=transforms.Compose(transform_list),
                                             target_transform=target_transform)
        return train_dataset, test_dataset
