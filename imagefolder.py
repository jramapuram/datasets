import os
import torch
import numpy as np
from torchvision import datasets, transforms

from .utils import create_loader


class ImageFolderLoader(object):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1, **kwargs):
        # first get the datasets
        train_dataset, test_dataset = self.get_datasets(path, transform, target_transform)

        # build the loaders
        kwargs_loader = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        self.train_loader = create_loader(train_dataset,
                                          train_sampler,
                                          batch_size,
                                          shuffle=True if train_sampler is None else False,
                                          **kwargs_loader)

        self.test_loader = create_loader(test_dataset,
                                         test_sampler,
                                         batch_size,
                                         shuffle=False,
                                         **kwargs_loader)
        self.batch_size = batch_size
        self.output_size = 0

        # iterate over the entire dataset to find the max label
        # but just one image to get the image sizing
        test_img, _ = self.train_loader.__iter__().__next__()
        self.img_shp = list(test_img.size()[1:])
        print("determined img_size: ", self.img_shp)
        if 'output_size' not in kwargs:
            for _, label in self.train_loader:
                if not isinstance(label, (float, int))\
                   and len(label) > 1:
                    for l in label:
                        if l > self.output_size:
                            self.output_size = l
                else:
                    if label > self.output_size:
                        self.output_size = label

            self.output_size = self.output_size.item() + 1 # Longtensor --> int
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

        transform_list.append(transforms.ToTensor())
        train_dataset = datasets.ImageFolder(root=os.path.join(path, 'train'),
                                             transform=transforms.Compose(transform_list),
                                             target_transform=target_transform)
        test_dataset = datasets.ImageFolder(root=os.path.join(path, 'test'),
                                            transform=transforms.Compose(transform_list),
                                            target_transform=target_transform)
        return train_dataset, test_dataset
