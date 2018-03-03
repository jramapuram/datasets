import torch
import numpy as np
from torchvision import datasets, transforms


class PermutedMNISTLoader(object):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1):
        # first get the datasets
        train_dataset, test_dataset = self.get_datasets(path, transform, target_transform)

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
        #print('test img = ', test_img.shape)
        self.img_shp = list(test_img.size()[1:])
        #print("derived image shape = ", self.img_shp)

    @staticmethod
    def _get_permutation_lambda():
        pixel_permutation = torch.randperm(28*28)         # add the permutation
        return transforms.Lambda(lambda x: x.view(-1,1)[pixel_permutation].view(1, 28, 28))

    @staticmethod
    def get_datasets(path, transform=None, target_transform=None):
        if transform:
            assert isinstance(transform, list)

        transform_list = []
        if transform:
            transform_list.extend(transform)

        transform_list.append(transforms.ToTensor())
        transform_list.append(PermutedMNISTLoader._get_permutation_lambda())
        train_dataset = datasets.MNIST(path, train=True, download=True,
                                       transform=transforms.Compose(transform_list),
                                       target_transform=target_transform)
        test_dataset = datasets.MNIST(path, train=False,
                                      transform=transforms.Compose(transform_list),
                                      target_transform=target_transform)
        return train_dataset, test_dataset
