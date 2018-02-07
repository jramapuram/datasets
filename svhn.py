from __future__ import print_function
import torch
from torchvision import datasets, transforms

from datasets.svhn_full import SVHNFull


class SVHNFullLoader(object):
    ''' This loads the original SVHN dataset (non-centered).

        The classes here are BCE classes where each bit
        signifies the presence of the digit in the img'''

    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1):
        # first grab the datasets
        train_dataset, test_dataset = self.get_datasets(path, transform,
                                                        target_transform)

        kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

        train_sampler = train_sampler(train_dataset) if train_sampler else None
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True if train_sampler is None else False,
            sampler=train_sampler
            **kwargs)
        print("successfully loaded SVHN training data...")

        test_sampler = test_sampler(test_dataset) if test_sampler else None
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=False,
            sampler=test_sampler,
            **kwargs)
        print("successfully loaded SVHN test data...")

        self.output_size = 10
        self.batch_size = batch_size
        self.img_shp = [3, 32, 32]

    @staticmethod
    def get_datasets(path, transform=None, target_transform=None):
        if transform:
            assert isinstance(transform, list)

        transform_list = []
        if transform:
            transform_list.extend(transform)

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        # normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
        #                                  std=(0.5, 0.5, 0.5))
        # transform_list.append(normalize)
        transform_list.append(transforms.ToTensor())
        train_dataset = SVHNFull(path, split='train', download=True,
                                 transform=transforms.Compose(transform_list),
                                 target_transform=target_transform)
        test_dataset = SVHNFull(path, split='test', download=True,
                                transform=transforms.Compose(transform_list),
                                target_transform=target_transform)
        return train_dataset, test_dataset


class SVHNCenteredLoader(object):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1):
        # first grab the datasets
        train_dataset, test_dataset = self.get_datasets(
            #path, target_transform=None #transforms.Lambda(lambda lbl: lbl - 1)
            path, transform, target_transform
        )

        kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
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
        self.img_shp = [3, 32, 32]

    @staticmethod
    def get_datasets(path, transform=None, target_transform=None):
        if transform:
            assert isinstance(transform, list)

        transform_list = []
        if transform:
            transform_list.extend(transform)

        # target_transform_list = [transforms.Lambda(lambda lbl: lbl - 1)]
        # if target_transform:
        #     target_transform_list.append(target_transform)

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        # normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
        #                                  std=(0.5, 0.5, 0.5))
        # transform_list.append(normalize)
        transform_list.append(transforms.ToTensor())

        train_dataset = datasets.SVHN(path, split='train', download=True,
                                      transform=transforms.Compose(transform_list),
                                      target_transform=target_transform)
                                      #target_transform=transforms.Compose(target_transform_list))
        test_dataset = datasets.SVHN(path, split='test', download=True,
                                     transform=transforms.Compose(transform_list),
                                     target_transform=target_transform)
                                     #target_transform=transforms.Compose(target_transform_list))
        return train_dataset, test_dataset
