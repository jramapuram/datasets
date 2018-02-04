import torch
import numpy as np
from torchvision import datasets, transforms
from datasets.cifar import CIFAR10Loader
from datasets.fashion_mnist import FashionMNISTLoader
from datasets.mnist_cluttered import ClutteredMNISTLoader
from datasets.mnist import MNISTLoader
from datasets.svhn import SVHNCenteredLoader, SVHNFullLoader
from datasets.utils import bw_2_rgb_lambda, resize_lambda

class MergedLoader(object):
    def __init__(self, datasets_list, path, batch_size,
                 train_sampler=None, test_sampler=None,
                 resize=[32, 32], convert_to_rgb=True,
                 use_cuda=1):
        dataset_map = {
            'mnist': MNISTLoader.get_datasets,
            'fashion': FashionMNISTLoader.get_datasets,
            'cifar10': CIFAR10Loader.get_datasets,
            'svhn_centered': SVHNCenteredLoader.get_datasets,
            'svhn_full': SVHNFullLoader.get_datasets,
            'clutter': ClutteredMNISTLoader.get_datasets
        }

        transform_list = []
        if resize:
            l = lambda img, sz=resize : resize_lambda(img, size=resize)
            transform_list.append(
                transforms.Lambda(l)
            )

        if convert_to_rgb:
            transform_list.append(transforms.Lambda(bw_2_rgb_lambda))

        if not resize and not convert_to_rgb:
            transform_list = None

        dataset_list \
            = [dataset_map[dataset_name](path=path, transform=transform_list)
               for dataset_name in datasets_list]

        # merge the datasets using the + operator
        train_dataset = dataset_list[0][0]
        test_dataset = dataset_list[0][1]
        for train, test in dataset_list:
            train_dataset += train
            test_dataset += test

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_sampler = train_sampler(train_dataset) if train_sampler else None
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=not train_sampler,
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
