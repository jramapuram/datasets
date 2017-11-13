from __future__ import print_function
import torch
from torchvision import datasets, transforms

from datasets.svhn_full import SVHNFull


class SubtractOneLambda(object):
    ''' simple lambda to subtract one from labels'''
    def  __call__(self, label):
        return label - 1


class SVHNFullLoader(object):
    ''' This loads the original SVHN dataset (non-centered).

        The classes here are BCE classes where each bit
        signifies the presence of the digit in the img'''

    def __init__(self, path, batch_size, use_cuda=1):
        kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                         std=(0.5, 0.5, 0.5))

        self.train_loader = torch.utils.data.DataLoader(
            SVHNFull(path, split='train', download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         normalize
                         #transforms.Normalize((0.45142,), (0.039718,))
                     ]),
                     target_transform=transforms.Lambda(SubtractOneLambda())),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True, **kwargs)
        print("successfully loaded SVHN training data...")

        self.test_loader = torch.utils.data.DataLoader(
            SVHNFull(path, split='test', download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         normalize
                         #transforms.Normalize((0.45142,), (0.039718,))
                     ]),
                     target_transform=transforms.Lambda(SubtractOneLambda())),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True, **kwargs)
        print("successfully loaded SVHN test data...")

        self.output_size = 10
        self.batch_size = batch_size
        self.img_shp = [3, 32, 32]


class SVHNCenteredLoader(object):
    def __init__(self, path, batch_size, use_cuda=1):
        kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                         std=(0.5, 0.5, 0.5))

        self.train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(path, split='train', download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              normalize
                              #transforms.Normalize((0.45142,), (0.039718,))
                          ])),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True, **kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(path, split='test', download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              normalize
                              #transforms.Normalize((0.45142,), (0.039718,))
                          ])),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True, **kwargs)

        self.output_size = 10
        self.batch_size = batch_size
        self.img_shp = [3, 32, 32]
