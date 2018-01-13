import torch
import numpy as np
from scipy.misc import imread, imresize
from torchvision import datasets, transforms

def resize_lambda(img, size=(64, 64)):
    return imresize(img, size)

def bw_2_rgb_lambda(img):
    expanded = np.expand_dims(img, -1)
    return  np.concatenate([expanded, expanded, expanded], axis=-1)

class CIFAR10Loader(object):
    def __init__(self, path, batch_size, sampler=None, use_cuda=1):
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_dataset = datasets.CIFAR10(path, train=True, download=True,
                                         transform=transforms.Compose([
                                             #transforms.Lambda(resize_lambda),
                                             #transforms.Lambda(bw_2_rgb_lambda),
                                             transforms.ToTensor(),
                                             # transforms.Normalize((0.1307,), (0.3081,))
                                         ]))
        train_sampler = sampler(train_dataset) if sampler else None
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=not sampler,
            sampler=train_sampler
            **kwargs)

        test_dataset = datasets.CIFAR10(path, train=False,
                                        transform=transforms.Compose([
                                            #transforms.Lambda(resize_lambda),
                                            #transforms.Lambda(bw_2_rgb_lambda),
                                            transforms.ToTensor(),
                                            #transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
        test_sampler = sampler(test_dataset) if sampler else None
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=not sampler,
            sampler=test_sampler,
            **kwargs)

        self.output_size = 10
        self.batch_size = batch_size

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        #print('test img = ', test_img.shape)
        self.img_shp = list(test_img.size()[1:])
        #print("derived image shape = ", self.img_shp)
