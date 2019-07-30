import os
import torch
import numpy as np
import torchvision.transforms.functional as F

from sklearn.preprocessing import LabelBinarizer
from torchvision import datasets, transforms


from .abstract_dataset import AbstractLoader
from .utils import create_loader


def generate_samples(num_samples, seq_len, output_size, max_digit):
    """ Helper to generate sampels between 0 and max_digit

    :param num_samples: the total number of samples to generate
    :param seq_len: length of each sequence
    :param output_size: the output size
    :param max_digit: the upper bound in the uniform distribution
    :returns: [B, seq_len*output_size]
    :rtype: torch.Tensor, torch.Tensor

    """
    data = np.random.uniform(0, max_digit, size=[num_samples, seq_len*output_size])
    labels = np.argsort(data, axis=-1)
    data = data.reshape(num_samples, seq_len, output_size)
    labels = labels.reshape(num_samples, seq_len*output_size)
    print('labels = ', labels.shape, " | data = ", data.shape)
    return [data.astype(np.float32), labels]


def get_output_size(**kwargs):
    return 1 if 'output_size' not in kwargs else kwargs['output_size']


class SortDataset(torch.utils.data.Dataset):
    def __init__(self, path, split='train', train=True, download=True,
                 transform=None, target_transform=None, **kwargs):
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # set the output size of sort to 512 if it is not set
        self.output_size = get_output_size(**kwargs)

        # set the number of samples to 2 million by default
        train_samples = 2000000 if 'num_samples' not in kwargs else kwargs['num_samples']
        self.num_samples = train_samples if split == 'train' else int(train_samples*0.2)

        # max sorting range U ~ [0, max_digit]
        self.max_digit = 1 if 'max_digit' not in kwargs else kwargs['max_digit']

        # set the sequence length if it isn't specified
        self.sequence_length = 10 if 'sequence_length' not in kwargs else kwargs['sequence_length']

        # load the sort dataset and labels
        self.data, self.labels = generate_samples(self.num_samples,
                                                  self.sequence_length,
                                                  self.output_size,
                                                  self.max_digit)
        print("[{}] {} samples".format(split, len(self.labels)))

    def __getitem__(self, index):
        target = self.labels[index]
        data = self.data[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)

        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target)

        return data, target

    def __len__(self):
        return len(self.labels)


class EmptyToTensor(object):
    ''' hack of ToTensor: since it is checked in superclass '''
    def __repr__(self):
        return 'ToTensor()'

    def __call__(self, x):
        return x


class SortLoader(AbstractLoader):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1, **kwargs):
        if isinstance(transform, list): # hack to do ToTensor()
            transform.extend(EmptyToTensor())
        else:
            transform = [EmptyToTensor()]

        # use the abstract class to build the loader with the above
        super(SortLoader, self).__init__(SortDataset, path=path,
                                         batch_size=batch_size,
                                         train_sampler=train_sampler,
                                         test_sampler=test_sampler,
                                         transform=transform,
                                         target_transform=target_transform,
                                         use_cuda=use_cuda, **kwargs)
        self.train_sequence_length, self.img_shp = self._get_seqlen_and_output_size(self.train_loader)
        self.test_sequence_length, _  = self._get_seqlen_and_output_size(self.test_loader)
        self.output_size = get_output_size(**kwargs) * self.train_sequence_length
        self.loss_type = 'l2' # fixed
        print("derived output_size = ", self.output_size)
        print("derived train_sequence_length = ", self.train_sequence_length)
        print("derived test_sequence_ength = ", self.test_sequence_length)

    def _get_seqlen_and_output_size(self, loader):
        """ Helper to get the lengths of train / test and output size

        :param loader: the dataloader
        :returns: seq_len and feature_size
        :rtype: int, int

        """
        for data, label in loader:
            assert len(data.shape) == 3
            _, seq_len, feature_size = data.shape
            return seq_len, feature_size
