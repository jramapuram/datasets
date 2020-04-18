import math
import torch
import numpy as np
import torch.distributed as dist

from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import Sampler

import datasets.utils as utils


class FixedRandomSampler(Sampler):
    """Does a SINGLE fixed random transform of the dataset."""

    def __init__(self, data_source):
        self.data_source = data_source
        with utils.temp_seed(1234):
            self.fixed_perm = np.random.permutation(len(self.data_source))

    def __iter__(self):
        return iter(self.fixed_perm)

    def __len__(self):
        return len(self.data_source)


class ClassSampler(Sampler):
    """Sampler that restricts data loading to a single class of the dataset.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        class_number: The class index to filter out.
                      This can be a list as well to handle
                      multiple classes.
    """

    def __init__(self, class_number, shuffle=True):
        assert class_number is not None
        self.class_number = class_number
        self.shuffle = shuffle

    def __call__(self, dataset, class_number=None):
        ''' helps to recompute indices '''
        if class_number is None:
            class_number = self.class_number

        # if we receive a list, then iterate over this sequentially
        if isinstance(class_number, list):
            self.indices = []
            self.num_samples = 0
            for cn in class_number:
                indices, num_samples = self._calc_indices(dataset, cn)
                self.indices += indices
                self.num_samples += num_samples
        else:
            self.indices, self.num_samples = self._calc_indices(dataset, class_number)

        # DEBUG print:
        # print("#indices for {} = {} | dataset = {}".format(self.class_number,
        #                                                    len(self.indices),
        #                                                    len(self.dataset)))

        # set the current dataset as a subset
        self.dataset = Subset(dataset, self.indices)
        return self.dataset

    @staticmethod
    def _calc_indices(dataset, class_number):
        indices = [i for i, (_, target) in enumerate(dataset) if target == class_number]
        return indices, len(indices)

    def __iter__(self):
        assert hasattr(self, 'indices'), "need to run __call__() on ClassSampler first"
        if self.shuffle:
            return (self.indices[i] for i in torch.randperm(len(self.indices)))

        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return self.num_samples


class GeneralDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
       Sourced from https://bit.ly/3eq7MP9 to enable padding.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        pad: pad data by replicating samples
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, pad=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.pad = pad
        self.epoch = 0
        self.shuffle = shuffle
        if self.pad:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.num_samples = int(math.ceil((len(self.dataset) - self.rank) * 1.0 / self.num_replicas))
            self.total_size = len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        if self.pad:
            indices += indices[:(self.total_size - len(indices))]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
