import math
import torch
import numpy as np
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler


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

    def __init__(self, dataset, class_number):
        assert class_number is not None
        self.class_number = class_number

        # place in call to provide compatibility
        self.__call__(dataset, class_number=class_number)

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

        # set the current dataset as a subset
        self.dataset = Subset(dataset, self.indices)
        # print("#indices for {} = {} | dataset = {}".format(self.class_number,
        #                                                    len(self.indices),
        #                                                    len(self.dataset)))

    @staticmethod
    def _calc_indices(dataset, class_number):
        indices = [i for i, (_, target) in enumerate(dataset) if target == class_number]
        return indices, len(indices)

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
