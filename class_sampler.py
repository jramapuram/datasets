import math
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler


class ClassSampler(Sampler):
    """Sampler that restricts data loading to a single class of the dataset.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        class_number: The class index to filter out
    """

    def __init__(self, dataset, class_number):
        assert class_number is not None
        self.dataset = dataset
        self.class_number = class_number
        self.indices, self.num_samples = self._calc_indices(dataset, class_number)

    @staticmethod
    def _calc_indices(dataset, class_number):
        indices = []
        for i, (_, target) in enumerate(dataset):
            if target == class_number:
                indices.append(i)

        return indices, len(indices)

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
