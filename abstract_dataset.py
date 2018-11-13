import torch
import numpy as np
from scipy.misc import imread, imresize
from torchvision import datasets, transforms

from .utils import binarize, create_loader


class AbstractLoader(object):
    def __init__(self, dataset_generator, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, binarize_images=True, use_cuda=1, **kwargs):
        ''' base class, set dataset_generator to be the DataSet, eg: datasets.MNIST '''
        self.dataset_generator = dataset_generator
        train_dataset, test_dataset = self.get_datasets(path, transform,
                                                        target_transform,
                                                        **kwargs)

        # build the loaders to wrap the datasets
        num_workers = kwargs['num_workers'] if 'num_workers' in kwargs \
            and kwargs['num_workers'] is not None else 8
        pin_memory = kwargs['pin_memory'] if 'pin_memory' in kwargs \
            and kwargs['pin_memory'] is not None else True
        kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
        self.train_loader = create_loader(train_dataset,
                                          train_sampler,
                                          batch_size,
                                          shuffle=True if train_sampler is None else False,
                                          **kwargs)
        self.test_loader = create_loader(test_dataset,
                                         test_sampler,
                                         batch_size,
                                         shuffle=False,
                                         **kwargs)

        # these need to be filled by the dataloader
        self.loss_type = None
        self.img_shp = None
        self.output_size = 0
        self.batch_size = batch_size


    @staticmethod
    def determine_output_size(train_loader, **kwargs):
        # determine output size
        if 'output_size' not in kwargs or kwargs['output_size'] is None:
            for _, label in self.train_loader:
                if not isinstance(label, (float, int)) and len(label) > 1:
                    l = np.array(label).max()
                    if l > self.output_size:
                        self.output_size = l
                else:
                    l = label.max().item()
                    if l > self.output_size:
                        self.output_size = l

            self.output_size = self.output_size + 1
        else:
            self.output_size = kwargs['output_size']

        print("determined output_size: ", self.output_size)

    def get_datasets(self, path, transform=None, target_transform=None, **kwargs):
        if transform:
            assert isinstance(transform, list), "transforms need to be in a list"

        transform_list = []
        if transform:
            transform_list.extend(transform)

        # add ToTensor if it isn't there
        transform_names = [str(tt) for tt in transform_list]
        if 'ToTensor()' not in transform_names:
            transform_list.append(transforms.ToTensor())

        # build the dataset objects
        train_dataset = self.dataset_generator(path=path, train=True, split='train',
                                               transform=transforms.Compose(transform_list),
                                               target_transform=target_transform, **kwargs)
        test_dataset = self.dataset_generator(path=path, train=False, split='test',
                                              transform=transforms.Compose(transform_list),
                                              target_transform=target_transform, **kwargs)
        return train_dataset, test_dataset
