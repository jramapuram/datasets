import functools
from torchvision import datasets

from .abstract_dataset import AbstractLoader


class FashionMNISTLoader(AbstractLoader):
    """Simple FashionMNISTLoader loader, there is no validation set."""

    def __init__(self, path, batch_size, num_replicas=1,
                 train_sampler=None, test_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 cuda=True, **kwargs):

        # Curry the train and test dataset generators.
        train_generator = functools.partial(datasets.FashionMNIST, root=path, train=True, download=True)
        test_generator = functools.partial(datasets.FashionMNIST, root=path, train=False, download=True)

        super(FashionMNISTLoader, self).__init__(batch_size=batch_size,
                                                 train_dataset_generator=train_generator,
                                                 test_dataset_generator=test_generator,
                                                 train_sampler=train_sampler,
                                                 test_sampler=test_sampler,
                                                 train_transform=train_transform,
                                                 train_target_transform=train_target_transform,
                                                 test_transform=test_transform,
                                                 test_target_transform=test_target_transform,
                                                 num_replicas=num_replicas, cuda=cuda, **kwargs)
        self.output_size = 10  # fixed
        self.loss_type = 'ce'  # fixed

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.input_shape = list(test_img.size()[1:])
        print("derived image shape = ", self.input_shape)
