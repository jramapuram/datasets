import os
import functools
from torchvision import datasets

from .abstract_dataset import AbstractLoader


class ImageFolderLoader(AbstractLoader):
    """Simple pytorch image-folder loader."""

    def __init__(self, path, batch_size, num_replicas=0,
                 train_sampler=None, test_sampler=None, valid_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 valid_transform=None, valid_target_transform=None,
                 cuda=True, **kwargs):
        # Curry the train and test dataset generators.
        train_generator = functools.partial(datasets.ImageFolder, root=os.path.join(path, 'train'))
        test_generator = functools.partial(datasets.ImageFolder, root=os.path.join(path, 'test'))
        valid_generator = None
        if os.path.isdir(os.path.join(path, 'valid')):
            valid_generator = functools.partial(datasets.ImageFolder, root=os.path.join(path, 'valid'))

        super(ImageFolderLoader, self).__init__(batch_size=batch_size,
                                                train_dataset_generator=train_generator,
                                                test_dataset_generator=test_generator,
                                                valid_dataset_generator=valid_generator,
                                                train_sampler=train_sampler,
                                                test_sampler=test_sampler,
                                                valid_sampler=valid_sampler,
                                                train_transform=train_transform,
                                                train_target_transform=train_target_transform,
                                                test_transform=test_transform,
                                                test_target_transform=test_target_transform,
                                                valid_transform=valid_transform,
                                                valid_target_transform=valid_target_transform,
                                                num_replicas=num_replicas, cuda=cuda, **kwargs)

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.input_shape = list(test_img.size()[1:])
        print("derived image shape = ", self.input_shape)

        # derive the output size using the imagefolder attr
        self.loss_type = 'ce'  # TODO: how to incorporate other features?
        self.output_size = len(self.train_loader.dataset.classes)
        print("derived output size = ", self.output_size)
