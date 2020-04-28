import functools
from torchvision import datasets, transforms

from .svhn_full import SVHNFull
from .abstract_dataset import AbstractLoader


class SVHNFullLoader(AbstractLoader):
    """Full SVHN loader (non-centered), there is no validation set."""

    def __init__(self, path, batch_size, num_replicas=1,
                 train_sampler=None, test_sampler=None, valid_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 valid_transform=None, valid_target_transform=None,
                 cuda=True, **kwargs):

        # Curry the train and test dataset generators.
        train_generator = functools.partial(SVHNFull, root=path, split='train', download=True)
        test_generator = functools.partial(SVHNFull, root=path, split='test', download=True)
        valid_generator = functools.partial(SVHNFull, root=path, split='extra', download=True)

        # Fix the stupid SVHN index from 1 issue
        offset_transform = transforms.Lambda(lambda lbl: lbl - 1)

        def _extend_or_append_xform(target_transform):
            if target_transform is not None:
                target_transform.extend(offset_transform)
            else:
                target_transform = [offset_transform]

            return target_transform

        train_target_transform = _extend_or_append_xform(train_target_transform)
        test_target_transform = _extend_or_append_xform(test_target_transform)
        valid_target_transform = _extend_or_append_xform(valid_target_transform)

        super(SVHNFullLoader, self).__init__(batch_size=batch_size,
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
        self.output_size = 10  # fixed
        self.loss_type = 'ce'  # fixed

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.input_shape = list(test_img.size()[1:])
        print("derived image shape = ", self.input_shape)


class SVHNCenteredLoader(AbstractLoader):
    """Simple centered SVHN loader (the default), there is no validation set."""

    def __init__(self, path, batch_size, num_replicas=1,
                 train_sampler=None, test_sampler=None, valid_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 valid_transform=None, valid_target_transform=None,
                 cuda=True, **kwargs):

        # Curry the train and test dataset generators.
        train_generator = functools.partial(datasets.SVHN, root=path, split='train', download=True)
        test_generator = functools.partial(datasets.SVHN, root=path, split='test', download=True)
        valid_generator = functools.partial(datasets.SVHN, root=path, split='extra', download=True)

        # Fix the stupid SVHN index from 1 issue
        offset_transform = transforms.Lambda(lambda lbl: lbl - 1)

        def _extend_or_append_xform(target_transform):
            if target_transform is not None:
                target_transform.extend(offset_transform)
            else:
                target_transform = [offset_transform]

            return target_transform

        train_target_transform = _extend_or_append_xform(train_target_transform)
        test_target_transform = _extend_or_append_xform(test_target_transform)
        valid_target_transform = _extend_or_append_xform(valid_target_transform)

        super(SVHNCenteredLoader, self).__init__(batch_size=batch_size,
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
        self.output_size = 10  # fixed
        self.loss_type = 'ce'  # fixed

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.input_shape = list(test_img.size()[1:])
        print("derived image shape = ", self.input_shape)
