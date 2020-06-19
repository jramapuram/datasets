import os
import functools
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from .abstract_dataset import AbstractLoader


class ImageFolderLoader(AbstractLoader):
    """Simple pytorch image-folder loader."""

    def __init__(self, path, batch_size, num_replicas=1,
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


class MultiAugmentImageDataset(datasets.ImageFolder):
    """Extends imagefolder to simply augment the same image num_augments times."""

    def __init__(self, root, transform=None, target_transform=None, non_augmented_transform=None,
                 loader=default_loader, is_valid_file=None, num_augments=2):
        assert num_augments > 1, "Use this dataset when you want >1 augmentations"
        self.num_augments = num_augments  # Number of times to augment the same image
        self.non_augment_transform = non_augmented_transform  # transform for non-augmented image (eg: resize)

        super(MultiAugmentImageDataset, self).__init__(
            root=root, transform=transform, target_transform=target_transform,
            loader=default_loader, is_valid_file=is_valid_file)

    def __getitem_non_transformed__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.non_augment_transform is not None:
            sample = self.non_augment_transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __getitem__(self, index):
        """Label is the same for index, so just run augmentations again."""
        sample0, target = self.__getitem_non_transformed__(index)
        samples = [sample0] + [super(MultiAugmentImageDataset, self).__getitem__(index)[0]
                               for _ in range(self.num_augments)]
        return samples + [target]


class MultiAugmentImageFolder(AbstractLoader):
    """Runs multiple augmentations PER image and returns."""

    def __init__(self, path, batch_size, num_replicas=1,
                 train_sampler=None, test_sampler=None, valid_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 valid_transform=None, valid_target_transform=None,
                 non_augmented_transform=None,  # The first image returned is non-augmented, useful for resize, etc.
                 cuda=True, num_augments=2, **kwargs):
        # Curry the train and test dataset generators.
        train_generator = functools.partial(MultiAugmentImageDataset,
                                            root=os.path.join(path, 'train'),
                                            non_augmented_transform=self.compose_transforms(non_augmented_transform),
                                            num_augments=num_augments)
        test_generator = functools.partial(MultiAugmentImageDataset,
                                           root=os.path.join(path, 'test'),
                                           non_augmented_transform=self.compose_transforms(non_augmented_transform),
                                           num_augments=num_augments)
        valid_generator = None
        if os.path.isdir(os.path.join(path, 'valid')):
            valid_generator = functools.partial(MultiAugmentImageDataset,
                                                root=os.path.join(path, 'valid'),
                                                num_augments=num_augments)

        super(MultiAugmentImageFolder, self).__init__(batch_size=batch_size,
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
        train_samples_and_labels = self.train_loader.__iter__().__next__()
        self.input_shape = list(train_samples_and_labels[0].size()[1:])
        print("derived image shape = ", self.input_shape)

        # derive the output size using the imagefolder attr
        self.loss_type = 'ce'  # TODO: how to incorporate other features?
        self.output_size = len(self.train_loader.dataset.classes)
        print("derived output size = ", self.output_size)
