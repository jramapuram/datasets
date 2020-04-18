import torch
import functools
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from .grid_generator import SampleSpec
from ..utils import temp_seed
from ..abstract_dataset import AbstractLoader

DEFAULT_PAIRS = 4
DEFAULT_CLASSES = 4
sample_spec = SampleSpec(num_pairs=DEFAULT_PAIRS, num_classes=DEFAULT_CLASSES, im_dim=76, min_cell=15, max_cell=18)


def set_sample_spec(num_pairs, num_classes, reset_every=None, im_dim=76):
    global sample_spec
    assert sample_spec.generators is None, 'attempting to redefine spec after it has been used'
    sample_spec = SampleSpec(num_pairs=num_pairs, num_classes=num_classes, im_dim=im_dim,
                             min_cell=15, max_cell=18, reset_every=reset_every)


def numpy_to_PIL(img):
    """Simple helper to converty a numpy image to PIL for transform operations."""
    return Image.fromarray(np.uint8(img.squeeze()*255))


class ToTensor(object):
    """simple override to add context to ToTensor"""
    def __init__(self, numpy_base_type=np.float32):
        self.numpy_base_type = numpy_base_type

    def __call__(self, img):
        result = img.astype(self.numpy_base_type)
        result = np.expand_dims(result, axis=0)
        result = torch.from_numpy(result)
        return result


class LenImplementor(object):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


class OnlineGridGenerator(Dataset):
    def __init__(self, batch_size, batches_per_epoch,
                 transform=None,
                 target_transform=None):
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.current_batch = 0
        self.transform = transform
        self.target_transform = target_transform

        # lots of other code does len(dataloader.dataset) to get size,
        self.dataset = LenImplementor(batches_per_epoch * batch_size)

    def __len__(self):
        return self.batches_per_epoch * self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        # Python 3 compatibility
        return self.next()

    def __getitem__(self, index):
        return self.next()

    def next(self):
        self.current_batch += 1
        img, target = sample_spec.generate(1)
        img, target = np.squeeze(img), np.squeeze(target)
        img = numpy_to_PIL(img)

        # apply our input transform
        if self.transform is not None:
            for transform in self.transform:
                if transform is not None:
                    img = transform(img)

        # apply our target transform
        if self.target_transform is not None:
            for transform in self.target_transform:
                if transform is not None:
                    target = transform(target)

        return img, target


class FixedGridGenerator(Dataset):
    """Simple helper to generate a fixed size dataset."""
    def __init__(self, total_samples, transform, target_transform):
        self.transform = transform
        self.target_transform = target_transform
        self.total_samples = total_samples

        # generate all test samples independently and store away
        print("starting fixed generation of %d samples. This might take a while..." % total_samples)
        with temp_seed(123456):
            self.dataset, self.labels, stats = sample_spec.blocking_generate_with_stats(total_samples)
            print("dataset = ", self.dataset.shape)
            self.dataset = [numpy_to_PIL(ti) for ti in self.dataset]
            print("generator retry rate = {}".format(stats['num_retries'] / float(stats['num_generated'])))
            print("test samples successfully generated...")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        target = self.labels[index]
        img = numpy_to_PIL(self.dataset[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = target.astype(np.int64)
        return img, target


class GridDataLoader(AbstractLoader):
    """Simple all-pairs loader, there is no validation set (can be easily done)."""

    def __init__(self, path, batch_size, num_replicas=0,
                 train_sampler=None, test_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 cuda=True, batches_per_epoch=500, test_frac=0.2, **kwargs):

        # Curry the train and test dataset generators.
        train_generator = functools.partial(
            OnlineGridGenerator, batch_size=batch_size, batches_per_epoch=batches_per_epoch)
        test_generator = functools.partial(
            FixedGridGenerator, total_samples=int(batches_per_epoch * batch_size * test_frac))

        # kwargs['num_workers'] = 0  # Over-ride this for the loaders
        super(GridDataLoader, self).__init__(batch_size=batch_size,
                                             train_dataset_generator=train_generator,
                                             test_dataset_generator=test_generator,
                                             train_sampler=train_sampler,
                                             test_sampler=test_sampler,
                                             train_transform=train_transform,
                                             train_target_transform=train_target_transform,
                                             test_transform=test_transform,
                                             test_target_transform=test_target_transform,
                                             num_replicas=num_replicas, cuda=cuda, **kwargs)
        self.output_size = 2   # fixed
        self.loss_type = 'ce'  # fixed

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.img_shp = list(test_img.size()[1:])
        print("derived image shape = ", self.img_shp)
