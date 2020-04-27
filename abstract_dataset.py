import torch
import functools
import numpy as np
from torchvision import transforms

from .utils import create_loader
from .samplers import GeneralDistributedSampler


def get_worker_init_fn(seed=42, same_seed_workers=False):
    """Helper to determine the worker_init_fn for dataloaders.

    :param seed: a seed provided by the user
    :param same_seed_workers: whether to set the same seed
    :returns: curried fn
    :rtype: lambda wid: set_seed(wid)

    """
    if same_seed_workers:

        def different_seed(seed, wid):
            np.random.seed(seed + wid)
            torch.cuda.manual_seed(seed + wid)

        def same_seed(sedd, wid):
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)

        if same_seed_workers:
            return functools.partial(same_seed_workers, seed=seed)

        return functools.partial(different_seed, seed=seed)


class AbstractLoader(object):
    def __init__(self, batch_size: int,
                 train_dataset_generator,
                 test_dataset_generator,
                 valid_dataset_generator=None,
                 train_sampler=None, test_sampler=None, valid_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 valid_transform=None, valid_target_transform=None,
                 workers_per_replica: int = 2, num_replicas: int = 0, seed: int = 42,
                 same_seed_workers: bool = False, timeout: int = 0,
                 pin_memory: bool = True, drop_last: bool = True, cuda: bool = False, **kwargs):
        """Load a dataset and wrap with loaders.

        :param batch_size: total batch size to use for datasets, auto-divided for dist-data-parallel
        :param train_dataset_generator: curried generator, accepts only transform/targ transform
        :param test_dataset_generator: curried generator, accepts only transform/targ transform
        :param valid_dataset_generator: [optional] curried generator, accepts only transform/targ transform
        :param train_sampler: train loader sampler
        :param test_sampler: test loader sampler
        :param valid_sampler: [optional] valid loader sampler
        :param train_transform: [optional] list of transforms for training
        :param train_target_transform: [optional] list of target transforms for training
        :param test_transform: [optional] list of transforms for testing
        :param test_target_transform: [optional] list of target transforms for testing
        :param valid_transform: [optional] list of transforms for validation
        :param valid_target_transform: [optional] list of target transforms for validation
        :param workers_per_replica: number of workers per replica
        :param num_replicas: number of replica devices
        :param same_seed_workers: seed the workers with the same seed?
        :param timeout: timeout for workers
        :param pin_memory: pin memory for GPU
        :param drop_last: create equal sized batches
        :param cuda: use cuda or not
        :returns: object with two (or three) internal dataloaders.
        :rtype: object

        """
        self.train_sampler = train_sampler    # Keep the samplers as members for set_epoch
        self.test_sampler = test_sampler
        self.valid_sampler = valid_sampler
        self.is_distributed = num_replicas > 1

        # Get the raw torch datasets
        train_dataset = self.get_dataset(train_dataset_generator,
                                         train_transform,
                                         train_target_transform)
        test_dataset = self.get_dataset(test_dataset_generator,
                                        test_transform,
                                        test_target_transform)
        valid_dataset = None
        if valid_dataset_generator is not None:
            valid_dataset = self.get_dataset(valid_dataset_generator,
                                             valid_transform,
                                             valid_target_transform)

        # If we are distributed handle RNG
        worker_init_fn = get_worker_init_fn(seed=seed, same_seed_workers=same_seed_workers) \
            if num_replicas > 0 else None

        # Build the distributed data samplers
        if self.is_distributed:
            assert train_sampler is None, "Can't use a custom train sampler with distributed-multiprocess."
            self.train_sampler = GeneralDistributedSampler(train_dataset)
            # TODO(jramapuram): setup logic for distributed testing
            # assert test_sampler is None, "Can't use a custom test sampler with distributed-multiprocess."
            # self.test_sampler = GeneralDistributedSampler(train_dataset, shuffle=False, pad=False)  # don't pad test
            if valid_dataset:
                assert valid_sampler is None, "Can't use a custom valid sampler with distributed-multiprocess."
                self.valid_sampler = GeneralDistributedSampler(valid_dataset)

        # Wrap the dataset with a loader with the (common) args below
        loader_kwargs = {'num_workers': workers_per_replica,
                         'pin_memory': pin_memory & cuda,
                         'worker_init_fn': worker_init_fn,
                         'timeout': timeout,
                         'drop_last': drop_last}
        self.train_loader = create_loader(dataset=train_dataset,
                                          sampler=self.train_sampler,
                                          batch_size=batch_size,
                                          shuffle=True if self.train_sampler is None else False,
                                          **loader_kwargs)
        assert len(self.train_loader.dataset) >= batch_size, "train-set has {} samples but {} bs requested.".format(
            len(self.train_loader.dataset), batch_size)

        self.test_loader = create_loader(dataset=test_dataset,
                                         sampler=self.test_sampler,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         **loader_kwargs)
        assert len(self.test_loader.dataset) >= batch_size, "test-set has {} samples but {} bs requested.".format(
            len(self.test_loader.dataset), batch_size)

        self.valid_loader = None
        if valid_dataset is not None:
            self.valid_loader = create_loader(dataset=valid_dataset,
                                              sampler=self.valid_sampler,
                                              batch_size=batch_size,
                                              shuffle=True if self.train_sampler is None else False,
                                              **loader_kwargs)
            assert len(self.valid_loader.dataset) >= batch_size, "valid-set has {} samples but {} bs requested.".format(
                len(self.valid_loader.dataset), batch_size)

        # Set the dataset lengths if they exist.
        self.num_train_samples = len(train_dataset)
        self.num_test_samples = len(test_dataset)
        self.num_valid_samples = len(valid_dataset) if valid_dataset is not None else 0
        print("train = {} | test = {} | valid = {}".format(
            self.num_train_samples, self.num_test_samples, self.num_valid_samples))

        # these need to be filled by the dataloader
        self.loss_type = None
        self.input_shape = None
        self.output_size = 0
        self.batch_size = batch_size

    def set_epoch(self, epoch, split):
        """Sets the epoch for fixed RNG shuffling for distributed multi-processing.

        :param epoch: the current epoch
        :param split: train test or valid
        :returns: nothing, but sets the sampler epoch internally
        :rtype: None

        """
        if self.is_distributed:  # Only distributed settings require setting the epoch
            split_map = {'train': self.train_sampler, 'test': self.test_sampler, 'valid': self.valid_sampler}
            split_map[split].set_epoch(epoch)

    def set_all_epochs(self, epoch):
        """Syntactic sugar to set all loader epochs."""
        if self.is_distributed:  # Only distributed settings require setting the epoch
            # splits = ['train', 'test', 'valid'] if self.valid_loader is not None else ['train', 'test']
            splits = ['train', 'valid'] if self.valid_loader is not None else ['train']
            for s in splits:
                self.set_epoch(epoch=epoch, split=s)

    def determine_output_size(self):
        """Iterate dataset to find the maximum output size."""
        for _, label in self.train_loader:
            if not isinstance(label, (float, int)) and len(label) > 1:
                lbl = np.array(label).max()
                if lbl > self.output_size:
                    self.output_size = lbl
            else:
                lbl = label.max().item()
                if lbl > self.output_size:
                    self.output_size = lbl

        self.output_size = self.output_size + 1
        print("auto-determined label size: ", self.output_size)
        return self.output_size

    def compose_transforms(self, transform):
        """Helper to compose a list of transforms while tentatively addding ToTensor() to it."""
        transform_list = []
        if transform:
            assert isinstance(transform, list), "transforms need to be in a list"
            transform_list.extend(transform)

        # add ToTensor if it isn't there
        transform_names = [str(tt) for tt in transform_list]
        if 'ToTensor()' not in transform_names:
            transform_list.append(transforms.ToTensor())

        return transforms.Compose(transform_list)

    def get_dataset(self, dataset_generator, transform=None, target_transform=None):
        """Helper to constrct the dataset object.

        :param dataset_generator: curried dataset generator, accepts transform and target_transform
        :param transform: the transform list for this dataset split
        :param target_transform: the transform list for the labels for this dataset split
        :returns: a torch dataset
        :rtype: torch.data.Dataset

        """
        target_transform_list = []
        if target_transform:
            assert isinstance(target_transform, list), "target_transforms need to be in a list"
            target_transform_list.extend(target_transform)

        # Build a torchvision Composed set of transforms
        transform = self.compose_transforms(transform)
        target_transform = transforms.Compose(target_transform_list)

        # build the dataset objects
        dataset = dataset_generator(transform=transform, target_transform=target_transform)
        return dataset
