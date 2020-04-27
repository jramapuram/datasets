import os
import numpy as np

from datasets.all_pairs.grid_loader import GridDataLoader
from datasets.samplers import ClassSampler
from datasets.cifar import CIFAR10Loader, CIFAR100Loader
from datasets.nih_chest_xray import NIHChestXrayLoader
from datasets.starcraft_predict_battle import StarcraftPredictBattleLoader
from datasets.fashion_mnist import FashionMNISTLoader
from datasets.mnist_cluttered import ClutteredMNISTLoader
from datasets.mnist import MNISTLoader
from datasets.binarized_mnist import BinarizedMNISTLoader
from datasets.omniglot import OmniglotLoader, BinarizedOmniglotLoader, BinarizedOmniglotBurdaLoader
from datasets.permuted_mnist import PermutedMNISTLoader
from datasets.sort import SortLoader
from datasets.celeb_a import CelebALoader, CelebASequentialLoader
from datasets.svhn import SVHNCenteredLoader, SVHNFullLoader
from datasets.imagefolder import ImageFolderLoader, MultiAugmentImageFolder
from datasets.multi_imagefolder import MultiImageFolderLoader
from datasets.utils import sequential_dataset_merger

# ensures we get the same permutation
PERMUTE_SEED = 1

loader_map = {
    'all_pairs': GridDataLoader,
    'sort': SortLoader,
    'nih_chest_xray': NIHChestXrayLoader,
    'starcraft_predict_battle': StarcraftPredictBattleLoader,
    'mnist': MNISTLoader,
    'binarized_mnist': BinarizedMNISTLoader,
    'binarized_omniglot': BinarizedOmniglotLoader,
    'binarized_omniglot_burda': BinarizedOmniglotBurdaLoader,
    'celeba': CelebALoader,
    'celeba_sequential': CelebASequentialLoader,
    'omniglot': OmniglotLoader,
    'permuted': PermutedMNISTLoader,
    'fashion': FashionMNISTLoader,
    'clutter': ClutteredMNISTLoader,
    'cifar10': CIFAR10Loader,
    'cifar100': CIFAR100Loader,
    'svhn': SVHNCenteredLoader,
    'svhn_centered': SVHNCenteredLoader,
    'svhn_full': SVHNFullLoader,
    'image_folder': ImageFolderLoader,
    'multi_augment_image_folder': MultiAugmentImageFolder,  # augments same image multiple times
    'multi_image_folder': MultiImageFolderLoader,  # reads in parallel from multiple folders
}


def get_samplers(num_classes):
    ''' builds samplers taking into account previous classes'''
    # NOTE: test datasets are now merged via sequential_test_set_merger
    test_samplers = [ClassSampler(class_number=j, shuffle=False) for j in range(num_classes)]
    train_samplers = [ClassSampler(class_number=j, shuffle=True) for j in range(num_classes)]
    valid_samplers = [ClassSampler(class_number=j, shuffle=True) for j in range(num_classes)]
    return train_samplers, test_samplers, valid_samplers


def get_loader(task: str, data_dir: str, batch_size: int, cuda: bool,
               workers_per_replica: int = 2, num_replicas: int = 0, seed: int = 42,
               same_seed_workers: bool = False, timeout: int = 0, pin_memory=True, drop_last=True,
               train_transform=None, train_target_transform=None,
               test_transform=None, test_target_transform=None,
               valid_transform=None, valid_target_transform=None,
               train_sampler=None, test_sampler=None, valid_sampler=None,
               increment_seed=True, **kwargs):
    """Returns a loader with .train_loader, .test_loader and optionally .valid_loader

    :param task: string task name
    :param data_dir: string data directory
    :param batch_size: batch size for each loaders
    :param cuda: bool flag to enable pin_memory, etc
    :param workers_per_replica: number of data loader threads per worker
    :param num_replicas: used with DistributedDataParallel
    :param seed: seed for same_seed_workers, **ignored otherwise**
    :param same_seed_workers: set the same seed on the workers
    :param timeout: timeout for worker
    :param pin_memory: pin memory for CUDA; default to true
    :param drop_last: ensures equal sized minibatches
    :param train_transform: (optional) list of data transforms
    :param train_target_transform: (optional) list of label transforms
    :param test_transform: (optional) list of data transforms
    :param test_target_transform: (optional) list of label transforms
    :param valid_transform: (optional) list of data transforms
    :param valid_target_transform: (optional) list of label transforms
    :param train_sampler: (optional) data sampler
    :param test_sampler: (optional) data sampler
    :param valid_sampler: (optional) data sampler
    :param increment_seed: modifies RNG for permutation dataset
    :returns: AbstractLoader instance
    :rtype:

    """
    global PERMUTE_SEED

    # overwrite data dir for fashion MNIST because it has
    # issues being in the same directory as regular MNIST
    if task == 'fashion':
        data_dir = os.path.join(data_dir, "fashion")
    else:
        data_dir = data_dir

    if '+' in task:  # the merge operand
        loaders = []
        for split in task.split('+'):
            loaders.append(get_loader(task=split,
                                      data_dir=data_dir,
                                      batch_size=batch_size,
                                      cuda=cuda,
                                      workers_per_replica=workers_per_replica,
                                      num_replicas=num_replicas,
                                      seed=seed,
                                      same_selfeed_workers=same_seed_workers,
                                      timeout=timeout,
                                      pin_memory=pin_memory,
                                      drop_last=drop_last,
                                      train_transform=train_transform,
                                      train_target_transform=train_target_transform,
                                      test_transform=test_transform,
                                      test_target_transform=test_target_transform,
                                      valid_transform=valid_transform,
                                      valid_target_transform=valid_target_transform,
                                      train_sampler=train_sampler,
                                      test_sampler=test_sampler,
                                      **kwargs))
            if increment_seed:
                PERMUTE_SEED += 1

        PERMUTE_SEED = 1  # Reset global seed here
        has_valid = np.all([hasattr(l, 'valid_loader') for l in loaders])
        splits = ['train', 'test'] if not has_valid else ['train', 'test', 'valid']
        for split in splits:
            loaders = sequential_dataset_merger(
                loaders, split, fixed_shuffle=(split == 'test'))  # fixed shuffle test set

        loader = loaders[-1]
    else:
        assert task in loader_map, "unknown task requested"
        if task == 'permuted':
            kwargs['seed'] = PERMUTE_SEED
        elif task == 'crop_dual_imagefolder':
            # Lazy load this because of PYVIPS issues.
            from datasets.crop_dual_imagefolder import CropDualImageFolderLoader
            loader_map['crop_dual_imagefolder'] = CropDualImageFolderLoader

        return loader_map[task](path=data_dir,
                                batch_size=batch_size,
                                workers_per_replica=workers_per_replica,
                                num_replicas=num_replicas,
                                seed=seed,
                                same_seed_workers=same_seed_workers,
                                timeout=timeout,
                                pin_memory=pin_memory,
                                drop_last=drop_last,
                                train_transform=train_transform,
                                train_target_transform=train_target_transform,
                                test_transform=test_transform,
                                test_target_transform=test_target_transform,
                                valid_transform=valid_transform,
                                valid_target_transform=valid_target_transform,
                                train_sampler=train_sampler,
                                test_sampler=test_sampler,
                                valid_sampler=valid_sampler,
                                cuda=cuda,
                                **kwargs)

    return loader


def _check_for_sublist(loaders):
    is_sublist = False
    for l in loaders:
        if isinstance(l, list):
            is_sublist = True
            break

    return is_sublist


def get_split_data_loaders(task: str, num_classes: int, data_dir: str, batch_size: int, cuda: bool,
                           workers_per_replica: int = 2, num_replicas: int = 0, seed: int = 42,
                           same_seed_workers: bool = False, timeout: int = 0, pin_memory=True, drop_last=True,
                           train_transform=None, train_target_transform=None,
                           test_transform=None, test_target_transform=None,
                           valid_transform=None, valid_target_transform=None,
                           sequentially_merge_test=True, **kwargs):
    '''Splits a loader into num_classes separate loaders (train_and_test) based on class idx.'''
    train_samplers, test_samplers, valid_samplers = get_samplers(num_classes=num_classes)
    global PERMUTE_SEED

    loaders = []
    if '+' in task:  # the merge operand
        for split in task.split('+'):
            loaders.extend([
                get_loader(task=split, batch_size=batch_size, cuda=cuda,
                           workers_per_replica=workers_per_replica,
                           num_replicas=num_replicas,
                           seed=seed,
                           same_seed_workers=same_seed_workers,
                           timeout=timeout,
                           pin_memory=pin_memory,
                           drop_last=drop_last,
                           train_transform=train_transform,
                           train_target_transform=train_target_transform,
                           test_transform=test_transform,
                           test_target_transform=test_target_transform,
                           valid_transform=valid_transform,
                           valid_target_transform=valid_target_transform,
                           train_sampler=tr,
                           test_sampler=te,
                           valid_sampler=va,
                           increment_seed=False,
                           sequentially_merge_test=False,
                           **kwargs)
                for tr, te, va in zip(train_samplers, test_samplers, valid_samplers)]
            )
            PERMUTE_SEED += 1
    else:
        # iterate over samplers and generate
        loaders = [get_loader(task=task, batch_size=batch_size, cuda=cuda,
                              workers_per_replica=workers_per_replica,
                              num_replicas=num_replicas,
                              seed=seed,
                              same_seed_workers=same_seed_workers,
                              timeout=timeout,
                              pin_memory=pin_memory,
                              drop_last=drop_last,
                              train_transform=train_transform,
                              train_target_transform=train_target_transform,
                              test_transform=test_transform,
                              test_target_transform=test_target_transform,
                              valid_transform=valid_transform,
                              valid_target_transform=valid_target_transform,
                              train_sampler=tr,
                              test_sampler=te,
                              valid_sampler=va,
                              sequentially_merge_test=False,
                              increment_seed=False,
                              **kwargs)
                   for tr, te, va in zip(train_samplers, test_samplers, valid_samplers)]

    if _check_for_sublist(loaders):
        loaders = [item for sublist in loaders for item in sublist]

    PERMUTE_SEED = 1
    if sequentially_merge_test:  # merge test sets sequentially
        return sequential_dataset_merger(loaders, 'test')

    return loaders
