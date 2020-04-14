import os

from datasets.crop_dual_imagefolder import CropDualImageFolderLoader
from datasets.all_pairs.grid_loader import GridDataLoader
from datasets.class_sampler import ClassSampler
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
from datasets.imagefolder import ImageFolderLoader
from datasets.multi_imagefolder import MultiImageFolderLoader
from datasets.utils import sequential_test_set_merger

# ensures we get the same permutation
PERMUTE_SEED = 1

loader_map = {
    'all_pairs': GridDataLoader,
    'sort': SortLoader,
    'nih_chest_xray': NIHChestXrayLoader,
    'starcraft_predict_battle': StarcraftPredictBattleLoader,
    'crop_dual_imagefolder': CropDualImageFolderLoader,
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
    'multi_image_folder': MultiImageFolderLoader
}


def get_samplers(num_classes):
    ''' builds samplers taking into account previous classes'''
    # NOTE: test datasets are now merged via sequential_test_set_merger
    test_samplers = [ClassSampler(class_number=j, shuffle=False) for j in range(num_classes)]
    train_samplers = [ClassSampler(class_number=j, shuffle=True) for j in range(num_classes)]
    valid_samplers = [ClassSampler(class_number=j, shuffle=True) for j in range(num_classes)]
    return train_samplers, test_samplers, valid_samplers


def get_loader(task, data_dir, batch_size, cuda,
               train_transform=None, train_target_transform=None,
               test_transform=None, test_target_transform=None,
               valid_transform=None, valid_target_transform=None,
               train_sampler=None, test_sampler=None, valid_sampler=None,
               increment_seed=True, sequentially_merge_test=True,
               **kwargs):
    ''' increment_seed: increases permutation rng seed,
        sequentially_merge_test: merge all the test sets sequentially '''
    global PERMUTE_SEED

    # overwrite data dir for fashion MNIST because it has issues being
    # in the same directory as regular MNIST
    if task == 'fashion':
        data_dir = os.path.join(data_dir, "fashion")
    else:
        data_dir = data_dir

    if '+' in task:  # the merge operand
        loader = []
        for split in task.split('+'):
            loader.append(get_loader(task=split,
                                     data_dir=data_dir,
                                     batch_size=batch_size,
                                     cuda=cuda,
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

        PERMUTE_SEED = 1
        if sequentially_merge_test:
            loader = sequential_test_set_merger(loader)
            # loader = simple_merger(loader, args.batch_size, args.cuda)
    else:
        assert task in loader_map, "unknown task requested"
        if task == 'permuted':
            kwargs['seed'] = PERMUTE_SEED

        return loader_map[task](path=data_dir, batch_size=batch_size,
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


def get_split_data_loaders(task, num_classes, data_dir, batch_size, cuda,
                           train_transform=None, train_target_transform=None,
                           test_transform=None, test_target_transform=None,
                           valid_transform=None, valid_target_transform=None,
                           sequentially_merge_test=True, **kwargs):
    ''' helper to return the model and the loader '''
    # we build 10 samplers as all of the below have 10 classes
    train_samplers, test_samplers, valid_samplers = get_samplers(num_classes=num_classes)
    global PERMUTE_SEED

    loaders = []
    if '+' in task:  # the merge operand
        for split in task.split('+'):
            loaders.extend([
                get_loader(task=split, batch_size=batch_size, cuda=cuda,
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
        return sequential_test_set_merger(loaders)

    return loaders
