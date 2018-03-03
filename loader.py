import os
import PIL
import numpy as np
import torchvision.transforms as transforms
from copy import deepcopy

from datasets.class_sampler import ClassSampler
from datasets.cifar import CIFAR10Loader
from datasets.fashion_mnist import FashionMNISTLoader
from datasets.mnist_cluttered import ClutteredMNISTLoader
from datasets.mnist import MNISTLoader
from datasets.permuted_mnist import PermutedMNISTLoader
from datasets.merger import MergedLoader
from datasets.svhn import SVHNCenteredLoader, SVHNFullLoader
from datasets.utils import bw_2_rgb_lambda, resize_lambda, simple_merger


def get_samplers(num_classes):
    ''' builds samplers taking into account previous classes'''
    test_samplers = []
    for i in range(num_classes):
        numbers = list(range(i + 1)) if i > 0 else 0
        test_samplers.append(lambda x, j=numbers: ClassSampler(x, class_number=j))

    train_samplers = [lambda x, j=j: ClassSampler(x, class_number=j)
                      for j in range(num_classes)]
    return train_samplers, test_samplers


def get_rotated_loader(task, args,
                       angles=[30, 70, 270],
                       num_classes=10):
    ''' returns a list of loaders that are rotated by angles '''
    loader_map = {
        'mnist': MNISTLoader,
        'fashion': FashionMNISTLoader,
        'clutter': ClutteredMNISTLoader,
        'cifar10': CIFAR10Loader,
        'svhn': SVHNCenteredLoader,
        'svhn_centered': SVHNCenteredLoader,
        'svhn_full': SVHNFullLoader
    }
    assert task in loader_map

    train_samplers, test_samplers = get_samplers(num_classes)
    rotation_transforms = [transforms.RandomRotation(
        (a, a), resample=PIL.Image.BILINEAR
    ) for a in angles]

    loaders = [loader_map[task](path=args.data_dir,
                                batch_size=args.batch_size,
                                train_sampler=tr,
                                test_sampler=te,
                                transform=[rt],
                                use_cuda=args.cuda)
               for rt in rotation_transforms
               for tr, te in zip(train_samplers, test_samplers)]

    # append the normal loaders too
    args_clone = deepcopy(args)
    args_clone.task = [task]
    loaders.extend(get_sequential_data_loaders(args_clone, num_classes))

    print("total rotated loaders: ", len(loaders))
    return loaders

def get_loader(args, transform=None, target_transform=None):
    task = args.task[0] if len(args.task) == 1 else 'merged'
    if task == 'cifar10':
        loader = CIFAR10Loader(path=args.data_dir,
                               batch_size=args.batch_size,
                               transform=transform,
                               target_transform=target_transform,
                               use_cuda=args.cuda)
    elif task == 'mnist':
        loader = MNISTLoader(path=args.data_dir,
                             batch_size=args.batch_size,
                             transform=transform,
                             target_transform=target_transform,
                             use_cuda=args.cuda)
    elif task == 'permuted':
        loader = PermutedMNISTLoader(path=args.data_dir,
                                     batch_size=args.batch_size,
                                     transform=transform,
                                     target_transform=target_transform,
                                     use_cuda=args.cuda)
    elif task == 'fashion':
        # NOTE: cant have MNIST & fashion in same dir, so workaround
        loader = FashionMNISTLoader(path=os.path.join(args.data_dir, "fashion"),
                                    batch_size=args.batch_size,
                                    transform=transform,
                                    target_transform=target_transform,
                                    use_cuda=args.cuda)
    elif task == 'clutter':
        loader = ClutteredMNISTLoader(path=args.data_dir,
                                      batch_size=args.batch_size,
                                      transform=transform,
                                      target_transform=target_transform,
                                      use_cuda=args.cuda)

    elif task == 'svhn_centered' or task == 'svhn':
        loader = SVHNCenteredLoader(path=args.data_dir,
                                    batch_size=args.batch_size,
                                    transform=transform,
                                    target_transform=target_transform,
                                    use_cuda=args.cuda)
    elif task == 'svhn_full':
        loader = SVHNFullLoader(path=args.data_dir,
                                batch_size=args.batch_size,
                                transform=transform,
                                target_transform=target_transform,
                                use_cuda=args.cuda)
    elif task == 'merged':
        loader = MergedLoader(args.task, path=args.data_dir,
                              batch_size=args.batch_size,
                              # transform=transform,
                              # target_transform=target_transform,
                              use_cuda=args.cuda)
    elif '+' in task:
        # TODO: merge this into merged loader
        loader = []
        for split in task.split('+'):
            args_clone = deepcopy(args)
            args_clone.task = [split]

            transform = [
                transforms.Resize((32, 32)), # XXX: parameterize
                transforms.Lambda(lambda img: bw_2_rgb_lambda(img))
            ]
            loader.append(get_loader(args_clone, transform=transform))

    elif 'rotated' in task:
        # print("""WARN: currently rotated dataset simply
        #          returns rotations only for sequential """)
        # args_clone = deepcopy(args)
        # args_clone.task = [task.split('_')[1]]
        # return get_loader(args_clone)
        task = task.split('_')[1]
        loaders = get_rotated_loader(task, args)
        loader = simple_merger(loaders, args.batch_size, args.cuda)
    else:
        raise Exception("unknown dataset provided / not supported yet")

    return loader


def get_sequential_data_loaders(args, num_classes=10):
    ''' helper to return the model and the loader '''
    # we build 10 samplers as all of the below have 10 classes
    train_samplers, test_samplers = get_samplers(num_classes=num_classes)
    task_name = args.task[0] if len(args.task) == 1 else 'merged'
    print(args.task)

    if task_name == 'cifar10':
        loaders = [CIFAR10Loader(path=args.data_dir,
                                 batch_size=args.batch_size,
                                 train_sampler=tr,
                                 test_sampler=te,
                                 use_cuda=args.cuda)
                   for tr, te in zip(train_samplers, test_samplers)]
    elif task_name == 'mnist':
        loaders = [MNISTLoader(path=args.data_dir,
                               batch_size=args.batch_size,
                               train_sampler=tr,
                               test_sampler=te,
                               use_cuda=args.cuda)
                   for tr, te in zip(train_samplers, test_samplers)]
    elif task_name == 'fashion':
        # NOTE: cant place MNIST and fashion MNIST in the same directory, so workaround
        loaders = [FashionMNISTLoader(path=os.path.join(args.data_dir, "fashion"),
                                      batch_size=args.batch_size,
                                      train_sampler=tr,
                                      test_sampler=te,
                                      use_cuda=args.cuda)
                   for tr, te in zip(train_samplers, test_samplers)]
    elif task_name == 'clutter':
        loaders = [ClutteredMNISTLoader(path=args.data_dir,
                                        batch_size=args.batch_size,
                                        train_sampler=tr,
                                        test_sampler=te,
                                        use_cuda=args.cuda)
                   for tr, te in zip(train_samplers, test_samplers)]
    elif task_name == 'permuted':
        loaders = [PermutedMNISTLoader(path=args.data_dir,
                                       batch_size=args.batch_size,
                                       train_sampler=tr,
                                       test_sampler=te,
                                       use_cuda=args.cuda)
                   for tr, te in zip(train_samplers, test_samplers)]
    elif task_name == 'svhn' or task_name == 'svhn_centered':
        loaders = [SVHNCenteredLoader(path=args.data_dir,
                                      batch_size=args.batch_size,
                                      train_sampler=tr,
                                      test_sampler=te,
                                      use_cuda=args.cuda)
                   for tr, te in zip(train_samplers, test_samplers)]
    elif task_name == 'svhn_full':
        loaders = [SVHNFullLoader(path=args.data_dir,
                                  batch_size=args.batch_size,
                                  train_sampler=tr,
                                  test_sampler=te,
                                  use_cuda=args.cuda)
                   for tr, te in zip(train_samplers, test_samplers)]
    elif task_name == 'merged':
        loaders = [MergedLoader(args.task, path=args.data_dir,
                                batch_size=args.batch_size,
                                train_sampler=tr,
                                test_sampler=te,
                                use_cuda=args.cuda)
                   for tr, te in zip(train_samplers, test_samplers)]
    elif '+' in task_name:
        loaders = []
        for split in task_name.split('+'):
            args_clone = deepcopy(args)
            args_clone.task = [split]
            loaders.extend(get_sequential_data_loaders(args_clone, num_classes))
    elif 'rotated' in task_name:
        task = task_name.split('_')[1]
        loaders = get_rotated_loader(task, args)
    else:
        raise Exception("unknown dataset provided / not supported yet")

    return loaders
