import cv2
import torch
import numpy as np

from copy import deepcopy
from PIL import Image
from collections import namedtuple
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from datasets.class_sampler import ClassSampler

# simple namedtuple loader
GenericLoader = namedtuple('GenericLoader', 'img_shp output_size train_loader test_loader')


def resize_lambda(img, size=(64, 64)):
    if not isinstance(img, (np.float32, np.float64)):
        img = np.asarray(img)

    if not isinstance(size, tuple):
        size = tuple(size)

    return cv2.resize(img, size)


def permute_lambda(img, pixel_permutation):
    if not isinstance(img, (np.float32, np.float64)):
        img = np.asarray(img)

    img_orig_shape = img.shape
    return Image.fromarray(
        img.reshape(-1, 1)[pixel_permutation].reshape(img_orig_shape)
    )


# def bw_2_rgb_lambda(img):
#     if not isinstance(img, (np.float32, np.float64)):
#         img = np.asarray(img)

#     return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

def bw_2_rgb_lambda(img):
    if img.mode == "RGB":
        return img

    return img.convert(mode="RGB")

def binarize(img, block_size=21):
    if not isinstance(img, (np.float32, np.float64)):
        img = np.asarray(img)

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, 0)
    return np.expand_dims(img, -1) if len(img.shape) < 3 else img

def simple_merger(loaders, batch_size, use_cuda=False):
    print("\nWARN [simplemerger]: no process in place for handling different classes\n")
    train_dataset = loaders[0].train_loader.dataset
    test_dataset = loaders[0].test_loader.dataset
    img_shp = loaders[0].img_shp
    output_size = loaders[0].output_size

    for loader in loaders[1:]:
        train_dataset += loader.train_loader.dataset
        test_dataset += loader.test_loader.dataset
        assert img_shp == loader.img_shp
        assert output_size == loader.output_size

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        **kwargs
    )

    return GenericLoader(img_shp,
                         output_size,
                         train_loader,
                         test_loader)


def create_loader(dataset, sampler, batch_size, shuffle, **kwargs):
    if sampler is not None and \
       not isinstance(sampler, (SequentialSampler, RandomSampler)):
        # re-run the operand to extract indices
        # XXX: might be another sampler, cant
        #      directly check due to lambda
        sampler = sampler(dataset)
        dataset = sampler.dataset
        sampler = None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=shuffle,
        sampler=sampler,
        **kwargs)

def sequential_test_set_merger(loaders):
    test_dataset = [loaders[0].test_loader.dataset]
    for loader in loaders[1:]:
        current_clone = deepcopy(loader.test_loader.dataset)
        for td in test_dataset:
            loader.test_loader.dataset += td

        # re-create the test loader
        # in order to get correct samples
        loader.test_loader \
            = create_loader(loader.test_loader.dataset,
                            None, #loader.test_loader.sampler,
                            loader.batch_size,
                            shuffle=False)
        test_dataset.append(current_clone)

    return loaders
