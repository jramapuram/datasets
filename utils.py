import cv2
import torch
import numpy as np

from copy import deepcopy
from PIL import Image
from collections import namedtuple
from torch.utils.data.dataset import Subset

from datasets.class_sampler import ClassSampler

# simple struct to hold properties of a loader
class GenericLoader(object):
    def __init__(self, img_shp, output_size, train_loader, test_loader):
        self.img_shp = img_shp
        self.output_size = output_size
        self.train_loader = train_loader
        self.test_loader = test_loader

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


def normalize_images(imgs, mu=None, sigma=None, eps=1e-9):
    ''' normalize imgs with provided mu /sigma
        or computes them and returns with the normalized
       images and tabulated mu / sigma'''
    if mu is None:
        if len(imgs.shape) == 4:
            chans = imgs.shape[1]
            mu = np.asarray(
                [np.mean(imgs[:, i, :, :]) for i in range(chans)]
            ).reshape(1, -1, 1, 1)
        elif len(imgs.shape) == 5:  # glimpses
            chans = imgs.shape[2]
            mu = np.asarray(
                [np.mean(imgs[:, :, i, :, :]) for i in range(chans)]
            ).reshape(1, 1, -1, 1, 1)
            sigma = np.asarray(
                [np.std(imgs[:, :, i, :, :]) for i in range(chans)]
            ).reshape(1, 1, -1, 1, 1)
        else:
            raise Exception("unknown number of dims for normalization")

    if sigma is None:
        if len(imgs.shape) == 4:
            chans = imgs.shape[1]
            sigma = np.asarray(
                [np.std(imgs[:, i, :, :]) for i in range(chans)]
            ).reshape(1, -1, 1, 1)
        elif len(imgs.shape) == 5:  # glimpses
            chans = imgs.shape[2]
            sigma = np.asarray(
                [np.std(imgs[:, :, i, :, :]) for i in range(chans)]
            ).reshape(1, 1, -1, 1, 1)
        else:
            raise Exception("unknown number of dims for normalization")

    return (imgs - mu) / (sigma + eps), [mu, sigma]


def normalize_train_test_images(train_imgs, test_imgs, eps=1e-9):
    ''' simple helper to take train and test images
        and normalize the test images by the train mu/sigma '''
    assert len(train_imgs.shape) == len(test_imgs.shape) >= 4

    train_imgs , [mu, sigma] = normalize_images(train_imgs, eps=eps)
    return [train_imgs,
            (test_imgs - mu) / (sigma + eps)]


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
    if isinstance(sampler, ClassSampler):
        # our sampler is hacky; just filters dataset
        # and nulls itself out for GC
        dataset = sampler(dataset)
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
                            loader.test_loader.batch_size,
                            shuffle=True)
        test_dataset.append(current_clone)

    return loaders


def trim_samples_in_loader(loader, num_samples, rand_selection=True, cuda=False):
    '''trims the samples in the loader to be num_samples
       by random indices (rand_selection=True) or first num_samples'''
    num_total_samples = len(loader.dataset)
    assert num_total_samples > num_samples
    indices = torch.randperm(num_total_samples)[0:num_samples] if rand_selection \
              else torch.arange(num_samples)
    loader.dataset = Subset(loader.dataset, indices)
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    shuffle = True if rand_selection else False
    loader = create_loader(loader.dataset, sampler=None,
                           batch_size=num_samples,
                           shuffle=shuffle, **kwargs)
    return loader
