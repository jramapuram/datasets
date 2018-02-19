import cv2
import torch
import numpy as np
from collections import namedtuple

# simple namedtuple loader
GenericLoader = namedtuple('GenericLoader', 'img_shp output_size train_loader test_loader')


def resize_lambda(img, size=(64, 64)):
    if not isinstance(img, (np.float32, np.float64)):
        img = np.asarray(img)

    if not isinstance(size, tuple):
        size = tuple(size)

    return cv2.resize(img, size)

# def bw_2_rgb_lambda(img):
#     if not isinstance(img, (np.float32, np.float64)):
#         img = np.asarray(img)

#     return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

def bw_2_rgb_lambda(img):
    if img.mode == "RGB":
        return img

    return img.convert(mode="RGB")


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
