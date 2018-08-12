import os
import gc
import torch
import numpy as np
import multiprocessing
import torchvision.transforms.functional as F

from PIL import Image
from copy import deepcopy
from torchvision import datasets, transforms
from torch.utils.data.dataloader import default_collate
from multiprocessing import Process, Queue, Pool
from contextlib import contextmanager
from joblib import Parallel, delayed
#from loky import get_reusable_executor

from .utils import create_loader

try:
    import pyvips
    # pyvips.leak_set(True)
    # pyvips.cache_set_max_mem(10000)
    pyvips.cache_set_max(0)
    USE_PYVIPS = True
    # print("using VIPS backend")
except e:
    # print("failure to load VIPS: using PIL backend")
    USE_PYVIPS = False


def collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    batch_standard = [(data, lbl) for data, _, lbl in batch]
    lbdas = [lbda for _, lbda, _ in batch]
    return lbdas, default_collate(batch_standard)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img_mode = img.mode
            if img_mode == '1':
                return img.convert('L')

            return img.convert(img_mode)


def vips_loader(path):
    img = pyvips.Image.new_from_file(path, access='sequential-unbuffered')
    height, width = img.height, img.width
    return np.array(img.write_to_memory()).reshape(height, width, -1)


class MultiImageFolder(datasets.ImageFolder):
    """Inherits from Imagefolder, but returns an multiple images (and ONLY one class label) """
    def __init__(self, roots, transform=None, target_transform=None, **kwargs):
        assert isinstance(roots, list), "multi-imagefolder needs a list of roots"
        loader = vips_loader if USE_PYVIPS is True else pil_loader
        super(MultiImageFolder, self).__init__(roots[0],
                                               transform=transform,
                                               target_transform=target_transform,
                                               loader=loader)
        self.num_roots = len(roots)
        self.num_extra_roots = self.num_roots - 1

        # determine the extension replacements
        # eg: original is .png, other is .tiff for pyramid tiff, etc, etc
        path_test, target_test = self.imgs[0]
        orig_extension = os.path.splitext(path_test)[-1]
        possible_exts = ['.tif', '.tiff', '.png', '.jpg', 'jpeg', '.bmp']

        # iterate over all paths from first root [used above] and add corresponding images
        self.imgs_other = [[] for _ in range(self.num_extra_roots)]
        for path, _ in self.imgs:
            for i, new_root in enumerate(roots[1:]):
                refactored_path = path.replace(roots[0], new_root)
                determined_full_path = None
                for ext_i in possible_exts:
                    target_without_ext = os.path.splitext(refactored_path)[0]
                    #full_path = os.path.join(refactored_path, target_without_ext + ext_i)
                    full_path = refactored_path.replace(orig_extension, ext_i)
                    if os.path.isfile(full_path):
                        determined_full_path = full_path
                        break

                assert determined_full_path is not None, \
                    "Could not find equivalent file in dataset folder {}".format(new_root)
                self.imgs_other[i].append(determined_full_path)

    def _get_and_transform_img(self, root_index, img_index):
        path = self.imgs_other[root_index][img_index]

        # test to match paths
        # p0, _ = self.imgs[img_index]
        # others = []
        # for rt_list in self.imgs_other:
        #     others.append(rt_list[img_index])
        # print("paths = ", [p0] + others)

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            (img, lambda, target): where target is class_index of the target class.
                                   and lambda is the cropping function
        """
        # first grab the main [eg: the downsampled]
        img, target = super(MultiImageFolder, self).__getitem__(index)

        # then grab the rest of the images
        other_imgs = [self._get_and_transform_img(root_idx, index)
                      for root_idx in range(self.num_extra_roots)]
        return [img] + other_imgs, target


class MultiImageFolderLoader(object):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1, **kwargs):
        ''' assumes that the '''
        # first get the datasets
        train_dataset, test_dataset = self.get_datasets(path, transform,
                                                        target_transform,
                                                        **kwargs)

        # build the loaders, note that pinning memory **deadlocks** this loader!
        #kwargs_loader = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

        kwargs_loader = {'num_workers': multiprocessing.cpu_count(),
                         'pin_memory': True} if use_cuda else {}
        self.train_loader = create_loader(train_dataset,
                                          train_sampler,
                                          batch_size,
                                          shuffle=True if train_sampler is None else False,
                                          **kwargs_loader)

        self.test_loader = create_loader(test_dataset,
                                         test_sampler,
                                         batch_size,
                                         shuffle=False,
                                         **kwargs_loader)
        # self.train_loader.pool = self.pool
        # self.test_loader.pool = self.pool
        self.batch_size = batch_size
        self.output_size = 0

        # just one image to get the image sizing
        test_imgs, _ = self.train_loader.__iter__().__next__()
        self.img_shp = list(test_imgs[0].size()[1:])
        print("determined img_size: ", self.img_shp)

        # iterate over the entire dataset to find the max label
        # if 'output_size' not in kwargs:
        #     for _, label in self.train_loader:
        #         if not isinstance(label, (float, int))\
        #            and len(label) > 1:
        #             for l in label:
        #                 if l > self.output_size:
        #                     self.output_size = l
        #         else:
        #             if label > self.output_size:
        #                 self.output_size = label

        #     self.output_size = self.output_size.item() + 1 # Longtensor --> int
        # else:
        #     self.output_size = kwargs['output_size']
        self.output_size = 10

        print("determined output_size: ", self.output_size)

    @staticmethod
    def get_datasets(path, transform=None, target_transform=None, **kwargs):
        if transform:
            assert isinstance(transform, list)

        transform_list = []
        if transform:
            transform_list.extend(transform)

        transform_list.append(transforms.ToTensor())

        # train roots are everything with train_ , test roots are everything with test_
        all_dirs = sorted([os.path.join(path, o) for o in os.listdir(path)
                           if os.path.isdir(os.path.join(path,o))])
        train_roots = [t for t in all_dirs if 'train' in t]
        test_roots = [te for te in all_dirs if 'test' in te]
        print("train_roots = ", train_roots)
        print("test_roots = ", test_roots)
        assert len(train_roots) == len(test_roots), "number of train sets needs to match test sets"
        assert len(train_roots) > 0, "no datasets detected!"

        train_dataset = MultiImageFolder(roots=train_roots,
                                         transform=transforms.Compose(transform_list),
                                         target_transform=target_transform, **kwargs)
        test_dataset = MultiImageFolder(roots=test_roots,
                                        transform=transforms.Compose(transform_list),
                                        target_transform=target_transform, **kwargs)
        return train_dataset, test_dataset
