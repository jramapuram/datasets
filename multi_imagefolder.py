import os
import functools
import numpy as np
import random

from PIL import Image
from torchvision import datasets, transforms

from .utils import temp_seed
from .abstract_dataset import AbstractLoader


try:
    import pyvips
    # pyvips.leak_set(True)
    # pyvips.cache_set_max_mem(10000)
    pyvips.cache_set_max(0)
    USE_PYVIPS = True
    # print("using VIPS backend")
except:
    # print("failure to load VIPS: using PIL backend")
    USE_PYVIPS = False


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # https://github.com/python-pillow/Pillow/issues/835
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img_mode = img.mode
            if img_mode == '1':
                return img.convert('L')

            return img.convert(img_mode)


def vips_loader(path):
    img = pyvips.Image.new_from_file(path, access='sequential')
    height, width = img.height, img.width
    img_np = np.array(img.write_to_memory()).reshape(height, width, -1)
    del img  # mitigate tentative memory leak in pyvips
    return np.transpose(img_np, (2, 0, 1))


class MultiImageFolder(datasets.ImageFolder):
    """Inherits from Imagefolder, but returns an multiple images (and ONLY one class label) """

    def __init__(self, roots, transform=None, aux_transform=None, target_transform=None):
        assert isinstance(roots, list), "multi-imagefolder needs a list of roots"
        loader = vips_loader if USE_PYVIPS is True else pil_loader
        super(MultiImageFolder, self).__init__(roots[0],
                                               transform=transform,
                                               target_transform=target_transform,
                                               loader=loader)
        self.num_roots = len(roots)
        self.num_extra_roots = self.num_roots - 1
        self.aux_transform = aux_transform

        # sort the images otherwise we will always read a folder at a time
        # this is problematic for the test-loader which generally doesnt shuffle!
        with temp_seed(1234):
            random.shuffle(self.imgs)

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
        if self.aux_transform is not None:
            sample = self.aux_transform(sample)

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


class MultiImageFolderLoader(AbstractLoader):
    """Loads data from train_*, test_* and valid_* folders simultaneously."""

    def __init__(self, path, batch_size, num_replicas=0,
                 train_sampler=None, test_sampler=None, valid_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 valid_transform=None, valid_target_transform=None,
                 cuda=True, output_size=None, **kwargs):
        # output_size=None uses automagic

        # Calculate the roots based on the path
        all_dirs = sorted([os.path.join(path, o) for o in os.listdir(path)
                           if os.path.isdir(os.path.join(path, o))])
        train_roots = [t for t in all_dirs if 'train' in t]
        test_roots = [te for te in all_dirs if 'test' in te]
        valid_roots = [va for va in all_dirs if 'valid' in va]
        print("train_roots = ", train_roots)
        print("test_roots = ", test_roots)
        print("valid_roots = ", valid_roots)
        assert len(train_roots) == len(test_roots), "number of train sets needs to match test sets"
        assert len(train_roots) > 0, "no datasets detected!"

        # Use the same train_transform for aux_transform
        aux_transform = self.compose_transforms(train_transform)

        # Curry the train and test dataset generators.
        train_generator = functools.partial(MultiImageFolder, aux_transform=aux_transform, roots=train_roots)
        test_generator = functools.partial(MultiImageFolder, roots=test_roots)
        valid_generator = None
        if len(valid_roots) > 0:
            valid_generator = functools.partial(MultiImageFolder,
                                                aux_transform=aux_transform,
                                                roots=valid_roots)

        super(MultiImageFolderLoader, self).__init__(batch_size=batch_size,
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
                                                     num_replicas=num_replicas, cuda=cuda, **kwargs)
        self.output_size = output_size if output_size is not None\
            else self.determine_output_size()            # automagic
        self.loss_type = 'ce'                            # fixed

        # grab a test sample to get the size
        test_imgs, _ = self.train_loader.__iter__().__next__()
        self.input_shape = list(test_imgs[0].size()[1:])
        print("derived image shape = ", self.input_shape)
        self.aux_input_shape = [list(ti.size()[1:]) for ti in test_imgs[1:]]
        print("determined aux img_sizes: ", self.aux_input_shape)
