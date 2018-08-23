import os
import gc
import torch
import numpy as np
import torchvision.transforms.functional as F

import random
random.seed(1234) # fix the seed for shuffling

from PIL import Image
from copy import deepcopy
from torchvision import datasets, transforms
from torch.utils.data.dataloader import default_collate
from contextlib import contextmanager
from joblib import Parallel, delayed
#from loky import get_reusable_executor

from .utils import create_loader

try:
    import pyvips
    # pyvips.leak_set(True)
    pyvips.cache_set_max(0)
    USE_PYVIPS = True
    # print("using VIPS backend")
except:
    # print("failure to load VIPS: using PIL backend")
    USE_PYVIPS = False

# import pytiff

def collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    batch_standard = [(data, lbl) for data, _, lbl in batch]
    lbdas = [lbda for _, lbda, _ in batch]
    return lbdas, default_collate(batch_standard)


class CropLambdaPool(object):
    def __init__(self, num_workers=8):
        self.num_workers = num_workers
        # self.executor = get_reusable_executor(max_workers=num_workers,
        #                                       timeout=100, kill_workers=True)

    def _apply(self, lbda, z_i):
        return lbda(z_i)

    # def __call__(self, list_of_lambdas, z_vec):
    #     #n_jobs=1 disables parallelization: return Parallel(n_jobs=1)(
    #     z_vec = np.ascontiguousarray(z_vec)
    #     return list(self.executor.map(self._apply, list_of_lambdas, z_vec))


    # def __call__(self, list_of_lambdas, z_vec):
    #     #with Pool(processes=1) as pool:
    #     with Pool(processes=len(list_of_lambdas)) as pool:
    #         return pool.starmap(self._apply, zip(list_of_lambdas, z_vec))

    def __call__(self, list_of_lambdas, z_vec):
        #n_jobs=1 disables parallelization: return Parallel(n_jobs=1)(
        return Parallel(n_jobs=self.num_workers, timeout=300)(
            delayed(self._apply)(list_of_lambdas[i], np.ascontiguousarray(z_vec[i]))
            for i in range(len(list_of_lambdas)))


class CropLambda(object):
    """Returns a lambda that crops to a region.

    Args:
        window_size: the resized return image [not related to img_percentage].
        max_img_percentage: the maximum percentage of the image to use for the crop.
    """

    def __init__(self, path, window_size, max_img_percentage=0.15):
        self.path = path
        self.window_size = window_size
        self.max_img_percent = max_img_percentage

    def scale(self, val, newmin, newmax):
        return (((val) * (newmax - newmin)) / (1.0)) + newmin

    def __call__(self, crop, transform=None, override=False):
        img = self.__call_pyvips__(crop, override) if USE_PYVIPS is True \
            else self.__call_PIL__(crop, override)

        if transform is not None:
            return F.to_tensor(transform(img))

        return F.to_tensor(img).unsqueeze(0)

    def _get_crop_sizing_and_centers(self, crop, img_size):
        # scale the (x, y) co-ordinates to the size of the image
        assert crop[1] >= 0 and crop[1] <= 1, "x needs to be \in [0, 1]"
        assert crop[2] >= 0 and crop[2] <= 1, "y needs to be \in [0, 1]"
        x, y = [int(self.scale(crop[1], 0, img_size[0])),
                int(self.scale(crop[2], 0, img_size[1]))]

        # calculate the scale of the true crop using the provided scale
        # Note: this is different from the return size, i.e. window_size
        crop_scale = min(crop[0], self.max_img_percent)
        crop_size = np.floor(img_size * crop_scale).astype(int)
        crop_size = [max(crop_size[0], 2), max(crop_size[1], 2)]

        # bound the (x, t) co-ordinates to be plausible
        # i.e < img_size - crop_size
        max_coords = img_size - crop_size
        x, y = min(x, max_coords[0]), min(y, max_coords[1])

        return x, y, crop_size

    # def __call_pytiff__(self, crop):
    #     with pytiff.Tiff("test_data/small_example_tiled.tif") as handle:
    #         img_size = handle.shape
    #         x, y, crop_size = self._get_crop_sizing_and_centers(crop, img_size)
    #         return handle
    #         part = handle[100:200, :]

    def __call_PIL__(self, crop, override):
        ''' converts [crop_center, x, y] to a 4-tuple
            defining the left, upper, right, and lower
            pixel coordinate and return a lambda '''
        with open(self.path, 'rb') as f:
            with Image.open(f) as img:
                img_size = np.array(img.size) # numpy-ize the img size (tuple)
                self.max_img_percent = 1.0 if override is True else self.max_img_percent
                x, y, crop_size = self._get_crop_sizing_and_centers(crop, img_size)

                # crop the actual image and then upsample it to window_size
                # resample = 2 is a BILINEAR transform, avoid importing PIL for enum
                # TODO: maybe also try 1 = ANTIALIAS = LANCZOS
                crop_img = img.crop((x, y, x + crop_size[0], y + crop_size[1]))
                return crop_img.resize((self.window_size, self.window_size), resample=2)

    def __call_pyvips__(self, crop, override):
        ''' converts [crop_center, x, y] to a 4-tuple
            defining the left, upper, right, and lower
            pixel coordinate and return a lambda '''
        #access = 'sequential' if 'tif' not in self.path or 'tiff' not in self.path else 'random'
        img = pyvips.Image.new_from_file(self.path, access='sequential-unbuffered')
        img_size = np.array([img.height, img.width]) # numpy-ize the img size (tuple)
        assert (img_size > 0).any(), "image [{}] had height[{}] and width[{}]".format(self.path, img.height, img.width)

        # get the crop dims
        self.max_img_percent = 1.0 if override is True else self.max_img_percent
        x, y, crop_size = self._get_crop_sizing_and_centers(crop, img_size)

        # crop the actual image and then upsample it to window_size
        crop_img = img.crop(x, y, crop_size[0], crop_size[1])
        crop_img = img.crop(x, y, crop_size[0], crop_size[1])
        crop_img_np = np.array(crop_img.resize(self.window_size / crop_img.width,
                                               vscale=self.window_size / crop_img.height).write_to_memory())

        #XXX: try to mitigate memory leak in VIPS
        del img
        del crop_img
        # gc.collect()

        return crop_img_np.reshape(self.window_size, self.window_size, -1)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img_mode = img.mode
            if img_mode == '1':
                return img.convert('L')

            return img.convert(mode)


class CropDualImageFolder(datasets.ImageFolder):
    """Inherits from Imagefolder and returns a lambda instead of an image"""
    def __init__(self, root, transform=None, target_transform=None, **kwargs):
        assert 'window_size' in kwargs, "crop dual dataset needs a window_size"
        assert 'max_img_percent' in kwargs, "crop dual dataset needs a max_img_percent"
        assert 'postfix' in kwargs, "crop dual dataset needs a postfix for second dataset"
        super(CropDualImageFolder, self).__init__(root,
                                                  transform=transform,
                                                  target_transform=target_transform,
                                                  loader=pil_loader)
        self.postfix = kwargs['postfix']
        self.window_size = kwargs['window_size']
        self.max_img_percent = kwargs['max_img_percent']

        # sort the images otherwise we will always read a folder at a time
        # this is problematic for the test-loader which generally doesnt shuffle!
        random.shuffle(self.imgs)

        # determine the extension replacement
        # eg: small is .png, large is .tiff for pyramid tiff
        path_test, target_test = self.imgs[0]
        orig_extension = os.path.splitext(path_test)[-1]
        new_base_path = os.path.splitext(root + self.postfix + path_test.replace(root, ""))[0]
        possible_exts = ['.tif', '.tiff', '.png', '.jpg', 'jpeg', '.bmp']
        ext = orig_extension
        for ext_i in possible_exts:
            ext = ext_i if os.path.isfile(new_base_path + ext_i) else ext

        print("determined secondary image format: ", ext)

        # imgs_lbda holds the path + target of the small distribution
        # while imgs holds the path + target of the true large distribution
        # self.lbda_loader = crop_loader
        self.imgs_lbda = []
        for path, target in self.imgs:
            img_path = path.replace(root, "")
            root_lbda = root + self.postfix
            lbda_path = os.path.splitext(root_lbda + img_path)[0] + ext
            self.imgs_lbda.append((lbda_path, target))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            (img, lambda, target): where target is class_index of the target class.
                                   and lambda is the cropping function
        """
        # first grab the main [eg: the downsampled]
        img, target = super(CropDualImageFolder, self).__getitem__(index)

        # then grab the cropping lambda
        path, _ = self.imgs_lbda[index]
        crop_lbda = CropLambda(path, window_size=self.window_size,
                               max_img_percentage=self.max_img_percent)
        return F.to_tensor(img), crop_lbda, target


class CropDualImageFolderLoader(object):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1, **kwargs):
        ''' assumes that the '''
        # first get the datasets
        train_dataset, test_dataset = self.get_datasets(path, transform,
                                                        target_transform,
                                                        **kwargs)

        # build the loaders, note that pinning memory **deadlocks** this loader!
        kwargs_loader = {'num_workers': 4, 'pin_memory': False, 'collate_fn': collate} \
            if use_cuda else {'collate_fn': collate}
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
        _, (test_img, _) = self.train_loader.__iter__().__next__()
        self.img_shp = list(test_img.size()[1:])
        print("determined img_size: ", self.img_shp)

        # iterate over the entire dataset to find the max label
        if 'output_size' not in kwargs:
            for _, (_, label) in self.train_loader:
                if not isinstance(label, (float, int))\
                   and len(label) > 1:
                    for l in label:
                        if l > self.output_size:
                            self.output_size = l
                else:
                    if label > self.output_size:
                        self.output_size = label

            self.output_size = self.output_size.item() + 1 # Longtensor --> int
        else:
            self.output_size = kwargs['output_size']

        print("determined output_size: ", self.output_size)

    @staticmethod
    def get_datasets(path, transform=None, target_transform=None, **kwargs):
        if transform:
            assert isinstance(transform, list)

        transform_list = []
        if transform:
            transform_list.extend(transform)

        train_dataset = CropDualImageFolder(root=os.path.join(path, 'train'),
                                            transform=transforms.Compose(transform_list),
                                            target_transform=target_transform, **kwargs)
        test_dataset = CropDualImageFolder(root=os.path.join(path, 'test'),
                                           transform=transforms.Compose(transform_list),
                                           target_transform=target_transform, **kwargs)
        return train_dataset, test_dataset
