import os
import gc
import torch
import warnings
import numpy as np
import torchvision.transforms.functional as F

import random
random.seed(1234) # fix the seed for shuffling

from cffi import FFI
from PIL import Image
from copy import deepcopy
from torchvision import datasets, transforms
from torch.utils.data.dataloader import default_collate
from contextlib import contextmanager
from joblib import Parallel, delayed


from datasets.utils import create_loader


# logic for loading:
#   1) Rust library [best mem + best speed]
#   2) pyvips       [poor mem + best speed]
if os.path.isfile('./libs/libparallel_image_crop.so'): #XXX: hardcoded
    USE_LIB = 'rust'
else:
    try:
        import pyvips
        # pyvips.leak_set(True)
        pyvips.cache_set_max(0)
        # warnings.warn("failure to load Rust : using VIPS backend")
        USE_LIB = 'pyvips'
    except:
        raise Exception("failed to load rust-lib and pyvips")


def collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    batch_standard = [(data, lbl) for data, _, lbl in batch]
    lbdas = [lbda for _, lbda, _ in batch]
    return lbdas, default_collate(batch_standard)


class RustParallelImageCrop(object):
    def __init__(self, window_size, chans, max_image_percentage,
                 num_threads=0, use_vips=True, lib_path='./libs/libparallel_image_crop.so'):
        '''Load the FFI rust library and can return image crops (and resizes) via __call__
           Accepts a list of paths to the images, a list of the crop defns.

            - window_size: resized return size
            - chans: image channels
            - max_image_percentage: max percent of image to crop
            - num_threads: 0 uses all CPU cores
            - use_vips: use VIPS backend instead of rust-image
            - lib_path: location of rust library for FFI
        '''
        self.window_size = window_size
        self.chans = chans
        self.max_image_percentage = max_image_percentage
        self.num_threads = num_threads
        self.use_vips = use_vips
        self.lib, self.ffi = self.create_and_set_ffi(lib_path)
        self.ptr = self.lib.initialize(num_threads, use_vips)

    def __call__(self, path_list, crops, override=False):
        ''' - path_list: list of string paths
            - crops: [[s0, x0, y0], [s1, x1, y1], ...]
            - override: when set to true returns s without truncating s to max_img_percent
        '''
        assert len(crops.shape) == 2 and crops.shape[-1] == 3, "expect dim(crops) = 2 && feature-dim of 3: [s, x, y]"

        # convert the paths to ascii-strings and then FFI convert
        path_keepalive = [self.ffi.new("char[]", p.encode('ascii')) for p in path_list]

        # calculate the batch size, contiguize/fp32-ize and build the return container
        batch_size = len(path_list)
        crops_output = np.ascontiguousarray(
            np.zeros(self.chans*self.window_size*self.window_size*batch_size, dtype=np.uint8)
        )
        crops = crops.astype(np.float32) # force fp32 for ffi conversion
        scales, x, y = [np.ascontiguousarray(crops[:, 0]),
                        np.ascontiguousarray(crops[:, 1]),
                        np.ascontiguousarray(crops[:, 2])]

        # execute the job in FFI
        self.lib.parallel_crop_and_resize(
            self.ptr, self.ffi.new("char* []", path_keepalive) ,
            self.ffi.cast("uint8_t*", crops_output.ctypes.data), # resultant crops
            self.ffi.cast("float*", scales.ctypes.data),         # scale
            self.ffi.cast("float*", x.ctypes.data),              # x
            self.ffi.cast("float*", y.ctypes.data),              # y
            self.window_size,
            self.chans,
            1.0 if override is True else self.max_image_percentage,
            batch_size
        )

        # reshape the resultant into pytorch format, convert to float and return
        # debug with: plt.imshow(crops[np.random.randint(batch_size)].squeeze()); plt.show()
        crops_output = crops_output.reshape([batch_size, self.window_size, self.window_size, self.chans])
        crops_output = crops_output.transpose(0, 3, 1, 2).astype(np.float32)
        # crops_output = crops_output / 255.0 # no need, handled in F.to_tensor
        return crops_output

    @staticmethod
    def create_and_set_ffi(lib_path):
        ffi = FFI()
        ffi.cdef("""
        void destroy(void*);
        void* initialize(uint64_t, bool);
        void parallel_crop_and_resize(void*, char**, uint8_t*, float*, float*, float*, uint32_t, uint32_t, float, size_t);
        """);

        lib = ffi.dlopen(lib_path)
        return lib, ffi

    def __del__(self):
        if self.lib is not None:
            print("shutting down RUST FFI")
            self.lib.destroy(self.ptr)


def _apply(lbda, z_i, override):
    return lbda(z_i, override=override)

class CropLambdaPool(object):
    def __init__(self, num_workers=0, **kwargs):
        self._handle_warnings()
        self.num_workers = num_workers
        if num_workers == 0 and USE_LIB != 'rust':
            # override because parallel(-1) uses all cpu cores
            # whereas in rust 0 uses all cpu cores
            self.num_workers = -1

        if USE_LIB == 'rust':
            self.rust_ffi = RustParallelImageCrop(**kwargs)

    def _handle_warnings(self):
        print("using {} image cropping-backend".format(USE_LIB))

    def _apply(self, lbda, top_left_i, bottom_right_i, override):
        return lbda(top_left_i, bottom_right_i, override=override)

    def __call__(self, list_of_lambdas_or_strs,
                 top_left_vec, bottom_right_vec,
                 override=False):
        # convert to fp32 first
        top_left_vec = top_left_vec.astype(np.float32)
        bottom_right_vec = bottom_right_vec.astype(np.float32)

        if USE_LIB != "rust":
            return Parallel(n_jobs=self.num_workers, timeout=300)( # n_jobs=1 disables parallelization
                delayed(self._apply)(list_of_lambdas_or_strs[i], np.ascontiguousarray(top_left_vec[i]),
                                     np.ascontiguousarray(bottom_right_vec[i]), override=override)
                for i in range(len(list_of_lambdas_or_strs)))

        return self.rust_ffi(list_of_lambdas_or_strs, z_vec, override=override)


class CropLambda(object):
    """Returns a lambda that crops to a region.

    Args:
        window_size: the resized return image [not related to img_percentage].
        max_img_percentage: the maximum percentage of the image to use for the crop.
    """

    def __init__(self, path, window_size, crop_padding, max_img_percentage=0.15):
        self.path = path
        self.window_size = window_size
        self.crop_padding = crop_padding
        self.max_img_percent = max_img_percentage

    def scale(self, val, newmin, newmax, oldmin, oldmax):
        return (((val - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin

    def _get_crop_sizing_and_centers(self, top_left, bottom_right, img_size):
        # scale the (x, y) co-ordinates to the size of the image
        assert top_left[0] >= -1 and top_left[1] <= 1, "top_left needs to be \in [-1, 1]"
        assert bottom_right[0] >= -1 and bottom_right[1] <= 1, "bottom_right needs to be \in [-1, 1]"
        x, y = [int(self.scale(top_left[0], 0, img_size[0]-1, -1, 1)),
                int(self.scale(top_left[1], 0, img_size[1]-1, -1, 1))]

        # tabulate the size of the crop
        br_x, br_y = [int(self.scale(bottom_right[0], 0, img_size[0]-1, -1, 1)),
                      int(self.scale(bottom_right[1], 0, img_size[1]-1, -1, 1))]
        crop_size = [br_x - x, br_y - y]

        # bound the (x, t) co-ordinates to be plausible
        # i.e < img_size - crop_size and > crop_size
        max_coords = img_size - crop_size
        x, y = min(x, max_coords[0]), min(y, max_coords[1]) # fix issue on bottom-right

        return x, y, crop_size

    @staticmethod
    def vimage_to_np(vimg, rescale=True, dtype=np.float32):
        vimg_size = np.array([vimg.width, vimg.height])
        vimg_np = np.array(vimg.write_to_memory(), dtype=dtype)
        vimg_np = vimg_np.reshape(vimg_size[0], vimg_size[1], -1)
        if rescale:
            vimg_np /= 255.0 # rescale to [0, 1]

        return vimg_np

    def __call__(self, top_left, bottom_right, override=False):
        ''' converts crop_location = [nw, se] to a crop
            and returns it after resizing to predefined window size

            Arguments:
              - crop_location: [ (nw_x, nw_y), (se_x, se_y) ]
              - override: returns entire image if True
        '''
        img = pyvips.Image.new_from_file(self.path, access='sequential-unbuffered')
        img_size, chans = np.array([img.width, img.height]), img.bands # numpy-ize the img size (tuple)
        assert (img_size > 0).any(), "image [{}] had height[{}] and width[{}]".format(
            self.path, img.height, img.width
        )

        # get the crop dims
        self.max_img_percent = 1.0 if override is True else self.max_img_percent
        x, y, crop_size = self._get_crop_sizing_and_centers(top_left, bottom_right, img_size)
        # print("x = ", x, " | y = ", y, " | crop_size = ", crop_size)

        # crop the actual image and then upsample it to window_size
        crop_img = img.crop(x, y, crop_size[0], crop_size[1])
        if not override: # only resize if not overriding
            crop_img = crop_img.resize(self.window_size / crop_img.width,
                                       vscale=self.window_size / crop_img.height)

        crop_img_np = self.vimage_to_np(crop_img)

        # try to mitigate memory leak
        del img; del crop_img
        # gc.collect() # XXX: too slow

        return F.to_tensor(crop_img_np).unsqueeze(0)


def pil_loader(path):
    # open path as file to avoid ResourceWarning :
    # https://github.com/python-pillow/Pillow/issues/835
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img_mode = img.mode
            if img_mode == '1':
                return img.convert('L')

            return img.convert(img_mode)


class CropDualImageFolder(datasets.ImageFolder):
    """Inherits from Imagefolder and returns a lambda instead of an image"""
    def __init__(self, root, transform=None, target_transform=None, **kwargs):
        assert 'window_size' in kwargs, "crop dual dataset needs a window_size"
        assert 'max_image_percentage' in kwargs, "crop dual dataset needs a max_image_percentage"
        assert 'crop_padding' in kwargs, "crop dual dataset needs crop_padding"
        assert 'postfix' in kwargs, "crop dual dataset needs a postfix for second dataset"
        super(CropDualImageFolder, self).__init__(root,
                                                  transform=transform,
                                                  target_transform=target_transform,
                                                  loader=pil_loader)
        self.postfix = kwargs['postfix']
        self.window_size = kwargs['window_size']
        self.max_img_percent = kwargs['max_image_percentage']
        self.crop_padding = kwargs['crop_padding']

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

        # then grab the cropping lambda (or path in the case of Rust)
        path, _ = self.imgs_lbda[index]
        if USE_LIB != 'rust': # we return the crop object
            crop_lbda = CropLambda(path, window_size=self.window_size,
                                   crop_padding=self.crop_padding,
                                   max_img_percentage=self.max_img_percent)
        else: # rust lib just needs a list of the paths
            crop_lbda = path

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
        kwargs_loader = {'num_workers': 4, 'pin_memory': True, 'collate_fn': collate} \
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
        self.batch_size = batch_size
        self.output_size = 0

        # just one image to get the image sizing
        _, (test_img, _) = self.train_loader.__iter__().__next__()
        self.img_shp = list(test_img.size()[1:])
        print("determined img_size: ", self.img_shp)

        # iterate over the entire dataset to find the max label
        if 'output_size' not in kwargs or kwargs['output_size'] is None:
            for _, (_, label) in self.train_loader:
                if not isinstance(label, (float, int)) and len(label) > 1:
                    l = np.array(label).max()
                    if l > self.output_size:
                        self.output_size = l
                else:
                    l = label.max().item()
                    if l > self.output_size:
                        self.output_size = l

            self.output_size = self.output_size + 1
        else:
            self.output_size = kwargs['output_size']

        print("determined output_size: ", self.output_size)
        assert self.output_size > 0, "could not determine output size"

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
