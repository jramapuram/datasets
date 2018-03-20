import os
import torch
import numpy as np
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader, \
    find_classes, make_dataset

from .utils import create_loader


class MultiImageFolderLoader(object):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1, **kwargs):
        # first get the datasets
        train_dataset, test_dataset = self.get_datasets(path, transform, target_transform)

        # build the loaders
        # kwargs_loader = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        kwargs_loader = {'pin_memory': True} if use_cuda else {}
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
        self.output_size = 10

        # iterate over the entire dataset to find the max label
        # but just one image to get the image sizing
        # test_imgs, _ = self.train_loader.__iter__().__next__()
        # test_img = test_imgs[0] # grab the first one from the list
        # self.img_shp = list(test_img.size()[1:])
        self.img_shp = [1, 32, 32]

        # if 'output_size' not in kwargs:
        #     print("trying to determine output size from loader...")
        #     for _, labels in self.train_loader:
        #         label = labels[0]
        #         if not isinstance(label, (float, int))\
        #            and len(label) > 1:
        #             for l in label:
        #                 if l > self.output_size:
        #                     self.output_size = l
        #         else:
        #             if label > self.output_size:
        #                 self.output_size = label

        #     self.output_size = self.output_size[0] + 1 # Longtensor --> int
        # else:
        #     self.output_size = kwargs['output_size']

        print("determined output_size: ", self.output_size)

    @staticmethod
    def get_datasets(paths, transform=None, target_transform=None):
        if transform:
            assert isinstance(transform, list)

        transform_list = []
        if transform:
            transform_list.extend(transform)

        paths = paths.split(',')
        transform_list.append(transforms.ToTensor())
        train_dataset = MultiImageFolder(roots=[os.path.join(path, 'train') for path in paths],
                                         transform=transforms.Compose(transform_list),
                                         target_transform=target_transform)
        test_dataset = MultiImageFolder(roots=[os.path.join(path, 'test') for path in paths],
                                        transform=transforms.Compose(transform_list),
                                        target_transform=target_transform)
        return train_dataset, test_dataset


class MultiImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root1/dog/xxx.png
        root1/dog/xxy.png
        root1/dog/xxz.png
        root2/dog/xxx.png
        root2/dog/xxy.png
        root2/dog/xxz.png


    Args:
        roots (tuple, list): Root directory paths.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes_list (list): List of list of the class names.
        class_to_idx_list (dict): list of dict with items (class_name, class_index).
        imgs_list (list): list of List of (image path, class_index) tuples
    """

    def __init__(self, roots, transform=None, target_transform=None,
                 loader=default_loader):
        assert isinstance(roots, (tuple, list))
        self.classes_list, self.class_to_idx_list, self.imgs_list = [], [], []
        for root in roots:
            classes, class_to_idx = find_classes(root)
            imgs = make_dataset(root, class_to_idx)
            if len(imgs) == 0:
                raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                   "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

            # add them to the list
            self.classes_list.append(classes)
            self.class_to_idx_list.append(class_to_idx)
            self.imgs_list.append(imgs)

        # sanity check that we have the same number of samples
        num_imgs = len(self.imgs_list[0])
        for imgs in self.imgs_list:
            assert len(imgs) == num_imgs

        self.roots = roots
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_ret, target_ret = [], []
        for imgs in self.imgs_list:
            path, target = imgs[index]
            img = self.loader(path)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            img_ret.append(img)
            target_ret.append(target)

        return img_ret, target_ret

    def __len__(self):
        return len(self.imgs_list[0])
