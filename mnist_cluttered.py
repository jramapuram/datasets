import os
import cv2
cv2.setNumThreads(0)

import torch
import numpy as np

from PIL import Image
from torchvision import datasets, transforms
from torch.utils.serialization import load_lua

from .utils import create_loader, normalize_train_test_images

def load_cluttered_mnist(path, segment='train'):
    full = load_lua(os.path.join(path, '%s.t7'%segment))
    data = [t[0].unsqueeze(1) for t in full]
    labels = []
    for t in full:
        _, index = torch.max(t[1], 0)
        labels.append(index.unsqueeze(0))

    return [torch.cat(data).type(torch.FloatTensor).numpy(),
            torch.cat(labels).squeeze().type(torch.LongTensor).numpy()]


class ClutteredMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, path, segment='train', transform=None, target_transform=None, **kwargs):

        self.path = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform
        self.segment = segment.lower().strip()  # train or test or val

        # load the images and labels
        self.imgs, self.labels = self._load_from_path()

    def _load_from_path(self):
        # load the tensor dataset from it's t7 binaries
        imgs, labels =  load_cluttered_mnist(self.path, segment=self.segment)
        print("imgs_%s = "%self.segment, imgs.shape,
              " | lbl_%s = "%self.segment, labels.shape)
        return imgs, labels

    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index]
        img = Image.fromarray(np.uint8(img.squeeze()*255))
        #img = np.transpose(img, (1, 2, 0))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        #img = np.transpose(img, (1, 2, 0))
        #img = Image.fromarray(np.uint8(img*255))
        return img, target

    def __len__(self):
        return len(self.imgs)


class ClutteredMNISTLoader(object):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1, **kwargs):
        # first get the datasets
        train_dataset, test_dataset = self.get_datasets(path, transform,
                                                        target_transform, **kwargs)

        # normalize the images
        # train_dataset.imgs, test_dataset.imgs = normalize_train_test_images(
        #     train_dataset.imgs, test_dataset.imgs
        # )

        # build the loaders
        kwargs = {'num_workers': os.cpu_count(), 'pin_memory': True} if use_cuda else {}
        self.train_loader = create_loader(train_dataset,
                                          train_sampler,
                                          batch_size,
                                          shuffle=True if train_sampler is None else False,
                                          **kwargs)

        self.test_loader = create_loader(test_dataset,
                                         test_sampler,
                                         batch_size,
                                         shuffle=False,
                                         **kwargs)
        self.output_size = 0
        self.batch_size = batch_size

        # determine output size
        if 'output_size' not in kwargs or kwargs['output_size'] is None:
            for _, label in self.train_loader:
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
        self.img_shp = [1, 100, 100]

    @staticmethod
    def get_datasets(path, transform=None, target_transform=None, **kwargs):
        if transform:
            assert isinstance(transform, list)

        transform_list = []
        if transform:
            transform_list.extend(transform)

        transform_list.append(transforms.ToTensor())
        train_dataset = ClutteredMNISTDataset(path, segment='train',
                                              transform=transforms.Compose(transform_list),
                                              target_transform=target_transform, **kwargs)
        test_dataset = ClutteredMNISTDataset(path, segment='test',
                                             transform=transforms.Compose(transform_list),
                                             target_transform=target_transform, **kwargs)
        return train_dataset, test_dataset
