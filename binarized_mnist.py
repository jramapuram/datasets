import os
import torch
import imageio
import functools
import numpy as np
from PIL import Image

from .utils import temp_seed
from .abstract_dataset import AbstractLoader


def _download_file(url, dest):
    import requests
    print("downloading {} to {}".format(url, dest))
    if not os.path.isdir(os.path.dirname(dest)):
        os.makedirs(os.path.dirname(dest))

    r = requests.get(url, allow_redirects=True)
    open(dest, 'wb').write(r.content)


# the original URLS
url_dict = {
    'train': 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat',
    'valid': 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat',
    'test': 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat'
}

# one-image URL
DATA_URL = 'https://github.com/jramapuram/datasets/releases/download/binary_mnist/binary_mnist.png'


class BinarizedMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, path, split='train', download=True,
                 transform=None, target_transform=None, **kwargs):
        self.split = split
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform

        # dest_filename = os.path.join(self.path, url_dict[split].split('/')[-1])
        # if download and not os.path.isfile(dest_filename): # pull the data
        #     _download_file(url_dict[split], dest_filename)
        dest_filename = os.path.join(self.path, DATA_URL.split('/')[-1])
        if download and not os.path.isfile(dest_filename): # pull the data
            _download_file(DATA_URL, dest_filename)

        # https://twitter.com/alemi/status/1042658244609499137
        imgs, labels = np.split(imageio.imread(dest_filename)[..., :3].ravel(), [-70000])
        imgs = np.unpackbits(imgs).reshape((-1, 28, 28))
        imgs, labels = [np.split(y, [50000, 60000]) for y in (imgs, labels)]
        if split == 'train':
            self.imgs, self.labels = [np.vstack([imgs[0], imgs[1]]),
                                      np.concatenate([labels[0], labels[1]])]
        elif split == 'test':
            with temp_seed(1234):
                self.imgs, self.labels = imgs[-1], labels[-1]
                rnd_perm = np.random.permutation(np.arange(len(self.imgs)))
                self.imgs, self.labels = self.imgs[rnd_perm], self.labels[rnd_perm]

        print("[{}] {} samples".format(split, len(self.labels)))

    def __getitem__(self, index):
        target = self.labels[index]
        img = Image.fromarray(self.imgs[index], mode='L')

        if self.transform is not None:
            img = self.transform(img)

        img[img > 0] = 1.0  # XXX: workaround to upsample / downsample

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = target.astype(np.int64)
        return img, target

    def __len__(self):
        return len(self.labels)


class BinarizedMNISTLoader(AbstractLoader):
    """Simple BinarizedMNIST loader, there is no validation set."""

    def __init__(self, path, batch_size, num_replicas=1,
                 train_sampler=None, test_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 cuda=True, **kwargs):

        # Curry the train and test dataset generators.
        train_generator = functools.partial(BinarizedMNISTDataset, path=path, split='train', download=True)
        test_generator = functools.partial(BinarizedMNISTDataset, path=path, split='test', download=True)

        super(BinarizedMNISTLoader, self).__init__(batch_size=batch_size,
                                                   train_dataset_generator=train_generator,
                                                   test_dataset_generator=test_generator,
                                                   train_sampler=train_sampler,
                                                   test_sampler=test_sampler,
                                                   train_transform=train_transform,
                                                   train_target_transform=train_target_transform,
                                                   test_transform=test_transform,
                                                   test_target_transform=test_target_transform,
                                                   num_replicas=num_replicas, cuda=cuda, **kwargs)
        self.output_size = 10  # fixed
        self.loss_type = 'ce'  # fixed

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.input_shape = list(test_img.size()[1:])
        print("derived image shape = ", self.input_shape)
