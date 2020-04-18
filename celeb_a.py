import os
import zipfile
import functools
import requests
import pandas as pd
import numpy as np
import torch

from PIL import Image

from .abstract_dataset import AbstractLoader


def pil_loader(path):
    # open path as file to avoid ResourceWarning :
    # https://github.com/python-pillow/Pillow/issues/835
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            # return img.convert('L')
            return img.convert('RGB')


def _download_file(url, dest):
    print("downloading {} to {}".format(url, dest))
    if not os.path.isdir(os.path.dirname(dest)):
        os.makedirs(os.path.dirname(dest))

    r = requests.get(url, allow_redirects=True)
    open(dest, 'wb').write(r.content)


def _extract_dataset(path):
    extract_path = os.path.join(path, "img_align_celeba")
    if not os.path.isdir(extract_path):
        print('unzipping celebA...', end='', flush=True)
        zip_path = os.path.join(path, 'celeba.zip')
        zip_ref = zipfile.ZipFile(zip_path, 'r')
        zip_ref.extractall(path)
        zip_ref.close()
        print('..done')


# Global to keep track of current feature
current_feature = 0


def read_sequential_dataset(path, split='train', features=['Bald', 'Male',
                                                           'Young', 'Eyeglasses',
                                                           'Wearing_Hat', 'Attractive']):
    """ Read the sequential datasets based on the features in the feature list.

        all_features  = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                         'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                         'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                         'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                         'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                         'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
                         'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                         'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
                         'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                         'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    :param path: the base path
    :param split: train / test /valid
    :param features: the features to extract
    :returns: filenames, labels
    :rtype: np.array, np.array

    """
    _extract_dataset(path)

    # read the eval datafame
    labels_csv_path = os.path.join(path, 'list_eval_partition.txt')
    eval_df = pd.read_csv(
        labels_csv_path, delim_whitespace=True, header=None
    ).values[:, 1].astype(np.int32)

    # read the labels and the paths
    labels_csv_path = os.path.join(path, 'list_attr_celeba.txt')
    df = pd.read_csv(labels_csv_path, skiprows=[0], delim_whitespace=True)

    global current_feature
    print("using feature {}".format(features[current_feature]))

    # extract only the features requested and make them a class
    dataset, filenames = [], []
    split_map = {'train': 0, 'valid': 1, 'test': 2}
    requested_df = df[eval_df == split_map[split]]
    filtered = requested_df[features[current_feature]][requested_df[features[current_feature]] == 1]
    dataset = np.zeros_like(filtered.index.values) + current_feature
    filenames = filtered.index.values

    if split == 'test':
        current_feature = current_feature + 1 if current_feature < 5 else 0

    return filenames, dataset


def read_filenames_and_labels(path, split='train'):
    """ Process all filenames and labels.

    :param path: the base path
    :param split: train / test /valid
    :returns: filenames, labels
    :rtype: np.array, np.array

    """
    _extract_dataset(path)

    # read the eval datafame
    labels_csv_path = os.path.join(path, 'list_eval_partition.txt')
    eval_df = pd.read_csv(
        labels_csv_path, delim_whitespace=True, header=None
    ).values[:, 1].astype(np.int32)

    # read the labels and the paths
    labels_csv_path = os.path.join(path, 'list_attr_celeba.txt')
    df = pd.read_csv(labels_csv_path, skiprows=[0], delim_whitespace=True)

    # extract only the features requested and make them a class
    split_map = {'train': 0, 'valid': 1, 'test': 2}
    requested_df = df[eval_df == split_map[split]]
    labels = requested_df.values

    # the final labels and filenames
    labels[labels == -1] = 0  # celeba likes to use -1 for 0 =/
    filenames = requested_df.index.values
    return filenames, labels


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, path, split='train',  download=True,
                 transform=None, target_transform=None,
                 is_sequential=False, dataset_to_memory=False, **kwargs):
        self.split = split
        self.is_in_memory = False
        self.path = os.path.join(os.path.expanduser(path), "img_align_celeba")
        self.transform = transform
        self.loader = pil_loader
        self.target_transform = target_transform

        if download and not os.path.isdir(os.path.join(path, "img_align_celeba")): # pull the data
            _download_file('https://gitlab.idiap.ch/bob/bob.db.celeba/raw/642079f38ee7639931dded3714039a77f8031304/bob/db/celeba/data/list_attr_celeba.txt',
                           os.path.join(path, 'list_attr_celeba.txt'))
            _download_file('https://gitlab.idiap.ch/bob/bob.db.celeba/raw/642079f38ee7639931dded3714039a77f8031304/bob/db/celeba/data/list_eval_partition.txt',
                           os.path.join(path, 'list_eval_partition.txt'))
            _download_file('https://s3-us-west-1.amazonaws.com/audacity-dlnfd/datasets/celeba.zip', os.path.join(path, 'celeba.zip'))

        # read the labels and filenames
        self.img_names, self.labels = read_sequential_dataset(path, split=split) if is_sequential \
            else read_filenames_and_labels(path, split=split)
        print("[{}] {} samples".format(split, len(self.labels)))

        # Load into memory if requested
        if dataset_to_memory:
            self.to_memory()

    def to_memory(self):
        if self.is_in_memory is False:
            print("Loading CelebA images into memory...", end=' ', flush=True)
            self.imgs = [self.loader(os.path.join(self.path, img_filename))
                         for img_filename in self.img_names]
            # NOTE: this is probably not what you want due to rng augmentations
            # if self.transform is not None:
            #     self.imgs = [self.transform(img) for img in self.imgs]

            self.is_in_memory = True
            print("completed!")

    def __getitem__(self, index):
        """Returns the online or in-memory getter."""
        if self.is_in_memory:
            return self._getitem_memory(index)

        return self._getitem_online(index)

    def _getitem_memory(self, index):
        """Simply returns transformed loaded image."""
        target = self.labels[index]
        img = self.imgs[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _getitem_online(self, index):
        """Normal getter: reads images via threads."""
        target = self.labels[index]
        img = self.loader(os.path.join(self.path, self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.labels)


class _CelebALoader(AbstractLoader):
    """Simple CelebA loader, there is no validation set."""

    def __init__(self, path, batch_size, num_replicas=0,
                 train_sampler=None, test_sampler=None, valid_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 valid_transform=None, valid_target_transform=None,
                 cuda=True, is_sequential=False, **kwargs):

        # Curry the train, test and valid dataset generators.
        train_generator = functools.partial(CelebADataset, is_sequential=is_sequential,
                                            path=path, split='train', download=True)
        valid_generator = functools.partial(CelebADataset, is_sequential=is_sequential,
                                            path=path, split='valid', download=True)
        test_generator = functools.partial(CelebADataset, is_sequential=is_sequential,
                                           path=path, split='test', download=True)

        super(_CelebALoader, self).__init__(batch_size=batch_size,
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
                                            valid_transform=valid_transform,
                                            valid_target_transform=valid_target_transform,
                                            num_replicas=num_replicas, cuda=cuda, **kwargs)
        self.output_size = 40   # fixed
        self.loss_type = 'bce'  # fixed

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.input_shape = list(test_img.size()[1:])
        print("derived image shape = ", self.input_shape)


class CelebALoader(_CelebALoader):
    """Simple CelebA loader with validation set incl"""

    def __init__(self, path, batch_size, num_replicas=0,
                 train_sampler=None, test_sampler=None, valid_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 valid_transform=None, valid_target_transform=None,
                 cuda=True, **kwargs):
        super(CelebALoader, self).__init__(path=path, batch_size=batch_size,
                                           train_sampler=train_sampler, test_sampler=test_sampler, valid_sampler=valid_sampler,
                                           train_transform=train_transform, train_target_transform=train_target_transform,
                                           test_transform=test_transform, test_target_transform=test_target_transform,
                                           valid_transform=valid_transform, valid_target_transform=valid_transform,
                                           num_replicas=num_replicas, cuda=cuda, is_sequential=False, **kwargs)


class CelebASequentialLoader(_CelebALoader):
    """Simple CelebA loader which splits dataset using features."""

    def __init__(self, path, batch_size, num_replicas=0,
                 train_sampler=None, test_sampler=None, valid_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 valid_transform=None, valid_target_transform=None,
                 cuda=True, **kwargs):
        super(CelebALoader, self).__init__(path=path, batch_size=batch_size,
                                           train_sampler=train_sampler, test_sampler=test_sampler, valid_sampler=valid_sampler,
                                           train_transform=train_transform, train_target_transform=train_target_transform,
                                           test_transform=test_transform, test_target_transform=test_target_transform,
                                           valid_transform=valid_transform, valid_target_transform=valid_transform,
                                           num_replicas=num_replicas, cuda=cuda, is_sequential=True, **kwargs)
