import os
import zipfile
import requests
#import urllib.request
import pandas as pd
import numpy as np
import torch
import numpy as np
import torchvision.transforms.functional as F

from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from torchvision import datasets, transforms


from .abstract_dataset import AbstractLoader
from .utils import create_loader

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

def vips_loader(path):
    img = pyvips.Image.new_from_file(path, access='sequential')
    height, width = img.height, img.width
    img_np = np.array(img.write_to_memory()).reshape(height, width, -1)
    del img # mitigate tentative memory leak in pyvips
    return img_np


def pil_loader(path):
    # open path as file to avoid ResourceWarning :
    # https://github.com/python-pillow/Pillow/issues/835
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            # return img.convert('L')
            return img.convert('RGB')


def one_hot_np(num_cols, indices):
    num_rows = len(indices)
    mat = np.zeros((num_rows, num_cols))
    mat[np.arange(num_rows), indices] = 1
    return mat


def one_hot(feature_matrix):
    assert len(feature_matrix.shape) == 2
    maxes = [feature_matrix[:, i].max() for i in range(feature_matrix.shape[-1])]
    column_features = [one_hot_np(max_val+1, col) for max_val, col in zip(maxes, feature_matrix.T)]
    stacked = np.concatenate(column_features, -1)
    #return to_binary(stacked)
    return stacked


def _download_file(url, dest):
    r = requests.get(url, allow_redirects=True)
    open(dest, 'wb').write(r.content)


def bool2int(x):
    # https://tinyurl.com/y7hgxbzs
    y = 0
    for i,j in enumerate(x):
        y += j << i

    return y

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
    extract_path = os.path.join(path, "img_align_celeba")
    if not os.path.isdir(extract_path):
        print('unzipping celebA...', end='', flush=True)
        zip_path = os.path.join(path, 'data.zip')
        zip_ref = zipfile.ZipFile(zip_path, 'r')
        zip_ref.extractall(path)
        zip_ref.close()
        print('..done')

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


def read_generative_dataset(path, split='train'):
    """ Read the generative dataset (no labels)

    :param path: the base path
    :param split: train/test /valid
    :returns: filenames, zeros for labels
    :rtype: np.array, np.array

    """
    extract_path = os.path.join(path, "img_align_celeba")
    if not os.path.isdir(extract_path):
        print('unzipping celebA...', end='', flush=True)
        zip_path = os.path.join(path, 'celeba.zip')
        zip_ref = zipfile.ZipFile(zip_path, 'r')
        zip_ref.extractall(path)
        zip_ref.close()
        print('..done')

    # read the labels and the paths
    labels_csv_path = os.path.join(path, 'list_attr_celeba.txt')
    df = pd.read_csv(labels_csv_path, skiprows=[0], delim_whitespace=True)

    # read the eval datafame
    labels_csv_path = os.path.join(path, 'list_eval_partition.txt')
    eval_df = pd.read_csv(
        labels_csv_path, delim_whitespace=True, header=None
    ).values[:, 1].astype(np.int32)

    # extract the correct split and return 0s for the labels
    dataset, filenames = [], []
    split_map = {'train': 0, 'valid': 1, 'test': 2}
    requested_df = df[eval_df == split_map[split]]
    dataset = np.zeros_like(requested_df.index.values)
    filenames = requested_df.index.values

    return filenames, dataset


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, path, split='train', train=True, download=True,
                 transform=None, target_transform=None, is_generative=True, **kwargs):
        self.split = split
        self.path = os.path.join(os.path.expanduser(path), "img_align_celeba")
        self.transform = transform
        #self.loader = vips_loader if USE_PYVIPS is True else pil_loader
        self.loader = pil_loader
        self.target_transform = target_transform

        if download and not os.path.isdir(self.path): # pull the data
            download_path = os.path.join(path, 'celeba.zip')
            _download_file('https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip', download_path)

        # read the labels
        self.img_names, self.labels = read_sequential_dataset(path, split=split) if not is_generative \
            else read_generative_dataset(path, split=split)
        self.output_size = 1 if is_generative else 255
        print("[{}] {} samples".format(split, len(self.labels)))

    def __getitem__(self, index):
        target = self.labels[index]
        img = self.loader(os.path.join(self.path, self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.labels)


class CelebALoader(AbstractLoader):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1,
                 is_generative=True, **kwargs):
        # use the abstract class to build the loader
        super(CelebALoader, self).__init__(CelebADataset, path=path,
                                           batch_size=batch_size,
                                           train_sampler=train_sampler,
                                           test_sampler=test_sampler,
                                           transform=transform,
                                           target_transform=target_transform,
                                           use_cuda=use_cuda,
                                           is_generative=is_generative,
                                           **kwargs)
        #self.output_size = 40
        #self.output_size = 255
        # self.output_size = 6
        self.output_size = 1 if is_generative else 6
        self.loss_type = 'sce' # fixed
        print("derived output size = ", self.output_size)

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.img_shp = list(test_img.size()[1:])
        print("derived image shape = ", self.img_shp)
