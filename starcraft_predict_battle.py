import os
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


def to_binary(arr):
    return arr.dot(2**np.arange(arr.shape[-1])[::-1])

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


def read_classes(csv_name='predictions.csv'):
    count_fields = ['marine_count',
                    'marauder_count',
                    'siegetank_count',
                    'siegetanksieged_count',
                    'zergling_present',
                    'baneling_present',
                    'hydralisk_present',
                    'zerg_lost',
                    'terran_lost']
    parsed = pd.read_csv(csv_name)
    # classes = np.concatenate([np.expand_dims(parsed[k], 1) for k in count_fields], 1)
    # classes = one_hot(classes)
    classes = parsed['marine_count'].values.astype(np.int64)

    # # filter out the 0 marine elements since it is heavy tailed
    # idx = classes > 0
    # classes = classes[idx]

    # remove the large classes which shouldn't be there
    idx2 = classes < 23
    classes = classes[idx2]

    filenames = {
        'relative_path': parsed['relative_img'].values[idx2],
        'fullscreen_path': parsed['fullscreen_img'].values[idx2],
        'minimap_path': parsed['minimap_img'].values[idx2]
        # 'relative_path': parsed['relative_img'].values[idx][idx2],
        # 'fullscreen_path': parsed['fullscreen_img'].values[idx][idx2],
        # 'minimap_path': parsed['minimap_img'].values[idx][idx2]
    }

    return classes, filenames


class StarcraftPredictBattleDataset(torch.utils.data.Dataset):
    def __init__(self, path, split='train', train=True, download=True,
                 transform=None, target_transform=None, **kwargs):
        self.split = split
        self.path = os.path.expanduser(path)
        self.transform = transform
        #self.loader = vips_loader if USE_PYVIPS is True else pil_loader
        self.loader = pil_loader
        self.target_transform = target_transform

        # hard-coded
        # self.output_size = 124
        self.output_size = 22 + 1 # 22 marines + 0 case

        # load the images-paths and labels
        self.labels, self.img_names = read_classes(os.path.join(self.path, "predictions.csv"))
        assert len(self.img_names['fullscreen_path']) == len(self.labels)

        # determine train-test split
        num_test = int(len(self.labels) * 0.2)
        num_train = len(self.labels) - num_test
        if split == 'train':
            self.img_names = {
                'relative_path': self.img_names['relative_path'][0:num_train],
                'fullscreen_path': self.img_names['fullscreen_path'][0:num_train],
                'minimap_path': self.img_names['minimap_path'][0:num_train]
            }
            self.labels = self.labels[0:num_train]
            # TODO: doesn't the dataloader handle this?
            # shuffle_indices = np.arange(len(self.labels))
            # np.random.shuffle(shuffle_indices)
            # self.img_names, self.labels = self.img_names[shuffle_indices], self.labels[shuffle_indices]
        else:
            self.img_names = {
                'relative_path': self.img_names['relative_path'][-num_test:],
                'fullscreen_path': self.img_names['fullscreen_path'][-num_test:],
                'minimap_path': self.img_names['minimap_path'][-num_test:]
            }
            self.labels = self.labels[-num_test:]

        print("[{}] {} samples".format(split, len(self.labels)))

    def __getitem__(self, index):
        target = self.labels[index]
        fullscreen = self.loader(os.path.join(self.path, self.img_names['fullscreen_path'][index]))
        minimap = self.loader(os.path.join(self.path, self.img_names['minimap_path'][index]))

        if self.transform is not None:
            minimap = self.transform(minimap)

        if hasattr(self, 'aux_transform') and self.aux_transform is not None:
            fullscreen = self.transform(fullscreen)

        if not isinstance(fullscreen, torch.Tensor):
            fullscreen = F.to_tensor(fullscreen)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [minimap, fullscreen], target

    def __len__(self):
        return len(self.labels)


def compute_sampler_weighting(path):
    ''' reads the classes, computes the weights and then does :
             1.0 - #samples / #total_samples                '''
    classes, _ = read_classes(os.path.join(path, "predictions.csv"))
    hist, _ = np.histogram(classes, classes.max()+1)
    num_samples = len(classes)
    # weights_unbalanced = [hist[i] for i in classes]
    # weights = [1.0 - (w / num_samples) for w in weights_unbalanced]
    weights = [hist[i] for i in classes]

    # compute train - test weighting
    num_test = int(num_samples * 0.2)
    num_train = num_samples - num_test
    weights_train = weights[0:num_train]
    weights_test = weights[-num_test:]

    # don't need this anymore
    del classes #XXX

    # return reciprocal weights
    return [1.0 / np.array(weights_train),
            1.0 / np.array(weights_test)]


class StarcraftPredictBattleLoader(AbstractLoader):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1, **kwargs):
        # derive the weighted samplers
        assert train_sampler is None, "sc2 loader uses weighted sampler"
        assert test_sampler is None, "sc2 loader uses weighted sampler"
        weights_train, weights_test = compute_sampler_weighting(path)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights_train,
                                                                       num_samples=len(weights_train))
        test_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights_test,
                                                                      num_samples=len(weights_test))

        # use the abstract class to build the loader
        super(StarcraftPredictBattleLoader, self).__init__(StarcraftPredictBattleDataset, path=path,
                                                           batch_size=batch_size,
                                                           train_sampler=train_sampler,
                                                           test_sampler=test_sampler,
                                                           transform=transform,
                                                           target_transform=target_transform,
                                                           use_cuda=use_cuda, **kwargs)
        # self.output_size = 124  # fixed
        # self.loss_type = 'bce' # fixed
        self.output_size = 22 + 1 # fixed
        self.loss_type = 'sce' # fixed
        print("derived output size = ", self.output_size)

        # grab a test sample to get the size
        [test_minimap, _], _ = self.train_loader.__iter__().__next__()
        self.img_shp = list(test_minimap.size()[1:])
        print("derived image shape = ", self.img_shp)
