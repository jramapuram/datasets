import os
import h5py
import torch
import pandas as pd
import numpy as np
import numpy as np
import torchvision.transforms.functional as F

from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from torchvision import datasets, transforms


from .abstract_dataset import AbstractLoader
from .utils import create_loader


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


def feature_name_to_idx(feature_name, csv_file):
    '''reads the csv, removes the filenames, returns the idx offset'''
    df = pd.read_csv(csv_file)
    non_img_path_keys = [k for k in df.keys() if '_img' not in k]
    return np.argmax(np.array(non_img_path_keys) == feature_name)


def read_classes(hdf5_file, base_path):
    parsed = hdf5_file['labels'][:]
    csv_file = os.path.join(base_path, 'predictions.csv')
    marine_count_idx = feature_name_to_idx('marine_count', csv_file)
    classes = parsed[:, marine_count_idx].astype(np.int64)

    # remove the large classes which shouldn't be there
    idx = classes < 23
    return classes[idx]


class StarcraftPredictBattleDataset(torch.utils.data.Dataset):
    def __init__(self, path, split='train', train=True, download=True,
                 transform=None, target_transform=None, **kwargs):
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform
        csv_file = os.path.join(path, 'predictions.csv')
        self.idx = feature_name_to_idx('marine_count', csv_file)

        # read the hdf5 file
        h5_path = os.path.join(self.path, '{}_dataset.hdf5'.format(split))
        self.h5_file = h5py.File(h5_path, 'r')

        # hard-coded
        # self.output_size = 124
        self.output_size = 22 + 1 # 22 marines + 0 case

        # debug prints
        print("[{}] {} samples".format(split, len(self.h5_file['labels'])))

    def __getitem__(self, index):
        target = self.h5_file['labels'][index][self.idx]
        fullscreen = F.to_tensor(self.h5_file['fullscreen_img'][index][:].transpose(1, 2, 0))
        minimap = F.to_tensor(self.h5_file['minimap_img'][index][:].transpose(1, 2, 0))
        return [minimap, fullscreen], target

        # fullscreen = F.to_pil_image(F.to_tensor(self.h5_file['fullscreen_img'][index]))
        # minimap = F.to_pil_image(F.to_tensor(self.h5_file['minimap_img'][index]))

        # if self.transform is not None:
        #     minimap = self.transform(minimap)

        # if hasattr(self, 'aux_transform') and self.aux_transform is not None:
        #     fullscreen = self.transform(fullscreen)

        # if not isinstance(fullscreen, torch.Tensor):
        #     fullscreen = F.to_tensor(fullscreen)

        # if not isinstance(minimap, torch.Tensor):
        #     minimap = F.to_tensor(minimap)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        # return [minimap, fullscreen], target

    def __len__(self):
        return len(self.h5_file['labels'])


def compute_sampler_weighting(path, split):
    ''' reads the classes, computes the weights and then does :
             1.0 - #samples / #total_samples                '''
    h5_path = os.path.join(path, '{}_dataset.hdf5'.format(split))
    with h5py.File(h5_path, 'r') as h5_file:
        classes = read_classes(h5_file, path)
        hist, _ = np.histogram(classes, classes.max()+1)
        num_samples = len(classes)
        # weights_unbalanced = [hist[i] for i in classes]
        # weights = [1.0 - (w / num_samples) for w in weights_unbalanced]
        weights = [hist[i] for i in classes]

        # return reciprocal weights
        return 1.0 / np.array(weights)


class StarcraftPredictBattleHDF5Loader(AbstractLoader):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1, **kwargs):
        # derive the weighted samplers
        assert train_sampler is None, "sc2 loader uses weighted sampler"
        assert test_sampler is None, "sc2 loader uses weighted sampler"
        weights_train = compute_sampler_weighting(path, split='train')
        weights_test = compute_sampler_weighting(path, split='test')
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights_train,
                                                                       num_samples=len(weights_train))
        test_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights_test,
                                                                      num_samples=len(weights_test))

        # use the abstract class to build the loader
        super(StarcraftPredictBattleHDF5Loader, self).__init__(StarcraftPredictBattleDataset, path=path,
                                                               batch_size=batch_size,
                                                               train_sampler=train_sampler,
                                                               test_sampler=test_sampler,
                                                               transform=transform,
                                                               target_transform=target_transform,
                                                               num_workers=1,
                                                               use_cuda=use_cuda,
                                                               **kwargs)
        # self.output_size = 124  # fixed
        # self.loss_type = 'bce' # fixed
        self.output_size = 22 + 1 # fixed
        self.loss_type = 'sce' # fixed
        print("derived output size = ", self.output_size)

        # grab a test sample to get the size
        [test_minimap, _], _ = self.train_loader.__iter__().__next__()
        self.img_shp = list(test_minimap.size()[1:])
        print("derived image shape = ", self.img_shp)
