import os
import functools
import pandas as pd
import numpy as np
import torch

from PIL import Image
from sklearn.preprocessing import LabelBinarizer

from .utils import temp_seed
from .abstract_dataset import AbstractLoader


def _split_classes_into_list(name):
    """This dataset provides the classes via a | in the str to signify an or for BCE."""
    classes = name.split("|")
    if len(classes) < 1:
        return [name]  # base case

    return classes


def read_classes(csv_name):
    parsed = pd.read_csv(csv_name).values
    classes = parsed[:, 1]
    filenames = parsed[:, 0]
    classes = [c.replace(" ", "_") for c in classes]
    classes = [_split_classes_into_list(c.lower())
               for c in classes]
    return np.array(classes), filenames


def pil_loader(path):
    # open path as file to avoid ResourceWarning :
    # https://github.com/python-pillow/Pillow/issues/835
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            # force convert for this dataset since
            # there are some which are RGB for some reason
            return img.convert('L')
            # return img.convert('RGB')


class NIHChestXrayDataset(torch.utils.data.Dataset):
    """NIH Chest Xray dataset, assumes downloaded to a folder specified by path."""

    def __init__(self, path, split='train', transform=None, target_transform=None, **kwargs):
        self.split = split
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform

        # hardcoded stuff for this dataset
        self.num_classes = 15
        all_labels = np.array(['infiltration', 'pneumothorax', 'nodule', 'cardiomegaly',
                               'emphysema', 'pneumonia', 'hernia', 'effusion', 'edema',
                               'atelectasis', 'no_finding', 'consolidation', 'mass',
                               'pleural_thickening', 'fibrosis'])
        self.label_encoder = LabelBinarizer()
        self.label_encoder.fit_transform(all_labels)

        # load the images-paths and labels
        self.labels, self.img_names = read_classes(os.path.join(self.path, "Data_Entry_2017.csv"))
        assert len(self.img_names) == len(self.labels), "need same number of files as classes"

        # Read the test files and not the indices to get the train files
        test_files = pd.read_csv(os.path.join(self.path, "test_list.txt"), header=None).values
        test_idx = np.in1d(self.img_names, test_files)
        train_idx = np.logical_not(test_idx)

        # Set the images and labels appropriately
        if split == 'test':
            self.img_names, self.labels = self.img_names[test_idx], self.labels[test_idx]
            with temp_seed(1234):  # Do a fixed shuffle of the test set
                rnd_perm = np.random.permutation(np.arange(len(self.img_names)))
                self.img_names, self.labels = self.img_names[rnd_perm], self.labels[rnd_perm]

        else:
            self.img_names, self.labels = self.img_names[train_idx], self.labels[train_idx]

        # extend the path of the each image filename
        self.img_names = [os.path.join(self.path, "images", img_name)
                          for img_name in self.img_names]
        print("[{}] {} samples".format(split, len(self.labels)))

    def one_hot_label(self, label):
        one_hot_matrix = self.label_encoder.transform(np.array(label).reshape(-1, 1))
        return np.sum(one_hot_matrix, axis=0).astype(np.float32)

    def __getitem__(self, index):
        img_path, target = self.img_names[index], self.labels[index]
        img = pil_loader(img_path)
        # target = np.expand_dims(self.one_hot_label(target), 0)
        target = self.one_hot_label(target)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img_names)


class NIHChestXrayLoader(AbstractLoader):
    """Simple NIHChestXrayLoader loader, there is no validation set."""

    def __init__(self, path, batch_size, num_replicas=1,
                 train_sampler=None, test_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 cuda=True, **kwargs):

        # Curry the train and test dataset generators.
        train_generator = functools.partial(NIHChestXrayDataset, path=path, split='train')
        test_generator = functools.partial(NIHChestXrayDataset, path=path, split='test')

        super(NIHChestXrayLoader, self).__init__(batch_size=batch_size,
                                                 train_dataset_generator=train_generator,
                                                 test_dataset_generator=test_generator,
                                                 train_sampler=train_sampler,
                                                 test_sampler=test_sampler,
                                                 train_transform=train_transform,
                                                 train_target_transform=train_target_transform,
                                                 test_transform=test_transform,
                                                 test_target_transform=test_target_transform,
                                                 num_replicas=num_replicas, cuda=cuda, **kwargs)
        self.output_size = 15   # fixed
        self.loss_type = 'bce'  # fixed

        # grab a test sample to get the size
        test_img, test_lbl = self.train_loader.__iter__().__next__()
        assert test_lbl.shape[-1] == self.output_size, "label size did not match: {} vs {}".format(
            test_lbl.shape, self.output_size)
        self.input_shape = list(test_img.size()[1:])
        print("derived image shape = ", self.input_shape)
