import os
import pandas as pd
import numpy as np
import torch
import numpy as np

from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from torchvision import datasets, transforms

from .abstract_dataset import AbstractLoader
from .utils import create_loader


def _split_classes_into_list(name):
    classes = name.split("|")
    if len(classes) < 1:
        return [name] # base case

    return classes


def read_classes(csv_name):
    parsed = pd.read_csv(csv_name).values
    classes = parsed[:, 1]
    filenames = parsed[:, 0]
    classes = [c.replace(" ", "_") for c in classes]
    classes = [_split_classes_into_list(c.lower())
               for c in classes]
    return classes, filenames


def find_unique_classes(class_list):
    return list(set([item for sublist in class_list
                     for item in sublist]))


def pil_loader(path):
    # open path as file to avoid ResourceWarning :
    # https://github.com/python-pillow/Pillow/issues/835
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            # force convert for this dataset since
            # there are some which are RGB for some reason
            return img.convert('L')
            #return img.convert('RGB')


class NIHChestXrayDataset(torch.utils.data.Dataset):
    def __init__(self, path, split='train', train=True, download=True,
                 transform=None, target_transform=None, **kwargs):
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

        # unique_classes = find_unique_classes(self.labels)
        # print("determined {} classes: {}".format(
        #     len(unique_classes), unique_classes)
        # )
        # print(unique_classes)
        # exit(0)


        # determine train-test split
        num_test = int(len(self.labels) * 0.2)
        num_train = len(self.labels) - num_test
        if split == 'train':
            self.img_names, self.labels = self.img_names[0:num_train], self.labels[0:num_train]
            # TODO: doesn't the dataloader handle this?
            # shuffle_indices = np.arange(len(self.labels))
            # np.random.shuffle(shuffle_indices)
            # self.img_names, self.labels = self.img_names[shuffle_indices], self.labels[shuffle_indices]
        else:
            self.img_names, self.labels = self.img_names[-num_test:], self.labels[-num_test:]

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
        #target = np.expand_dims(self.one_hot_label(target), 0)
        target = self.one_hot_label(target)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img_names)


class NIHChestXrayLoader(AbstractLoader):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1, **kwargs):
        super(NIHChestXrayLoader, self).__init__(NIHChestXrayDataset, path=path,
                                                 batch_size=batch_size,
                                                 train_sampler=train_sampler,
                                                 test_sampler=test_sampler,
                                                 transform=transform,
                                                 target_transform=target_transform,
                                                 use_cuda=use_cuda, **kwargs)
        self.output_size = 15  # fixed
        self.loss_type = 'bce' # fixed

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.img_shp = list(test_img.size()[1:])
        print("derived image shape = ", self.img_shp)
