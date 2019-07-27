from __future__ import print_function
import torch
import os
import os.path
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity


def one_hot_np(num_cols, indices):
    num_rows = len(indices)
    mat = np.zeros((num_rows, num_cols))
    mat[np.arange(num_rows), indices] = 1
    return mat


class SVHNFull(data.Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train.tar.gz",
                  "train.tar.gz", "a649f4cb15c35520e8a8c342d4c0005a"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test.tar.gz",
                 "test.tar.gz", "790d9c8d42f1fcbd219b59956c853a81"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra.tar.gz",
                  "extra.tar.gz", "606f41243d71ca4d5fe66dbaf1f02bee"]}

    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        basedir = os.path.join(self.root, split)
        if split == 'test':
            num_samples = 13068
            matlab_file = 'train_32x32.mat'
        elif split == 'train':
            num_samples = 33402
            matlab_file = 'test_32x32.mat'

        # extract the tar.gz if all the files dont exist
        all_files_exist = True
        for r in range(1, num_samples):
            if not os.path.isfile(os.path.join(basedir, '%d.png' % r)):
                all_files_exist = False

        if not all_files_exist:
            print("extracting files...")
            import tarfile
            tar = tarfile.open(os.path.join(self.root, self.filename), "r:gz")
            tar.extractall(path=self.root)
            tar.close()
        else:
            print("skipping extraction")

        # load the h5py file and pull the correct key
        import h5py
        self.root_mat = h5py.File(os.path.join(basedir, 'digitStruct.mat'), 'r')
        self.df = self.create_dataset(self.root_mat, basedir)
        self.labels = self.find_number_in(self.df['full_num'].values)

        # load the images and resize to [32x32]
        imgs = [np.expand_dims(numpy.array(Image.fromarray(arr).resize((32, 32), Image.BILINEAR)), 0)
                for f in range(1, num_samples+1)]
        self.data = np.vstack(imgs)
        self.data = np.transpose(self.data, (0, 3, 1, 2))

    def get_box_data(self, index):
        '''helper to read matlab file'''
        meta_data = dict()
        meta_data['height'] = []
        meta_data['label'] = []
        meta_data['left'] = []
        meta_data['top'] = []
        meta_data['width'] = []

        def print_attrs(name, obj):
            vals = []
            if obj.shape[0] == 1:
                vals.append(obj[0][0])
            else:
                for k in range(obj.shape[0]):
                    vals.append(int(self.root_mat[obj[k][0]][0][0]))
            meta_data[name] = vals

        box = self.root_mat['/digitStruct/bbox'][index]
        self.root_mat[box[0]].visititems(print_attrs)
        return meta_data

    def find_number_in(self, arr):
        ''' takes a numpy array of [123, 45, ..] and returns a 10 way vector ''
            eg:  133 --> [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]'''
        str_list = [list(str(k)) for k in arr]
        split_list = []
        for list_item in str_list:
            split_list.append([int(arr_elem) for arr_elem in list_item])

        oh = [one_hot_np(10, li) for li in split_list]
        reduced_oh = [np.minimum(np.sum(np.vstack(oh_li), 0), 1) for oh_li in oh]
        return np.asarray(reduced_oh, dtype=np.int32)

    def create_dataset(self, data, path):
        import h5py
        import pandas as pd
        from tqdm import tqdm

        df = []
        for i in tqdm(range(len(data['/digitStruct/name']))):
            meta_data = self.get_box_data(i)
            num_length = len(meta_data['label'])
            if num_length < 6:
                dd = {'filename': '%s/%d.png' % (path, i+1), 'len': num_length}
                for i in range(5):
                    dd['num%d' % (i+1)] = -1
                    dd['bbox%d' % (i+1)] = (0, 0, 0, 0)

                full_num = []
                for i in range(num_length):
                    dd['num%d' % (i+1)] = int(meta_data['label'][i])
                    full_num += [int(meta_data['label'][i])]
                    dd['bbox%d' % (i+1)] = (meta_data['left'][i],
                                            meta_data['top'][i],
                                            meta_data['width'][i],
                                            meta_data['height'][i])

                dd['full_num'] = self.with_map(full_num)

            df.append(dd)

        df = pd.DataFrame(df)
        for i in range(1, 6):
            df.set_value(df[df['num%d'%i] == 10].index, 'num%d'%i, 0)

        for i in range(1, 6):
            df.set_value(df['num%d'%i].isnull(), 'num%d'%i, 10)

        for i in range(1, 6):
            for j in df['bbox%d'%i][df['bbox%d'%i].isnull()].index:
                df.set_value(j, 'bbox%d'%(i+1), (0, 0, 0, 0))

        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def with_map(nums):
        ''' Helper to convert a list of ints to a single int'''
        return int(''.join(map(str, nums)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)
