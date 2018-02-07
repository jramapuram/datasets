import os
import cv2
import torch
from torchvision import datasets, transforms
from torch.utils.serialization import load_lua


def load_cluttered_mnist(path, segment='train'):
    full = load_lua(os.path.join(path, '%s.t7'%segment))
    data = [t[0].unsqueeze(1) for t in full]
    labels = []
    for t in full:
        _, index = torch.max(t[1], 0)
        labels.append(index)

    return [torch.cat(data).type(torch.FloatTensor).numpy(),
            torch.cat(labels).type(torch.LongTensor).numpy()]


class ClutteredMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, path, segment='train', transform=None, target_transform=None):
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform
        self.segment = segment.lower().strip()  # train or test or val

        # load the images and labels
        self.imgs, self.labels = self._load_from_path()

    def _load_from_path(self):
        # load the tensor dataset from it's t7 binaries
        imgs, labels =  load_cluttered_mnist(self.path, segment=self.segment)
        print("imgs_%s = "%self.segment, imgs.size(),
              " | lbl_%s = "%self.segment, labels.size())
        return imgs, labels

    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ClutteredMNISTLoader(object):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1):
        # first get the datasets
        train_dataset, test_dataset = self.get_datasets(path, transform,
                                                        target_transform)

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        train_sampler = train_sampler(train_dataset) if train_sampler else None
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True if train_sampler is None else False,
            sampler=train_sampler,
            **kwargs)

        # self.val_loader = torch.utils.data.DataLoader(
        #     val_dataset,
        #     batch_size=batch_size,
        #     drop_last=True,
        #     shuffle=False, **kwargs)

        test_sampler = test_sampler(test_dataset) if test_sampler else None
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=False,
            sampler=test_sampler,
            **kwargs)

        self.output_size = 10
        self.batch_size = batch_size
        #self.img_shp = [28, 28]
        self.img_shp = [100, 100]

    @staticmethod
    def get_datasets(path, transform=None, target_transform=None):
        if transform:
            assert isinstance(transform, list)

        transform_list = []
        if transform:
            transform_list.extend(transform)

        transform_list.append(transforms.ToTensor())
        train_dataset = ClutteredMNISTDataset(path, segment='train',
                                              transform=transforms.Compose(transform_list),
                                              target_transform=target_transform)
        test_dataset = ClutteredMNISTDataset(path, segment='test',
                                             transform=transforms.Compose(transform_list),
                                             target_transform=target_transform)
        return train_dataset, test_dataset
