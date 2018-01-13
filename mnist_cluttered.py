import os
import torch
from torchvision import datasets, transforms
from torch.utils.serialization import load_lua


def load_cluttered_mnist(path):
    train = load_lua(os.path.join(path, 'train.t7'))
    val = load_lua(os.path.join(path, 'valid.t7'))
    test = load_lua(os.path.join(path, 'test.t7'))

    train_data = [t[0].unsqueeze(1) for t in train]
    train_labels = []
    for t in train:
        _, index = torch.max(t[1], 0)
        train_labels.append(index)

    valid_data = [v[0].unsqueeze(1) for v in val]
    valid_labels = []
    for t in val:
        _, index = torch.max(t[1], 0)
        valid_labels.append(index)

    test_data = [t[0].unsqueeze(1) for t in test]
    test_labels = []
    for t in test:
        _, index = torch.max(t[1], 0)
        test_labels.append(index)

    return [torch.cat(train_data).type(torch.FloatTensor),
            torch.cat(valid_data).type(torch.FloatTensor),
            torch.cat(test_data).type(torch.FloatTensor),
            torch.cat(train_labels).type(torch.LongTensor),
            torch.cat(valid_labels).type(torch.LongTensor),
            torch.cat(test_labels).type(torch.LongTensor)]

class ClutteredMNISTLoader(object):
    def __init__(self, path, batch_size, sampler=None, use_cuda=1):
        # load the tensor dataset from it's t7 binaries
        imgs_train, imgs_val, imgs_test, \
            labels_train, labels_val, labels_test = load_cluttered_mnist(path=path)
        print("train = ", imgs_train.size(), " | lbl = ", labels_train.size())
        print("val = ", imgs_val.size(), " | lbl = ", labels_val.size())
        print("test = ", imgs_test.size(), " | lbl = ", labels_test.size())
        train_dataset = torch.utils.data.TensorDataset(imgs_train,
                                                       labels_train)
        # val_dataset = torch.utils.data.TensorDataset(imgs_val,
        #                                              labels_val)
        test_dataset = torch.utils.data.TensorDataset(imgs_test,
                                                      labels_test)

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        train_sampler = sampler(train_dataset) if sampler else None
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=not sampler,
            sampler=train_sampler,
            **kwargs)

        # self.val_loader = torch.utils.data.DataLoader(
        #     val_dataset,
        #     batch_size=batch_size,
        #     drop_last=True,
        #     shuffle=False, **kwargs)

        test_sampler = sampler(test_dataset) if sampler else None
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
