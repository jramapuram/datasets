import functools
import torch
import numpy as np

from .abstract_dataset import AbstractLoader


def generate_samples(num_samples, seq_len, output_size, max_digit):
    """ Helper to generate sampels between 0 and max_digit

    :param num_samples: the total number of samples to generate
    :param seq_len: length of each sequence
    :param output_size: the output size
    :param max_digit: the upper bound in the uniform distribution
    :returns: [B, seq_len*output_size]
    :rtype: torch.Tensor, torch.Tensor

    """
    data = np.random.uniform(0, max_digit, size=[num_samples, seq_len*output_size])
    labels = np.argsort(data, axis=-1)
    data = data.reshape(num_samples, seq_len, output_size)
    labels = labels.reshape(num_samples, seq_len*output_size)
    print('labels = ', labels.shape, " | data = ", data.shape)
    return [data.astype(np.float32), labels]


class SortDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None, target_transform=None,
                 num_samples=2000000, max_digit=1, sequence_length=10, output_size=1):
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # set the output size of sort to 1 if it is not set
        self.output_size = output_size

        # set the number of samples to 2 million by default
        self.num_samples = num_samples

        # max sorting range U ~ [0, max_digit]
        self.max_digit = max_digit

        # set the sequence length if it isn't specified
        self.sequence_length = sequence_length

        # load the sort dataset and labels
        self.data, self.labels = generate_samples(self.num_samples,
                                                  self.sequence_length,
                                                  self.output_size,
                                                  self.max_digit)
        print("[{}] {} samples".format(split, len(self.labels)))

    def __getitem__(self, index):
        target = self.labels[index]
        data = self.data[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # ToTensor() seems to add an extra dimension, fix below.
        if len(data.shape) > 2 and data.shape[0] == 1:
            data = data.view(data.shape[1:])

        return data, target

    def __len__(self):
        return len(self.labels)


class SortLoader(AbstractLoader):
    """Simple sort loader, there is no validation set."""

    def __init__(self, path, batch_size, num_replicas=1,
                 train_sampler=None, test_sampler=None, valid_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 valid_transform=None, valid_target_transform=None, cuda=True,

                 # Custom kwargs for the sort dataset below
                 num_train_samples=2000000, num_test_samples=int(0.2*2000000),
                 num_valid_samples=int(0.2*2000000), max_digit=1,
                 sequence_length=10, output_size=1, **kwargs):

        # Curry the train and test dataset generators.
        train_generator = functools.partial(SortDataset, split='train', num_samples=num_train_samples,
                                            max_digit=max_digit, sequence_length=sequence_length,
                                            output_size=output_size)
        valid_generator = functools.partial(SortDataset, split='valid', num_samples=num_valid_samples,
                                            max_digit=max_digit, sequence_length=sequence_length,
                                            output_size=output_size)
        test_generator = functools.partial(SortDataset, split='test', num_samples=num_test_samples,
                                           max_digit=max_digit, sequence_length=sequence_length,
                                           output_size=output_size)

        super(SortLoader, self).__init__(batch_size=batch_size,
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

        # Determine input-output sizing
        test_sample, b = self.train_loader.__iter__().__next__()
        print("label = ", b.shape)
        assert len(test_sample.shape) == 3, "expect [B, T, F], got {}.".format(test_sample.shape)
        _, seq_len, feature_size = test_sample.shape
        self.input_shape = [seq_len, feature_size]
        self.output_size = output_size * seq_len
        print("derived input shape = ", self.input_shape)
        print("derived output size = ", self.output_size)
