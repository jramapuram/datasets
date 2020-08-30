"""Data saved using dataloader from: https://bit.ly/34qn0Bb , used for GQN TFRecord datasets."""

import os
import functools
import torch
import numpy as np

from torchvision.datasets.folder import default_loader

from .abstract_dataset import AbstractLoader


class DMLABMazesDataset(torch.utils.data.Dataset):
    """Uses the standard image reader which changes for episode_length."""

    def __init__(self, path, segment='train', transform=None, target_transform=None, episode_length=18):
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.episode_length = episode_length
        self.target_transform = target_transform
        self.segment = segment.lower().strip()  # train or test or val

        # use the default image loader
        self.loader = default_loader

        # create the datareader object and a seccion
        self.num_samples_per_episode = 300  # per episode
        self.num_episodes = 10000 if segment == 'train' else 1000
        self.episode_folder = [os.path.join(self.path, 'images', segment, str(idx))
                               for idx in range(self.num_episodes)]

    def _sample_idxs(self):
        last = self.num_samples_per_episode - 1
        begin = np.random.randint(0, last - self.episode_length)
        end = begin + self.episode_length
        return begin, end

    def __getitem__(self, index):
        episode_path = self.episode_folder[index]
        begin, end = self._sample_idxs()
        query = [self.loader(os.path.join(episode_path, '{}.png'.format(idx)))
                 for idx in np.arange(begin, end)]
        target = self.loader(os.path.join(episode_path, 'target.png'))

        if self.transform is not None:
            query = [self.transform(q) for q in query]
            target = self.transform(target)

        # The target is also an image so don't use target_transform as it is typically for labels
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return (torch.cat([q.unsqueeze(0) for q in query], 0),
                target)

    def __len__(self):
        return len(self.episode_folder)


class DMLabMazesLoader(AbstractLoader):
    """DMLab Mazes Loader from GQN datasets."""

    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 num_replicas=1, cuda=True,

                 # Custom dataset kwargs
                 episode_length=18, **kwargs):

        # Curry the train and test dataset generators.
        train_generator = functools.partial(DMLABMazesDataset, path=path, segment='train', episode_length=episode_length)
        test_generator = functools.partial(DMLABMazesDataset, path=path, segment='test', episode_length=episode_length)

        super(DMLabMazesLoader, self).__init__(batch_size=batch_size,
                                               train_dataset_generator=train_generator,
                                               test_dataset_generator=test_generator,
                                               train_sampler=train_sampler,
                                               test_sampler=test_sampler,
                                               train_transform=train_transform,
                                               train_target_transform=train_target_transform,
                                               test_transform=test_transform,
                                               test_target_transform=test_target_transform,
                                               num_replicas=num_replicas, cuda=cuda, **kwargs)
        self.output_size = 0
        self.loss_type = 'pixels'  # fixed

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.input_shape = list(test_img.size()[-3:])
        print("derived image shape = ", self.input_shape)
