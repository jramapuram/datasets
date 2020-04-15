import cv2

import torch
import contextlib
import numpy as np

from typing import Tuple
from copy import deepcopy
from PIL import Image

from datasets.samplers import ClassSampler, FixedRandomSampler

cv2.setNumThreads(0)  # since we use pytorch workers


def resize_lambda(img, size: Tuple[int, int]):
    """converts np image to cv2 and resize."""
    if not isinstance(img, (np.float32, np.float64)):
        img = np.asarray(img)

    if not isinstance(size, tuple):
        size = tuple(size)

    return cv2.resize(img, size)


def permute_lambda(img, pixel_permutation):
    """Permute pixels using provided pixel_permutation"""
    if not isinstance(img, (np.float32, np.float64)):
        img = np.asarray(img)

    img_orig_shape = img.shape
    return Image.fromarray(
        img.reshape(-1, 1)[pixel_permutation].reshape(img_orig_shape)
    )


# from https://tinyurl.com/yy3hyz4d
# sets a temporary numpy seed in scoped context
# eg: with temp_seed(1234):
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def normalize_images(imgs, mu=None, sigma=None, eps=1e-9):
    """Normalize imgs with provided mu /sigma
       or computes them and returns with the normalized
       images and tabulated mu / sigma

    :param imgs: list of images
    :param mu: (optional) provided mean
    :param sigma: (optional) provided sigma
    :param eps: tolerance
    :returns: normalized images
    :rtype: type(imgs), [mu, sigma]

    """
    if mu is None:
        if len(imgs.shape) == 4:
            chans = imgs.shape[1]
            mu = np.asarray(
                [np.mean(imgs[:, i, :, :]) for i in range(chans)]
            ).reshape(1, -1, 1, 1)
        elif len(imgs.shape) == 5:  # glimpses
            chans = imgs.shape[2]
            mu = np.asarray(
                [np.mean(imgs[:, :, i, :, :]) for i in range(chans)]
            ).reshape(1, 1, -1, 1, 1)
            sigma = np.asarray(
                [np.std(imgs[:, :, i, :, :]) for i in range(chans)]
            ).reshape(1, 1, -1, 1, 1)
        else:
            raise Exception("unknown number of dims for normalization")

    if sigma is None:
        if len(imgs.shape) == 4:
            chans = imgs.shape[1]
            sigma = np.asarray(
                [np.std(imgs[:, i, :, :]) for i in range(chans)]
            ).reshape(1, -1, 1, 1)
        elif len(imgs.shape) == 5:  # glimpses
            chans = imgs.shape[2]
            sigma = np.asarray(
                [np.std(imgs[:, :, i, :, :]) for i in range(chans)]
            ).reshape(1, 1, -1, 1, 1)
        else:
            raise Exception("unknown number of dims for normalization")

    return (imgs - mu) / (sigma + eps), [mu, sigma]


def normalize_train_test_images(train_imgs, test_imgs, eps=1e-9):
    ''' simple helper to take train and test images
        and normalize the test images by the train mu/sigma '''
    train_imgs, [mu, sigma] = normalize_images(train_imgs, eps=eps)
    return [train_imgs, normalize_images(test_imgs, mu=mu, sigma=sigma, eps=eps)]


def bw_2_rgb_lambda(img):
    """simple helper to convert BG to RGB."""
    if img.mode == "RGB":
        return img

    return img.convert(mode="RGB")


def binarize(img, block_size: int = 21):
    """Uses Otsu-thresholding to binarize an image."""
    if not isinstance(img, (np.float32, np.float64)):
        img = np.asarray(img)

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, block_size, 0)
    return np.expand_dims(img, -1) if len(img.shape) < 3 else img


def find_max_label(loader):
    """iterate over loader and find the max label size."""
    max_label = 0
    for _, lbls in loader:
        max_seen_lbl = max(lbls)
        if max_seen_lbl > max_label:
            max_label = max_seen_lbl

    return max_label


def label_offset_merger(loaders, batch_size, use_cuda=False):
    ''' iterate over all the loaders and:
           1. finds the max labels
           2. increments loader2 with +loader1_max_label
           3. build a new loader with all the data [uses simple_merger]'''
    # step 1
    max_labels_train = [find_max_label(loader.train_loader) for loader in loaders]
    max_labels_test = [find_max_label(loader.test_loader) for loader in loaders]
    max_labels = np.maximum(max_labels_test, max_labels_train) + 1
    for j in range(1, len(max_labels)):
        max_labels[j] += max_labels[j - 1]

    print('determined offset max_labels: ', max_labels)
    max_labels = torch.from_numpy(max_labels.astype(np.int32))

    # step 2
    def _extract_and_increment(loader, idx):
        data_container, lbl_container = [], []
        for data, labels in loader:  # extract all the data
            data_container.append(data)
            lbl_container.append(labels)

        # handle data concat
        if isinstance(data_container[0], torch.Tensor):
            data_container = torch.cat(data_container, 0)
        elif isinstance(data_container[0], np.array):
            data_container = torch.from_numpy(np.vstack(data_container))
        else:
            raise Exception("unknown data type")

        # handle label concat
        if isinstance(lbl_container[0], torch.Tensor):
            lbl_container = torch.cat(lbl_container, 0)
        elif isinstance(lbl_container[0], np.array):
            lbl_container = torch.from_numpy(np.vstack(lbl_container))
        else:
            raise Exception("unknown label type")

        # do the actual incrementing
        lbl_container += max_labels[idx - 1]
        dataset = torch.utils.data.TensorDataset(data_container, lbl_container)
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=loader.batch_size,
            drop_last=True,
            shuffle=True,
            **kwargs
        )

    # recreate the fucking loaders
    for i in range(1, len(max_labels)):
        loaders[i].train_loader = _extract_and_increment(loaders[i].train_loader, i)
        loaders[i].test_loader = _extract_and_increment(loaders[i].test_loader, i)
        loaders[i].output_size = max_labels[i].cpu().item()

    # step3: finally merge them with simpleMerger
    return simple_merger(loaders, batch_size, use_cuda)


def simple_merger(loaders):
    """Merges train and test datasets given a list of loaders."""
    print("""\nWARN [simplemerger]: no process in place for handling different classes,
ignore this if you called label_offset_merger\n""")

    has_valid = np.all([hasattr(l, 'valid_loader') for l in loaders])
    splits = ['train', 'test'] if not has_valid else ['train', 'test', 'valid']
    for split in splits:
        loaders = sequential_dataset_merger(
            loaders, split, fixed_shuffle=(split == 'test'))  # fixed shuffle test set

    return loaders[-1]


def create_loader(dataset, sampler, batch_size, shuffle,
                  pin_memory=True, drop_last=True,
                  num_workers=0, timeout=0, worker_init_fn=None):
    """Given a dataset and a sampler creates a torch dataloader.
       A  little extra wizardry for ClassSampler.

    :param dataset: the dataset to wrap
    :param sampler: what sampler to use
    :param batch_size: batch size for dataloader
    :param shuffle: whether to shuffle or not
    :param pin_memory: pin memory to CUDA
    :param drop_last: drop the last elems to not have smaller batch size
    :param num_workers: >0 if distributed
    :param timeout: timeout for collecting batch from worker
    :param worker_init_fn: lambda wid: do_something(wid)
    :returns: a dataloader
    :rtype: torch.utils.data.Dataloader

    """
    if isinstance(sampler, ClassSampler):
        # our sampler is hacky; just filters dataset
        # and nulls itself out for GC
        dataset = sampler(dataset)
        sampler = None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        sampler=sampler,
        num_workers=num_workers,
        timeout=timeout,
        worker_init_fn=worker_init_fn)


def sequential_dataset_merger(loaders, split='test', fixed_shuffle=False):
    """Given a list of loaders, merge their test/train/valid sets in each new loader.
       Other splits (split != split) are kept the same.

       Eg: [L1, L2, L3] --> [split(L1), split(L1+L2), split(L1+L2+L3)].

    :param loaders: list of loaders with .'split'_loader member populated
    :param split: dataset split, eg: test, train, valid
    :param fixed_shuffle: forces a single FIXED shuffle (useful when merging test sets).
    :returns: the list of loaders with the merge completed.
    :rtype: list

    """
    # Grab the underlying dataset
    datasets = [getattr(l, split + "_loader").dataset for l in loaders]

    for idx in range(len(datasets)):
        current_dataset = deepcopy(datasets[idx])  # copy to create new
        for ds in datasets[0:idx]:                 # add all previous datasets
            current_dataset += ds

        # Get the current data loader and its sampler
        current_loader = getattr(loaders[idx], split + "_loader")
        current_sampler = current_loader.sampler

        # Handle the sampler and shuffling
        is_shuffled = isinstance(current_loader.sampler, torch.utils.data.RandomSampler)
        new_sampler = current_sampler
        if is_shuffled and not fixed_shuffle:  # ds is shuffled, but dont require fixed shuffle
            new_sampler = None
        elif fixed_shuffle:                    # require fixed shuffle
            new_sampler = FixedRandomSampler(current_dataset)
        else:
            raise ValueError("Unknown sampler / fixed_shuffle combo.")

        # Build the new loader using the existing dataloader
        new_dataloader = create_loader(current_dataset,
                                       sampler=new_sampler,
                                       batch_size=current_loader.batch_size,
                                       shuffle=is_shuffled and not fixed_shuffle,
                                       pin_memory=current_loader.pin_memory,
                                       drop_last=current_loader.drop_last,
                                       num_workers=current_loader.num_workers,
                                       timeout=current_loader.timeout,
                                       worker_init_fn=current_loader.worker_init_fn)
        setattr(loaders[idx], split + "_loader", new_dataloader)

    return loaders


def sequential_test_set_merger(loaders):
    """Given a list of loaders, merge their test sets in each new loader
       while keeping the other sets the same. Syntactic sygar for sequential_dataset_set_merger.

       Eg: [L1, L2, L3] --> [L1, L1+L2(test), L1+L2+L3(test)].

    :param loaders: list of loaders with .test_loader member populated
    :returns: the list of loaders with the merge completed.
    :rtype: list

    """
    return sequential_dataset_merger(loaders, split='train', fixed_shuffle=True)
