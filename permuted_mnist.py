import numpy as np
from torchvision import transforms

from .mnist import MNISTLoader
from .utils import permute_lambda


def generate_permutation(seed, image_size=28*28):
    """Creates a pixel-permutation using the seed and size."""
    orig_seed = np.random.get_state()
    np.random.seed(seed)
    # print("USING SEED ", seed)
    perms = np.random.permutation(image_size)
    np.random.set_state(orig_seed)
    return perms


def get_permutation_lambda_transform(pixel_permutation):
    """Creates a torchvision transform for pixel permutation."""
    return transforms.Lambda(lambda x: permute_lambda(x, pixel_permutation=pixel_permutation))


class PermutedMNISTLoader(MNISTLoader):
    """Adds a unique (fixed) pixel permutation to the entire dataset."""

    def __init__(self, path, batch_size, num_replicas=1,
                 train_sampler=None, test_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 cuda=True, **kwargs):
        # generate the unique permutation for this loader
        seed = kwargs.get('seed', np.random.randint(1, 9999))
        perm = generate_permutation(seed)
        perm_transform = get_permutation_lambda_transform(perm)

        def _append_perm_transform(transform_list):
            """Adds the perm transform"""
            if transform_list is not None:
                assert isinstance(transform_list, (list, tuple))
                transform_list.insert(0, perm_transform)
            else:
                transform_list = [perm_transform]

            return transform_list

        # adds the permutation transform to both train and test.
        train_transform = _append_perm_transform(train_transform)
        test_transform = _append_perm_transform(test_transform)

        super(PermutedMNISTLoader, self).__init__(path=path, batch_size=batch_size,
                                                  train_transform=train_transform, test_transform=test_transform,
                                                  train_sampler=train_sampler, train_target_transform=train_target_transform,
                                                  test_sampler=test_sampler, test_target_transform=test_target_transform,
                                                  num_replicas=num_replicas, cuda=cuda, **kwargs)
