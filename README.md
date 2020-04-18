# datasets

Pytorch dataset loaders that can be cloned into any project.
Currently provides:

    - MNIST
    - BinarizedMNIST
    - ClutteredMNIST
    - PermutedMNIST
    - FashionMNIST
    - CelebA
    - CelebASequential (creates dataset based on specific features)
    - CIFAR10
    - CIFAR100
    - SVHN Centered
    - SVHN Non-Centered
    - Omniglot
    - Binarized Omniglot
    - Imagefolder (use for imagenet, etc)
    - MultiImagefolder (load multiple folders: train_* and test_*)
    - AllPairs (online generator)
    - Sort dataset (see DAB paper)
    - StarcraftPredictBattle (see Variational Saccading paper)
    - Dataset Operators (see section below)

## loader.py

This is the main entrypoint to the datasets.
Loaders can be created with many optional parameters, so don't fret the extra params:

``` python
def get_loader(task: str, data_dir: str, batch_size: int, cuda: bool,
               workers_per_replica: int = 4, num_replicas: int = 0, seed: int = 42,
               same_seed_workers: bool = False, timeout: int = 0, pin_memory=True, drop_last=True,
               train_transform=None, train_target_transform=None,
               test_transform=None, test_target_transform=None,
               valid_transform=None, valid_target_transform=None,
               train_sampler=None, test_sampler=None, valid_sampler=None,
               increment_seed=True, sequentially_merge_test=True,
               **kwargs):
    """Returns a loader with .train_loader, .test_loader and optionally .valid_loader

    :param task: string task name
    :param data_dir: string data directory
    :param batch_size: batch size for each loaders
    :param cuda: bool flag to enable pin_memory, etc
    :param workers_per_replica: number of data loader threads per worker
    :param num_replicas: used with DistributedDataParallel
    :param seed: seed for same_seed_workers, **ignored otherwise**
    :param same_seed_workers: set the same seed on the workers
    :param timeout: timeout for worker
    :param pin_memory: pin memory for CUDA; default to true
    :param drop_last: ensures equal sized minibatches
    :param train_transform: (optional) list of data transforms
    :param train_target_transform: (optional) list of label transforms
    :param test_transform: (optional) list of data transforms
    :param test_target_transform: (optional) list of label transforms
    :param valid_transform: (optional) list of data transforms
    :param valid_target_transform: (optional) list of label transforms
    :param train_sampler: (optional) data sampler
    :param test_sampler: (optional) data sampler
    :param valid_sampler: (optional) data sampler
    :param increment_seed: used internally for get_split_data_loaders; increases permutation rng seed
    :param sequentially_merge_test: used internally for get_split_data_loaders: merge all the test sets sequentially
    :returns: AbstractLoader instance
    :rtype:

    """
```

**Note**: all simple datasets are auto-downloaded into `data_dir` if they dont exist there.
For larger datasets (eg: imagenet) the loader will expect the data to exist in `data_dir`.


## Dataloader structure

Each dataloader has a `.train_loader`, `.test_loader` and an optional `.valid_loader` that iterates the data-split.

```python
from datasets.loader import get_loader

# use argparse here to extract required members
kwargs = {'batch_size': args.batch_size, ...}
mnist = get_loader(**kwargs)
for data, label in mnist.train_loader:
    # Do whatever you want with the training data.
    # Similar for .test_loader and .valid_loader
```

Since all the loaders are also operating over images they have the following members:

    - loader.input_shape : the size of the image as [C, H, W]
    - loader.batch_size : the batch size of the loader
    - loader.output_size : dimension of labels (eg: 10 for MNIST)

## Dataset Operators

### Sequentially Split Datasets

You can get a sequential loader by using the `get_sequential_data_loaders` function.
This takes the dataset and splits it into several datasets (eg: MNIST --> 10 datasets with each individual digit).

### Merged Datasets

Appending `+` between datasets in `args.task` will return a merged dataset, eg: `mnist+fashion` returns a mixture dataset.
Currently it is hardcoded to reshape all data to (32, 32) (to be fixed in future). This can be used with batch OR sequential datasets.

### Dataset Transformations

Use standard `torchvision` transforms for each dataset, passed into the loader as a list, eg for CelebA one might:

``` python
    train_transform = [transforms.CenterCrop(160),
                       transforms.Resize((32, 32)),
                       transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.2)], p=0.8),
                       transforms.RandomGrayscale(p=0.2),
                       transforms.ToTensor()]
    test_transform = [transforms.CenterCrop(160),
                      transforms.Resize((32, 32))]
```
