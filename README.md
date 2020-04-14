# datasets

Pytorch dataset loaders that can be cloned into any project.
Currently provides:

    - MNIST
    - FashionMNIST
    - ClutteredMNIST
    - PermutedMNIST
    - CIFAR10
    - SVHN
    - Omniglot
    - SVHN Original (non-centered)
    - Dataset Operators (see section below)

## loader.py

This is the main entrypoint to the datasets.
Loaders can be created with many optional parameters:

``` python
def get_loader(task, data_dir, batch_size, cuda,
               train_transform=None, train_target_transform=None,
               test_transform=None, test_target_transform=None,
               valid_transform=None, valid_target_transform=None,
               train_sampler=None, test_sampler=None, valid_sampler=None,
               increment_seed=True, sequentially_merge_test=True,
               **kwargs):
```

Some field descriptions:

    - args.task : what dataset to load (eg: 'mnist')
    - args.data_dir : where to save / load data from
    - args.batch_size : batch size for train and test loaders
    - args.cuda : this is needed in order to pin memory to GPU for data loaders and to create more workers

**Note**: all simple datasets are auto-downloaded into `data_dir` if they dont exist there.
For larger datasets (eg: mini-imagenet) the loader will expect the data to exist in `data_dir`.


## Dataloader structure

Each dataloader has a `.train_loader` and `.test_loader` that can be utilized in order to iterate over data, eg:

```python
from datasets.loader import get_loader

# use argparse here to extract required members

mnist = get_loader(**kwargs)
for data, label in mnist.train_loader:
    # do whatever you want with the training data
```

Since all the loaders are also operating over images they have the following members:

    - loader.input_shape : the size of the image as [C, H, W]
    - loader.batch_size : the batch size of the loader
    - loader.output_size : dimension of labels (eg: 10 for MNIST)

## Dataset Operators

### Sequentially Split Datasets

You can get a sequential loader by using the `get_sequential_data_loaders` function. This takes the dataset and splits it into several datasets (eg: MNIST --> 10 datasets with each individual digit).

### Merged Datasets

Appending `+` between datasets in `args.task` will return a merged dataset, eg: `mnist+fashion` returns a mixture dataset. Currently it is hardcoded to reshape all data to (32, 32) (to be fixed in future). This can be used with batch OR sequential datasets

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
