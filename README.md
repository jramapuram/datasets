# datasets

Pytorch dataset loaders that can be cloned into any project.
Currently provides:

    - MNIST
    - RotatedMNIST
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
It expects an argparse object as input with the following members:

    - args.task : what dataset to load (eg: 'mnist')
    - args.data_dir : where to save / load data from
    - args.batch_size : batch size for train and test loaders
    - args.cuda : this is needed in order to pin memory to GPU for data loaders and to create more workers

**Note**: all simple datasets are auto-downloaded into  `data_dir` if they dont exist there. For larger datasets (eg: mini-imagenet) the loader will expect the data to exist in `data_dir`.


## Dataloader structure

Each dataloader has a `.train_loader` and `.test_loader` that can be utilized in order to iterate over data, eg:

```python
from datasets.loader import get_loader

# use argparse here to extract required members

mnist = get_loader(args)
for data, label in mnist.train_loader:
    # do whatever you want with the training data
```

Since all the loaders are also operating over images they have the following members:

    - loader.img_shp : the size of the image as [C, H, W]
    - loader.batch_size : the batch size of the loader
    - loader.output_size : dimension of labels (eg: 10 for MNIST)

## Dataset Operators

### Sequentially Split Datasets

You can get a sequential loader by using the `get_sequential_data_loaders` function. This takes the dataset and splits it into several datasets (eg: MNIST --> 10 datasets with each individual digit).

### Merged Datasets

Appending `+` between datasets in `args.task` will return a merged dataset, eg: `mnist+fashion` returns a mixture dataset. Currently it is hardcoded to reshape all data to (32, 32) (to be fixed in future). This can be used with batch OR sequential datasets

### Rotated Datasets

Appending `rotated_` to `args.task` with rotate that entire dataset. This can be used with sequential or normal batch datasets.

### Dataset Transformations

The `utils.py` file houses dataset transformations such as `bw_2_rgb` and `resize` which allow to convert black and white images to RGB. The resize transform allows to resize all images in the dataset (particularly useful if you are merging two datasets)
