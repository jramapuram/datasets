import os
import torch.distributed as dist
import nvidia.dali.ops as ops
import nvidia.dali.types as types

from typing import Optional
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

from .abstract_dataset import AbstractLoader


# For reference
IMAGENET_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]


class Mux(object):
    """DALI doesn't support probabilistic augmentations, so use muxing."""

    def __init__(self, prob=0.5):
        self.to_bool = ops.Cast(dtype=types.DALIDataType.BOOL)
        self.rng = ops.CoinFlip(probability=prob)

    def __call__(self, true_case, false_case):
        """Use masking to mux."""
        condition = self.to_bool(self.rng())
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case


class RandomGrayScale(object):
    """Parallels RandomGrayscale from torchvision. Currently BROKEN!"""

    def __init__(self, prob=0.5, cuda=True):
        raise NotImplementedError("Can't do this with DALI yet.")
        self.mux = Mux(prob=prob)
        self.op = ops.ColorSpaceConversion(device="gpu" if cuda else "cpu",
                                           image_type=types.RGB,
                                           output_type=types.GRAY)

    def __call__(self, images):
        return self.mux(true_case=self.op(images), false_case=images)


class RandomHorizontalFlip(object):
    """Parallels RandomHorizontalFlip from torchvision."""

    def __init__(self, prob=0.5, cuda=True):
        self.mux = Mux(prob=prob)
        self.op = ops.Flip(device="gpu" if cuda else "cpu",
                           horizontal=1,
                           depthwise=0,
                           vertical=0)

    def __call__(self, images):
        return self.mux(true_case=self.op(images), false_case=images)


class ColorJitter(object):
    """Parallels torchvision ColorJitter."""

    def __init__(self, brightness=0.8, contrast=0.8, saturation=0.2, hue=0, prob=0.8, cuda=True):
        """Parallels the torchvision color-jitter transform.

        Args:
            brightness (float or tuple of float (min, max)): How much to jitter brightness.
                brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
                or the given [min, max]. Should be non negative numbers.
            contrast (float or tuple of float (min, max)): How much to jitter contrast.
                contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
                or the given [min, max]. Should be non negative numbers.
            saturation (float or tuple of float (min, max)): How much to jitter saturation.
                saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
                or the given [min, max]. Should be non negative numbers.
            hue (float or tuple of float (min, max)): How much to jitter hue.
                hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
                Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            prob (float): probability of applying the ColorJitter transform at all.
            cuda (bool): if true uses the GPU

        """
        # This RNG doesn't actually work dynamically
        self.mux = Mux(prob=prob)

        # Generates uniform values within appropriate ranges
        self.brightness = ops.Uniform(range=(max(0, 1.0 - brightness), 1.0 + brightness))
        self.contrast = ops.Uniform(range=(max(0, 1.0 - contrast), 1.0 + contrast))
        self.saturation = ops.Uniform(range=(max(0, 1.0 - saturation), 1.0 + saturation))
        self.hue = ops.Uniform(range=(-hue, hue))

        # The actual transform
        self.op = ops.ColorTwist(device="gpu" if cuda else "cpu",
                                 image_type=types.RGB)

    def __call__(self, images):
        true_case = self.op(images,
                            brightness=self.brightness(),
                            saturation=self.saturation(),
                            contrast=self.contrast(),
                            hue=self.hue())
        return self.mux(true_case=true_case, false_case=images)


class CropMirrorNormalize(object):
    """A cleaner version of crop-mirror-normalize."""

    def __init__(self, crop=None,
                 cuda=True,
                 mean=[0.0, 0.0, 0.0],
                 std=[1.0, 1.0, 1.0],
                 flip_prob=0.5):
        """Crops, mirrors horizontally (with prob flip_prob) and normalizes with (x-mean)/std.

        :param crop: tuple for cropping or None for not Cropping
        :param cuda: are we using cuda?
        :param mean: mean to subtract
        :param std: std-dev to divide by
        :param flip_prob: horizon
        :returns: operator
        :rtype: object

        """
        if crop is not None:
            assert isinstance(crop, (tuple, list)), "crop needs to be a tuple/list: (h, w)."

        self.cmnp = ops.CropMirrorNormalize(device="gpu" if cuda else "cpu",
                                            crop=crop,
                                            # output_dtype=types.UINT8, #FLOAT,
                                            output_layout=types.NHWC,
                                            image_type=types.RGB,
                                            mean=mean, std=std)
        self.coin = ops.CoinFlip(probability=flip_prob)

    def __call__(self, images):
        rng = self.coin()
        return self.cmnp(images, mirror=rng)


class HybridPipeline(Pipeline):
    """A simple DALI image pipeline."""

    def __init__(self, data_dir: str, batch_size: int, shuffle: bool = False, device: str = "gpu",
                 transforms=None, target_transform=None, workers_per_replica: int = 2,
                 rank: int = 0, num_replicas: int = 1, seed: Optional[int] = None, **kwargs):
        """Hybrid NVIDIA-DALI pipeline.

        :param data_dir: directory where images are stored.
        :param batch_size: batch size
        :param shuffle: shuffle dataset?
        :param device: cpu or gpu
        :param transforms: a list of nvidia dali ops.
        :param target_transform: same as pytorch target_transform
        :param workers_per_replica: local dataloader threads to use
        :param rank: global rank in a DDP setting (or 0 for local)
        :param num_replicas: total replicas in the pool
        :param seed: optional seed for dataloader
        :returns: Dali pipeline
        :rtype: nvidia.dali.pipeline.Pipeline

        """
        super(HybridPipeline, self).__init__(batch_size=batch_size,
                                             num_threads=workers_per_replica,
                                             device_id=rank,
                                             seed=seed if seed is not None else -1)
        transform_list = []
        if transforms is not None:
            assert isinstance(transforms, (tuple, list)), "transforms need to be a list/tuple or None."
            transform_list.extend(transforms)

        # Convert to CHW for pytorch
        transform_list.append(ops.Transpose(device=device, perm=(2, 0, 1)))

        self.transforms = transform_list
        self.target_transform = target_transform

        # The base file reader
        self.file_reader = ops.FileReader(file_root=data_dir,
                                          shard_id=rank,
                                          num_shards=num_replicas,
                                          random_shuffle=shuffle)

        # The nv-decoder and magic numbers from: https://bit.ly/3cSi359
        # Stated there that these sizes reqd for 'full-sized' image net images.
        device = "mixed" if device == "gpu" else device
        device_memory_padding = 211025920 if device == 'mixed' else 0  # magic numbers
        host_memory_padding = 140544512 if device == 'mixed' else 0    # magic numbers
        self.decode = ops.ImageDecoder(device=device,
                                       device_memory_padding=device_memory_padding,
                                       host_memory_padding=host_memory_padding,
                                       output_type=types.RGB)

        # Set the output_size based on the number of folders in the directory
        self.output_size = sum([1 for d in os.listdir(data_dir)
                                if os.path.isdir(os.path.join(data_dir, d))])

    def define_graph(self):
        # First just read the image path and labels and then decode them.
        images, labels = self.file_reader(name="Reader")
        images = self.decode(images)

        # Now apply the transforms
        if self.transforms:
            for transform in self.transforms:
                images = transform(images)

        # transform the labels if applicable
        if self.target_transform:
            labels = self.target_transform(labels)

        return images, labels


def get_local_rank(num_replicas):
    """Helper to return the current distributed rank."""
    rank = 0
    if num_replicas > 1:
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")

        rank = dist.get_rank()

    return rank


class DALIClassificationIteratorLikePytorch(DALIClassificationIterator):
    def __next__(self):
        """Override this to return things like pytorch."""

        sample = super(DALIClassificationIteratorLikePytorch, self).__next__()

        if sample is not None and len(sample) > 0:
            if isinstance(sample[0], dict):
                images = sample[0]["data"]
                labels = sample[0]["label"]
            else:
                images, labels = sample

            return images.float() / 255, labels.squeeze().long()


class DALIImageFolderLoader(AbstractLoader):
    """Simple DALI image-folder loader, but doesn't follow normal AbstractLoader."""

    def __init__(self, path, batch_size, num_replicas=1,
                 train_sampler=None, test_sampler=None, valid_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 valid_transform=None, valid_target_transform=None,
                 cuda=True, **kwargs):
        rank = get_local_rank(num_replicas)

        # Build the train dataset and loader
        train_dataset = HybridPipeline(data_dir=os.path.join(path, 'train'),
                                       batch_size=batch_size,
                                       shuffle=True,
                                       device="gpu" if cuda else "cpu",
                                       transforms=train_transform,
                                       target_transform=train_target_transform,
                                       rank=rank, num_replicas=num_replicas, **kwargs)
        train_dataset.build()
        self.train_loader = DALIClassificationIteratorLikePytorch(
            train_dataset, size=train_dataset.epoch_size("Reader") // num_replicas,
            fill_last_batch=True,
            last_batch_padded=True,
            auto_reset=True
        )

        # Build the test dataset and loader
        test_dataset = HybridPipeline(data_dir=os.path.join(path, 'test'),
                                      batch_size=batch_size,
                                      shuffle=False,
                                      device="gpu" if cuda else "cpu",
                                      transforms=test_transform,
                                      target_transform=test_target_transform,
                                      rank=rank, num_replicas=1,  # Use FULL test set on each replica
                                      **kwargs)
        test_dataset.build()
        self.test_loader = DALIClassificationIteratorLikePytorch(test_dataset, size=test_dataset.epoch_size("Reader"),
                                                                 fill_last_batch=True,
                                                                 last_batch_padded=True,
                                                                 auto_reset=True)

        # Build the valid dataset and loader
        self.valid_loader = None
        if os.path.isdir(os.path.join(path, 'valid')):
            valid_dataset = HybridPipeline(data_dir=os.path.join(path, 'valid'),
                                           batch_size=batch_size,
                                           shuffle=True,
                                           device="gpu" if cuda else "cpu",
                                           transforms=valid_transform,
                                           target_transform=valid_target_transform,
                                           rank=rank, num_replicas=num_replicas, **kwargs)
            valid_dataset.build()
            self.valid_loader = DALIClassificationIteratorLikePytorch(
                valid_dataset, size=valid_dataset.epoch_size("Reader") // num_replicas,
                fill_last_batch=True,
                last_batch_padded=True,
                auto_reset=True
            )

        # Set the dataset lengths if they exist.
        self.num_train_samples = train_dataset.epoch_size("Reader") // num_replicas
        self.num_test_samples = test_dataset.epoch_size("Reader")
        self.num_valid_samples = valid_dataset.epoch_size("Reader") // num_replicas \
            if self.valid_loader is not None else 0
        print("train = {} | test = {} | valid = {}".format(
            self.num_train_samples, self.num_test_samples, self.num_valid_samples))

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.input_shape = list(test_img.size()[1:])
        print("derived image shape = ", self.input_shape)

        # derive the output size using the imagefolder attr
        self.loss_type = 'ce'  # TODO: try to automagic this later.
        self.output_size = train_dataset.output_size
        print("derived output size = ", self.output_size)

    def set_all_epochs(self, epoch):
        """No-op here as it is handled via the pipeline already."""
        pass

    def set_epoch(self, epoch, split):
        """No-op here as it is handled via the pipeline already."""
        pass
