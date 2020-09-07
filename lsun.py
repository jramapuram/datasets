import functools
from torchvision import datasets

from .abstract_dataset import AbstractLoader


class LSUNLoader(AbstractLoader):
    """LSUN loader from torchvision."""

    def __init__(self, path, batch_size, num_replicas=1,
                 train_sampler=None, test_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 cuda=True, classes=None, **kwargs):
        """Classes can be a list, eg:

           classes = ['bedroom', 'bridge', 'church_outdoor', 'classroom',
                      'conference_room', 'dining_room', 'kitchen',
                      'living_room', 'restaurant', 'tower']
        """
        if classes is None:
            classes_train = 'train'
            classes_test = 'val'
        else:
            classes_train = [c + "_train" for c in classes]
            classes_test = [c + "_val" for c in classes]

        # Some sanity checks for LMDB
        num_replicas = kwargs.get('num_replicas', 1)
        workers_per_replica = kwargs.get('workers_per_replica', None)
        if num_replicas > 1:
            # https://github.com/pytorch/vision/issues/689
            assert workers_per_replica == 0, "LMDB only supports workers_per_replica=0"

        # Curry the train and test dataset generators.
        train_generator = functools.partial(datasets.LSUN, root=path, classes=classes_train)
        test_generator = functools.partial(datasets.LSUN, root=path, classes=classes_test)

        super(LSUNLoader, self).__init__(batch_size=batch_size,
                                         train_dataset_generator=train_generator,
                                         test_dataset_generator=test_generator,
                                         train_sampler=train_sampler,
                                         test_sampler=test_sampler,
                                         train_transform=train_transform,
                                         train_target_transform=train_target_transform,
                                         test_transform=test_transform,
                                         test_target_transform=test_target_transform,
                                         num_replicas=num_replicas, cuda=cuda, **kwargs)
        self.output_size = 10  # fixed
        self.loss_type = 'ce'  # fixed

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.input_shape = list(test_img.size()[1:])
        print("derived image shape = ", self.input_shape)
