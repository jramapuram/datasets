# coding: utf-8

import os
import sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

import matplotlib.pyplot as plt
from crop_dual_imagefolder import CropDualImageFolderLoader

def test_crops():
    import matplotlib.pyplot as plt
    batch_size = 1
    data_dir = os.path.join(os.path.expanduser("~"), "datasets/fivek_6class")
    #data_dir = os.path.join(os.path.expanduser("~"), "datasets/cluttered_imagefolder_ptiff_v2")
    c = CropDualImageFolderLoader(path=data_dir, batch_size=1, train_sampler=None,
                                  test_sampler=None, transform=None, target_transform=None,
                                  use_cuda=0, window_size=100, max_img_percent=0.25, postfix="_large")
    crop_lbda, (downsampled_img, lbl) = c.train_loader.__iter__().__next__()
    print('downsampled_image [{}] = {}'.format(type(downsampled_img), downsampled_img.shape))
    # plt.imshow(torch.transpose(downsampled_img[0], 0, 2)); plt.show()

    # grab the full image first
    z = np.array([1.0, 0.0, 0.0])
    full_img = crop_lbda[0](z, override=True)
    print('full_image pre [{}] = {}'.format(type(full_img), full_img.shape))
    full_img = torch.transpose(full_img[0], 2, 0).squeeze()
    print('full_image post [{}] = {}'.format(type(full_img), full_img.shape))
    plt.imsave('full.png', full_img)
    #plt.imshow(full_img); plt.show()

    # iterate through a bunch of z's and save them
    for x in [0, 0.25, 0.75, 1.0]:
        for y in [0, 0.25, 0.75, 1.0]:
            full_img = crop_lbda[0]([0.25, x, y])
            full_img = torch.transpose(full_img[0], 2, 0).squeeze()
            plt.imsave('crop_0.25_{}_{}.png'.format(str(x), str(y)), full_img)

if __name__ == "__main__":
    test_crops()


# (base) ➜  tests git:(master) ✗ python test_crop_dual_image.py
# determined secondary image format:  .jpg
# determined secondary image format:  .jpg
# determined img_size:  [3, 32, 32]
# determined output_size:  6
# downsampled_image [<class 'torch.Tensor'>] = torch.Size([1, 3, 32, 32])
# 0 0 [3522, 2348] [3522 2348]
# full_image pre [<class 'torch.Tensor'>] = torch.Size([1, 3, 100, 100])
# full_image post [<class 'torch.Tensor'>] = torch.Size([100, 100, 3])
# 0 0 [880, 587] [3522 2348]
# 0 587 [880, 587] [3522 2348]
# 0 1761 [880, 587] [3522 2348]
# 0 1761 [880, 587] [3522 2348]
# 880 0 [880, 587] [3522 2348]
# 880 587 [880, 587] [3522 2348]
# 880 1761 [880, 587] [3522 2348]
# 880 1761 [880, 587] [3522 2348]
# 2641 0 [880, 587] [3522 2348]
# 2641 587 [880, 587] [3522 2348]
# 2641 1761 [880, 587] [3522 2348]
# 2641 1761 [880, 587] [3522 2348]
# 2642 0 [880, 587] [3522 2348]
# 2642 587 [880, 587] [3522 2348]
# 2642 1761 [880, 587] [3522 2348]
# 2642 1761 [880, 587] [3522 2348]
