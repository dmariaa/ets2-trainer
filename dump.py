import glob
import os

import matplotlib as plt
from PIL import Image

from scipy.interpolate import LinearNDInterpolator
import numpy as np

from utils import colorize


def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity


files = glob.glob("../resultados/*.groundtruth.png")
cmapper = plt.cm.get_cmap('magma')

for i, file in enumerate(files):
    # depth_map = np.asarray(file)
    depth_map = np.asarray(Image.open(file)) / 256 / 80
    y, x = np.where(depth_map > 0)
    d = depth_map[depth_map != 0]

    xyd = np.stack((x, y, d)).T

    gt = lin_interp(depth_map.shape, xyd)
    # gt = depth_map / 80
    gt = cmapper(1 - gt)
    gt = np.uint8(gt * 255)
    gt = gt[:, :, :3]

    file_name = os.path.splitext(file)[0]
    im = Image.fromarray(gt)
    im.save(f"{file_name}.groundtruth.depth.jpg")
