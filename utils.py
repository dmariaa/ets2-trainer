import glob
import os
import struct
import random
import matplotlib
import matplotlib.cm
import numpy as np
import rawutil


def linearize_depth(depth, near=0.1, far=3000.0):
    p33 = far / (far - near)
    p43 = -(far * near) / (far - near)
    return p43 / (depth - p33)


def depth_norm(depth, near=0.1, far=3000.0):
    depth = far / depth
    return depth


def depth_norm_old(depth, maxdepth=None):
    if maxdepth is None:
        return depth
    else:
        return maxdepth / depth


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def colorize(value, vmin=None, vmax=None, cmap='magma'):
    value = value.cpu().numpy()[0, :, :]
    value = normalize_data(value)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def normalize_data(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max())
    mi = float(x.min())
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def generate_split(base_path, train_ratio=0.01, batch_size=1, image_type="jpg"):
    dirs = glob.glob(f"{base_path}/*/")
    files = []

    for directory in dirs:
        filenames_base = os.path.join(directory, f"*.{image_type}")
        f = sorted(glob.glob(filenames_base))
        files.extend(f)

    files = [os.path.splitext(f)[0] for f in files]

    total_number_of_files = len(files)
    val_number_of_files = int(total_number_of_files * (1 - train_ratio))
    train_number_of_files = total_number_of_files - val_number_of_files

    remainder = train_number_of_files % batch_size
    train_number_of_files += remainder
    val_number_of_files -= remainder

    random.shuffle(files)

    train_files = files[:-val_number_of_files]
    validation_files = files[-val_number_of_files:]

    print(f"training number of files: {total_number_of_files - val_number_of_files}")
    print(f"validation number of files: {val_number_of_files}")

    return {
        'train': train_files,
        'validation': validation_files
    }


def read_depth_file(file):
    data = np.fromfile(file, dtype='byte')
    header = get_header(data)
    return {
        'header': header,
        'data': get_data(data, header)
    }


def get_header(data):
    header = {
        "magic": bytes(struct.unpack('bb', data[0:2])).decode('utf-8'),
        "size": struct.unpack('<l', data[2:6])[0],
        "width": struct.unpack('<l', data[6:10])[0],
        "height": struct.unpack('<l', data[10:14])[0],
        "min_val": struct.unpack('<f', data[14:18])[0],
        "max_val": struct.unpack('<f', data[18:22])[0],
        "bit_depth": struct.unpack('<h', data[22:24])[0],
        "offset": struct.unpack('<l', data[24:28])[0]
    }
    return header


def get_data(file_data, header):
    data_bytes = header['bit_depth'] // 8
    format = '<%se' % int((header['size'] - 28) / data_bytes)
    data = np.array(rawutil.unpack(format, file_data[header['offset']:]))
    return data
