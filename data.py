import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random
from utils import generate_split, read_depth_file
from skimage.transform import resize
import PILETS2Tools


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

            if _is_pil_image(depth):
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                depth = np.fliplr(depth)

        return {'image': image, 'depth': depth}


class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))

        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}


class ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        image = transforms.Resize((480, 640))(image)
        image = self.to_tensor(image)

        if _is_pil_image(depth):
            depth = depth.resize((240, 320))
            depth = self.to_tensor(depth)
        else:
            depth = resize(depth, (240, 320), preserve_range=True, mode='reflect', anti_aliasing=True)
            depth = self.to_tensor(depth)

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        # print("Pic before {}".format(np.array(pic).shape))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


def getNoTransform(is_test=False):
    return transforms.Compose([
        ToTensor(is_test=is_test)
    ])


def getDefaultTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])


def getTrainingTestingData(batch_size, path, num_workers, split_ratio):
    # split = generate_split(path, split_ratio, batch_size)
    split_data = np.load(os.path.join(path, 'dataset_split.npz'), allow_pickle=True)
    transformed_training = ETS2Dataset(split_data['train'], path, True, transform=getDefaultTrainTransform())
    transformed_testing = ETS2Dataset(split_data['val'],path, True, transform=getNoTransform())

    return DataLoader(transformed_training, batch_size, shuffle=True, num_workers=num_workers), \
           DataLoader(transformed_testing, batch_size, shuffle=False, num_workers=num_workers)


class ETS2Dataset(Dataset):
    def __init__(self, files, data_path, is_train=False, transform=None):
        super(ETS2Dataset, self).__init__()
        self.data_path = data_path
        self.is_train = is_train
        self.data = files
        self.transform = transform

    def __getitem__(self, item):
        file = self.data[item]
        file_path = os.path.join(self.data_path, file[0], file[1])
        image = Image.open(f"{file_path}.jpg")

        depth_file = read_depth_file(f"{file_path}.depth.raw")
        header = depth_file['header']
        depth = -depth_file['data']
        depth[depth == 0] = 3000.0
        size = (header['height'], header['width'], 1)
        depth.shape = size

        sample = {'image': image, 'depth': depth }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data)
