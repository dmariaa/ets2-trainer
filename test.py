import argparse
import glob
import os

import numpy as np
import torch

from PIL import Image
from torchvision import transforms, datasets
from model import Model
from utils import colorize

base_dir = os.path.dirname(__file__)


def parse_args():
    parser = argparse.ArgumentParser("Testing ets2-dataset trained model")

    parser.add_argument('files',
                        help='file_name(s) to test')

    parser.add_argument('--images-path',
                        type=str,
                        help='directory with images to test (default=test)',
                        default=os.path.join(base_dir, 'test'))

    parser.add_argument('--model-name',
                        type=str,
                        help='model to load (default=ets2)',
                        default='ets2')

    parser.add_argument('--no-cuda',  # dest='no_cuda',
                        help='disable CUDA',
                        action='store_true')

    return parser.parse_args()


class Tester():
    def __init__(self, options):
        self.opt = options
        self.model = Model(pretrained_encoder=False)

        self.device = torch.device(
            "cpu" if self.opt.no_cuda or not torch.cuda.is_available() else f"cuda:{self.opt.device_number}")

    def run(self):
        self.load_model()

        images = glob.glob(os.path.join(self.opt.images_path, self.opt.files))
        with torch.no_grad():
            for i, image_file in enumerate(images):
                print(f"predicting depth for {image_file}")
                input_image = Image.open(image_file).convert('RGB')
                original_size = input_image.size
                input_image = input_image.resize((640,480), Image.ANTIALIAS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)
                input_image.to(self.device)
                output = self.model(input_image)

                self.save_output(output, original_size, image_file)

    def save_output(self, output, original_size, original_name):
        orig_width, orig_height = original_size
        depth_name = os.path.splitext(os.path.basename(original_name))[0]

        print(f"Output shape {output.shape}")

        # save depth
        depth = torch.nn.functional.interpolate(output, (orig_height, orig_width), mode="bilinear", align_corners=False)

        print(f"Depth range ({depth.min()},{depth.max()})")

        print(f"Depth shape {depth.shape}")
        np.save(os.path.join(self.opt.images_path, f"{depth_name}.npy"), depth[0].data)


        # save depth colormap
        colorized_depth = colorize(depth[0])

        colorized_depth = colorized_depth.transpose(1, 2, 0)
        print(f"Colorized depth {colorized_depth.shape}")
        im = Image.fromarray(colorized_depth)
        im.save(os.path.join(self.opt.images_path, f"{depth_name}.jpg"))

    def load_model(self):
        models_folder = os.path.join(base_dir, "models", self.opt.model_name)

        if not os.path.exists(models_folder):
            print(f"Model {models_folder} does not exist")
            raise FileNotFoundError

        model_path = os.path.join(models_folder, 'model.pth')

        model_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(model_dict['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()


if __name__ == '__main__':
    options = parse_args()
    tester = Tester(options)
    tester.run()
