import argparse
import os

import numpy as np
import torch

from model import Model

base_dir = os.path.dirname(__file__)


def parse_args():
    parser = argparse.ArgumentParser("Testing ets2-dataset trained model")

    parser.add_argument('--images-path',
                        type=str,
                        help='directory with images to test (default=test)',
                        default=os.path.join(base_dir, 'eval'))

    parser.add_argument('--model-name',
                        type=str,
                        help='model to load (default=ets2)',
                        default='ets2')

    parser.add_argument('--no-cuda',  # dest='no_cuda',
                        help='disable CUDA',
                        action='store_true')

    return parser.parse_args()


def compute_errors(depth, pred):
    thresh = np.maximum((depth / pred), (pred / depth))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (depth - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(depth) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(depth - pred) / depth)

    sq_rel = np.mean(((depth - pred) ** 2) / depth)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


class Evaluator():
    def __init__(self, options):
        self.opt = options
        self.model = Model(pretrained_encoder=False)

        self.device = torch.device(
            "cpu" if self.opt.no_cuda or not torch.cuda.is_available() else f"cuda:{self.opt.device_number}")

    def evaluate(self):

        pass

    def evaluate(self):
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