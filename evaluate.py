import argparse
import os

import numpy as np
import torch

from data import getEvaluationData, getKittiEvaluationData
from model import Model
from skimage.transform import resize

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

    parser.add_argument('--no-cuda',
                        help='disable CUDA',
                        action='store_true')

    parser.add_argument('--cuda-device', dest='device_number',
                        type=int,
                        help='cuda device/GPU number to use, (default=0)',
                        default=0)

    parser.add_argument('--evaluate-dataset',
                        help='Dataset used to perform evaluation',
                        choices=['kitti', 'ets2'],
                        default='ets2')

    parser.add_argument("--disable-median-scaling",
                        help="if set disables median scaling in evaluation",
                        action="store_true")

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

        self.device = torch.device(
            "cpu" if self.opt.no_cuda or not torch.cuda.is_available() else f"cuda:{self.opt.device_number}")
        print(f"Using device {self.device}")

        self.model = Model(pretrained_encoder=False)
        self.load_model()

        self.device = torch.device(
            "cpu" if self.opt.no_cuda or not torch.cuda.is_available() else f"cuda:{self.opt.device_number}")

        if self.opt.evaluate_dataset == 'kitti':
            self.dataset_loader = getKittiEvaluationData(6, self.opt.images_path, num_workers=2)
        else:
            self.dataset_loader = getEvaluationData(6, self.opt.images_path, num_workers=2)

    def evaluate(self):
        self.model.to(self.device)
        self.model.eval()

        pred_depths = []
        gt_depths = []
        errors = []
        ratios = []

        with torch.no_grad():
            for batch_idx, data in enumerate(self.dataset_loader):
                image = torch.autograd.Variable(data['image'].to(self.device))
                depth = torch.autograd.Variable(data['depth'].to(self.device))
                preds = self.model(image)

                for i, pred in enumerate(preds):
                    pred = pred[0].cpu().numpy()
                    d = depth[i].cpu().numpy()

                    s = d.shape

                    mask = np.logical_and(d > 1e-3, d < 80)

                    crop = np.array([0.40810811 * s[0], 0.99189189 * s[0],
                                     0.03594771 * s[1], 0.96405229 * s[1]]).astype(np.int32)
                    crop_mask = np.zeros(mask.shape)
                    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                    mask = np.logical_and(mask, crop_mask)

                    d = d[mask]
                    d = np.clip(d, 1e-3, 80)

                    pred = resize(pred, s, preserve_range=True, mode='reflect', anti_aliasing=True)
                    pred = pred[mask]
                    pred = np.clip(80 / pred, 1e-3, 80)

                    if not self.opt.disable_median_scaling:
                        ratio = np.median(d) / np.median(pred)
                        ratios.append(ratio)
                        pred *= ratio

                    # print(f"Prediction: min({pred.min()}), max({pred.max()}), GT Depth: min({d.min()}), max({d.max()})")
                    errors.append(compute_errors(d, pred))

                print(f"Processed batch {batch_idx}", end='\r')
                # mean_errors = np.array(errors).mean(0)
                # print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
                # print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
                # break

        if not self.opt.disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

        mean_errors = np.array(errors).mean(0)
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")

    def load_model(self):
        models_folder = os.path.join(base_dir, "models", self.opt.model_name)
        if not os.path.exists(models_folder):
            print(f"Model {models_folder} does not exist")
            raise FileNotFoundError

        model_path = os.path.join(models_folder, 'model.pth')
        model_dict = torch.load(model_path, map_location=self.device)

        self.model.load_state_dict(model_dict['model_state_dict'])
        self.model.to(self.device)

        checkpoint_date = model_dict['checkpoint']
        print(f"model {self.opt.model_name} generated on {checkpoint_date.strftime('%d/%m/%Y %H:%M:%S')}")


if __name__ == "__main__":
    options = parse_args()
    evaluator = Evaluator(options)
    evaluator.evaluate()
