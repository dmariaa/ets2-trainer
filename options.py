import argparse
import os

base_dir = os.path.dirname(__file__)


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='ETS2 dataset training options')

        # directories
        self.parser.add_argument('--data-path',
                                 type=str,
                                 help='data directory (default=data)',
                                 default=os.path.join(base_dir, 'data'))
        self.parser.add_argument('--log-dir',
                                 type=str,
                                 help='log directory (default=runs)',
                                 default=os.path.join(base_dir, 'runs')
                                 )

        # training options
        self.parser.add_argument('--model-name',
                                 type=str,
                                 help='name of the folder to save the model in',
                                 default='ets2')
        self.parser.add_argument('--resume-training',
                                 help='resume last checkpoint of the model',
                                 action='store_true')
        self.parser.add_argument('--no-cuda',
                                 help='disable CUDA',
                                 action='store_true')
        self.parser.add_argument('--cuda-device', dest='device_number',
                                 type=int,
                                 help='cuda device/GPU number to use, (default=0)',
                                 default=0)
        self.parser.add_argument("--num-workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)
        self.parser.add_argument("--train-split-ratio",
                                 type=float,
                                 help="train/validation split ratio",
                                 default=0.95)

        # optimization options
        self.parser.add_argument('--epochs',
                                 type=int,
                                 help='number of epochs to run',
                                 default=20)
        self.parser.add_argument('--lr', '--learning-rate', dest='learning_rate',
                                 type=float,
                                 help='initial learning rate',
                                 default=1e-4)
        self.parser.add_argument('--bs', '--batch-size', dest='batch_size',
                                 type=int,
                                 help='batch size',
                                 default=16)

        # logging options
        self.parser.add_argument('--log-frequency',# dest='log_frequency',
                                 type=int,
                                 help='number of batches between each tensorboard log',
                                 default=250)
        self.parser.add_argument('--save-frequency',# dest='save_frequency',
                                 type=int,
                                 help='number of epochs between saves',
                                 default=1)

    def parse(self):
        return self.parser.parse_args()
