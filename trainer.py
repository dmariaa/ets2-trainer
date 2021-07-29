import datetime
import glob
import json

import numpy as np
import os
import time
import torch
from utils import AverageMeter, depth_norm, normalize_data, colorize

from loss import ssim
from torch import nn
from tensorboardX import SummaryWriter
from data import getTrainingTestingData

from model import Model

from torchinfo import summary

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.model_resumed = None
        self.epoch = 0
        self.step = 0
        self.training_start_time = None

        self.depth_min_meters = 0.1
        self.depth_max_meters = 3000.0
        self.depth_delta_meters = self.depth_max_meters / self.depth_min_meters

        self.device = torch.device(
            "cpu" if self.opt.no_cuda or not torch.cuda.is_available() else f"cuda:{self.opt.device_number}")
        print(f"Using device {self.device}")

        # create model
        self.model = Model()

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.opt.learning_rate)

        total_params = 0
        for parameter in self.model.parameters():
            total_params += parameter.numel()

        #print(f"Total trainable parameters: {total_params}")
        #exit()

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        if os.path.exists(self.log_path):
            if self.opt.resume_training:
                self.resume_model()
            else:
                now = datetime.datetime.now()
                backup_name = f"{self.log_path}_{now.strftime('%Y%m%d%H%M%S')}"
                print(f"Making a backup copy of model {self.log_path} in {backup_name}")
                os.rename(self.log_path, backup_name)
                os.makedirs(self.log_path)
        else:
            os.makedirs(self.log_path)

        self.model = self.model.to(self.device)

        #summary(self.model, (1, 3, 480, 640))
        #exit()

        self.l1_loss = nn.L1Loss()
        self.loss_balancer = 0.1

        # get data
        self.train_loader, self.test_loader = getTrainingTestingData(batch_size=self.opt.batch_size,
                                                                     path=self.opt.data_path,
                                                                     split_ratio=self.opt.train_split_ratio,
                                                                     num_workers=self.opt.num_workers)
        self.train_batches = len(self.train_loader)
        self.test_batches = len(self.test_loader)
        self.val_iter = iter(self.test_loader)
        self.step = self.epoch * self.train_batches
        self.total_steps = self.opt.epochs * self.train_batches

        # tensorboard writers
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode), flush_secs=30)

        # save model options
        self.save_opts()

    def train(self):
        self.training_start_time = time.time()

        for self.epoch in range(self.epoch, self.opt.epochs):
            self.run_epoch()

    def run_epoch(self):
        batch_time_meter = AverageMeter()
        loss_meter = AverageMeter()
        start_time = time.time()

        self.model.train()

        for batch_idx, sample_batched in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # prepare image and depth
            image = torch.autograd.Variable(sample_batched['image'].to(self.device))
            depth = torch.autograd.Variable(sample_batched['depth'].to(self.device, non_blocking=True))

            # Normalize depth????
            depth_n = depth_norm(depth)

            # Predict and compute loss
            output = self.model(image)
            loss = self.compute_loss(depth_n, output)

            # Update
            loss_meter.update(loss.data.item(), image.size(0))

            # backward pass
            loss.backward()
            self.optimizer.step()

            # time measure and log
            batch_time_meter.update(time.time() - start_time, image.size(0))

            early_logging = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_logging = self.step % 2000 == 0

            if early_logging or late_logging:
                self.log_time(batch_idx, self.train_batches, batch_time_meter, loss_meter)
                self.log_tensorboard('train', image.data, depth.data, output, loss_meter.val)
                self.val()

            start_time = time.time()
            self.step += 1

        if (self.epoch + 1) % self.opt.save_frequency == 0:
            self.save_model(self.epoch, loss_meter.val)

    def val(self):
        """Validate model
        """
        self.model.eval()

        try:
            sample_batched = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.test_loader)
            sample_batched = self.val_iter.next()

        with torch.no_grad():
            image = torch.autograd.Variable(sample_batched['image'].to(self.device))
            depth = torch.autograd.Variable(sample_batched['depth'].to(self.device, non_blocking=True))

            # Normalize depth????
            # depth = depth_norm(depth)

            # Predict and compute loss
            output = self.model(image)
            loss = self.compute_loss(depth, output)

            # Update
            self.log_tensorboard('val', image.data, depth.data, output, loss.data.item())
            del sample_batched, image, depth, output, loss

        self.model.train()

    def compute_loss(self, inputs, outputs):
        depth_loss = self.l1_loss(outputs, inputs)
        ssim_loss = torch.clamp((1 - ssim(outputs, inputs, val_range=3000.0)) * 0.5, 0, 1)
        loss = ssim_loss + (self.loss_balancer * depth_loss)
        # print(f"depth range: ({inputs.min()},{inputs.max()}), predicted range: ({outputs.min()},{outputs.max()}) depth_loss: {depth_loss}, ssim_loss: {ssim_loss}, loss: {loss}")
        return loss

    def log_time(self, batch_idx, number_of_batches, batch_time_meter, loss_meter):
        """ Terminal output log
        """
        remaining_batches = self.train_batches - batch_idx
        remaining_epochs = self.opt.epochs - self.epoch
        remaining_steps = remaining_batches + (remaining_epochs - 1) * self.train_batches

        epoch_eta_seconds = int(batch_time_meter.avg * remaining_batches)
        total_eta_seconds = int(batch_time_meter.avg * remaining_steps)

        eta = datetime.timedelta(seconds=total_eta_seconds)
        samples_per_sec = batch_time_meter.avg

        print('epoch: [{0}/{total_epochs}][{1}/{2}] | '
              'step: [{step}/{remaining_steps}] | '
              'examples/s {3:5.1f} | '
              'batch {batch_time.val:.3f} ({batch_time.sum:.3f}) | '
              'eta {eta} | '
              'loss {loss.val:.4f} ({loss.avg:.4f})'
              .format(self.epoch, batch_idx, number_of_batches,
                      samples_per_sec,
                      remaining_steps=self.total_steps,
                      step=self.step,
                      total_epochs=self.opt.epochs,
                      batch_time=batch_time_meter,
                      loss=loss_meter,
                      eta=eta), flush=True)

    def log_tensorboard(self, mode, image, depth, output, loss):
        """Tensorboard logging
        """
        writer = self.writers[mode]
        writer.add_scalar('loss', loss, self.step)

        #print(f"{mode}: image ({image.min()},{image.max()}), depth ({depth.min()},{depth.max()}), "
        #      f"output ({output.min()},{output.max()})")

        # save as much as 4 images of batch
        for j in range(min(4, len(image))):
            writer.add_image(f"color_{j}", image[j].data, self.step)
            writer.add_image(f"depth_{j}", colorize(depth[j].data), self.step)
            writer.add_image(f"depth_pred_{j}", colorize(output[j].data), self.step)
            writer.add_image(f"diff_{j}", colorize(torch.abs(output[j].data - depth[j].data)), self.step)

    def save_opts(self):
        save_folder = os.path.join(self.log_path, "models")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        opts_data = self.opt.__dict__.copy()
        with open(os.path.join(save_folder, "options.json"), 'w') as json_file:
            json.dump(opts_data, json_file, indent=2)

    def save_model(self, epoch, loss):
        save_folder = os.path.join(self.log_path, "models")

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_file = os.path.join(save_folder, f"model_epoch_{epoch}.pth")

        model_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'checkpoint': datetime.datetime.now(),
            'additional_data': {
                'samples': {
                    'train': len(self.train_loader),
                    'validation': len(self.test_loader)
                },
                'checkpoint_loss_value': loss,
                'model_resumed': self.model_resumed
            }
        }

        torch.save(model_dict, save_file)

    def resume_model(self):
        load_folder = os.path.join(self.log_path, "models")

        if not os.path.exists(load_folder):
            print(f"Can't resume model, {self.opt.model_name}, {load_folder} not found")
            raise FileNotFoundError()

        checkpoints = glob.glob(os.path.join(load_folder, "model_epoch_*.pth"))

        if len(checkpoints) == 0:
            print("No checkpoint to load, start training from zero")
            return

        checkpoints.sort(key=lambda x: int(os.path.basename(x)[len("model_epoch_"):-4]), reverse=True)
        load_file = os.path.join(load_folder, checkpoints[0])
        checkpoint = torch.load(load_file)

        self.epoch = checkpoint['epoch'] + 1
        self.opt.epochs += self.epoch
        self.resume_data = checkpoint['additional_data']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        checkpoint_date = checkpoint['checkpoint'].strftime('%d/%m/%Y %H:%M:%S')
        print(f"model {self.opt.model_name} resumed from checkpoint [{checkpoint_date}]\n"
              f"   going to run from epoch: {self.epoch} to {self.opt.epochs}")
