import datetime
import os
import time
import torch
from utils import AverageMeter, DepthNorm, normalize_image

from loss import ssim
from torch import nn
from tensorboardX import SummaryWriter
from data import getTrainingTestingData

from model import Model


class Trainer:
    def __init__(self, options):
        self.opt = options

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self.device = torch.device(
            "cpu" if self.opt.no_cuda or not torch.cuda.is_available() else f"cuda:{self.opt.device_number}")

        print(f"Using device {self.device}")

        # create model
        self.model = Model()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.opt.learning_rate)
        self.l1_loss = nn.L1Loss()
        self.loss_balancer = 0.9

        # get data
        self.train_loader, self.test_loader = getTrainingTestingData(batch_size=self.opt.batch_size,
                                                                     path=self.opt.data_path,
                                                                     num_workers=self.opt.num_workers)
        self.train_batches = len(self.train_loader)
        self.test_batches = len(self.test_loader)
        self.val_iter = iter(self.test_loader)

        # tensorboard writers
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode), flush_secs=30)

    def train(self):
        self.epoch = 0
        self.step = 0
        self.training_start_time = time.time()

        for self.epoch in range(self.opt.epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequence == 0:
                self.save_model()

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
            # depth = DepthNorm(depth)

            # Predict and compute loss
            output = self.model(image)
            loss = self.compute_loss(depth, output)

            # Update
            loss_meter.update(loss.data.item(), image.size(0))

            # backward pass
            loss.backward()
            self.optimizer.step()

            # time measure and log
            batch_time_meter.update(time.time() - start_time)

            early_logging = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_logging = self.step % 2000 == 0

            if early_logging or late_logging:
                self.log_time(batch_idx, self.train_batches, batch_time_meter, loss_meter)
                self.log_tensorboard('train', image.data, depth.data, output, loss_meter.val)
                self.val()

            start_time = time.time()
            self.step += 1

    def compute_loss(self, inputs, outputs):
        depth_loss = self.l1_loss(outputs, inputs)
        ssim_loss = torch.clamp((1 - ssim(outputs, inputs, val_range=1000.0 / 10.0)) * 0.5, 0, 1)
        loss = (self.loss_balancer * ssim_loss) + ((1 - self.loss_balancer) * depth_loss)
        return loss

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
            # depth = DepthNorm(depth)

            # Predict and compute loss
            output = self.model(image)
            loss = self.compute_loss(depth, output)

            # Update
            self.log_tensorboard('val', image.data, depth.data, output, loss.data.item())
            del sample_batched, image, depth, output, loss

        self.model.train()


    def save_model(self):
        pass

    def log_time(self, batch_idx, number_of_batches, batch_time_meter, loss_meter):
        """ Terminal output log
        """
        remaining_batches = self.train_batches - batch_idx
        remaining_epochs = self.opt.epochs - self.epoch
        epoch_eta_seconds = int(batch_time_meter.val * remaining_batches)
        total_eta_seconds = epoch_eta_seconds * self.opt.epochs
        eta = datetime.timedelta(seconds=total_eta_seconds)
        samples_per_sec = self.opt.batch_size / batch_time_meter.val

        print('epoch: [{0}][{1}/{2}] | '
              'examples/s {3:5.1f} | '
              'batch {batch_time.val:.3f} ({batch_time.sum:.3f}) | '
              'eta {eta} | '
              'loss {loss.val:.4f} ({loss.avg:.4f})'
              .format(self.epoch, batch_idx, number_of_batches,
                      samples_per_sec,
                      batch_time=batch_time_meter,
                      loss=loss_meter,
                      eta=eta))

    def log_tensorboard(self, mode, image, depth, output, loss):
        """Tensorboard logging
        """
        writer = self.writers[mode]
        writer.add_scalar('loss', loss, self.step)

        # save as much as 4 images of batch
        for j in range(min(4, self.opt.batch_size)):
            writer.add_image("color_{}", image[j].data, self.step)
            writer.add_image("depth_{}", normalize_image(depth[j].data), self.step)
            writer.add_image("depth_pred_{}", normalize_image(output[j].data), self.step)
            writer.add_image("diff_{}", normalize_image(torch.abs(output[j].data - depth[j].data)), self.step)