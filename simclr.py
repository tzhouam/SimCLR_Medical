# import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
import copy
torch.manual_seed(0)
import json
import datetime

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        self.writer = SummaryWriter(log_dir=f'runs/simclr/{self.args.arch}/{current_time}')
        # logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        args_path=os.path.join(self.writer.log_dir, 'args.json')
        arg_log = copy.deepcopy(self.args.__dict__)
        del arg_log['device']
        with open(args_path, 'w') as f:
            json.dump(arg_log, f, indent=2)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        # save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        # logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        # logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        min_loss = 100000
        max_acc = 0
        best_epoch = 0
        for epoch_counter in range(1, self.args.epochs+1):
            for images, _ in tqdm(train_loader):
                # print(images)
                images = torch.cat(images, dim=0)       #2*(B,C,H,W) ->(2B,C,H,W)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                # if n_iter % 5 == 0:
                top1, top5 = accuracy(logits, labels, topk=(1, 5)) # Calculate the accuracy of classify positive and negative images
                self.writer.add_scalar('loss', loss, global_step=n_iter)
                self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            if loss < min_loss:
                min_loss = loss
                best_epoch = epoch_counter
            if max_acc < top1[0]:
                max_acc = top1[0]
            tqdm.write(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
            if epoch_counter % self.args.checkpoint == 0:
                save_checkpoint({
                    'epoch': epoch_counter,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.writer.log_dir, f'checkpoint_{epoch_counter}.pth.tar'))
        tqdm.write("Training has finished.")
        # save model checkpoints
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, f'checkpoint_{self.args.epochs}.pth.tar'))
        self.writer.close()
        
