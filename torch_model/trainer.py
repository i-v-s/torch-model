import json

import numpy as np
from typing import Optional

from os.path import join

import torch
from torch.nn import Module
from torch.utils.data import DataLoader


class Epoch:
    def __init__(self, trainer, optimizer, data, train):
        self.trainer = trainer
        self.data = data
        self.losses = []
        self.optimizer = optimizer
        self.train = train
        self.worst_loss = None
        
    def on_loss(self, loss, get_worst=None):
        self.losses += loss.detach().cpu().flatten().tolist()

        if self.train:
            loss.mean().backward()
            self.optimizer.step()
            
        if get_worst == 'local':
            wl, wi = loss.detach().max(0)
            return wi.item()
        elif get_worst == 'global':
            wl, wi = loss.detach().max(0)
            if self.worst_loss is None or wl > self.worst_loss:
                self.worst_loss = wl
                return wi.item()
            else:
                return None


class Trainer:
    def __init__(self, model, model_name, state=None):
        if state is None:
            state = {}
        self.model_name = model_name
        self.model = model
        self.epoch = state.get('epoch', 0) + 1
        self.best_loss = state.get('best_loss', None)
        self.logger = None

    def report_quality(self, epoch: Epoch):
        losses = np.array(epoch.losses)
        print(f'Epoch {self.epoch} {"train" if epoch.train else "eval"} mean loss: {losses.mean()}, worst loss: {losses.max()}')

    def save_model(self, directory='models'):
        name = self.model_name
        torch.save(self.model.state_dict(), join(directory, name, '%s.pt' % name))
        with open(join(directory, name, 'state.json'), 'w') as f:
            json.dump({'best_loss': self.best_loss, 'epoch': self.epoch}, f, indent='  ')

    def epochs(self, optimizer, train_loader: DataLoader, eval_loader: Optional[DataLoader] = None):
        k = None
        while k != ord('q'):
            self.model.train()
            train_epoch = Epoch(self, optimizer, train_loader, True)
            yield train_epoch
            self.report_quality(train_epoch)

            if eval_loader is not None:
                self.model.eval()
                eval_epoch = Epoch(self, optimizer, eval_loader, False)
                yield eval_epoch
                self.report_quality(eval_epoch)
            self.epoch += 1
            self.save_model()
