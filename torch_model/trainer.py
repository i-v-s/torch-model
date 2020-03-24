import json

import numpy as np
from typing import List, Dict, Generator, NamedTuple

from os import mkdir
from os.path import join, isdir, isfile

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class StatInfo(NamedTuple):
    min: float
    max: float
    mean: float
    median: float


class Epoch:
    def __init__(self, trainer, optimizer, data, train, name):
        self.trainer = trainer
        self.data = data
        self.losses = []
        self.optimizer = optimizer
        self.train = train
        self.worst_loss = None
        self.name = name

    def __int__(self):
        return self.trainer.epoch
        
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

    def stats(self):
        losses = np.array(self.losses)
        return StatInfo(
            losses.min().item(),
            losses.max().item(),
            losses.mean().item(),
            np.median(losses).item()
        )


class Trainer:
    def __init__(self, model, model_name, state=None, root_dir='models', with_tb=True):
        if state is None:
            state = {}
        self.root_dir = join(root_dir, model_name)
        self.model_name = model_name
        self.model: Module = model
        self.epoch = state.get('epoch', 0) + 1

        self.best_metrics = {}
        if isfile(join(self.root_dir, 'best_metrics.json')):
            with open(join(self.root_dir, 'best_metrics.json'), 'r') as file:
                self.best_metrics = json.load(file)
        self.logger = None
        if with_tb:
            log_dir = join(root_dir, model_name, 'log')
            if not isdir(log_dir):
                mkdir(log_dir)
            self.tb_writer = SummaryWriter(log_dir)
        else:
            self.tb_writer = None

    def report_quality(self, epoch: Epoch, stats: StatInfo):
        print(f'Epoch {self.epoch} {epoch.name} mean loss: {stats.mean}, worst loss: {stats.max}')

    def save_checkpoint(self, name):
        torch.save({
            'model': self.model.state_dict(),
            'epoch': self.epoch
        }, join(self.root_dir, name + '.pt'))
        with open(join(self.root_dir, 'best_metrics.json'), 'w') as f:
            json.dump(self.best_metrics, f, indent='  ')

    def resume(self, name='last'):
        fn = join(self.root_dir, name + '.pt')
        if not isfile(fn):
            return False
        cp = torch.load(fn)
        self.epoch = cp['epoch']
        self.model.load_state_dict(cp['model'])
        return True

    def check_metrics(self, phase: str, metrics: StatInfo):
        name = phase
        bm = self.best_metrics.get(name, None)
        if bm is None or bm > metrics.mean:
            self.best_metrics[name] = metrics.mean
            self.save_checkpoint('best-' + name)

    def write_hparams(self, optimizer, train_loader, eval_loaders):
        params = {k: v for k, v in optimizer.defaults.items() if type(v) is not tuple}
        for name, loader in [('train', train_loader)] + eval_loaders:
            ds = loader.dataset
            if isinstance(ds, Dataset):
                params[name + '_size'] = len(ds)
        self.tb_writer.add_hparams(params, {})

    def epochs(self, optimizer, train_loader: DataLoader,
               *loaders: List[DataLoader],
               **kw_loaders: Dict[str, DataLoader]) -> Generator[Epoch, None, None]:

        k = None
        eval_loaders = [(f'eval_{i + 1}', l) for i, l in enumerate(loaders)] \
            if len(loaders) != 1 else [('eval', loaders[0])]
        eval_loaders += list(kw_loaders.items())
        if self.tb_writer is not None:
            self.write_hparams(optimizer, train_loader, eval_loaders)
            self.tb_writer.add_graph(self.model, torch.zeros((1, 3, 512, 96), device='cuda:0'))

        while k != ord('q'):
            all_stats = []
            self.model.train()
            train_epoch = Epoch(self, optimizer, train_loader, True, 'train')
            yield train_epoch
            stats = train_epoch.stats()
            self.report_quality(train_epoch, stats)
            all_stats.append(('train', stats))
            self.check_metrics('train', stats)

            for name, el in eval_loaders:
                self.model.eval()
                eval_epoch = Epoch(self, optimizer, el, False, name)
                yield eval_epoch
                stats = eval_epoch.stats()
                self.check_metrics(name, stats)
                self.report_quality(eval_epoch, stats)
                all_stats.append((name, stats))
            self.epoch += 1
            self.save_checkpoint('last')

            if self.tb_writer is not None:
                self.tb_writer.add_scalars('Losses/Mean', {name: stats.mean for name, stats in all_stats}, self.epoch)
                self.tb_writer.add_scalars('Losses/Worst', {name: stats.max for name, stats in all_stats}, self.epoch)
                params = torch.cat(list(map(lambda p: p.detach().view(-1), self.model.parameters())))
                self.tb_writer.add_histogram('Model weights', params, self.epoch)
                # self.tb_writer.add_scalar(f'Mean_loss/{epoch.name}', stats.mean, self.epoch)
                # self.tb_writer.add_scalar(f'Worst_loss/{epoch.name}', stats.mean, self.epoch)
