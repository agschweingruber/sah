from datetime import datetime
import numpy as np
import torch
import os
import time
import sys
from sklearn.metrics import roc_auc_score
from glob import glob

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Fitter:
    def __init__(self, model, device, config, isClassification=True):
        self.config = config
        self.epoch = 0
        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5
        
        self.best_auc = 0
        self.device = device
        self.model = model

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        self.loss = torch.nn.CrossEntropyLoss() 
        self.isClassification = isClassification

        self.log(f'Fitter prepared. Device is {self.device}. Optimizer is {self.optimizer}.')

    def fit(self, train_loader, validation_loader, fold):
        _tr = []
        _val = []
        _auc = []

        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')
            t = time.time()

            summary_loss = self.train_one_epoch(train_loader)
            _tr.append(summary_loss.avg)
            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')

            t = time.time()
            
            summary_loss, auc = self.validation(validation_loader)
            _val.append(summary_loss.avg)
            _auc.append(auc)
            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, auc {auc} time: {(time.time() - t):.5f}')

            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-fold{fold}-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-fold{fold}-*epoch.bin'))[:-1]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)
                
            
            np.save(self.base_dir+"/log.npy", np.array([_tr, _val, _auc]))
            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        gt = []
        pe = []
        for step, (images, masks) in enumerate(val_loader):
            images = images.to(self.device, dtype=torch.float)

            masks = masks.to(self.device)

            sys.stdout.write('\r'+
                f'Val Step {step}/{len(val_loader)}, ' + \
                f'summary_loss: {summary_loss.avg:.5f}, ' + \
                f'time: {(time.time() - t):.5f}'
            )

            with torch.no_grad():
                batch_size = images.shape[0]
                output = self.model(images)
                pe.extend(output.detach().cpu().numpy())
                gt.extend(masks.detach().cpu().numpy())
                loss = self.loss(output, masks)
                summary_loss.update(loss.detach().item(), batch_size)
                
        pe = np.array(pe)
        gt = np.array(gt)

        auc = roc_auc_score(np.array(gt), np.array(pe)[:,1])

        return summary_loss, auc

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, masks) in enumerate(train_loader):
            images = images.to(self.device, dtype=torch.float)
            
            masks = masks.to(self.device)

            sys.stdout.write('\r'+
                f'Train Step {step}/{len(train_loader)}, ' + \
                f'summary_loss: {summary_loss.avg:.5f}, ' + \
                f'time: {(time.time() - t):.5f}'
            )

            batch_size = images.shape[0]

            self.optimizer.zero_grad()
            output = self.model(images)

            loss = self.loss(output, masks)
            loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)
            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()
        print("")
        return summary_loss

    def make_prediction(self, crops):
        self.model.eval()
        crops = crops.to(self.device)
        masks = []
        with torch.no_grad():
            masks.append(self.model(crops))
        return masks

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')

