from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import networks
import trainer

class Trainer(trainer.GenericTrainer):
    def __init__(self, IncrementalLoader, model, args):
        super().__init__(IncrementalLoader, model, args)
        self.loss = torch.nn.CrossEntropyLoss(reduce = False)
    
    def train_main(self, logger):
        
        self.model_exp = copy.deepcopy(self.model)
        self.model_exp.eval()
        for param in self.model_exp.parameters():
            param.requires_grad = False
        
        self.model = copy.deepcopy(self.model_fixed)
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        
        logger.model = self.model
        
        schedule = self.args.schedule[:-1] + [self.args.schedule[-1] - 10]
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr, momentum=self.args.momentum,
                                         weight_decay=self.args.decay, nesterov=False)
        
        self.optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, schedule, gamma=0.1)
        
        for epoch in range(self.args.nepochs-20):
            
            self.incremental_loader.mode = 'train'
            self.train(epoch, mode = 'main')
            self.optimizer_scheduler.step()
            
            if epoch % 10 == (10 - 1) and self.args.debug:
                
                logger.evaluate(mode='train', get_results = False)
                logger.evaluate(mode='test', get_results = False)
    
    def balance_fine_tune(self):
        
        self.fc_optimizer = torch.optim.SGD(self.model.module.fc.parameters(), 0.01, weight_decay=self.args.decay)
        self.fc_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.fc_optimizer, [10,15], gamma=0.1)
        for epoch in range(20):
            self.train(epoch, mode = 'bft')
            self.fc_optimizer_scheduler.step()
    
    
    def train(self, epoch, mode = 'None'):
        
        T=2
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        mid = self.seen_classes[-1]
        lamb = mid / end
        start = 0
        
        cls_num = torch.Tensor(self.incremental_loader.get_cls_num_list()).cuda()
        
        for data, target in tqdm(self.train_iterator):
            data, target = data.cuda(), target.cuda()

            output = self.model(data)[:, :end]
            curr_idx = target>=mid
            prev_idx = target<mid
            
            loss_CE = 0
            dw_CE = torch.ones(end).cuda()
            if mode == 'init':
                if curr_idx.float().sum() > 0:
                    loss_CE = (self.loss(output[curr_idx][:,mid:end], target[curr_idx]%(end-mid)) * \
                               dw_CE[target[curr_idx]]).mean()
                if prev_idx.float().sum() > 0:
                    loss_CE -= F.log_softmax(output[prev_idx][:,mid:end], dim=1).mean()
                
            else:
                if mode == 'bft':
                    dw_CE = cls_num.sum() / (cls_num * end)
                loss_CE = (self.loss(output, target) * dw_CE[target]).mean()
            
            loss_KD = 0
            if tasknum > 0 and mode != 'init':
                score = self.model_fixed(data).data
                score_exp = self.model_exp(data).data
                
                # distillation from P
                soft_target = F.softmax(score[:,:mid] / T, dim=1)
                output_log = F.log_softmax(output[:,:mid] / T, dim=1)
                dw_KD = torch.ones(end).cuda()
                if mode == 'bft':
                    dw_KD[:mid] = cls_num[:mid].sum() / (cls_num[:mid] * mid)
                dw_KD = dw_KD[target].view(-1,1)
                loss_KD = (F.kl_div(output_log, soft_target, reduce=False) * lamb * dw_KD).sum(dim=1).mean() * (T**2)
                
                # distillation from C
                soft_target = F.softmax(score_exp[:,mid:end] / T, dim=1)
                output_log = F.log_softmax(output[:,mid:end] / T, dim=1)
                dw_KD_exp = torch.ones(end).cuda()
                if mode == 'bft':
                    dw_KD_exp[mid:] = cls_num[mid:].sum() / (cls_num[mid:] * (end - mid))
                dw_KD_exp = dw_KD_exp[target].view(-1,1)
                loss_KD_exp = (F.kl_div(output_log, soft_target, reduce=False) * (1-lamb) * dw_KD_exp).sum(dim=1).mean() * (T**2)
                
                loss_KD = loss_KD + loss_KD_exp
                
            if mode == 'bft':
                self.fc_optimizer.zero_grad()
                (loss_KD + loss_CE).backward()
                self.fc_optimizer.step()
                
            else:
                
                self.optimizer.zero_grad()
                (loss_KD + loss_CE).backward()
                self.optimizer.step()
