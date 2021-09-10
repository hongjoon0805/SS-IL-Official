from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import networks
import trainer


class Trainer(trainer.GenericTrainer):
    def __init__(self, IncrementalLoader, model, args):
        super().__init__(IncrementalLoader, model, args)
        
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        
    def train(self, epoch):
        
        T=2
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        mid = end - self.args.step_size
        
        for data, target in tqdm(self.train_iterator):
            data, target = data.cuda(), target.cuda()
            
            output = self.model(data)
            
            if tasknum > 0 and self.args.SS:
                loss_CE_curr = 0
                loss_CE_prev = 0
                curr_mask = target >= mid
                prev_mask = target < mid
                curr_num = (curr_mask).sum().int()
                prev_num = (prev_mask).sum().int()
                batch_size = curr_num + prev_num
                
                loss_CE_curr = self.loss(output[curr_mask,mid:end], target[curr_mask]%(end-mid)) * curr_num
                loss_CE_prev = 0
                if prev_num > 0:
                    loss_CE_prev = self.loss(output[prev_mask,:mid], target[prev_mask]) * prev_num
                loss_CE = (loss_CE_curr + loss_CE_prev) / batch_size
                
            else:
                loss_CE = self.loss(output[:,:end], target)
            
            loss_KD = 0
            if tasknum > 0:
                # loss_KD
                score = self.model_fixed(data)[:,:mid].data
                
                if self.args.distill == 'local':
                    
                    loss_KD = torch.zeros(tasknum).cuda()
                    for t in range(tasknum):

                        # local distillation
                        start_KD = (t) * self.args.step_size
                        end_KD = (t+1) * self.args.step_size

                        soft_target = F.softmax(score[:,start_KD:end_KD] / T, dim=1)
                        output_log = F.log_softmax(output[:,start_KD:end_KD] / T, dim=1)
                        loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)
                    loss_KD = loss_KD.sum()
                
                if self.args.distill == 'global':
                    soft_target = F.softmax(score / T, dim=1)
                    output_log = F.log_softmax(output[:,:mid] / T, dim=1)
                    loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean') * (T ** 2)
            
            self.optimizer.zero_grad()
            (loss_CE + loss_KD).backward()
            self.optimizer.step()
            