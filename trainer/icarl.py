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
        mid = self.seen_classes[-1]
        start = 0
        
        for data, target in tqdm(self.train_iterator):
            data, target = data.cuda(), target.cuda()
            
            output = self.model(data)
            
            loss_CE = self.loss(output[:,:end], target)
            
            loss_KD = 0
            if tasknum > 0:
                score = self.model_fixed(data).data
                loss_KD = torch.zeros(tasknum).cuda()
                for t in range(tasknum):
                    
                    # local distillation
                    start_KD = (t) * self.args.step_size
                    end_KD = (t+1) * self.args.step_size

                    soft_target = F.softmax(score[:,start_KD:end_KD] / T, dim=1)
                    output_log = F.log_softmax(output[:,start_KD:end_KD] / T, dim=1)
                    loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)
                
                loss_KD = loss_KD.sum()
                
            self.optimizer.zero_grad()
            (loss_KD + loss_CE).backward()
            self.optimizer.step()