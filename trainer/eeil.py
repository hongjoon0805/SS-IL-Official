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
    
    def balance_fine_tune(self):
        self.update_frozen_model()
        self.setup_training(self.args.lr / 10)
        
        self.incremental_loader.update_bft_buffer()
        self.incremental_loader.mode = 'b-ft'
        
        schedule = np.array(self.args.schedule)
        bftepoch = int(self.args.nepochs*3/4)
        for epoch in range(bftepoch):
            self.update_lr(epoch, schedule)
            self.train(epoch, bft=True)
        
    def train(self, epoch, bft=False):
        
        self.model.train()
        print("Epochs %d"%epoch)
        T=2
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        mid = self.seen_classes[-1]
        start = 0
        
        for data, target in tqdm(self.train_iterator):
            data, target = data.cuda(), target.cuda()
            try:
                output = self.model(data)[:,:end]
            except:
                continue
            loss_CE = self.loss(output, target)
            
            loss_KD = 0
            if tasknum > 0:
                score = self.model_fixed(data).data
                
                if bft is False:
                    loss_KD = torch.zeros(tasknum).cuda()
                    
                    for t in range(tasknum):

                        # local distillation
                        start_KD = (t) * self.args.step_size
                        end_KD = (t+1) * self.args.step_size

                        soft_target = F.softmax(score[:,start_KD:end_KD] / T, dim=1)
                        output_log = F.log_softmax(output[:,start_KD:end_KD] / T, dim=1)
                        loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)
                    loss_KD = loss_KD.sum()
                else:
                    soft_target = F.softmax(score[:,mid:end] / T, dim=1)
                    output_log = F.log_softmax(output[:,mid:end] / T, dim=1)
                    loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)
                
            self.optimizer.zero_grad()
            (loss_KD + loss_CE).backward()
            self.optimizer.step()
           