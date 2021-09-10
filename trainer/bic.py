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

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha =  nn.Parameter(torch.Tensor(1).uniform_(1,1))
        self.beta = nn.Parameter(torch.Tensor(1).uniform_(0,0))
    def forward(self, x):
        return x*self.alpha + self.beta

class Trainer(trainer.GenericTrainer):
    def __init__(self, IncrementalLoader, model, args):
        super().__init__(IncrementalLoader, model, args)
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        
        self.bias_correction_layer = BiasLayer().cuda()
        self.bias_correction_layer_arr = []
        
    def update_bias_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp]*2 == epoch:
                for param_group in self.bias_optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from %0.4f to %0.4f"%(self.current_lr,
                                                                        self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]


    def setup_training(self, lr):
        
        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.4f"%lr)
            param_group['lr'] = lr
            self.current_lr = lr
        
        lr = lr/100
        self.bias_correction_layer_arr.append(self.bias_correction_layer)
        self.bias_correction_layer = BiasLayer().cuda()
        self.bias_optimizer = torch.optim.SGD(self.bias_correction_layer.parameters(), self.args.lr)
        
        for param_group in self.bias_optimizer.param_groups:
            print("Setting LR to %0.4f"%lr)
            param_group['lr'] = lr
            self.current_lr = lr
            
    def train(self, epoch):
        
        T=2
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        mid = self.seen_classes[-1]
        start = 0
        lamb = mid / end
        
        for data, target in tqdm(self.train_iterator):
            data, target = data.cuda(), target.cuda()

            output, feature_curr = self.model(data, feature_return=True)
            score, feature_prev = self.model_fixed(data, feature_return=True)
                
            output = output[:, :end]

            loss_CE = self.loss(output, target)

            loss_KD = 0
            if tasknum > 0:
                end_KD = mid
                start_KD = end_KD - self.args.step_size

                layer = self.bias_correction_layer_arr[-1] # last bias correction layer

                score = score[:,:end_KD].data
                score = torch.cat([score[:,:start_KD], layer(score[:,start_KD:end_KD])], dim=1)
                soft_target = F.softmax(score / T, dim=1)
                output_log = F.log_softmax(output[:,:end_KD] / T, dim=1)
                loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean')
                
            self.optimizer.zero_grad()
            (lamb*loss_KD + (1-lamb)*loss_CE).backward()
            self.optimizer.step()

    def train_bias_correction(self):
        self.incremental_loader.mode = 'bias'
        schedule = np.array(self.args.schedule)
        total_epochs = self.args.nepochs
        for e in range(total_epochs*2):
            self.bias_correction(self.train_iterator)
            self.update_bias_lr(e, schedule)
    
    def bias_correction(self, bias_iterator):
        
        self.model.eval()
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        start = end-self.args.step_size
        
        for data, target in tqdm(bias_iterator):
            data, target = data.cuda(), target.cuda()
            
            output = self.model(data)[:,:end]
            
            # bias correction
            output_new = self.bias_correction_layer(output[:,start:end])
            output = torch.cat((output[:,:start], output_new), dim=1)
            
            loss_CE = self.loss(output, target)
            
            self.bias_optimizer.zero_grad()
            (loss_CE).backward()
            self.bias_optimizer.step()
            
