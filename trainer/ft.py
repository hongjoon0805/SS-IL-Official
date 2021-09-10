from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import networks
import trainer

class NormedLinear(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

    def forward(self, x):
        #out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=1))
        out = F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))
        return out

class Trainer(trainer.GenericTrainer):
    def __init__(self, IncrementalLoader, model, args):
        super().__init__(IncrementalLoader, model, args)
        if 'Hinge' in self.args.date:
            self.model.module.fc = NormedLinear(512, 1004).cuda()
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        
    def balance_fine_tune(self):
        self.update_frozen_model()
        
        self.incremental_loader.update_bft_buffer()
        self.incremental_loader.mode = 'b-ft'
        
        for epoch in range(15):
            self.train(epoch)
        
    def train(self, epoch):
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        mid = self.seen_classes[-1]
        
        for data, target in tqdm(self.train_iterator):
            data, target = data.cuda(), target.cuda()
            
            output = self.model(data)
            
            if tasknum > 0 and self.args.prev_new:
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
            
            self.optimizer.zero_grad()
            (loss_CE).backward()
            self.optimizer.step()
            
    def weight_align(self):
        end = self.train_data_iterator.dataset.end
        start = end-self.args.step_size
        weight = self.model.module.fc.weight.data
        
        prev = weight[:start, :]
        new = weight[start:end, :]
        print(prev.shape, new.shape)
        mean_prev = torch.mean(torch.norm(prev, dim=1)).item()
        mean_new = torch.mean(torch.norm(new, dim=1)).item()

        gamma = mean_prev/mean_new
        print(mean_prev, mean_new, gamma)
        new = new * gamma
        result = torch.cat((prev, new), dim=0)
        weight[:end, :] = result
        print(torch.mean(torch.norm(self.model.module.fc.weight.data[:start], dim=1)).item())
        print(torch.mean(torch.norm(self.model.module.fc.weight.data[start:end], dim=1)).item())

