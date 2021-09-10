from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import networks

class Trainer(GenericTrainer):
    def __init__(self, trainDataIterator, model, args):
        super().__init__(trainDataIterator, model, args)

    def train(self, epoch):
        
        T=2
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        start = end-self.args.step_size
        
        for data, target in tqdm(self.train_iterator):
            data, target = data.cuda(), target.cuda()
            
        
            output = self.model(data)
            loss_CE = self.loss(output[:,start:end], target[:,start:end])
            
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
                    loss_KD[t] = F.kl_div(output_log, soft_target) * (T**2)
                
                loss_KD = loss_KD.sum()
                
            self.optimizer.zero_grad()
            (loss_KD + loss_CE).backward()
            self.optimizer.step()

    def add_model(self):
        model = copy.deepcopy(self.model_single)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.models.append(model)
        print("Total Models %d"%len(self.models))

