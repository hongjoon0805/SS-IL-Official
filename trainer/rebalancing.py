from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from networks.layers import NormedLinear, SplitNormedLinear, LSCLinear, SplitLSCLinear
import trainer

class Trainer(trainer.GenericTrainer):
    def __init__(self, IncrementalLoader, model, args):
        super().__init__(IncrementalLoader, model, args)
        
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.lamb_base = args.lamb_base
        
    
    def setup_training(self, lr):
        tasknum = self.incremental_loader.t
        
        if tasknum == 0:
            in_features = self.model.module.fc.in_features
            out_features = self.args.base_classes
            
            self.model.module.fc = NormedLinear(in_features, out_features).cuda()
        
        elif tasknum == 1:
            in_features = self.model.module.fc.in_features
            out_features = self.model.module.fc.out_features # 100
            
            # 100 & 10
            new_fc = SplitNormedLinear(in_features, out_features, self.args.step_size).cuda()
            new_fc.fc1.weight.data = self.model.module.fc.weight.data
            new_fc.eta.data = self.model.module.fc.eta.data
            self.model.module.fc = new_fc
            
        elif tasknum>1:
            in_features = self.model.module.fc.in_features
            out_features1 = self.model.module.fc.fc1.out_features
            out_features2 = self.model.module.fc.fc2.out_features
            
            new_fc = SplitNormedLinear(in_features, out_features1+out_features2, self.args.step_size).cuda()
            new_fc.fc1.weight.data[:out_features1] = self.model.module.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = self.model.module.fc.fc2.weight.data
            new_fc.eta.data = self.model.module.fc.eta.data
            self.model.module.fc = new_fc
            
        
        if tasknum > 0:
        
            self.imprint_weights()
        
            ignored_params = list(map(id, self.model.module.fc.fc1.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, \
                self.model.parameters())
            params =[{'params': base_params, 'lr': self.args.lr, 'weight_decay': self.args.decay}, \
                    {'params': self.model.module.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
            
        else:
            params = self.model.parameters()
        
        self.optimizer = torch.optim.SGD(params, self.args.lr, momentum=self.args.momentum,
                                         weight_decay=self.args.decay, nesterov=False)
        
        
    def balance_fine_tune(self):
        fc1_params = self.model.module.fc.fc1.parameters()
        fc2_params = self.model.module.fc.fc2.parameters()
        params = [{'params': fc1_params},{'params': fc2_params}]
        
        self.fc_optimizer = torch.optim.SGD(params, 0.01, weight_decay = self.args.decay)
        self.fc_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.fc_optimizer, milestones=[10], gamma=0.1)
        self.incremental_loader.update_bft_buffer()
        self.incremental_loader.mode = 'b-ft'
        
        for epoch in range(20):
            self.train(epoch, bft = True)
            self.fc_scheduler.step()
        
    def train(self, epoch, bft=False):
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        mid = self.seen_classes[-1]
        lamb = 0
        
        for data, target in tqdm(self.train_iterator):
            data, target = data.cuda(), target.cuda()
            
            old_idx = target < mid
            new_idx = target >= mid
            
            output, features = self.model(data, feature_return=True)
            loss_CE = self.loss(output[:,:end], target) # end < 100
            loss_dis = 0
            loss_mr = 0
            
            if tasknum > 0 and bft == False:
                lamb = self.lamb_base * np.sqrt(mid/(end-mid))
                _, old_features = self.model_fixed(data, feature_return=True)
                loss_dis = nn.CosineEmbeddingLoss()(features, old_features.detach(), torch.ones(features.shape[0]).cuda())
                
                eta = self.model.module.fc.eta.data
                old_output = output[old_idx] / eta
                old_data = data[old_idx]
                old_target = target[old_idx]
                old_samples = old_output.shape[0]
                
                if old_samples > 0:
                    K = 2
                    m = 0.5
                    gt_scores = old_output[torch.arange(old_samples),old_target].view(-1, 1).repeat(1, K)
                    max_novel_scores = old_output[:,mid:].topk(K, dim=1)[0]
                    loss_mr = nn.MarginRankingLoss(margin=m)(gt_scores.view(-1,1), \
                                                             max_novel_scores.view(-1, 1), \
                                                             torch.ones(old_samples*K).cuda())
                    
            
            self.optimizer.zero_grad()
            (loss_CE + loss_dis * lamb + loss_mr).backward()
            if bft:
                self.fc_optimizer.step()
            else:
                self.optimizer.step()
        
            
    def imprint_weights(self):
        
        self.model.eval()
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        mid = self.seen_classes[-1]
        
        old_embedding_norm = self.model.module.fc.fc1.weight.data.norm(dim=1, keepdim=True)
        average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0)
        
        num_features = self.model.module.fc.in_features
        novel_embedding = torch.zeros((self.args.step_size, num_features)).cuda()
        totalFeatures = torch.zeros((self.args.step_size, 1)).cuda()
        
        with torch.no_grad():
            for data, target in tqdm(self.train_iterator):
                data, target = data.cuda(), target.cuda()
                
                _, features = self.model.forward(data, feature_return=True)
                
                new_idx = target >= mid

                new_features = features[new_idx]
                new_target_idx = target[new_idx] - mid
                
                novel_embedding.index_add_(0, new_target_idx, new_features.data)
                totalFeatures.index_add_(0, new_target_idx, torch.ones_like(new_target_idx.unsqueeze(1)).float().cuda())
                
            novel_embedding = F.normalize(novel_embedding / totalFeatures, dim = 1) * average_old_embedding_norm
            
            self.model.module.fc.fc2.weight.data = novel_embedding
