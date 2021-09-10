from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from networks.layers import NormedLinear, SplitNormedLinear, LSCLinear, SplitLSCLinear
import trainer
from sklearn.cluster import KMeans


class Trainer(trainer.GenericTrainer):
    def __init__(self, IncrementalLoader, model, args):
        super().__init__(IncrementalLoader, model, args)
        
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.lamb_c = args.lamb_c
        self.lamb_f = args.lamb_f
        
    def setup_training(self, lr):
        tasknum = self.incremental_loader.t
        
        if tasknum == 0:
            in_features = self.model.module.fc.in_features
            out_features = self.args.base_classes
            
            self.model.module.fc = LSCLinear(in_features, out_features).cuda()
        
        elif tasknum == 1:
            in_features = self.model.module.fc.in_features
            out_features = self.model.module.fc.out_features
            
            new_fc = SplitLSCLinear(in_features, out_features, self.args.step_size).cuda()
            new_fc.fc1.weight.data = self.model.module.fc.weight.data
            self.model.module.fc = new_fc
            
        elif tasknum>1:
            in_features = self.model.module.fc.in_features
            out_features1 = self.model.module.fc.fc1.out_features
            out_features2 = self.model.module.fc.fc2.out_features
            K = self.model.module.fc.fc2.K
            
            new_fc = SplitLSCLinear(in_features, out_features1+out_features2, self.args.step_size).cuda()
            new_fc.fc1.weight.data[:out_features1*K] = self.model.module.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1*K:] = self.model.module.fc.fc2.weight.data
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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.nepochs)
    
    def balance_fine_tune(self):
        fc1_params = self.model.module.fc.fc1.parameters()
        fc2_params = self.model.module.fc.fc2.parameters()
        params = [{'params': fc1_params},{'params': fc2_params}]
        
        self.fc_optimizer = torch.optim.SGD(params, 0.01, weight_decay = self.args.decay)
        # There is no lr scheduler in BFT phase
        
        self.incremental_loader.update_bft_buffer()
        self.incremental_loader.mode = 'b-ft'
        
        for epoch in range(20):
            self.train(epoch, bft = True)
        
    def train(self, epoch, bft = False):
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        mid = self.seen_classes[-1]
        lamb_c = 0
        lamb_f = 0
        
        for data, target in tqdm(self.train_iterator):
            data, target = data.cuda(), target.cuda()
            batch_size = data.shape[0]
            
            output, features, acts = self.model(data, feature_return = True, podnet=True)
            scale = self.model.module.fc.factor
            loss_CE = self.nca(output, target, scale=scale)
            
            loss_spatial = 0
            loss_flat = 0
            if tasknum > 0 and bft == False:
                score, old_features, old_acts = self.model_fixed(data, feature_return = True, podnet=True)
                
                loss_flat = (features-old_features).norm(2).pow(2).sum() / batch_size
                loss_width, loss_height = 0, 0
                
                for (act, old_act) in zip(acts, old_acts):
                    act = act.pow(2)
                    old_act = old_act.pow(2)
                    # act: (B x C x H x W)
                    act_width = F.normalize(act.sum(dim=3).view(act.shape[0], -1), dim=1, p=2)
                    old_act_width = F.normalize(old_act.sum(dim=3).view(old_act.shape[0], -1), dim=1, p=2)
                    loss_width += torch.mean(torch.frobenius_norm(act_width - old_act_width, dim=-1))
                    
                    act_height = F.normalize(act.sum(dim=2).view(act.shape[0], -1), dim=1, p=2)
                    old_act_height = F.normalize(old_act.sum(dim=2).view(old_act.shape[0], -1), dim=1, p=2)
                    loss_width += torch.mean(torch.frobenius_norm(act_height - old_act_height, dim=-1))
                
                loss_spatial = loss_width + loss_height
                
                lamb_c = self.lamb_c / len(acts)
                lamb_f = self.lamb_f
                
            self.optimizer.zero_grad()
            (loss_CE + lamb_c * loss_spatial + lamb_f * loss_flat).backward()
            if bft:
                self.fc_optimizer.step()
            else:
                self.optimizer.step()
            
        if bft == False:
            self.scheduler.step()
        
    def nca(self, similarities, targets, scale = 1, margin=0.6):
        
        margins = torch.zeros_like(similarities)
        margins[torch.arange(margins.shape[0]), targets] = margin
#         similarities = scale * (similarities - margins)
        
        similarities = scale * (similarities - margin)
    
        similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
                    targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        

        losses = -losses
        loss = torch.mean(losses)
#         print('factor:', scale)
#         print('loss: ', loss)
        
        return loss
        
#         similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

#         disable_pos = torch.ones_like(similarities).cuda()
#         disable_pos[torch.arange(len(similarities)),targets] = 0
        
#         numerator = similarities[torch.arange(similarities.shape[0]), targets]
        
#         losses = -numerator + torch.log((torch.exp(similarities)*disable_pos).sum(-1))
#         losses = torch.clamp(losses, min=0.)
#         loss = torch.mean(losses)
#         print('scale:', scale)
#         print('loss: ', loss)
        
        return loss
    
    def imprint_weights(self):
        
        self.model.eval()
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        mid = self.seen_classes[-1]
        
        num_features = self.model.module.fc.in_features
        novel_embedding = torch.zeros((self.args.step_size, num_features)).cuda()
        totalFeatures = torch.zeros((self.args.step_size, 1)).cuda()
        
        features_arr = []
        targets_arr = []
        
        with torch.no_grad():
            
            weights_norm = self.model.module.fc.fc1.weight.data.norm(dim=1, keepdim=True)
            avg_weights_norm = torch.mean(weights_norm, dim=0).cpu()
            
            print('Imprint weights')
            self.incremental_loader.mode = 'train'
            for data, target in tqdm(self.train_iterator):
                data, target = data.cuda(), target.cuda()
                
                _, features = self.model.forward(data, feature_return=True)
                
                new_idx = target >= mid

                new_features = features[new_idx]
                new_target_idx = target[new_idx] - mid
                
                features_arr.append(new_features)
                targets_arr.append(new_target_idx)
                
            features = torch.cat(features_arr, dim=0)
            targets = torch.cat(targets_arr, dim=0)
            
            new_weights = []
            for c in range(end-mid):
                class_features = features[targets == c]
                clusterizer = KMeans(n_clusters=10)
                clusterizer.fit(class_features.cpu().numpy())
                
                for center in clusterizer.cluster_centers_:
                    new_weights.append(torch.tensor(center) * avg_weights_norm)
            
            new_weights = torch.stack(new_weights)
            self.model.module.fc.fc2.weight.data = new_weights.cuda()
                
                
            