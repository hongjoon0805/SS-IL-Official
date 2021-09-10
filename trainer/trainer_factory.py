import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as td
from PIL import Image

class TrainerFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(train_iterator, myModel, args):
        
        if args.trainer == 'lwf':
            import trainer.lwf as trainer
        elif args.trainer == 'ssil':
            import trainer.ssil as trainer
        elif args.trainer == 'ft' or args.trainer == 'il2m':
            import trainer.ft as trainer
        elif args.trainer == 'icarl':
            import trainer.icarl as trainer
        elif args.trainer == 'bic':
            import trainer.bic as trainer
        elif args.trainer == 'eeil':
            import trainer.eeil as trainer
        elif args.trainer == 'vanilla':
            import trainer.vanilla as trainer
        elif args.trainer == 'rebalancing':
            import trainer.rebalancing as trainer
        elif args.trainer == 'podnet':
            import trainer.podnet as trainer
        elif args.trainer == 'wild':
            import trainer.wild as trainer
        
        return trainer.Trainer(train_iterator, myModel, args)
    

class ExemplarLoader(td.Dataset):
    def __init__(self, train_dataset):
        
        self.data = train_dataset.dataset.train_data
        self.labels = train_dataset.dataset.train_labels
        self.exemplar = train_dataset.exemplar
        self.transform = train_dataset.dataset.train_transform
        self.mem_sz = len(self.exemplar)

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        index = self.exemplar[index % self.mem_sz]
        img = self.data[index]
        try:
            img = Image.fromarray(img)
        except:
            img = Image.open(img, mode='r').convert('RGB')
            
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[index]

class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''

    def __init__(self, IncrementalLoader, model, args):
        self.incremental_loader = IncrementalLoader
        self.model = model
        self.args = args
        self.kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.incremental_loader.mode = 'train'
        self.train_iterator = torch.utils.data.DataLoader(self.incremental_loader,
                                                   batch_size=self.args.batch_size, drop_last=True, 
                                                          shuffle=True, **self.kwargs)
        self.optimizer = torch.optim.SGD(self.model.parameters(), args.lr, momentum=args.momentum,
                                         weight_decay=args.decay, nesterov=False)
        self.model_fixed = copy.deepcopy(self.model)
        self.model_exp = copy.deepcopy(self.model)
        for param in self.model_fixed.parameters():
            param.requires_grad = False
        self.current_lr = args.lr
        self.seen_classes = [0]
        
    def update_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from %0.4f to %0.4f"%(self.current_lr,
                                                                        self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]
                    
    def increment_classes(self):
        
        self.seen_classes.append(self.incremental_loader.end)
        self.incremental_loader.update_exemplar()
        self.incremental_loader.task_change()
        
        
    def setup_training(self, lr):
        
        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.4f"%lr)
            param_group['lr'] = lr
            self.current_lr = lr

    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False
