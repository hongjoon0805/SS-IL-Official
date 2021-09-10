from trainer.trainer_factory import *

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import inv
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pickle
import os


class ResultLogger():
    def __init__(self, trainer, incremental_loader, args):
        
        self.tasknum = (incremental_loader.classes-args.base_classes)//args.step_size+1
        
        self.trainer = trainer
        self.model = trainer.model
        self.incremental_loader = incremental_loader
        self.args = args
        self.kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.option = 'Euclidean'
        self.result = {}
        
        # For IL2M
        classes = incremental_loader.classes
        self.init_class_means = torch.zeros(classes).cuda()
        self.current_class_means = torch.zeros(classes).cuda()
        self.model_confidence = torch.zeros(classes).cuda()
        
    def evaluate(self, mode = 'train', get_results = False):
        self.model.eval()
        t = self.incremental_loader.t
        if mode == 'train':
            self.incremental_loader.mode = 'evaluate'
            iterator = torch.utils.data.DataLoader(self.incremental_loader,
                                                   batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
            
        elif mode == 'test':
            self.incremental_loader.mode = 'test'
            iterator = torch.utils.data.DataLoader(self.incremental_loader,
                                                   batch_size=100, shuffle=False, **self.kwargs)
        
        with torch.no_grad():
            out_matrix = []
            target_matrix = []
            features_matrix = []
            
            for data, target in tqdm(iterator):
                data, target = data.cuda(), target.cuda()
                try:
                    out, features = self.make_output(data)
                except:
                    continue
                
                if data.shape[0] < 4:
                    continue
                
                out_matrix.append(out)
                features_matrix.append(features)
                target_matrix.append(target)
                
            print(len(out_matrix))
            out = torch.cat(out_matrix, dim=0)
            features = torch.cat(features_matrix, dim=0)
            target = torch.cat(target_matrix, dim=0)
            self.get_accuracy(mode, out, target)
            
            if mode == 'test' and get_results and t > 0:
                self.get_statistics(out, target)
                self.get_confusion_matrix(out, target)
                try:
                    self.get_cosine_similarity_score_softmax_average(out, features, target)
                except:
                    pass

            self.print_result(mode, t)
            
    def evaluate_large(self, mode = 'train'):
        
        if mode+'-top-1' not in self.result:
            self.result[mode + '-top-1'] = np.zeros(self.tasknum)
            self.result[mode + '-top-5'] = np.zeros(self.tasknum)
        
        self.model.eval()
        t = self.incremental_loader.t
        if mode == 'train':
            self.incremental_loader.mode = 'evaluate'
            iterator = torch.utils.data.DataLoader(self.incremental_loader,
                                                   batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
            
        elif mode == 'test':
            self.incremental_loader.mode = 'test'
            iterator = torch.utils.data.DataLoader(self.incremental_loader,
                                                   batch_size=100, shuffle=False, **self.kwargs)
        
        with torch.no_grad():
            out_matrix = []
            target_matrix = []
            features_matrix = []
            
            total = 0
            correct_1_sum = 0
            correct_5_sum = 0
            
            for data, target in tqdm(iterator):
                data, target = data.cuda(), target.cuda()
                try:
                    out, features = self.make_output(data)
                except:
                    continue
                
                total += data.shape[0]
                
                pred_1 = out.data.max(1, keepdim=True)[1]
                pred_5 = torch.topk(out, 5, dim=1)[1]
                
                correct_1 = pred_1.eq(target.data.view_as(pred_1)).sum().item()
                correct_5 = pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).sum().item()
                
                correct_1_sum += correct_1
                correct_5_sum += correct_5
        
        self.result[mode + '-top-1'][t] = round(100.*(correct_1_sum / total), 2)
        self.result[mode + '-top-5'][t] = round(100.*(correct_5_sum / total), 2)
    
    def make_output(self, data):
        end = self.incremental_loader.end
        start = self.incremental_loader.start
        step_size = self.args.step_size
        tasknum = self.incremental_loader.t
        
        if 'bic' in self.args.trainer:
            bias_correction_layer = self.trainer.bias_correction_layer
            out, features = self.model(data, feature_return = True)
            out = out[:,:end]
            if tasknum>0:
                out_new = bias_correction_layer(out[:,start:end])
                out = torch.cat((out[:,:start], out_new), dim=1)
            
        elif self.args.trainer == 'il2m':
            
            out, features = self.model(data, feature_return = True)
            out = out[:,:end]
            if tasknum>0:
                pred = out.data.max(1, keepdim=True)[1]
                mask = (pred >= start).int()
                prob = F.softmax(out[:,:end], dim=1)
                rect_prob = prob * (self.init_class_means[:end] / self.current_class_means[:end]) \
                                 * (self.model_confidence[end-1] / self.model_confidence[:end])
                out = (1-mask).float() * prob + mask.float() * rect_prob
            
        elif self.args.trainer == 'icarl' or 'nem' in self.args.trainer or self.args.trainer == 'gda':
            _, features = self.model.forward(data, feature_return=True)
            batch_vec = (features.data.unsqueeze(1) - self.class_means.unsqueeze(0))
            temp = torch.matmul(batch_vec, self.precision)
            out = -torch.matmul(temp.unsqueeze(2),batch_vec.unsqueeze(3)).squeeze()
            
        else:
            out, features = self.model(data, feature_return = True)
            out = out[:,:end]
        
        return out, features
    
    def get_accuracy(self, mode, out, target):
        if mode+'-top-1' not in self.result:
            self.result[mode + '-top-1'] = np.zeros(self.tasknum)
            self.result[mode + '-top-5'] = np.zeros(self.tasknum)
            
        t = self.incremental_loader.t
        
        pred_1 = out.data.max(1, keepdim=True)[1]
        pred_5 = torch.topk(out, 5, dim=1)[1]
        
        correct_1 = pred_1.eq(target.data.view_as(pred_1)).sum().item()
        correct_5 = pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).sum().item()
        
        self.result[mode+'-top-1'][t] = round(100.*(correct_1 / target.shape[0]), 2)
        self.result[mode+'-top-5'][t] = round(100.*(correct_5 / target.shape[0]), 2)
        
        
    def get_statistics(self, out, target):
        if 'statistics' not in self.result:
            self.result['statistics'] = []
        
        stat = {}
        
        t = self.incremental_loader.t
        end = self.incremental_loader.end
        start = self.incremental_loader.start
        
        samples_per_classes = target.shape[0] // end
        old_samples = (start) * samples_per_classes 
        new_samples = out.shape[0] - old_samples
        out_old, out_new = out[:old_samples], out[old_samples:]
        target_old, target_new = target[:old_samples], target[old_samples:]
        pred_old, pred_new = out_old.data.max(1, keepdim=True)[1], out_new.data.max(1, keepdim=True)[1]
        
        # statistics
        cp = pred_old.eq(target_old.data.view_as(pred_old)).sum().item()
        epn = (pred_old >= start).int().sum().item()
        epp = (old_samples-(cp + epn))
        cn = pred_new.eq(target_new.data.view_as(pred_new)).sum().item()
        enp = (pred_new < start).int().sum().item()
        enn = (new_samples-(cn + enp))
        
        stat['cp'], stat['epp'], stat['epn'], stat['cn'], stat['enn'], stat['enp'] = cp, epp, epn, cn, enn, enp
        
        # save statistics
        self.result['statistics'].append(stat)
        
    def get_confusion_matrix(self, out, target):
        # Get task specific confusion matrix
        if 'confusion_matrix' not in self.result:
            self.result['confusion_matrix'] = []
            
        task_pred = out.data.max(1, keepdim=True)[1] // self.args.step_size
        task_target = target // self.args.step_size
        
        matrix = confusion_matrix(task_target.data.cpu().numpy(), task_pred.data.cpu().numpy())
        
        self.result['confusion_matrix'].append(matrix)
        
    def get_cosine_similarity_score_softmax_average(self, out, features, target):
        # Cosine similarity between ground truth and prediction
        if 'cosine_similarity' not in self.result:
            self.result['score'] = {}
            self.result['softmax'] = {}
            self.result['cosine_similarity'] = {}
            
            self.result['score']['old_class_pred'] = []
            self.result['score']['new_class_pred'] = []
            self.result['score']['old_classes_mean'] = []
            self.result['score']['new_classes_mean'] = []
            self.result['score']['epn'] = []
            self.result['score']['enp'] = []
            
            
            self.result['softmax']['old_class_pred'] = []
            self.result['softmax']['new_class_pred'] = []
            self.result['softmax']['old_classes_mean'] = []
            self.result['softmax']['new_classes_mean'] = []
            self.result['softmax']['epn'] = []
            self.result['softmax']['enp'] = []
            
            self.result['cosine_similarity']['old_class_pred'] = []
            self.result['cosine_similarity']['new_class_pred'] = []
            self.result['cosine_similarity']['old_classes_mean'] = []
            self.result['cosine_similarity']['new_classes_mean'] = []
            self.result['cosine_similarity']['epn'] = []
            self.result['cosine_similarity']['enp'] = []
        
        t = self.incremental_loader.t
        end = self.incremental_loader.end
        start = self.incremental_loader.start
        mid = end - self.args.step_size
        samples_per_classes = target.shape[0] // end
        old_samples = (start) * samples_per_classes 
        
        weight = self.model.module.fc.weight
        sample_size = out.shape[0]
        pred = out[:,:end].data.max(1)[1]
        normalized_features = features / torch.norm(features, 2, 1).unsqueeze(1)
        normalized_weight = weight / torch.norm(weight, 2, 1).unsqueeze(1)
        cos_sim_matrix = torch.matmul(normalized_features, normalized_weight.transpose(0,1))
        
        softmax = F.softmax(out[:,:end], dim=1)
        
        old_class_pred, new_class_pred = pred < start, pred >= start
        
        pred_score = out[torch.arange(sample_size), pred]
        pred_softmax = softmax[torch.arange(sample_size), pred]
        pred_cos_sim = cos_sim_matrix[torch.arange(sample_size), pred]
        
        old_score_avg, new_score_avg = pred_score[old_class_pred].mean(), pred_score[new_class_pred].mean()
        old_softmax_avg, new_softmax_avg = pred_softmax[old_class_pred].mean(), pred_softmax[new_class_pred].mean()
        old_cos_sim_avg, new_cos_sim_avg = pred_cos_sim[old_class_pred].mean(), pred_cos_sim[new_class_pred].mean()
        
        epn_mask, enp_mask = pred[:old_samples] >= mid, pred[old_samples:] < mid
        
        epn_score_avg, enp_score_avg = pred_score[:old_samples][epn_mask].mean(), pred_score[old_samples:][enp_mask].mean()
        epn_softmax_avg = pred_softmax[:old_samples][epn_mask].mean()
        enp_softmax_avg = pred_softmax[old_samples:][enp_mask].mean()
        epn_cos_sim_avg = pred_cos_sim[:old_samples][epn_mask].mean()
        enp_cos_sim_avg = pred_cos_sim[old_samples:][enp_mask].mean()
        
        self.result['score']['old_class_pred'].append(old_score_avg.item())
        self.result['score']['new_class_pred'].append(new_score_avg.item())
        self.result['score']['old_classes_mean'].append(out[:,:mid].mean().item())
        self.result['score']['new_classes_mean'].append(out[:,mid:end].mean().item())
        self.result['score']['epn'].append(epn_score_avg.item())
        self.result['score']['enp'].append(enp_score_avg.item())
        
        self.result['softmax']['old_class_pred'].append(old_softmax_avg.item())
        self.result['softmax']['new_class_pred'].append(new_softmax_avg.item())
        self.result['softmax']['old_classes_mean'].append(softmax[:,:mid].mean().item())
        self.result['softmax']['new_classes_mean'].append(softmax[:,mid:end].mean().item())
        self.result['softmax']['epn'].append(epn_softmax_avg.item())
        self.result['softmax']['enp'].append(enp_softmax_avg.item())
        
        self.result['cosine_similarity']['old_class_pred'].append(old_cos_sim_avg.item())
        self.result['cosine_similarity']['new_class_pred'].append(new_cos_sim_avg.item())
        self.result['cosine_similarity']['old_classes_mean'].append(cos_sim_matrix[:,:mid].mean().item())
        self.result['cosine_similarity']['new_classes_mean'].append(cos_sim_matrix[:,mid:end].mean().item())
        self.result['cosine_similarity']['epn'].append(epn_cos_sim_avg.item())
        self.result['cosine_similarity']['enp'].append(enp_cos_sim_avg.item())
        
        return
        
    def get_task_accuracy(self, start, end, t, iterator):
        if 'task_accuracy' not in self.result:
            self.result['task_accuracy'] = np.zeros((self.tasknum, self.tasknum))
        
        with torch.no_grad():
        
            out_matrix = []
            target_matrix = []
            for data, target in tqdm(iterator):
                data, target = data.cuda(), target.cuda()
                target = target % (end-start)
                out = self.model(data)[:,start:end]
                out_matrix.append(out)
                target_matrix.append(target)

            out = torch.cat(out_matrix)
            target = torch.cat(target_matrix)
            
            pred = out.data.max(1, keepdim=True)[1]

            correct = pred.eq(target.data.view_as(pred)).sum().item()
            task = iterator.dataset.t
            self.result['task_accuracy'][t][task] = 100.*(correct / target.shape[0])
            
        return
    
    def update_moment(self):
        self.model.eval()
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        with torch.no_grad():
            # compute means
            classes = end
            class_means = torch.zeros((classes,512)).cuda()
            totalFeatures = torch.zeros((classes, 1)).cuda()
            total = 0
            
            self.incremental_loader.mode = 'evaluate'
            iterator = torch.utils.data.DataLoader(self.incremental_loader,
                                                   batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
            for data, target in tqdm(iterator):
                data, target = data.cuda(), target.cuda()
                if data.shape[0]<4:
                    continue
                total += data.shape[0]
                try:
                    _, features = self.model.forward(data, feature_return=True)
                except:
                    continue
                    
                class_means.index_add_(0, target, features.data)
                totalFeatures.index_add_(0, target, torch.ones_like(target.unsqueeze(1)).float().cuda())
                
            class_means = class_means / totalFeatures
            
            # compute precision
            covariance = torch.zeros(512,512).cuda()
            euclidean = torch.eye(512).cuda()

            if self.option == 'Mahalanobis':
                for data, target in tqdm(iterator):
                    data, target = data.cuda(), target.cuda()
                    _, features = self.model.forward(data, feature_return=True)

                    vec = (features.data - class_means[target])
                    
                    cov = torch.matmul(vec.unsqueeze(2), vec.unsqueeze(1)).sum(dim=0)
                    covariance += cov

                #avoid singular matrix
                covariance = covariance / totalFeatures.sum() + torch.eye(512).cuda() * 1e-9
                precision = covariance.inverse()

            self.class_means = class_means
            if self.option == 'Mahalanobis':
                self.precision = precision
            else:
                self.precision = euclidean
        
        return
        
    def update_mean(self):
        self.model.eval()
        classes = self.incremental_loader.classes
        end = self.incremental_loader.end
        end = self.incremental_loader.start
        step_size = self.args.step_size
        with torch.no_grad():
            class_means = torch.zeros(classes).cuda()
            class_count = torch.zeros(classes).cuda()
            current_count = 0
            
            self.incremental_loader.mode = 'evaluate'
            iterator = torch.utils.data.DataLoader(self.incremental_loader,
                                                   batch_size=self.args.batch_size,
                                                   shuffle=True, **self.kwargs)
            for data, target in tqdm(iterator):
                data, target = data.cuda(), target.cuda()
                out = self.model(data)
                prob = F.softmax(out[:,:end], dim=1)
                confidence = prob.max(dim=1)[0] * (target >= (start)).float()
                class_means.index_add_(0, target, prob[torch.arange(data.shape[0]),target])
                class_count.index_add_(0, target, torch.ones_like(target).float().cuda())
                
                self.model_confidence[start:end] += confidence.sum()
                current_count += (target >= (start)).float().sum()

            self.init_class_means[start:end] = class_means[start:end] / class_count[start:end]
            self.current_class_means[:end] = class_means[:end] / class_count[:end]
            self.model_confidence[start:end] /= current_count
    
    def make_log_name(self):
        self.first_task_name = '{}_step_{}_nepochs_{}_{}_task_0.pt'.format(
                self.args.dataset,
                self.args.base_classes,
                self.args.nepochs,
                self.args.trainer)
        
        self.log_name = '{}_{}_{}_{}_memsz_{}_base_{}_step_{}_batch_{}_epoch_{}'.format(
            self.args.date,
            self.args.dataset,
            self.args.trainer,
            self.args.seed,
            self.args.memory_budget,
            self.args.base_classes,
            self.args.step_size,
            self.args.batch_size,
            self.args.nepochs,
        )
        
        if self.args.memory_growing:
            self.log_name += '_growing'
        
        if self.args.trainer == 'ssil':
            self.log_name += '_replay_{}'.format(self.args.replay_batch_size)
        
        if self.args.trainer == 'rebalancing':
            self.log_name += '_lamb_base_{}'.format(self.args.lamb_base)
            
        if self.args.trainer == 'podnet':
            self.log_name += '_lamb_c_{}'.format(self.args.lamb_c) + '_lamb_f_{}'.format(self.args.lamb_f)
            
    def print_result(self, mode, t):
        print(mode + " top-1: %0.2f"%self.result[mode + '-top-1'][t])
        print(mode + " top-5: %0.2f"%self.result[mode + '-top-5'][t])
        
        return
        
    def save_results(self):
        
        if not os.path.isdir('./result_data'):
            os.mkdir('./result_data')
        
        path = self.log_name + '.pkl'
        with open('result_data/' + path, "wb") as f:
            pickle.dump(self.result, f)
        
    def save_model(self, add_name = '', running = False, epoch = None):
        
        if not os.path.isdir('./models'):
            os.mkdir('./models')
        
        t = self.incremental_loader.t
        if t==0:
            torch.save(self.model.state_dict(),'./models/'+ self.first_task_name)
        torch.save(self.model.state_dict(), './models/' + self.log_name + add_name + '_task_{}.pt'.format(t))
        if 'bic' in self.args.trainer:
            torch.save(self.trainer.bias_correction_layer.state_dict(), 
                       './models/' + self.log_name + '_bias' + '_task_{}.pt'.format(t))
