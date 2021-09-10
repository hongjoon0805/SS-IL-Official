import torch
import numpy

def load_models(args, trainer, t):
    
    if t==0:
        name = 'models/{}_step_{}_nepochs_{}_{}'.format(args.dataset, 
                                                                      args.base_classes, 
                                                                      args.nepochs, 
                                                                      args.trainer)
        if args.trainer == 'il2m':
            name = 'models/{}_step_{}_nepochs_{}_{}'.format(args.dataset, args.base_classes, args.nepochs, 'ft')
        if args.trainer == 'vanilla':
            name = 'models/{}_step_{}_nepochs_{}_{}'.format(args.dataset, args.base_classes, args.nepochs, 'ssil')
        
    elif t>0:
        trainer_name = args.trainer if args.trainer != 'il2m' else 'ft'
        
        name = 'models/{}_{}_{}_{}_memsz_{}_base_{}_step_{}_batch_{}_epoch_{}'.format(
            args.date,
            args.dataset, 
            trainer_name, 
            args.seed, 
            args.memory_budget, 
            args.base_classes, 
            args.step_size, 
            args.batch_size, 
            args.nepochs)
        
        if args.memory_growing:
            name += '_growing'
            
        if args.trainer == 'ssil':
            name += '_replay_{}'.format(args.replay_batch_size)
        
        if args.trainer == 'rebalancing':
            name += '_lamb_base_{}'.format(args.lamb_base)
            
        if args.trainer == 'podnet':
            name += '_lamb_c_{}'.format(args.lamb_c) + '_lamb_f_{}'.format(args.lamb_f)
        
        
        print(name)
    
    try:
        state_dict = torch.load(name + '_task_{}.pt'.format(t))
        trainer.model.load_state_dict(state_dict)
        if args.trainer == 'bic' and t>0:
            state_dict = torch.load(name + '_bias' + '_task_{}.pt'.format(t))
            trainer.bias_correction_layer.load_state_dict(state_dict)
            print('Load bias correction layer')
        flag = 1
    except:
        print('Failed to load Pre-trained model')
        print('Model training start')
        flag = 0 
    
    return flag
