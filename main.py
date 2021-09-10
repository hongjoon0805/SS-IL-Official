import argparse

import torch
import numpy as np
import utils

import data_handler
import networks
import trainer
import arguments
from sklearn.utils import shuffle


args = arguments.get_args()

dataset = data_handler.DatasetFactory.get_dataset(args.dataset)
seed = args.seed
m = args.memory_budget

# Fix the seed.
args.seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Loader used for training data
shuffle_idx = shuffle(np.arange(dataset.classes), random_state=args.seed)
dataset.shuffle_data(shuffle_idx)
print(shuffle_idx)
print("Label shuffled")

# Use Multi-GPU model
myModel = networks.ModelFactory.get_model(args.dataset, args.trainer)
myModel = torch.nn.DataParallel(myModel).cuda()

incremental_loader = data_handler.IncrementalLoader(dataset, args)
result_loader = data_handler.ResultLoader(dataset, args)
result_loader.reset()

# Get the required model
print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")

# Trainer object used for training
myTrainer = trainer.TrainerFactory.get_trainer(incremental_loader, myModel, args)

schedule = np.array(args.schedule)
tasknum = (dataset.classes-args.base_classes)//args.step_size+1
total_epochs = args.nepochs

# initialize result logger
logger = trainer.ResultLogger(myTrainer, incremental_loader, args)
logger.make_log_name()

for t in range(tasknum):
    
    print("SEED:", seed, "MEMORY_BUDGET:", m, "tasknum:", t)
    # Add new classes to the train, and test iterator
    lr = args.lr
    if 'ft' in  args.trainer or 'ssil' in args.trainer:
        lr = args.lr / (t+1)
        if t==1:
            total_epochs = args.nepochs // args.factor
            schedule = schedule // args.factor
    
    myTrainer.update_frozen_model()
    myTrainer.setup_training(lr)
    
    # Load pre-trained model
    flag = utils.load_models(args, myTrainer, t)
        
    # Running nepochs epochs
    
    for epoch in range(0, total_epochs):
        if flag == 1:
            print('Evaluation!')
            break
        if args.trainer != 'podnet':
            myTrainer.update_lr(epoch, schedule)
        if args.trainer == 'il2m':
            break
        else:
            incremental_loader.mode = 'train'
            myTrainer.train(epoch)
            
        if epoch % 10 == (10 - 1) and args.debug:
            if args.trainer == 'icarl' or 'nem' in args.trainer:
                logger.update_moment()
            logger.evaluate(mode='train', get_results = False)
            logger.evaluate(mode='test', get_results = False)
    
    # iCaRL prototype update
    if args.trainer == 'icarl' or 'nem' in args.trainer:
        logger.update_moment()
        print('Moment update finished')
    
    # IL2M mean update
    if args.trainer == 'il2m':
        logger.update_mean()
        print('Mean update finished')
    
    # balanced fine-tuning
    if t > 0 and (args.trainer in ['eeil', 'podnet']) and flag == 0:
        logger.save_model(add_name = '_before_bft')
        logger.evaluate(mode='test', get_results = False)
        myTrainer.balance_fine_tune()
        
    
    # BiC Bias correction
    if t > 0 and 'bic' in args.trainer and flag == 0:
        myTrainer.train_bias_correction()
    
    if args.dataset != 'Google_Landmark_v2_10K':
        logger.evaluate(mode='train', get_results = False)
        logger.evaluate(mode='test', get_results = True)
    
    else:
        logger.evaluate_large(mode='train')
        logger.evaluate_large(mode='test')
    
    # Get task-wise accuracy
    start = 0
    end = args.base_classes

    result_loader.reset()
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    iterator = torch.utils.data.DataLoader(result_loader, batch_size=100, **kwargs)
    for i in range(t+1):
        logger.get_task_accuracy(start, end, t, iterator)

        start = end
        end += args.step_size

        result_loader.task_change()
        
    # Save results
    logger.save_results()
    if args.trainer != 'il2m':
        logger.save_model()
    myTrainer.increment_classes()
