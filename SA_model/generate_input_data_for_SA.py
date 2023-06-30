##########################################################
# functions  generating ML training data
##########################################################

import numpy as np
import torch
from torch.utils.data import DataLoader
from misc.torchutils import seed_worker
from data_processing.sliding_window import apply_sliding_window
import pandas as pd

def ml_generate_train_data(data, args, sbj):

    # config dictionary containing setting parameters
    config= vars(args)  
    
    # reading data
    if args.dataset == 'wetlab':
        ml_train_data = np.loadtxt(rf'./ml_training_data/wetlab/train_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')
    elif args.dataset == 'rwhar':
        ml_train_data = np.loadtxt(fr'./ml_training_data/rwhar/train_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')

    #stacking up GT data as second column and training data as first column
    train_pred = ml_train_data[:,0]
    train_gt = ml_train_data[:,1] 
    ml_train_gt_pred = np.vstack((train_pred, train_gt)).T
    return ml_train_gt_pred


def ml_generate_train_data_exp_gt(data, args, sbj):
    
    # config dictionary containing setting parameters
    config= vars(args)   
    
    # reading data 
    if args.dataset == 'wetlab':
        ml_train_data = np.loadtxt(rf'./ml_training_data/wetlab/train_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')
    elif args.dataset == 'rwhar':
        ml_train_data = np.loadtxt(fr'ml_training_data/rwhar/train_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')

    #stacking up GT data as second column and training data as first column
    train_pred = ml_train_data[:,0]
    train_gt = ml_train_data[:,1] 
    ml_train_gt_gt = np.vstack((train_gt, train_gt)).T
    
    return ml_train_gt_gt


