##################################################
# # Skip heuristics algorithm with trainable hyperparameters using validation predictions after training.
##################################################

from skip_heuristics_scripts.data_skipping import data_skipping
from sklearn.metrics import f1_score
import numpy as np

def skip_heuristics(activity, args, window_threshold, skip_windows, tolerance_value, all_eval_output):
    """
    Applies a data skipping technique over the validation predictions to get modified predictions,
    calculates and returns several evaluation metrics and saving ratios for each activity.
    
    Parameters:
    activity (int): The activity for which to calculate evaluation metrics and saving ratios.
    args (Namespace): A namespace containing the arguments.
    window_threshold (float): The threshold to apply for data skipping.
    skip_windows (int): The number of windows to skip during data skipping.
    tolerance_value (float): The tolerance value to use during data skipping.
    all_eval_output (ndarray): An ndarray containing the evaluation outputs for all activities.
    
    Returns:
    f_one_gt_mod_val (float): The f1-score for ground truth modified validation predictions.
    f_one_gt_val (float): The f1-score for ground truth validation predictions.
    f_one_val_mod_val (float): The f1-score for modified validation predictions.
    f_one_gt_mod_val_avg (float): The average f1-score for ground truth modified validation predictions.
    f_one_gt_val_avg (float): The average f1-score for ground truth validation predictions.
    comp_saved_ratio (float): The computation saving ratio for the specified activity.
    data_saved_ratio (float): The data saving ratio for the specified activity.
    computations_saved (ndarray): An ndarray containing the computation saved for each activity.
    data_saved (ndarray): An ndarray containing the data saved for each activity.
    comp_windows (ndarray): An ndarray containing the total number of windows for each activity.
    data_windows (ndarray): An ndarray containing the total amount of data for each activity.
    all_mod_val_preds (ndarray): An ndarray containing modifieed validation predictions. 
    """  
    
    config = vars(args)
    # trainable hyperparameters
    config["saving_window_threshold"] = window_threshold
    config["saving_tolerance"] = tolerance_value
    config["saving_skip_windows"] = skip_windows

    # savings 
    computations_saved = np.zeros(config["nb_classes"])
    data_saved = np.zeros(config["nb_classes"])
    
    # defining validation predictions, validation gt and madified validation predictions
    all_val_preds = all_eval_output[:,0]
    all_mod_val_preds = np.copy(all_eval_output[:,0])
    all_val_gt = all_eval_output[:,1]

    # count total observations for each activity in validation gt 
    count_array = np.zeros(config["nb_classes"])
    for k in range(config["nb_classes"]):
        count_array[k] = (all_val_gt == k).sum()

    # apply data skip over validation predictions to get modified predictions
    all_mod_val_preds, data_saved, computations_saved = data_skipping(activity,
        all_mod_val_preds, config, data_saved, computations_saved,  apply_best=False
    )

    # calculate f1 scores for the modified predictions
    f_one_gt_mod_val = f1_score(all_val_gt, all_mod_val_preds, labels = np.array([activity]), average= None) 
    f_one_gt_val = f1_score(all_val_gt, all_val_preds, labels = np.array([activity]), average= None) 
    f_one_val_mod_val = f1_score(all_val_preds, all_mod_val_preds, labels = np.array([activity]), average= None) 
    f_one_gt_mod_val_avg = f1_score(all_val_gt, all_mod_val_preds, average= 'macro')
    f_one_gt_val_avg = f1_score(all_val_gt, all_val_preds, average= 'macro')

    return f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val, f_one_gt_mod_val_avg, f_one_gt_val_avg, computations_saved, data_saved, all_mod_val_preds
    