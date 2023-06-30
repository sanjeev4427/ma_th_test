
import numpy as np
import time
from sklearn.metrics import f1_score
from SA_model.generate_input_data_for_SA import ml_generate_train_data
from SA_model.simulated_annealing import simulated_annealing
from SA_model_not_activity_wise.simulated_annealing_all_act import simulated_annealing_all_act
from log_data_scripts.save_csv_results import activity_save_best_results_to_csv, all_act_save_best_results_to_csv
from ml_validation import ml_validation

# number of windows to time function
def window_to_time(window, config):
    """ Convert a window to time duration.

    Args:
        window (int): sliding window
        config (dict): General setting dictionary

    Returns:
        float: time duration of n number of windows
    """
    # number of windows to time equation
    # n is number of windows
    # one window is 1 sec
    one_window_duration = 1 
    t = (window-1)*(1-config["sw_overlap"]/100)*one_window_duration + one_window_duration 
    return t

# SA training for exp 1, activity independent training
def sim_ann_all_activity(args, window_threshold, skip_windows, tol_value, max_step_size_win_thr, max_step_size_skip_win, max_step_size_tol_val, \
                           init_temp, ann_rate,  log_folder_name, data):
    """Apply simulated annealing algorithm for each activity and genrate optimized window values. 

    Args:
        args (_type_): _description_
        window_threshold (int): threshold window
        skip_windows (int): skip windows 
        tol_value (int): tolerance window
        max_step_size_win_thr (int): maximum step size for generating new candidate in window threshold
        max_step_size_skip_win (int): maximum step size for generating new candidate in skip windows
        max_step_size_tol_val (int): maximum step size for generating new candidate in tolerance windows
        ann_rate (int): rate of temperature decay
        log_date (str): logging current date 
        log_timestamp (str): logging current time 
        data (NDArray): contains accelerometer, subject, activity information 
    """
    
    config = vars(args)
    log_dir = log_folder_name

    for _, sbj in enumerate(np.unique(data[:, 0])):
        # generating training data (validation data -> leave-one-out)
        ml_train_gt_pred = ml_generate_train_data(data, args, sbj)

        # creating empty lists to save best settings, performance metrics 
        best_thrs_for_activity_lst = []
        best_skip_win_for_activity_lst = []
        best_tol_val_for_activity_lst = []
        best_f1_for_activity_lst = []
        f_one_target_lst = list()
        best_data_saved_for_activity_lst = []
        best_comp_saved_for_activity_lst = []
        best_loss_for_activity_lst = []
        elapsed_time_lst = []
        acitivity_name_lst = []
        f_one_gt_mod_val_lst = []
        f_one_gt_val_lst = []
        f_one_val_mod_val_lst = []
        f_one_gt_mod_val_avg_lst = []
        f_one_gt_val_avg_lst = []

        # training for all activities at once  
        # get the start time
        start_time = time.time()
        # running SA metaheuristic
        best, loss, best_f1, f_one_target, best_data_saved, best_comp_saved,\
            f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val,\
                f_one_gt_mod_val_avg, f_one_gt_val_avg =\
            simulated_annealing_all_act(args, window_threshold,\
                                skip_windows, tol_value, max_step_size_win_thr, max_step_size_skip_win, max_step_size_tol_val,\
                                    init_temp, ann_rate, log_folder_name, ml_train_gt_pred,sbj)
        # get the end time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # printing best settings for each activity
        print(f"Done! for all activities")
        print(f"Optimum window threshold is: {best[:,0]} \n", 
                    f"Optimum skip windows is: {best[:,1]} \n",
                    f"Optimum tolerance value is: {best[:,2]} \n",
                    f"F1 score (for particular activity) at optimum hyperparameter: {best_f1} target was {f_one_target} \n",
                    f"Avg. modified f1 score (for all activities) at optimum hyperparameter: {f_one_gt_mod_val_avg} target was {1} \n",
                    f"Data saved at optimum hyperparameter: {best_data_saved} \n",
                    f"Computation saved at optimum hyperparameter: {best_comp_saved} \n",
                    f"Lowest loss value : {loss} \n",
                    f"Total time elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} \n\n")
        print(f"optimum time after which device will be switched off: {window_to_time(best[:,0],config)} seconds")
        print(f" optimum switch off duration: {window_to_time(best[:,1], config)} seconds")

        # appending best settings
        best_thrs_for_activity_lst.append(float(best[:,0]))
        best_skip_win_for_activity_lst.append(float(best[:,1]))
        best_tol_val_for_activity_lst.append(float(best[:,2]))
        best_f1_for_activity_lst.append(best_f1)
        f_one_target_lst.append(f_one_target)
        best_data_saved_for_activity_lst.append(best_data_saved)
        best_comp_saved_for_activity_lst.append(best_comp_saved)
        best_loss_for_activity_lst.append(loss)
        # acitivity_name_lst.append(activity_name)
        # get the execution time
        elapsed_time_lst.append(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        f_one_gt_mod_val_lst.append(f_one_gt_mod_val)
        f_one_gt_val_lst.append(f_one_gt_val)
        f_one_val_mod_val_lst.append(f_one_val_mod_val)
        f_one_gt_mod_val_avg_lst.append(f_one_gt_mod_val_avg)
        f_one_gt_val_avg_lst.append(f_one_gt_val_avg)
        # saving best settings for each subject 
        algo_name = 'SA'
        filename_best_csv = all_act_save_best_results_to_csv(best_thrs_for_activity_lst, 
                                    best_skip_win_for_activity_lst,
                                    best_tol_val_for_activity_lst,
                                    best_f1_for_activity_lst, f_one_target_lst, best_data_saved_for_activity_lst, 
                                    best_comp_saved_for_activity_lst,
                                    best_loss_for_activity_lst, elapsed_time_lst, f_one_gt_mod_val_lst,
                                    f_one_gt_val_lst, f_one_val_mod_val_lst,f_one_gt_mod_val_avg_lst,f_one_gt_val_avg_lst, log_dir, args, algo_name, sbj)
