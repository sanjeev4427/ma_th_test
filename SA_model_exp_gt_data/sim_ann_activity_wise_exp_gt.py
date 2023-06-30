
import numpy as np
import time
import os
from sklearn.metrics import f1_score
from SA_model.generate_input_data_for_SA import ml_generate_train_data, ml_generate_train_data_exp_gt
from SA_model.simulated_annealing import simulated_annealing
from log_data_scripts.save_csv_results import activity_save_best_results_to_csv
from ml_evaluate import mod_bar_graph
from ml_validation import ml_validation

# number of windows to time equation
def window_to_time(window, config):
    # n is number of windows
    # one window is 1 sec
    one_window_duration = 1 
    t = (window-1)*(1-config["sw_overlap"]/100)*one_window_duration + one_window_duration 
    return t

# get trained window threshold and skip windows 
def apply_best_no_tol(activity_label, filename):
    best = np.loadtxt(filename, skiprows=1, usecols=(1,2,3),delimiter=',').T
    best_threshold = best[0]
    best_win_skip = best[1]
    best_tolerance = best[2]
    window_threshold = best_threshold[int(activity_label)]
    skip_window = best_win_skip[int(activity_label)]
    tolerance_value = best_tolerance[int(activity_label)] 
    return int(window_threshold), int(skip_window), int(tolerance_value)

# funciton to train tolerance hyperparameter
def train_tolerance_hyp_sa(args, max_step_size_win_thr, max_step_size_skip_win, max_step_size_tol_val, \
                           init_temp, ann_rate, data, log_dir, sbj, filename_best_csv, exp_name):

        config = vars(args)
        print("--------training tolerance hyp-----------")
        if config["dataset"] == 'rwhar':
            label_name = ['climbing_down', 'climbing_up', 'jumping', 'lying',\
                        'running', 'sitting', 'standing', 'walking']
            activity_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        if config["dataset"] == 'wetlab':
            label_name = ['null_class', 'cutting', 'inverting', 'peeling', 'pestling',\
                        'pipetting', 'pouring', 'stirring', 'transfer']
            activity_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        
        # generating training data (validation data -> leave-one-out)
        ml_train_gt_pred= ml_generate_train_data(data, args, sbj)
        
        # reading hyp values from csv file
        best_filename_no_tol = filename_best_csv
        # distinguishing file names
        args.name = exp_name + '_' + 'tol_train'
           
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

        # training for each activity  
        for labels in activity_labels:
            activity = labels
            activity_name = label_name[activity]
            
            # extracting best settings for each activity
            window_threshold,skip_windows,_ = apply_best_no_tol(labels, best_filename_no_tol)
            
            # setting hyp param values (window threshold, skip windows are fixed and training tolerance value)
            window_threshold = np.array([window_threshold,window_threshold])
            skip_windows = np.array([skip_windows,skip_windows])
            tol_value = np.array([0, args.max_win_tol])
            max_step_size_win_thr = 0
            max_step_size_skip_win = 0
            
            
            # get the start time
            start_time = time.time()
            # running SA metaheuristic
            best, loss, best_f1, f_one_target, best_data_saved, best_comp_saved,\
                f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val,\
                    f_one_gt_mod_val_avg, f_one_gt_val_avg =\
                simulated_annealing(activity, activity_name, args, window_threshold,\
                                    skip_windows, tol_value, max_step_size_win_thr, max_step_size_skip_win, max_step_size_tol_val,\
                                        init_temp, ann_rate, log_dir, ml_train_gt_pred, sbj)
            # get the end time
            end_time = time.time()
            elapsed_time = end_time - start_time

            # printing best settings for each activity
            print(f"Done! for activity: {label_name[activity]}")
            print(f"Optimum window threshold is: {best[:,0]} \n", 
                        f"Optimum skip windows is: {best[:,1]} \n",
                        f"Optimum tolerance value is: {best[:,2]} \n",
                        f"F1 score (for particular activity) at optimum hyperparameter: {best_f1} target was {f_one_target} \n",
                        f"Avg. modified f1 score (for all activities) at optimum hyperparameter: {f_one_gt_mod_val_avg} target was {1} \n",
                        # f"Data saved at optimum hyperparameter: {best_data_saved} \n",
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
            acitivity_name_lst.append(activity_name)
            # get the execution time
            elapsed_time_lst.append(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            f_one_gt_mod_val_lst.append(f_one_gt_mod_val)
            f_one_gt_val_lst.append(f_one_gt_val)
            f_one_val_mod_val_lst.append(f_one_val_mod_val)
            f_one_gt_mod_val_avg_lst.append(f_one_gt_mod_val_avg)
            f_one_gt_val_avg_lst.append(f_one_gt_val_avg)
        # saving best settings for each subject 
        algo_name = 'SA'
        # save best results to csv file
        filename_best_csv = activity_save_best_results_to_csv(best_thrs_for_activity_lst, 
                                    best_skip_win_for_activity_lst,
                                    best_tol_val_for_activity_lst,
                                    best_f1_for_activity_lst, f_one_target_lst, best_data_saved_for_activity_lst, 
                                    best_comp_saved_for_activity_lst,
                                    best_loss_for_activity_lst, elapsed_time_lst, acitivity_name_lst,f_one_gt_mod_val_lst,
                                    f_one_gt_val_lst, f_one_val_mod_val_lst,f_one_gt_mod_val_avg_lst,f_one_gt_val_avg_lst, log_dir, args, algo_name, sbj)
    

# SA to train on GT data while keeping the tolerance = 0    
def sim_ann_activity_wise_gt(args, window_threshold, skip_windows, tol_value, max_step_size_win_thr, max_step_size_skip_win, max_step_size_tol_val, \
                           init_temp, ann_rate,  log_folder_name, data):
    config = vars(args)
    log_dir = log_folder_name

    if config["dataset"] == 'rwhar':
        label_name = ['climbing_down', 'climbing_up', 'jumping', 'lying',\
                       'running', 'sitting', 'standing', 'walking']
        activity_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    if config["dataset"] == 'wetlab':
        label_name = ['null_class', 'cutting', 'inverting', 'peeling', 'pestling',\
                       'pipetting', 'pouring', 'stirring', 'transfer']
        activity_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    for _, sbj in enumerate(np.unique(data[:, 0])):
        # generating training data (validation data -> leave-one-out)
        ml_train_gt_gt = ml_generate_train_data_exp_gt(data, args, sbj)
        
        # setting tol = 0
        window_threshold = np.array([1,args.max_win_thr])
        skip_windows = np.array([1,args.max_win_skip])
        tol_value = np.array([0, 0]) #! tol = 0
        max_step_size_tol_val = 0
        # extracting name of the experiment, to properly manage tol_zero and tol_train name
        if sbj == 0:
            temp_name = args.name
        exp_name = temp_name
        args.name = exp_name + '_' + 'tol_zero'
        
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

        # training for each activity  
        for labels in activity_labels:
            activity = labels
            activity_name = label_name[activity] 
            # get the start time
            start_time = time.time()
            # running SA metaheuristic
            best, loss, best_f1, f_one_target, best_data_saved, best_comp_saved,\
                f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val,\
                    f_one_gt_mod_val_avg, f_one_gt_val_avg =\
                simulated_annealing(activity, activity_name, args, window_threshold,\
                                    skip_windows, tol_value, max_step_size_win_thr, max_step_size_skip_win, max_step_size_tol_val,\
                                        init_temp, ann_rate, log_dir, ml_train_gt_gt, sbj)
            # get the end time
            end_time = time.time()
            elapsed_time = end_time - start_time

            # printing best settings for each activity
            print(f"Done! for activity: {label_name[activity]}")
            print(f"Optimum window threshold is: {best[:,0]} \n", 
                        f"Optimum skip windows is: {best[:,1]} \n",
                        f"Optimum tolerance value is: {best[:,2]} \n",
                        f"F1 score (for particular activity) at optimum hyperparameter: {best_f1} target was {f_one_target} \n",
                        f"Avg. modified f1 score (for all activities) at optimum hyperparameter: {f_one_gt_mod_val_avg} target was {1} \n",
                        # f"Data saved at optimum hyperparameter: {best_data_saved} \n",
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
            acitivity_name_lst.append(label_name[activity])
            # get the execution time
            elapsed_time_lst.append(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            f_one_gt_mod_val_lst.append(f_one_gt_mod_val)
            f_one_gt_val_lst.append(f_one_gt_val)
            f_one_val_mod_val_lst.append(f_one_val_mod_val)
            f_one_gt_mod_val_avg_lst.append(f_one_gt_mod_val_avg)
            f_one_gt_val_avg_lst.append(f_one_gt_val_avg)
        # saving best settings for each subject 
        algo_name = 'SA'
        filename_best_csv = activity_save_best_results_to_csv(best_thrs_for_activity_lst, 
                                    best_skip_win_for_activity_lst,
                                    best_tol_val_for_activity_lst,
                                    best_f1_for_activity_lst, f_one_target_lst, best_data_saved_for_activity_lst, 
                                    best_comp_saved_for_activity_lst,
                                    best_loss_for_activity_lst, elapsed_time_lst, acitivity_name_lst,f_one_gt_mod_val_lst,
                                    f_one_gt_val_lst, f_one_val_mod_val_lst,f_one_gt_mod_val_avg_lst,f_one_gt_val_avg_lst, log_dir, args, algo_name, sbj)

        # keeping thr, skip fix, training tol, for each subj
        train_tolerance_hyp_sa(args, max_step_size_win_thr, max_step_size_skip_win, max_step_size_tol_val,\
                                            init_temp, ann_rate, data, log_dir, sbj, filename_best_csv, exp_name)
