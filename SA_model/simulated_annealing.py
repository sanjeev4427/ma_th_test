##################################################
# Simulated algorithm to generate optimized hyperparameters for skip-heuristics.
##################################################
# Author: Sanjeev Kumar 
# Email: sanjeev.kumar(at)student.uni-siegen.de
##################################################

import numpy as np
from sklearn.metrics import f1_score
from log_data_scripts.save_csv_results import activity_save_ml_train_to_csv
from skip_heuristics_scripts.skip_heuristics import skip_heuristics
from log_data_scripts.activity_plotting import activity_plot_f1_iter, activity_plot_threshold_iter, activity_plot_tol_value_iter, \
                                    activity_plot_data_saved_iter, activity_plot_comp_saved_iter, \
                                        activity_plot_loss_anneal, activity_plot_skip_windows_iter, activity_plot_temperature_iter

# defining loss function
def loss_function(f_alpha, c_alpha, d_alpha, f_one, f_one_target, f_one_gt_val, f_one_gt_mod_val_avg, comp_saved_ratio,data_saved_ratio):
     
    """
    Computes the loss function based on the given parameters.

    Args:
    f_alpha (float): Weight of the F1 score difference between the predicted and target value.
    c_alpha (float): Weight of the computaiton saved component of loss funciton.
    d_alpha (float): Weight of the data saved component of loss funciton.
    f_one (float): F1 score of the predicted value.
    f_one_target (float): Target F1 score value.
    f_one_gt_val (float): F1 score value of the ground truth.
    f_one_gt_mod_val_avg (float): Average F1 score value of the modified ground truth.
    comp_saved_ratio (float): Computation saved ratio in percentage.
    data_saved_ratio (float): Data saved ratio in percentage.

    Returns:
    float: The loss value computed based on the given weights and parameters.
    """
    f_loss_diff = f_one_gt_val - f_one   
    
    # loss function with f1 as prority
    if f_loss_diff < 0: # f_one more than target
        loss = f_alpha*np.abs(f_one_target-f_one) + c_alpha*np.abs(1-comp_saved_ratio*0.01) + d_alpha*np.abs(1-data_saved_ratio*0.01) + f_alpha*np.abs(f_one_target-f_one_gt_mod_val_avg) - 5 
    elif f_loss_diff*100 < 1: # f1 less than target but differnce is less than 1 percent
        loss = f_alpha*np.abs(f_one_target-f_one) + c_alpha*np.abs(1-comp_saved_ratio*0.01) +  d_alpha*np.abs(1-data_saved_ratio*0.01) + f_alpha*np.abs(f_one_target-f_one_gt_mod_val_avg) 
    else:
        loss = f_alpha*np.abs(f_one_target-f_one) + c_alpha*np.abs(1-comp_saved_ratio*0.01) +  d_alpha*np.abs(1-data_saved_ratio*0.01) + f_alpha*np.abs(f_one_target-f_one_gt_mod_val_avg) + 5
    
    # # maximizing data saving is priority    
    #! uncomment this if maximizing data is priority 
    # loss_part = f_alpha*np.abs(f_one_target-f_one) + c_alpha*np.abs(1-comp_saved_ratio*0.01) + d_alpha*np.abs(1-data_saved_ratio*0.01) + f_alpha*np.abs(f_one_target-f_one_gt_mod_val_avg)
    
    # if data_saved_ratio > 95: 
    #     loss = loss_part - 6 
    # elif data_saved_ratio > 90: 
    #     loss = loss_part - 5
    # elif data_saved_ratio > 85:
    #     loss = loss_part - 4
    # elif data_saved_ratio > 80:
    #     loss = loss_part - 3
    # elif data_saved_ratio > 75:
    #     loss = loss_part - 2
    # elif data_saved_ratio > 70:
    #     loss = loss_part - 1
    # else:
    #     loss = loss_part + (65 - data_saved_ratio) // 5
    
    return loss

# defining objective function
def anneal_objective(activity, args, h_param, all_mod_eval_output):
    """Computes the objective value for the annealing algorithm based on the given parameters.

    Args:
    activity (numpy.ndarray): The input activity signal.
    args (argparse.Namespace): The configuration settings.
    h_param (numpy.ndarray): The hyperparameters.
    all_mod_eval_output (numpy.ndarray): Evaluation output for all modified activity signals.

    Returns:
    tuple: A tuple containing the following elements:
    - loss (float): The loss value computed based on the given weights and parameters.
    - f_one (float): F1 score of the modified activity signal.
    - f_one_target (float): Target F1 score value.
    - data_saved_ratio (float): Data saved ratio in percentage.
    - comp_saved_ratio (float): Compression saved ratio in percentage.
    - f_one_gt_mod_val (float): F1 score value of the modified ground truth.
    - f_one_gt_val (float): F1 score value of the ground truth.
    - f_one_val_mod_val (float): F1 score value of the modified activity signal against the modified ground truth.
    - f_one_gt_mod_val_avg (float): Average F1 score value of the modified ground truth.
    - f_one_gt_val_avg (float): Average F1 score value of the ground truth.
    """
    config = vars(args)
    window_threshold = h_param[:,0]
    skip_windows = h_param[:,1]
    tolerance_value = h_param[:,2]
    
    # applying skip heuristics to get modified predictions    
    f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val, f_one_gt_mod_val_avg, f_one_gt_val_avg, \
        comp_saved_ratio, data_saved_ratio,_ = skip_heuristics(activity, args, window_threshold, skip_windows, tolerance_value, all_mod_eval_output)  

    # for activity wise optimization 
    # f_one, comp_saved_ratio, data_saved_ratio \
    #     = skip_heuristics(activity, args, window_threshold, skip_windows, tolerance_value, all_mod_eval_output)  


     #defining computation saved calculation
    lam = args.sw_overlap /100
    comp_saved_ratio = skip_windows/(window_threshold + skip_windows)*100
    data_saved_ratio = 100*(skip_windows - lam*(skip_windows+1))/(skip_windows - lam*(skip_windows+1) + window_threshold + (window_threshold-1)*(1 - lam))

    # f_alpha = 10
    # c_alpha = 1
    # d_alpha = 2
    
    # changing the weights of the loss function
    f_alpha = args.f_alpha
    c_alpha = args.c_alpha
    d_alpha = args.d_alpha
    
    #set f1 target
    f_one_target = 1
    f_one = f_one_gt_mod_val

    loss = loss_function(f_alpha, c_alpha, d_alpha, f_one, f_one_target, f_one_gt_val, f_one_gt_mod_val_avg, comp_saved_ratio, data_saved_ratio)

    return loss, f_one, f_one_target, float(data_saved_ratio), float(comp_saved_ratio), f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val,\
                                                                                        f_one_gt_mod_val_avg, f_one_gt_val_avg


def simulated_annealing(activity, activity_name, args, window_threshold, skip_windows,\
                          tol_value, max_step_size_win_thr, max_step_size_skip_win, max_step_size_tol_val, \
                            init_temp, ann_rate, log_dir, all_mod_eval_output,sbj):
    
    """
    Implements simulated annealing algorithm to optimize window threshold, skip windows, and tolerance value.

    Args:
    activity (ndarray): Contains accelerometer, subject, activity information.
    activity_name (str): Name of activity to optimize.
    args (argparse.Namespace): Argument object containing all relevant hyperparameters and settings.
    window_threshold (int): Threshold window.
    skip_windows (int): Skip windows.
    tol_value (int): Tolerance window.
    max_step_size_win_thr (int): Maximum step size for window threshold.
    max_step_size_skip_win (int): Maximum step size for skip windows.
    max_step_size_tol_val (int): Maximum step size for tolerance value.
    init_temp (float): Initial temperature.
    ann_rate (float): Annealing rate.
    log_date (str): Date of logging.
    log_timestamp (str): Timestamp of logging.
    all_mod_eval_output (dict): Dictionary of all model evaluation output data.

    Returns:
    ndarray: Array containing the best values of window threshold, skip windows, and tolerance value.
    """

    config= vars(args)

    print("Running simulated algorithm...")
    print("initializing parameters...")
    # intialising hyperparameters
    best_window_threshold = np.random.randint(window_threshold[0], window_threshold[1]+1)

    # randint low inclusive, high exclusive
    best_skip_windows = np.random.randint(skip_windows[0], skip_windows[1]+1)
    best_tol_value = np.random.randint(tol_value[0], tol_value[1]+1)

    best = np.array([best_window_threshold, best_skip_windows, best_tol_value])
    best = best.reshape(1,3)
    
    # objective function to evaluate current candidate solution
    best_eval, best_f1, f_one_target, best_data_saved, best_comp_saved,\
                        f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val,\
                             f_one_gt_mod_val_avg, f_one_gt_val_avg = anneal_objective(activity, args, best, all_mod_eval_output)
    
    curr, curr_eval = np.copy(best), best_eval
    curr_f1, curr_comp_saved, curr_data_saved = best_f1, best_comp_saved, best_data_saved

    best_f_one_gt_mod_val,  best_f_one_gt_val, best_f_one_val_mod_val = f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val
    best_f_one_gt_mod_val_avg, best_f_one_gt_val_avg = f_one_gt_mod_val_avg, f_one_gt_val_avg

    candidate = np.zeros((1,3))
    loss_array = []
    window_threshold_array = []
    skip_windows_array = []
    tol_value_array = []
    fscore_array = []
    data_saved_array = []
    comp_saved_array = []
    temp_array = []
    acceptance_ratio_array = []

    #run the algorithm
    i = 0
    n_iter = 0
    acceptance_ratio = -1 
    
    while True:
        # break 
        
        # for first loop initializoing temperature
        if i == 0:
            temp = init_temp  
        else:
            # update temperature
            temp = init_temp * (ann_rate)**i #! updated init instead of temp
            
        # condition to stop the algorithm four consecutive non-improvement at temp value, 
        # setting upper bound for temp states 15 to prevent from gettign stuck in inf loop
        if (i >= 4 and np.average(acceptance_ratio_array[i-4:i]) <= 0.05) or i > 15:
            break

        print("-"*20)
        # resetting states 
        accepted_states = 0
        rejected_states = 0   
        acceptance_ratio = -1  
        # inner loop counter
        j = 0
        # saving acceptance ratio at particular temperature
        acceptance_ratio_at_temp = []
        
        while acceptance_ratio < 0.1:
            # generating random candidate states
            candidate[:,0] = curr[:,0] + np.random.randint(-max_step_size_win_thr, max_step_size_win_thr + 1)
            candidate[:,1] = curr[:,1] + np.random.randint(-max_step_size_skip_win, max_step_size_skip_win + 1)
            candidate[:,2] = curr[:,2] + np.random.randint(-max_step_size_tol_val, max_step_size_tol_val + 1)
            
            # for negative values make it positive # +1 for zero value 
            if candidate[:,0] <= 0:
                candidate[:,0] = np.abs(candidate[:,0]) + 1
            # if crossing upper bound, make it equal to upper bound    
            if candidate[:,0] > window_threshold[1]:
                candidate[:,0] = window_threshold[1]
            # for negative values make it positive # +1 for zero value
            if candidate[:,1] <= 0:
                candidate[:,1] = np.abs(candidate[:,1]) + 1
            # if crossing upper bound, make it equal to upper bound
            if candidate[:,1] > skip_windows[1]:
                candidate[:,1] = skip_windows[1]
            # for negative values make it positive # +1 for zero value
            if candidate[:,2] <= 0:
                candidate[:,2] = np.abs(candidate[:,2]) + 1
            # if crossing upper bound, make it equal to upper bound
            if candidate[:,2] > tol_value[1]:
                candidate[:,2] = tol_value[1]

            # evaluate candidate point
            # minimize loss function returned 
            candidate_eval, candidate_f1, f_one_target, candidate_data_saved, candidate_comp_saved, \
                                    f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val,\
                                    f_one_gt_mod_val_avg, f_one_gt_val_avg \
                                        = anneal_objective(activity, args, candidate, all_mod_eval_output)
            # check for new best solution
            if candidate_eval < best_eval:
                # report progress
                print("*"*10)
                 # print(f"improvement at iteration {i+1} of {n_iter}", '\n'*2)
                print(f"best loss improved from {best_eval} to {candidate_eval}")
                print(f"f1 score moved from {best_f1} to {candidate_f1} target is {f_one_target}. \n", 
                    f"Data saved moved from {best_data_saved} to {candidate_data_saved}. \n",
                    f"Computation saved moved from {best_comp_saved} to {candidate_comp_saved}. \n",
                    f"Candidate solution: {candidate}. \n")
                print("*"*10)

                # store new best point
                best, best_eval = np.copy(candidate), candidate_eval    
                best_f1, best_comp_saved, best_data_saved = candidate_f1, candidate_comp_saved, candidate_data_saved
                best_f_one_gt_mod_val,  best_f_one_gt_val, best_f_one_val_mod_val = f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val
                best_f_one_gt_mod_val_avg, best_f_one_gt_val_avg = f_one_gt_mod_val_avg, f_one_gt_val_avg
            # difference between candidate and current point evaluation
            diff = candidate_eval - curr_eval 

            # calculate metropolis acceptance criterion
            metropolis = np.exp(-diff / temp)
            # check if we should keep the new point
            if diff <= 0:
                curr, curr_eval = np.copy(candidate), candidate_eval
                curr_f1, curr_comp_saved, curr_data_saved = candidate_f1, candidate_comp_saved, candidate_data_saved
                accepted_states += 1

            elif np.random.rand() <= metropolis: # compared when diff is more than zero
                # store the new current point
                curr, curr_eval = np.copy(candidate), candidate_eval
                curr_f1, curr_comp_saved, curr_data_saved = candidate_f1, candidate_comp_saved, candidate_data_saved
                accepted_states += 1 #! should we also consider this as accepted?

            else:
                # loss is higher and probaility is more than metropolis
                rejected_states += 1
            
            # calculate acceptance ratio only afer some iterations
            # to give some time for exploration at higer temperature
            allow_acceptance_after_n_iter = 100
            acceptance_ratio_brak_value = 0.05
            if j > allow_acceptance_after_n_iter:
                acceptance_ratio = accepted_states / (accepted_states + rejected_states)
                acceptance_ratio_at_temp.append(acceptance_ratio) 
                if j%100 == 0:
                    print(f"iteration running @ {j}, annealing rate @ {ann_rate}, acceptance ratio @ {acceptance_ratio}")
                
                if j > 1000:
                    print(f"iteration exceeded limit...")
                    break

                # breaking condition
                if acceptance_ratio < acceptance_ratio_brak_value:
                    print(f"No major improvement, breaking...")
                    break
            # count number of iterations
            n_iter += 1
            
            # save training logs
            loss_array.append(curr_eval)
            window_threshold_array.append(float(curr[:,0]))
            skip_windows_array.append(float(curr[:,1]))
            tol_value_array.append(float(curr[:,2]))
            fscore_array.append(curr_f1)
            data_saved_array.append(curr_data_saved)
            comp_saved_array.append(curr_comp_saved)
            temp_array.append(temp)
            
            # update inner loop counter
            j = j + 1
        print(f"Temperature level : {temp}")
        # saving last acceptance ratio as ratio for whole temperature level
        acceptance_ratio_array.append(acceptance_ratio)
        print(f"last acceptance_ratio : {acceptance_ratio}")
        print (f"total iterations : {j}")
        # increase outer loop counter, temp states
        i = i + 1
    print(f"total number of temperature states: {i} ", "\n"*2)

    # plot trainig logs vs iterations
    # plot_loss_anneal(loss_array, n_iter, config, ann_rate)
    # plot_f1_iter(fscore_array, n_iter, config, ann_rate)
    # plot_data_saved_iter(data_saved_array, n_iter, config, ann_rate)
    # plot_comp_saved_iter(comp_saved_array, n_iter, config, ann_rate)
    # plot_threshold_iter(window_threshold_array, n_iter, config, ann_rate)
    # # plot_tolerance_iter(skip_windows_array, n_iter, config)
    # plot_skip_windows_iter(skip_windows_array, n_iter, config, ann_rate)
    # plot_temperature_iter(temp_array, n_iter, config, ann_rate)
    # save_sim_ann_results_to_csv(loss_array, window_threshold_array, skip_windows_array, fscore_array, 
    #                                             data_saved_array, comp_saved_array, temp_array, config, ann_rate)
    # log_dir = os.path.join('logs', log_date, log_timestamp)
    # activity_plot_loss_anneal(loss_array, n_iter, config, ann_rate, activity_name,args, log_dir)
    # activity_plot_f1_iter(fscore_array, n_iter, config, ann_rate, activity_name,args, log_dir)
    # activity_plot_data_saved_iter(data_saved_array, n_iter, config, ann_rate, activity_name, args, log_dir)
    # activity_plot_comp_saved_iter(comp_saved_array, n_iter, config, ann_rate, activity_name, args, log_dir)
    # activity_plot_threshold_iter(window_threshold_array, n_iter, config, ann_rate, activity_name, args, log_dir)
    # # activity_plot_tolerance_iter(skip_windows_array, n_iter, config)
    # activity_plot_skip_windows_iter(skip_windows_array, n_iter, config, ann_rate, activity_name, args, log_dir)
    # activity_plot_tol_value_iter(tol_value_array, n_iter, config, ann_rate, activity_name, args, log_dir)
    # activity_plot_temperature_iter(temp_array, n_iter, config, ann_rate, activity_name, args, log_dir)
       
    # save the training logs to csv
    # activity_save_ml_train_to_csv(loss_array, window_threshold_array, skip_windows_array, tol_value_array, fscore_array, 
    #                                             data_saved_array, comp_saved_array, temp_array, args, log_dir, sbj, activity_name)
    
    return best, best_eval[0], best_f1, f_one_target, round(best_data_saved,2), round(best_comp_saved,2), best_f_one_gt_mod_val[0],  best_f_one_gt_val[0], best_f_one_val_mod_val[0],\
                                                                                                best_f_one_gt_mod_val_avg, best_f_one_gt_val_avg
    


