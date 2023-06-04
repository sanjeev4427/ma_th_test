import csv
import numpy as np
import pandas as pd
import os
from misc.close_excel import close_all_excel_files
from misc.osutils import mkdir_if_missing 
# def save_sim_ann_results_to_csv(loss_array, threshold_value_array, skip_windows_array, fscore_array, 
#                                                 data_saved_array, comp_saved_array, temp_array, config, ann_rate):

#     data_dict = {'loss': loss_array, 
#             'threshold_value': threshold_value_array, 
#             'skip_windows': skip_windows_array, 
#             'fscore' : fscore_array, 
#             'data_saved' : data_saved_array, 
#             'comp_saved' : comp_saved_array, 
#             'temp' : temp_array
#             }

#     df = pd.DataFrame(data_dict)
#     df.to_csv(rf'C:\Users\minio\Box\Thesis- Marius\csv_data\sim_ann_data_for_ann_rate_{ann_rate}_{config["dataset"]}.csv', index=False, header=True)           
#     return None

# def save_best_sim_ann_results_to_csv(ann_rate_array, best_thrs_for_ann_rate_lst, 
#                                         best_skip_win_for_ann_rate_lst,
#                                         best_f1_for_ann_rate_lst, best_data_saved_for_ann_rate_lst, 
#                                         best_comp_saved_for_ann_rate_lst,
#                                         best_loss_for_ann_rate_lst, elapsed_time_lst, DATASET):

#     best_dict_for_ann_rate = { 'Annealing rate' : ann_rate_array,
#                               'Threshold': best_thrs_for_ann_rate_lst,
#                               'Skip Windows': best_skip_win_for_ann_rate_lst,
#                               'f1 score': best_f1_for_ann_rate_lst,
#                               'data saved': best_data_saved_for_ann_rate_lst,
#                               'computation saved': best_comp_saved_for_ann_rate_lst,
#                               'lowest loss': best_loss_for_ann_rate_lst,
#                               'elapsed time': elapsed_time_lst
#                               }
#     best_df = pd.DataFrame(best_dict_for_ann_rate)
#     best_df.to_csv (rf'C:\Users\minio\Box\Thesis- Marius\csv_data\best_results_for_ann_rate_{DATASET}.csv', index = False, header=True) 
#     print(best_df)

def activity_save_ml_train_to_csv(loss_array, window_threshold_array, skip_windows_array, tol_value_array, fscore_array, 
                                                data_saved_array, comp_saved_array, temp_array, args, log_dir, sbj):

    data_dict = {'loss': loss_array, 
                'threshold_value': window_threshold_array, 
                'skip_windows': skip_windows_array, 
                'tolerance_window': tol_value_array,
                'fscore' : fscore_array, 
                'data_saved' : data_saved_array, 
                'comp_saved' : comp_saved_array, 
                'temp/gen' : temp_array
                }

    df = pd.DataFrame(data_dict)
    mkdir_if_missing(log_dir)
    if args.name:
        filename = os.path.join(log_dir, f'ml_train_data__for_{args.dataset}_{args.algo_name}' + '_' + f'{int(sbj)+1}' + '_' + args.name + '.csv')
    else:
        filename = os.path.join(log_dir, f'ml_train_data__for_{args.dataset}_{args.algo_name}' + '_' + f'{int(sbj)+1}' + '_' + '.csv')
    df.to_csv (filename, index = False, header=True)


def activity_save_best_results_to_csv(best_thrs_for_activity_lst, 
                                        best_skip_win_for_activity_lst,
                                        best_tol_val_for_activity_lst,
                                        best_f1_for_activity_lst, f_one_target_lst, best_data_saved_for_activity_lst, 
                                        best_comp_saved_for_activity_lst,
                                        best_loss_for_activity_lst, elapsed_time_lst, acitivity_name_lst,
                                        f_one_gt_mod_val_lst,f_one_gt_val_lst, 
                                        f_one_val_mod_val_lst, f_one_gt_mod_val_avg_lst, 
                                        f_one_gt_val_avg_lst,log_dir, args, algo_name, sbj):
    
    """
    Save the best results for each activity to a CSV file.

    Parameters:
    -----------
    best_thrs_for_activity_lst : list of floats
        List of best thresholds for each activity.
    best_skip_win_for_activity_lst : list of integers
        List of best skip windows for each activity.
    best_tol_val_for_activity_lst : list of floats
        List of best tolerance values for each activity.
    best_f1_for_activity_lst : list of floats
        List of best f1 scores for each activity.
    f_one_target_lst : list of floats
        List of target f1 scores for each activity.
    best_data_saved_for_activity_lst : list of floats
        List of amount of data saved for each activity.
    best_comp_saved_for_activity_lst : list of floats
        List of amount of computation saved for each activity.
    best_loss_for_activity_lst : list of floats
        List of lowest loss values for each activity.
    elapsed_time_lst : list of floats
        List of elapsed times for each activity.
    acitivity_name_lst : list of strings
        List of activity names.
    f_one_gt_mod_val_lst : list of floats
        List of f1 scores with modified ground truth for each activity.
    f_one_gt_val_lst : list of floats
        List of f1 scores with original ground truth for each activity.
    f_one_val_mod_val_lst : list of floats
        List of f1 scores with modified predictions for each activity.
    f_one_gt_mod_val_avg_lst : list of floats
        List of average f1 scores with modified ground truth for each activity.
    f_one_gt_val_avg_lst : list of floats
        List of average f1 scores with original ground truth for each activity.
    log_dir : str
        Path to directory where the CSV file will be saved.
    args : argparse.Namespace
        Namespace containing the arguments passed to the script.
    algo_name : str
        Name of the algorithm.
    sbj : str
        Subject number.

    Returns:
    --------
    filename : str
        Path to the saved CSV file.
    """
    
    best_dict_for_ann_rate = {'Acitivity': acitivity_name_lst,
                              'Threshold': best_thrs_for_activity_lst,
                              'Skip Windows': best_skip_win_for_activity_lst,
                              'Tolerance value': best_tol_val_for_activity_lst,
                              'f1 score': best_f1_for_activity_lst,
                              'Target f1': f_one_target_lst,
                              'Data saved': best_data_saved_for_activity_lst,
                              'Computation saved': best_comp_saved_for_activity_lst,
                              'Lowest loss': best_loss_for_activity_lst,
                              'Elapsed time': elapsed_time_lst,
                              'f1_gt_train': f_one_gt_val_lst,
                              'f1_gt_mod_train':f_one_gt_mod_val_lst,
                              'f1_val_mod_train':f_one_val_mod_val_lst,
                              'f_one_gt_mod_train_avg': f_one_gt_mod_val_avg_lst,
                              'f_one_gt_train_avg': f_one_gt_val_avg_lst
                              }
    best_df = pd.DataFrame(best_dict_for_ann_rate)
    mkdir_if_missing(log_dir)
    if args.name:
        filename = os.path.join(log_dir, f'best_results_for_{args.dataset}_{algo_name}' + '_' + f'{int(sbj)+1}' + '_' + args.name + '.csv')
    else:
        filename = os.path.join(log_dir, f'best_results_for_{args.dataset}_{algo_name}' + '_' + f'{int(sbj)+1}' + '.csv')
    
    best_df.to_csv (filename, index = False, header=True) 
    print(best_df)

    return filename

def save_act_avg_std_results(algo_name, avg_f1_activity, std_f1_activity, avg_f1_unmod_activity, std_f1_unmod_activity, \
                            avg_comp_saved_activity, std_comp_saved_activity, avg_data_saved_activity, \
                                std_data_saved_activity,eval_criterion, filepath, args):
    
    activity = args.class_names
    capital_activity = [word.replace('_', ' ').title() for word in activity]    
    activities = capital_activity + ['Average']
    
    
    data = {
            'Activity': activities,
            'avg_f1_activity': avg_f1_activity,
            'std_f1_activity': std_f1_activity,
            'avg_f1_unmod_activity': avg_f1_unmod_activity,
            'std_f1_unmod_activity': std_f1_unmod_activity,
            'avg_comp_saved_activity': avg_comp_saved_activity,
            'std_comp_saved_activity': std_comp_saved_activity,
            'avg_data_saved_activity': avg_data_saved_activity,
            'std_data_saved_activity': std_data_saved_activity,
            'delta (f1-f1_unmod)': np.abs(avg_f1_activity - avg_f1_unmod_activity),
            'eval_criterion': eval_criterion            
            }
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)
    df = df.set_index('Activity')
        
    if args.name:
        filename = os.path.join(filepath, f'activity_avg_std_results_for_{args.dataset}_{algo_name}' + '_' + args.name + '.csv')
    else:
        filename = os.path.join(filepath, f'activity_avg_std_results_for_{args.dataset}_{algo_name}' + '.csv')
        
    # Save the DataFrame to a CSV file
    df.to_csv (filename, header=True) 
    
def save_subj_avg_std_results(algo_name, avg_f1_sub, std_f1_sub, avg_f1_unmod_sub, std_f1_unmod_sub, \
                            avg_comp_saved_sub, std_comp_saved_sub, avg_data_saved_sub, \
                                std_data_saved_sub, filepath, args):
    
    subjects = [f'Subject {x+1}' for x in range(len(avg_f1_sub))]

    data = {
            'Subject': subjects,
            'avg_f1_sub': avg_f1_sub,
            'std_f1_sub': std_f1_sub,
            'avg_f1_unmod_sub': avg_f1_unmod_sub,
            'std_f1_unmod_sub': std_f1_unmod_sub,
            'avg_comp_saved_sub': avg_comp_saved_sub,
            'std_comp_saved_sub': std_comp_saved_sub,
            'avg_data_saved_sub': avg_data_saved_sub,
            'std_data_saved_sub': std_data_saved_sub
            }
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)
    df = df.set_index('Subject')
        
    if args.name:
        filename = os.path.join(filepath, f'subj_avg_std_results_for_{args.dataset}_{algo_name}' + '_' + args.name + '.csv')
    else:
        filename = os.path.join(filepath, f'subj_avg_std_results_for_{args.dataset}_{algo_name}' + '.csv')
        
    # Save the DataFrame to a CSV file
    df.to_csv (filename, header=True) 
    
def save_exp_eval_csv(comp_saved, data_saved, percent_diff, args, filepath): 
    # evaluating performance of experimets performed using the delta avg. f1 score, avg. data saved and avg. comp saved
    exp_eval = (comp_saved[-1] + data_saved[-1])/(np.abs(percent_diff[-1])*0.01) 
    
    if args.name:
        filename = os.path.join(filepath, f'exp_eval_score_{args.dataset}_{args.algo_name}_' + args.name + '.csv')
    else:
        filename = os.path.join(filepath, f'best_results_for_{args.dataset}_{args.algo_name}_.csv')

    # Open the file in write mode and create a CSV writer object
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        # Write the data to the CSV file
        writer.writerow([float(exp_eval)])
