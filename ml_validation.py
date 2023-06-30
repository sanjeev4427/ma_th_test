 # apply best settings on validation data
import os
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from SA_model_not_activity_wise.data_skipping_all_act import data_skipping_all_act
from data_processing.data_analysis import plot_sensordata_and_labels
from misc.osutils import mkdir_if_missing
from model.evaluate import evaluate_mod_participant_scores
from skip_heuristics_scripts.data_skipping import data_skipping

def  apply_best_sett_exp_one_hy(sbj, args,log_dir,*arg):  
    """
    Apply the best settings to the given subject's training and validation data.

    Args:
        sbj (str): The subject ID.
        args (argparse.Namespace): The command-line arguments passed to the script.
        log_dir (str): The directory where the log files are stored.
        *arg (str): Variable length argument list; the first argument should be the name of the file containing the best settings.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following six elements:
            mod_train_preds (np.ndarray): The modified predictions for the subject's training data after applying the best settings.
            mod_val_preds (np.ndarray): The modified predictions for the subject's validation data after applying the best settings.
            train_output (np.ndarray): The original predictions for the subject's training data.
            val_output (np.ndarray): The original predictions for the subject's validation data.
            mod_val_comp_saved (np.ndarray): The computation saved for each activity in the subject's validation data after applying the best settings.
            mod_val_data_saved (np.ndarray): The data saved for each activity in the subject's validation data after applying the best settings.
    """
        
    config=vars(args)   
    # comapare f1 after applying best setting
    if args.dataset == 'rwhar':
        train_output = np.loadtxt(rf'./ml_training_data/rwhar/train_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')
        val_output = np.loadtxt(rf'./ml_training_data/rwhar/val_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')
        activity_labels = range(8)
    elif args.dataset == 'wetlab':
        train_output = np.loadtxt(rf'./ml_training_data/wetlab/train_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')
        val_output = np.loadtxt(rf'./ml_training_data/wetlab/val_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')
        activity_labels = range(10)

    computations_saved = np.zeros(args.nb_classes)
    data_saved = np.zeros(args.nb_classes)

    # activate apply_best mode in dat_skipping
    best_filename = arg[0]
    apply_best = True

    # apply best on training data
    mod_train_preds, mod_train_preds_copy, train_data_saved,train_comp_saved = data_skipping_all_act(np.copy(train_output[:,0]), config, data_saved, computations_saved, apply_best, best_filename)

    # apply best on validation data
    mod_val_preds, mod_val_preds_copy, _,_ = data_skipping_all_act(np.copy(val_output[:,0]), config, data_saved, computations_saved, apply_best, best_filename)
         
         
    # re-calculating the data saved and CS for each activity
    lam = args.sw_overlap*0.01
    
    val_gt = val_output[:,1]
    val_pred = np.copy(val_output[:,0])
    
    modval_val_pred = np.vstack((mod_val_preds_copy, val_pred)).T
    
    
    # saving the modified prediction to csv file
    # File path to save the CSV file
    lam_file_path = os.path.join(log_dir, 'lam.csv')

    # Save the NumPy array to CSV
    np.savetxt(lam_file_path, np.array([lam]), delimiter=',')

    # print(f"Array saved to '{lam_file_path}' successfully.")

    # calcultating DS and CS for each activity
    skip_count_act = np.zeros(args.nb_classes)
    threshold_count_act = np.zeros(args.nb_classes)
    
    df = pd.DataFrame(modval_val_pred)
    for act in range(args.nb_classes):
        # Filter the DataFrame based on the conditions
        skip_df = df[(df.iloc[:, 1] == act) & (df.iloc[:, 0] == -5)]

        # Count the number of rows in the filtered DataFrame
        skip_count = len(skip_df)
        skip_count_act[act] = skip_count
        total_count_act = len(df[(df.iloc[:, 1] == act)])
        threshold_count =  total_count_act - skip_count
        threshold_count_act[act] =  threshold_count
        
    #calculating comp saved            
    mod_val_comp_saved = np.nan_to_num(skip_count_act/(skip_count_act + threshold_count_act)*100, nan=0)
    
    #calculating data saved
    zero_mask = np.logical_or(skip_count_act == 0, threshold_count_act == 0)
    mod_val_data_saved = np.nan_to_num(
        np.where(
            zero_mask,  # Condition to check if skip_count_act and threshold_count_act are both zero
            np.nan,  # Value to replace when the condition is True (both are zero)
            np.clip(
                100 * (skip_count_act - lam * (skip_count_act + 1)) / (
                        skip_count_act - lam * (skip_count_act + 1) + threshold_count_act + (threshold_count_act - 1) * (
                        1 - lam)
                ),
                a_min=0,
                a_max=100
            )
        ),
        nan=0
    )
    return mod_train_preds, mod_val_preds, mod_val_preds_copy, train_output, val_output, mod_val_comp_saved, mod_val_data_saved

def  apply_best_settings(sbj, args,*arg):  
    """
    Apply the best settings to the given subject's training and validation data.

    Args:
        sbj (str): The subject ID.
        args (argparse.Namespace): The command-line arguments passed to the script.
        *arg (str): Variable length argument list; the first argument should be the name of the file containing the best settings.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following six elements:
            mod_train_preds (np.ndarray): The modified predictions for the subject's training data after applying the best settings.
            mod_val_preds (np.ndarray): The modified predictions for the subject's validation data after applying the best settings.
            train_output (np.ndarray): The original predictions for the subject's training data.
            val_output (np.ndarray): The original predictions for the subject's validation data.
            mod_val_comp_saved (np.ndarray): The computation saved for each activity in the subject's validation data after applying the best settings.
            mod_val_data_saved (np.ndarray): The data saved for each activity in the subject's validation data after applying the best settings.
    """
        
    config=vars(args)   
    # comapare f1 after applying best setting
    if args.dataset == 'rwhar':
        train_output = np.loadtxt(rf'./ml_training_data/rwhar/train_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')
        val_output = np.loadtxt(rf'./ml_training_data/rwhar/val_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')
        activity_labels = range(8)
    elif args.dataset == 'wetlab':
        train_output = np.loadtxt(rf'./ml_training_data/wetlab/train_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')
        val_output = np.loadtxt(rf'./ml_training_data/wetlab/val_pred_sbj_{int(sbj)+1}.csv', dtype=float, delimiter=',')
        activity_labels = range(10)

    computations_saved = np.zeros(args.nb_classes)
    data_saved = np.zeros(args.nb_classes)

    # activate apply_best mode in dat_skipping
    best_filename = arg[0]
    apply_best = True

    # apply best on training data
    mod_train_preds, mod_val_train_copy, train_data_saved,train_comp_saved= data_skipping(-2, np.copy(train_output[:,0]), config, data_saved, computations_saved, apply_best, best_filename)

    # apply best on validation data
    mod_val_preds, mod_val_preds_copy, _,_ = data_skipping(-2, np.copy(val_output[:,0]), config, data_saved, computations_saved, apply_best, best_filename)
    
    
    # re-calculating the data saved and CS for each activity
    lam = args.sw_overlap*0.01
    
    val_gt = val_output[:,1]
    val_pred = np.copy(val_output[:,0])
    
    modval_val_pred = np.vstack((mod_val_preds_copy, val_pred)).T
    
    
    # # saving the modified prediction to csv file
    # # File path to save the CSV file
    # mod_val_val_pred_file_path = r'./temp_files_del_later/mod_val_val_pred.csv'

    # # Save the NumPy array to CSV
    # np.savetxt(mod_val_val_pred_file_path, modval_val_pred, delimiter=',')

    # print(f"Array saved to '{mod_val_val_pred_file_path}' successfully.")

    # calcultating DS and CS for each activity
    skip_count_act = np.zeros(args.nb_classes)
    threshold_count_act = np.zeros(args.nb_classes)
    
    df = pd.DataFrame(modval_val_pred)
    for act in range(args.nb_classes):
        # Filter the DataFrame based on the conditions
        skip_df = df[(df.iloc[:, 1] == act) & (df.iloc[:, 0] == -5)]

        # Count the number of rows in the filtered DataFrame
        skip_count = len(skip_df)
        skip_count_act[act] = skip_count
        total_count_act = len(df[(df.iloc[:, 1] == act)])
        threshold_count =  total_count_act - skip_count
        threshold_count_act[act] =  threshold_count
        
    #calculating comp saved            
    mod_val_comp_saved = np.nan_to_num(skip_count_act/(skip_count_act + threshold_count_act)*100, nan=0)
    
    #calculating data saved
    zero_mask = np.logical_or(skip_count_act == 0, threshold_count_act == 0)
    mod_val_data_saved = np.nan_to_num(
        np.where(
            zero_mask,  # Condition to check if skip_count_act and threshold_count_act are both zero
            np.nan,  # Value to replace when the condition is True (both are zero)
            np.clip(
                100 * (skip_count_act - lam * (skip_count_act + 1)) / (
                        skip_count_act - lam * (skip_count_act + 1) + threshold_count_act + (threshold_count_act - 1) * (
                        1 - lam)
                ),
                a_min=0,
                a_max=100
            )
        ),
        nan=0
    )

    return mod_train_preds, mod_val_preds, train_output, val_output, mod_val_comp_saved, mod_val_data_saved


def ml_validation(args, data, algo_name, log_folder_name):
    """
    Calculate cross-participant scores using Leave-One-Subject-Out Cross Validation (LOSO CV).
    
    Args:
        args: A Namespace object containing the parsed command line arguments.
        data: A numpy array, where each row contains the label and features of a sample.
        algo_name: A string representing the name of the algorithm being used.
        log_folder_name: A string representing the path of logging folder.
        
    Returns:
        None
    """
    print('\nCALCULATING CROSS-PARTICIPANT SCORES USING LOSO CV.\n')

    lst_sbj = np.unique(data[:, 0])
    all_mod_eval_output = None
    mod_cp_scores = np.zeros((4, args.nb_seeds, args.nb_classes, len(lst_sbj)))
    cp_scores = np.zeros((4, args.nb_seeds, args.nb_classes, len(lst_sbj)))
    log_dir = log_folder_name 
    mod_cp_savings = np.zeros((2, args.nb_seeds, args.nb_classes, len(lst_sbj)))
    
    # various experiments that needs to be validated
    exp_one_hyp_all_act = False  # exp 1
    exp_train_gt = True # exp 3
    exp_loss_mx_data = False # exp 4
    exp_loss_mx_f1 = False # exp 4

    for seed_i in range(0,args.nb_seeds): 
    # for seed_i in range(0,1): 
        for sbj in lst_sbj:
            # paths of the trained huperparamenters for various experiments, else part is exp 2 (act wise)
            if exp_loss_mx_data:
                filename_best_csv = fr"./best_files_exp/exp_loss_fun/exp_mx_data_saved/best_files/best_results_for_{args.dataset}_{algo_name.upper()}_{int(sbj)+1}_{args.name}_seed_{seed_i+1}.csv"

            elif exp_loss_mx_f1:
                filename_best_csv = fr"./best_files_exp/exp_loss_fun/exp_mx_fscore/best_files/best_results_for_{args.dataset}_{algo_name.upper()}_{int(sbj)+1}_{args.name}_seed_{seed_i+1}.csv"

            elif exp_one_hyp_all_act:
                filename_best_csv = fr"./best_files_exp/exp_one_hyper_set/best_exp_one_hy_set/best_results_for_all_{args.dataset}_{algo_name.upper()}_{int(sbj)+1}_{args.name}_seed_{seed_i+1}.csv"
            
            elif exp_train_gt:
                filename_best_csv = fr"best_files_exp/best_gt_data_train_exp/best_files/best_results_for_{args.dataset}_{algo_name.upper()}_{int(sbj)+1}_{args.name}_seed_{seed_i+1}_tol_train.csv"
            else:
                filename_best_csv = fr"./best_files_exp/exp_search_space/best_files_exp_search_space/best_results_for_{args.dataset}_{algo_name.upper()}_{int(sbj)+1}_{args.name}_seed_{seed_i+1}.csv"

            # for plotting sensor data.
            val_data = data[data[:, 0] == sbj]

            # # modified training and validation predictions
            _, mod_val_preds, train_output, val_output, mod_val_comp_saved, mod_val_data_saved \
                                                    = apply_best_settings(sbj, args,filename_best_csv)

            # use this for exp_one_hyp_all_act experiment 
            #! uncomment for exp 1
            # if exp_one_hyp_all_act: 
            # # modified training and validation predictions
            #     mod_train_preds, mod_val_preds, mod_val_preds_copy, train_output, val_output, mod_val_comp_saved, mod_val_data_saved \
            #                                         = apply_best_sett_exp_one_hy(sbj, args,log_dir,filename_best_csv)
                                                    
            val_gt = val_output[:,1]
            val_pred = np.copy(val_output[:,0])

            train_gt = train_output[:,1]
            train_pred = np.copy(train_output[:,0])


            if all_mod_eval_output is None:
                all_mod_eval_output = np.vstack((mod_val_preds, val_gt)).T
            else:
                all_mod_eval_output = np.concatenate((all_mod_eval_output, np.vstack((mod_val_preds, val_gt)).T), axis=0)

            # plot sensor data and subject wise activity and modified activity
            # to create activity plot wrt time from best settings 
            # mkdir_if_missing(log_dir)
            if args.name:
                plot_sensordata_and_labels(val_data, sbj, val_gt, args.class_names, val_pred,
                                            mod_val_preds,
                                            figname=os.path.join(log_dir, 'sbj_' + str(int(sbj)+1) + '_' + args.dataset + '_' + args.algo_name + '_' + args.name + f'_seed_{seed_i+1}.png'))
            else:
                plot_sensordata_and_labels(val_data, sbj, val_gt, args.class_names, val_pred,
                                        mod_val_preds,
                                        figname=os.path.join(log_dir, 'sbj_' + str(int(sbj)) + '.png')) 

            # fill values for normal evaluation
            labels = list(range(0, args.nb_classes))
            cp_scores[0, seed_i, :, int(sbj)] = jaccard_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)
            cp_scores[1, seed_i, :, int(sbj)] = precision_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)
            cp_scores[2, seed_i, :, int(sbj)] = recall_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)
            cp_scores[3, seed_i, :, int(sbj)] = f1_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)

            mod_cp_scores[0, seed_i, :, int(sbj)] = jaccard_score(val_gt, mod_val_preds, average=None, labels=labels)
            mod_cp_scores[1, seed_i, :, int(sbj)] = precision_score(val_gt, mod_val_preds, average=None, labels=labels)
            mod_cp_scores[2, seed_i, :, int(sbj)] = recall_score(val_gt, mod_val_preds, average=None, labels=labels)
            mod_cp_scores[3, seed_i, :, int(sbj)] = f1_score(val_gt, mod_val_preds, average=None, labels=labels)

            mod_cp_savings[0, seed_i, :, int(sbj)] = mod_val_comp_saved
            mod_cp_savings[1, seed_i, :, int(sbj)] = mod_val_data_saved


                    # print("SUBJECT {0} VALIDATION RESULTS: ".format(int(sbj) + 1))
                    # # print("Accuracy: {0}".format(jaccard_score(val_gt, val_pred, average=None, labels=labels)))
                    # # print("Precision: {0}".format(precision_score(val_gt, val_pred, average=None, labels=labels)))
                    # # print("Recall: {0}".format(recall_score(val_gt, val_pred, average=None, labels=labels)))
                    # print("F1: {0}".format(f1_score(val_gt, val_pred, average=None, labels=labels)))
                    # print("Average F1: {0}".format(f1_score(val_gt, val_pred, average='macro')), '\n')


                    # print("SUBJECT {0} MODIFIED VALIDATION RESULTS: ".format(int(sbj) + 1))
                    # # print("Accuracy: {0}".format(jaccard_score(val_gt, mod_val_preds, average=None, labels=labels)))
                    # # print("Precision: {0}".format(precision_score(val_gt, mod_val_preds, average=None, labels=labels)))
                    # # print("Recall: {0}".format(recall_score(val_gt, mod_val_preds, average=None, labels=labels)))
                    # print("F1: {0}".format(f1_score(val_gt, mod_val_preds, average=None, labels=labels)))
                    # print("Average F1: {0}".format(f1_score(val_gt, mod_val_preds, average='macro')), '\n')

                    # print("SUBJECT {0} ML TRAINING RESULTS: ".format(int(sbj) + 1))
                    # print("F1: {0}".format(f1_score(train_gt, train_pred, average=None, labels=labels)))
                    # print("Average F1: {0}".format(f1_score(train_gt, train_pred, average='macro')), '\n')

                    # print("SUBJECT {0} ML MODIFIED TRAINING RESULTS: ".format(int(sbj) + 1))
                    # print("F1: {0}".format(f1_score(train_gt, mod_train_preds, average=None, labels=labels)))
                    # print("Average F1: {0}".format(f1_score(train_gt, mod_train_preds, average='macro')), '\n')

                    # print("SUBJECT {} COMPUTATION SAVINGS: {} %".format(int(sbj) + 1, mod_val_comp_saved))

        if args.save_analysis:
            mkdir_if_missing(log_dir)
            mod_cp_score_acc = pd.DataFrame(mod_cp_scores[0, seed_i, :, :], index=None)
            mod_cp_score_acc.index = args.class_names
            mod_cp_score_prec = pd.DataFrame(mod_cp_scores[1, seed_i, :, :], index=None)
            mod_cp_score_prec.index = args.class_names
            mod_cp_score_rec = pd.DataFrame(mod_cp_scores[2, seed_i, :, :], index=None)
            mod_cp_score_rec.index = args.class_names
            mod_cp_score_f1 = pd.DataFrame(mod_cp_scores[3, seed_i, :, :], index=None)
            mod_cp_score_f1.index = args.class_names
            mod_cp_savings_comp = pd.DataFrame(mod_cp_savings[0, seed_i, :, :], index=None)
            mod_cp_savings_comp.index = args.class_names
            mod_cp_savings_data = pd.DataFrame(mod_cp_savings[1, seed_i, :, :], index=None)
            mod_cp_savings_data.index = args.class_names

            if args.name:
                mod_cp_score_acc.to_csv(os.path.join(log_dir,f'mod_cp_scores_acc_{args.dataset}_{args.algo_name}_{args.name}_seed_{seed_i}.csv'))
                mod_cp_score_prec.to_csv(os.path.join(log_dir, 'mod_cp_scores_prec_{}_{}_{}_seed_{}.csv').format(args.dataset, args.algo_name, args.name,seed_i))
                mod_cp_score_rec.to_csv(os.path.join(log_dir, 'mod_cp_scores_rec_{}_{}_{}_seed_{}.csv').format(args.dataset, args.algo_name, args.name,seed_i))
                mod_cp_score_f1.to_csv(os.path.join(log_dir, 'mod_cp_scores_f1_{}_{}_{}_seed_{}.csv').format(args.dataset, args.algo_name, args.name,seed_i))
                mod_cp_savings_comp.to_csv(os.path.join(log_dir, 'mod_cp_savings_comp_{}_{}_{}_seed_{}.csv').format(args.dataset, args.algo_name, args.name,seed_i))
                mod_cp_savings_data.to_csv(os.path.join(log_dir, 'mod_cp_savings_data_{}_{}_{}_seed_{}.csv').format(args.dataset, args.algo_name, args.name,seed_i))
            else:
                mod_cp_score_acc.to_csv(os.path.join(log_dir, 'mod_cp_scores_acc.csv'))
                mod_cp_score_prec.to_csv(os.path.join(log_dir, 'mod_cp_scores_prec.csv'))
                mod_cp_score_rec.to_csv(os.path.join(log_dir, 'mod_cp_scores_rec.csv'))
                mod_cp_score_f1.to_csv(os.path.join(log_dir, 'mod_cp_scores_f1.csv'))
                mod_cp_savings_comp.to_csv(os.path.join(log_dir, 'mod_cp_savings_comp.csv'))
                mod_cp_savings_data.to_csv(os.path.join(log_dir, 'mod_cp_savings_data.csv'))
    
    # uncomment these lines for debugging
    # nb_seeds=3
    # nb_classes = 8
    # nb_subj = 15
    # mod_cp_scores = np.random.rand(4, nb_seeds, nb_classes, nb_subj)
    # cp_scores = np.random.rand(4, nb_seeds, nb_classes, nb_subj)
    # mod_cp_savings = np.random.rand(2, nb_seeds, nb_classes, nb_subj)
    evaluate_mod_participant_scores(algo_name, savings_scores=mod_cp_savings,
                            participant_scores=mod_cp_scores,
                            participant_scores_unmod=cp_scores,
                            # gen_gap_scores=mod_train_val_gap,
                            gen_gap_scores=None,                            
                            input_cm=all_mod_eval_output,
                            class_names=args.class_names,
                            nb_subjects=len(lst_sbj),
                            filepath=log_folder_name,
                            filename='mod-cross-participant',
                            args=args
                            )

    

    
