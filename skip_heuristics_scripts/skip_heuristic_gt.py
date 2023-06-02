##################################################
# Skip heuristics algorithm with fixed hyperparameters using ground truth prediction.
##################################################
# Author: Sanjeev Kumar 
# Email: sanjeev.kumar(at)student.uni-siegen.de
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
# Author: Michael Moeller
# Email: michael.moeller(at)uni-siegen.de
##################################################

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from misc.torchutils import seed_worker
from model.data_skipping import data_skipping
from data_processing.sliding_window import apply_sliding_window
from model.plotting import plot_stems_f1, plot_stems_data_saving, plot_stems_comp_saving, plot_stems_f1_act_wise,plot_stems_data_act_wise,plot_stems_comp_act_wise

def activity_wise_performance(val_gt, mod_val_preds, computations_saved, data_saved, count_array):

    mod_val_gt = np.copy(val_gt)
    '''RWHAR dataset activity labels:
        1 = stairs down
        2 = stairs up
        3 = jumping
        4 = lying
        5 = standing 
        6 = sitting
        7 = running/jogging 
        8 = walking'''
    # store each activity predications and activity ground truth array
    acitivity_pred_array = []
    activity_gt_array = []
    f1_score_array = []
    for activity in np.unique(mod_val_gt):
        idx = np.where(activity == mod_val_gt)[0]
        activity_pred = mod_val_preds[idx]
        activity_gt = mod_val_gt[idx]
        # calculate f1 score for each activity for each subject
        f1_score_array.append(round(f1_score(activity_gt, activity_pred, average='macro'), 2))
        # store each activity predications and activity ground truth array for each subject
        acitivity_pred_array.append(activity_pred)
        activity_gt_array.append(activity_gt)

    # calculating data saved activity wise for each subject
    data_saved_percent_sbj = (data_saved/count_array) * 100

    # calculating computation saved activity wise
    computations_saved_percent_sbj = (computations_saved/count_array) * 100
        
    return f1_score_array, data_saved_percent_sbj, computations_saved_percent_sbj

def skip_heuristic_gt(data, args):
    """Method to perform skip heuristics algorithm with fixed hyperparameters on ground truth predictions.

    Args:
        data (ndarray): processed data having sensor data and corresponding activity labels.
        args (): for passing command line arguments

    Returns:
        None: None
    """
    # config dictionary containing setting parameters
    config= vars(args)
    threshold_values_lst=[0.1, 0.2, 0.4, 0.6, 0.8, 1]
    # threshold_values_lst=[0.1]
    f1_scores_thrs_array=[]
    data_saved_thrs_array=[]
    comp_saved_thrs_array=[]
    f1_scores_act_thrs_array=[]
    data_saved_act_thrs_array=[]
    comp_saved_act_thrs_array=[]
    for threshold_val in threshold_values_lst:       
        config['saving_threshold'] = threshold_val
        print('x'*100)
        print(f'saving threshold: {config["saving_threshold"]}')
        f1_scores_sbj_array=[]
        data_saved_sbj_array=[]
        comp_saved_sbj_array=[]
        f1_act_wise_sbj_array = []
        data_saved_act_sbj_array= []
        comp_saved_act_sbj_array= []
        for i, sbj in enumerate(np.unique(data[:, 0])):
            print('-'*50)
            # loading data
            print('\n DATA SKIPPING APPLIED ON VALIDATION DATASET: SUBJECT {0} OF {1}'.format(int(sbj) + 1, int(np.max(data[:, 0])) + 1))
            train_data = data[data[:, 0] != sbj] # training data from all but one subject
            val_data = data[data[:, 0] == sbj]  # validaaton data from one subject

            # calculate concurrent windows
            curr_label = None
            curr_window = 0
            windows = []
            for sbj_id in np.unique(train_data[:, 0]): # first column is subject id
                sbj_label = train_data[train_data[:, 0] == sbj_id][:, -1] # label column for each subject in training data
                for label in sbj_label:
                    if label != curr_label and curr_label is not None:
                        windows.append([curr_label, curr_window / args.sampling_rate, curr_window]) #? what does dividing by sampling rate do?
                        curr_label = label                                                         #? I think it calculates activity duration, total windows/num of wind in 1 sec                    
                        curr_window = 1   # reset curr_window to 1
                    elif label == curr_label:
                        curr_window += 1
                    else:
                        curr_label = label
                        curr_window += 1
            windows = np.array(windows) # probably keeping track of after how many windows new activity/label are added

            # calculate savings array, calculates activity duration for each class/activity
            saving_array = np.zeros(args.nb_classes)
            for label in range(args.nb_classes):
                label_windows = windows[windows[:, 0] == label] #accessing windows label wise
                if label_windows.size != 0:
                    if args.saving_type == 'mean':
                        saving_array[int(label)] = np.mean(label_windows[:, 1].astype(float)) # mean of activity duration of each activity across all subjects
                    elif args.saving_type == 'median':
                        saving_array[int(label)] = np.median(label_windows[:, 1].astype(float)) # median of activity duration of each activity across all subjects
                    elif args.saving_type == 'min':
                        saving_array[int(label)] = np.min(label_windows[:, 1].astype(float))
                    elif args.saving_type == 'max':
                        saving_array[int(label)] = np.max(label_windows[:, 1].astype(float))
                    elif args.saving_type == 'first_quartile':
                        saving_array[int(label)] = np.percentile(label_windows[:, 1].astype(float), 25)

            args.saving_array = saving_array
            print(f'{args.saving_type} activity duaration for all activities: \n', 
                        saving_array)

            # Sensor data is segmented using a sliding window mechanism
            X_val, y_val = apply_sliding_window(val_data[:, :-1], val_data[:, -1],
                                                sliding_window_size=args.sw_length,
                                                unit=args.sw_unit,
                                                sampling_rate=args.sampling_rate,
                                                sliding_window_overlap=args.sw_overlap,
                                                )

            X_val = X_val[:, :, 1:] # removing subj_id from X_val
            
            val_features, val_labels = X_val, y_val
            
            g = torch.Generator()
            g.manual_seed(config['seed'])

            dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_labels))
            valloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
                            worker_init_fn=seed_worker, generator=g, pin_memory=True)

            # helper objects
            val_gt = []
            with torch.no_grad():
                computations_saved = np.zeros(config['nb_classes'])
                data_saved = np.zeros(config['nb_classes'])
                # iterate over validation dataset
                for i, (x, y) in enumerate(valloader):
                    # send x and y to GPU
                    inputs, targets = x.to(config['gpu']), y.to(config['gpu'])
                    y_true = targets.cpu().numpy().flatten()
                    val_gt = np.concatenate((np.array(val_gt, int), np.array(y_true, int)))

                # mod_val_gt will change (= mod_val_preds) when passed to data_skipping 
                # use val_gt for f1 score calculation
                mod_val_gt = np.copy(val_gt)
                
                # feed validation data predictions, skip over some data 
                mod_val_preds, data_saved, computations_saved = data_skipping(mod_val_gt, config, data_saved, computations_saved)

                count_array = np.zeros(config['nb_classes'])
                for k in range(config['nb_classes']):
                    count_array[k] = (val_gt == k).sum()

                # calculating activity wise f1 scores for each subject 
                f1_act_wise_sbj, data_saved_percent_sbj, computations_saved_percent_sbj \
                                = activity_wise_performance(val_gt, mod_val_preds, computations_saved, data_saved, count_array)
                

                # print epoch evaluation results for train and validation dataset
                print(
                    "\nComputations saved (per Class)\n",
                    computations_saved, "\n",
                    count_array, "\n"*2,
                    "Computations saved (A): {} of {} windows\n".format(sum(computations_saved), sum(count_array)),
                    "Computations saved: {:.2%}\n".format(sum(computations_saved) / sum(count_array)),
                    "Data saved (per Class)\n",
                    data_saved, "\n",
                    count_array * (1 - config['sw_overlap'] * 0.01) * config['sw_length'], "\n",
                    "Data saved (A): {} of {} seconds\n".format(sum(data_saved), sum(count_array) * (1 - config['sw_overlap'] * 0.01) * config['sw_length']),
                    "Data saved (R): {:.2%}\n".format(sum(data_saved) / (sum(count_array) * (1 - config['sw_overlap'] * 0.01) * config['sw_length']))
                    )

                # comp_windows and count_array are same so comp saved above and here are same
                #? data_windows and count_array are not same so data saved above and here should not be same 
                #? ...but output shows that they are same

                comp_windows, data_windows = count_array, count_array * (1 - config['sw_overlap'] * 0.01) * config['sw_length']

                print("SUBJECT {} COMPUTATION SAVINGS: {:.2%}".format(int(sbj) + 1, sum(computations_saved) / sum(comp_windows)))
                print("SUBJECT {} DATA SAVINGS: {:.2%}".format(int(sbj) + 1, sum(data_saved) / sum(data_windows)))
                print("Mod Valid F1 (M): {:.4f} for subject {}".format(f1_score(val_gt, mod_val_preds, average='macro'), int(sbj) + 1))
                print('\n', f'Saving F1 score for subject {int(sbj) + 1}...')
                f1_scores_sbj_array.append(f1_score(val_gt, mod_val_preds, average='macro'))
                data_saved_sbj_array.append(sum(data_saved) / sum(data_windows))
                comp_saved_sbj_array.append(sum(computations_saved) / sum(comp_windows))
                f1_act_wise_sbj_array.append(f1_act_wise_sbj)
                data_saved_act_sbj_array.append(data_saved_percent_sbj)
                comp_saved_act_sbj_array.append(computations_saved_percent_sbj)
 
        avg_f1_score_sbj = round(np.mean(f1_scores_sbj_array), 2)
        avg_data_saved_sbj = round(np.mean(data_saved_sbj_array)*100, 2)
        avg_comp_saved_sbj = round(np.mean(comp_saved_sbj_array)*100, 2)
        avg_f1_score_act_sbj = np.mean(f1_act_wise_sbj_array, axis=0)
        avg_data_saved_act_sbj = np.mean(data_saved_act_sbj_array, axis=0)
        avg_comp_saved_act_sbj = np.mean(comp_saved_act_sbj_array, axis=0)
        print(f"Average data saves for all subjects with threshold value {config['saving_threshold']} : {avg_data_saved_sbj}")
        print(f"Average computation saves for all subjects with threshold value {config['saving_threshold']} : {avg_comp_saved_sbj}")
        print(f"Average f1 score for all subjects with threshold value {config['saving_threshold']} : {avg_f1_score_sbj}")        
        print(f"Average f1 score for each activity with threshold value {config['saving_threshold']} : {avg_f1_score_act_sbj}")
        print(f"Averae data saved for each activity with threshold value {config['saving_threshold']} : {avg_data_saved_act_sbj}")
        print(f"Averae computations saved for each activity with threshold value {config['saving_threshold']} : {avg_comp_saved_act_sbj}")
        f1_scores_thrs_array.append(avg_f1_score_sbj)
        data_saved_thrs_array.append(avg_data_saved_sbj)
        comp_saved_thrs_array.append(avg_comp_saved_sbj)
        f1_scores_act_thrs_array.append(avg_f1_score_act_sbj)
        data_saved_act_thrs_array.append(avg_data_saved_act_sbj)
        comp_saved_act_thrs_array.append(avg_comp_saved_act_sbj)

    for i,val in enumerate(threshold_values_lst):
        print(f"Average f1 score for threshold value {val} : {f1_scores_thrs_array[i]}")
        print(f"Average data saved for threshold value {val} : {data_saved_thrs_array[i]}")
        print(f"Average computation saved for threshold value {val} : {comp_saved_thrs_array[i]}")
        # print(f"Average f1 score for each activity for threshold value {val} : {f1_scores_act_thrs_array[i]}", '\n'*2)
    # plot_stems_f1(threshold_values_lst, f1_scores_thrs_array, config)
    # plot_stems_data_saving(threshold_values_lst, data_saved_thrs_array, config)
    # plot_stems_comp_saving(threshold_values_lst, comp_saved_thrs_array, config)
    plot_stems_data_act_wise(threshold_values_lst, data_saved_act_thrs_array, config)
    plot_stems_comp_act_wise(threshold_values_lst, comp_saved_act_thrs_array, config)
    return None