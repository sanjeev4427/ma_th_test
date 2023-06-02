##################################################
# All functions related to plotting  
##################################################
# Author: Sanjeev kumar
# Email: sanjeev.kumar(at)student.uni-siegen.de
##################################################

import matplotlib.pyplot as plt
import numpy as np

def plot_stems_f1(threshold_values_lst, f1_scores_thrs_array, config):
    """Method to plot and save the stem graphs for the f1 scores versus threshold values.
    Args:
        threshold_values_lst (list): stores threshold values
        f1_scores_thrs_array (ndarray): stores f1 scores
        config (dict): General setting dictionary
    Returns:
        None: None
    """
    markerline, stemlines, baseline = plt.stem(
        threshold_values_lst, f1_scores_thrs_array, bottom=0, use_line_collection=True
    )
    markerline.set_markerfacecolor("red")
    plt.xlabel("Threshold values")
    plt.ylabel("Average F1 score")
    plt.title("Average F1 score vs threshold values")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\figures\f1_score_thrs_array_{config["dataset"]}.png',
        format="png",
        bbox_inches="tight",
    )
    #plt.show()
    return None

def plot_stems_f1_act_wise(threshold_values_lst, f1_scores_act_thrs_array, config):
    '''RWHAR dataset activity labels:
        1 = stairs down
        2 = stairs up
        3 = jumping
        4 = lying
        5 = standing 
        6 = sitting
        7 = running/jogging 
        8 = walking'''

    f1_activity_wise_arr = []
    label_name_rwhar = ['stairs down', 'stairs up', 'jumping', 'lying', 'standing', 'sitting', 'running-jogging', 'walking']
    label_name_wetlab = []
    for i in range(config['nb_classes']):
        f1_activity_wise = [] # stores temporarily f1 scores for act, reset at start of new activity 
        for j in range(len(threshold_values_lst)):
            f1_activity_wise.append(f1_scores_act_thrs_array[j][i])
        f1_activity_wise_arr.append(f1_activity_wise)

        plt.figure()
        markerline, stemlines, baseline = plt.stem(
        threshold_values_lst, f1_activity_wise_arr[i], bottom=0, use_line_collection=True
        )
        markerline.set_markerfacecolor("green")
        plt.xlabel("Threshold values")
        plt.ylabel(f"Avg. f1 score for {label_name_rwhar[i]}")
        plt.title(f"Average f1 score for {label_name_rwhar[i]} vs threshold values")
        plt.savefig(
            rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\figures\f1_activity_wise_label_{label_name_rwhar[i]}_{config["dataset"]}.png',
            format="png",
            bbox_inches="tight",
        )
        #plt.show()

    return None

def plot_stems_data_act_wise(threshold_values_lst, data_saved_act_thrs_array, config):
    data_activity_wise_arr = []
    label_name_rwhar = ['stairs down', 'stairs up', 'jumping', 'lying', 'standing', 'sitting', 'running-jogging', 'walking']
    for i in range(config['nb_classes']):
        data_activity_wise = [] # stores temporarily f1 scores for act, reset at start of new activity 
        for j in range(len(threshold_values_lst)):
            data_activity_wise.append(data_saved_act_thrs_array[j][i])
        data_activity_wise_arr.append(data_activity_wise)

        plt.figure()
        markerline, stemlines, baseline = plt.stem(
        threshold_values_lst, data_activity_wise_arr[i], bottom=0, use_line_collection=True
        )
        markerline.set_markerfacecolor("green")
        plt.xlabel("Threshold values")
        if config['dataset'] == 'rwhar':
            plt.ylabel(f"Avg. data saved (%) for {label_name_rwhar[i]}")
            plt.title(f"Average data saved (%) for {label_name_rwhar[i]} vs threshold values")
            plt.savefig(
                rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\figures\data_activity_wise_label_{label_name_rwhar[i]}_{config["dataset"]}.png',
                format="png",
                bbox_inches="tight",
            )

        # #plt.show()
    return None

def plot_stems_comp_act_wise(threshold_values_lst, comp_saved_act_thrs_array, config):
    comp_activity_wise_arr = []
    label_name_rwhar = ['stairs down', 'stairs up', 'jumping', 'lying', 'standing', 'sitting', 'running-jogging', 'walking']
    for i in range(config['nb_classes']):
        comp_activity_wise = [] # stores temporarily f1 scores for act, reset at start of new activity 
        for j in range(len(threshold_values_lst)):
            comp_activity_wise.append(comp_saved_act_thrs_array[j][i])
        comp_activity_wise_arr.append(comp_activity_wise)

        plt.figure()
        markerline, stemlines, baseline = plt.stem(
        threshold_values_lst, comp_activity_wise_arr[i], bottom=0, use_line_collection=True
        )
        markerline.set_markerfacecolor("green")
        plt.xlabel("Threshold values")
        plt.ylabel(f"Avg. comp saved (%) for {label_name_rwhar[i]}")
        plt.title(f"Average comp saved (%) for {label_name_rwhar[i]} vs threshold values")
        plt.savefig(
            rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\figures\comp_activity_wise_label_{label_name_rwhar[i]}_{config["dataset"]}.png',
            format="png",
            bbox_inches="tight",
        )
        #plt.show()
    return None

def plot_stems_data_saving(threshold_values_lst, data_saved_thrs_array, config):
    """Method to plot and save the stem graphs for the data saved versus threshold values.
    Args:
        threshold_values_lst (list): stores threshold values
        data_saved_thrs_array (ndarray): stores data saved values
        config (dict): General setting dictionary
    Returns:
        None: None
    """
    markerline, stemlines, baseline = plt.stem(
        threshold_values_lst, data_saved_thrs_array, bottom=0, use_line_collection=True
    )
    markerline.set_markerfacecolor("green")
    plt.xlabel("Threshold values")
    plt.ylabel("Avg. data saving (%)")
    plt.title("Average data saving (%) vs threshold values")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\figures\data_saving_thrs_array_{config["dataset"]}.png',
        format="png",
        bbox_inches="tight",
    )
    #plt.show()
    return None


def plot_stems_comp_saving(threshold_values_lst, comp_saved_thrs_array, config):
    """Method to plot and save the stem graphs for the computation saved versus threshold values.
    Args:
        threshold_values_lst (list): stores threshold values
        comp_saved_thrs_array (ndarray): stores computation saved values
        config (dict): General setting dictionary
    Returns:
        None: None
    """
    markerline, stemlines, baseline = plt.stem(
        threshold_values_lst, comp_saved_thrs_array, bottom=0, use_line_collection=True
    )
    markerline.set_markerfacecolor("black")
    plt.xlabel("Threshold values")
    plt.ylabel("Avg. computaion saving (%)")
    plt.title("Average computaion saving (%) vs threshold values")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\figures\computaion_saving_thrs_array_{config["dataset"]}.png',
        format="png",
        bbox_inches="tight",
    )
    #plt.show()
    return None

def plot_loss_anneal(loss_array, n_iter, config, ann_rate):
    plt.figure()
    plt.plot(range(n_iter), loss_array)
    plt.xlabel("Iteration")
    plt.ylabel("Loss ")
    plt.title(f"Loss vs iteration \n annealing rate: {ann_rate}")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\figures\loss_iter_{ann_rate}_{config["dataset"]}.png', format="png", bbox_inches="tight"
    )
    #plt.show()
    return None

def plot_f1_iter(fscore_array, n_iter, config, ann_rate):
    """Method to plot and save graphs for f1 score versus number of iterations. 
        Used for monitoring in optimization algorithms.
    Args:
        fscore_array (ndarray): stores f1 scores
        n_iter (int): stores number of iterations
        config (dict): General setting dictionary
    Returns:
        None: None
    """
    plt.figure()
    plt.plot(range(n_iter), fscore_array)
    plt.xlabel("Iteration")
    plt.ylabel("F1 score")
    plt.title(f"F1 score vs iteration \n annealing rate: {ann_rate}")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\figures\f1score_iter_{ann_rate}_{config["dataset"]}.png', format="png", bbox_inches="tight"
    )
    #plt.show()
    return None

def plot_data_saved_iter(data_saved_array, n_iter, config, ann_rate):
    #change this
    """Method to plot and save graphs for f1 score versus number of iterations. 
        Used for monitoring in optimization algorithms.
    Args:
        fscore_array (ndarray): stores f1 scores
        n_iter (int): stores number of iterations
        config (dict): General setting dictionary
    Returns:
        None: None
    """
    plt.figure()
    plt.plot(range(n_iter), data_saved_array)
    plt.xlabel("Iteration")
    plt.ylabel("Average data saved")
    plt.title(f"Average data saved vs iteration \n annealing rate: {ann_rate}")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\figures\data_saved_iter_{ann_rate}_{config["dataset"]}.png', format="png", bbox_inches="tight"
    )
    #plt.show()
    return None

def plot_comp_saved_iter(comp_saved_array, n_iter, config, ann_rate):
    #change this
    """Method to plot and save graphs for f1 score versus number of iterations. 
        Used for monitoring in optimization algorithms.
    Args:
        fscore_array (ndarray): stores f1 scores
        n_iter (int): stores number of iterations
        config (dict): General setting dictionary
    Returns:
        None: None
    """
    plt.figure()
    plt.plot(range(n_iter), comp_saved_array)
    plt.xlabel("Iteration")
    plt.ylabel("Average computation saved")
    plt.title(f"Average computation saved vs iteration \n annealing rate: {ann_rate}")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\figures\comp_saved_iter_{ann_rate}_{config["dataset"]}.png', format="png", bbox_inches="tight"
    )
    #plt.show()
    return None

def plot_threshold_iter(threshold_array, n_iter, config, ann_rate):
    """Method to plot and save graphs for threshold values versus number of iterations. 
        Used for monitoring in optimization algorithms.
    Args:
        fscore_array (ndarray): stores f1 scores
        n_iter (int): stores number of iterations
        config (dict): General setting dictionary
    Returns:
        None: None
    """
    plt.figure()
    plt.plot(range(n_iter), threshold_array)
    plt.xlabel("Iteration")
    plt.ylabel("Threshold value")
    plt.title(f"Threshold value vs iteration \n annealing rate: {ann_rate}")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\figures\threshold_iter_{ann_rate}_{config["dataset"]}.png', format="png", bbox_inches="tight"
    )
    #plt.show()
    return None


def plot_tolerance_iter(tolerance_array, n_iter, config, ann_rate):
    """Method to plot and save graphs for tolerance values versus number of iterations. 
        Used for monitoring in optimization algorithms.
    Args:
        tolerance_array (ndarray): stores tolerance values
        n_iter (int): stores number of iterations
        config (dict): General setting dictionary
    Returns:
        None: None
    """
    plt.figure()
    plt.plot(range(n_iter), tolerance_array)
    plt.xlabel("Iteration")
    plt.ylabel("Tolerance value")
    plt.title(f"Tolerance value vs iteration \n annealing rate: {ann_rate}")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\figures\tolerance_iter_{ann_rate}_{config["dataset"]}.png', format="png", bbox_inches="tight"
    )
    #plt.show()
    return None

def plot_skip_windows_iter(skip_windows_array, n_iter, config, ann_rate):
    """Method to plot and save graphs for skip_windows values versus number of iterations. 
        Used for monitoring in optimization algorithms.
    Args:
        skip_windows_array (ndarray): stores skip_windows values
        n_iter (int): stores number of iterations
        config (dict): General setting dictionary
    Returns:
        None: None
    """
    plt.figure()
    plt.plot(range(n_iter), skip_windows_array)
    plt.xlabel("Iteration")
    plt.ylabel("Skip windows value")
    plt.title(f"Skip_windows value vs iteration \n annealing rate: {ann_rate}")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\figures\skip_windows_iter_{ann_rate}_{config["dataset"]}.png', format="png", bbox_inches="tight"
    )
    #plt.show()
    return None

def plot_temperature_iter(temperature_array, n_iter, config, ann_rate):

    plt.figure()
    plt.plot(range(n_iter), temperature_array)
    plt.xlabel("Iteration")
    plt.ylabel("Temperature value")
    plt.title(f"Temperature value vs iteration \n annealing rate: {ann_rate}")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\figures\temperature_iter_{ann_rate}_{config["dataset"]}.png', format="png", bbox_inches="tight"
    )
    return None

def plot_finding_initial_temp(temperature_array, config, n_iter_name):
    
    plt.figure()
    plt.plot(range(len(temperature_array)), temperature_array)
    plt.xlabel("Warming up iterations")
    plt.ylabel("Temperature value")
    plt.title(f"Temperature value vs Warming up iterations \n final temp = {temperature_array[-1]}")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\figures\finding_initial_temp_iter_at_each_temp_{n_iter_name}_{config["dataset"]}.png', format="png", bbox_inches="tight"
    )
    np.savetxt(rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\csv_data\finding_initial_temp_iter_at_each_temp_{n_iter_name}_{config["dataset"]}.csv',
            temperature_array, 
            fmt = '%s',
           delimiter =",")
    return None

def plot_acceptance_ratio_finding_initial_temp(acceptance_ratio_array, config, n_iter_name):
    
    plt.figure()
    plt.plot(range(len(acceptance_ratio_array)), acceptance_ratio_array)
    plt.xlabel("Warming up iterations")
    plt.ylabel("Acceptance ratio")
    plt.title(f"Acceptance ratio vs Warming up iterations \n iteration at each Temp: {n_iter_name}")
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\figures\acc_ratio_finding_initial_temp_iter_at_each_temp_{n_iter_name}_{config["dataset"]}.png', format="png", bbox_inches="tight"
    )
    np.savetxt(rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\csv_data\acc_ratio_finding_initial_temp_iter_at_each_temp_{n_iter_name}_{config["dataset"]}.csv',
           acceptance_ratio_array, 
            fmt = '%s',
           delimiter =",")
    return None

