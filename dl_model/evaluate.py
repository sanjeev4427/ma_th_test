##################################################
# All functions related to evaluating training and testing results
##################################################


import matplotlib.pyplot as plt
import numpy as np
import itertools

import os
from sklearn.metrics import confusion_matrix
from log_data_scripts.save_csv_results import save_act_avg_std_results, save_subj_avg_std_results

from misc.osutils import mkdir_if_missing
from ml_evaluate import mod_bar_plot_activity, mod_bar_plot_sbj


def plot_confusion_matrix(input, target_names, title='Confusion matrix', cmap=None, normalize=True, output_path=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    input:        confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    cm = confusion_matrix(input[:, 1], input[:, 0])
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if output_path is not None:
        plt.savefig(output_path)


def evaluate_participant_scores(participant_scores, gen_gap_scores, input_cm, class_names, nb_subjects, filepath, filename, args):
    """
    Function which prints evaluation metrics of each participant, overall average and saves confusion matrix

    :param participant_scores: numpy array
        Array containing all results
    :param gen_gap_scores:
        Array containing generalization gap results
    :param input_cm: confusion matrix
        Confusion matrix of overall results
    :param class_names: list of strings
        Class names
    :param nb_subjects: int
        Number of subjects in dataset
    :param filepath: str
        Directory where to save plots to
    :param filename: str
        Name of plot
    :param args: dict
        Overall settings dict
    """
    print('\nPREDICTION RESULTS')
    print('-------------------')
    print('Average results')
    avg_acc = np.mean(participant_scores[0, :, :])
    std_acc = np.std(participant_scores[0, :, :])
    avg_prc = np.mean(participant_scores[1, :, :])
    std_prc = np.std(participant_scores[1, :, :])
    avg_rcll = np.mean(participant_scores[2, :, :])
    std_rcll = np.std(participant_scores[2, :, :])
    avg_f1 = np.mean(participant_scores[3, :, :])
    std_f1 = np.std(participant_scores[3, :, :])
    print('Avg. Accuracy {:.4f} (±{:.4f}), '.format(avg_acc, std_acc),
          'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
          'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
          'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1))
    if args.include_null:
        print('Average results (no null)')
        avg_acc = np.mean(participant_scores[0, 1:, :])
        std_acc = np.std(participant_scores[0, 1:, :])
        avg_prc = np.mean(participant_scores[1, 1:, :])
        std_prc = np.std(participant_scores[1, 1:, :])
        avg_rcll = np.mean(participant_scores[2, 1:, :])
        std_rcll = np.std(participant_scores[2, 1:, :])
        avg_f1 = np.mean(participant_scores[3, 1:, :])
        std_f1 = np.std(participant_scores[3, 1:, :])
        print('Avg. Accuracy {:.4f} (±{:.4f}), '.format(avg_acc, std_acc),
              'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
              'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
              'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1))
    print('Average class results')
    for i, class_name in enumerate(class_names):
        avg_acc = np.mean(participant_scores[0, i, :])
        std_acc = np.std(participant_scores[0, i, :])
        avg_prc = np.mean(participant_scores[1, i, :])
        std_prc = np.std(participant_scores[1, i, :])
        avg_rcll = np.mean(participant_scores[2, i, :])
        std_rcll = np.std(participant_scores[2, i, :])
        avg_f1 = np.mean(participant_scores[3, i, :])
        std_f1 = np.std(participant_scores[3, i, :])
        print('Class {}: Avg. Accuracy {:.4f} (±{:.4f}), '.format(class_name, avg_acc, std_acc),
              'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
              'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
              'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1))
    print('Subject-wise results')
    for subject in range(nb_subjects):
        print('Subject ', subject + 1, ' results: ')
        for i, class_name in enumerate(class_names):
            acc = participant_scores[0, i, subject]
            prc = participant_scores[1, i, subject]
            rcll = participant_scores[2, i, subject]
            f1 = participant_scores[3, i, subject]
            print('Class {}: Accuracy {:.4f}, '.format(class_name, acc),
                  'Precision {:.4f}, '.format(prc),
                  'Recall {:.4f}, '.format(rcll),
                  'F1-Score {:.4f}'.format(f1))

    print('\nGENERALIZATION GAP ANALYSIS')
    print('-------------------')
    print('Average results')
    avg_acc = np.mean(gen_gap_scores[0, :])
    std_acc = np.std(gen_gap_scores[0, :])
    avg_prc = np.mean(gen_gap_scores[1, :])
    std_prc = np.std(gen_gap_scores[1, :])
    avg_rcll = np.mean(gen_gap_scores[2, :])
    std_rcll = np.std(gen_gap_scores[2, :])
    avg_f1 = np.mean(gen_gap_scores[3, :])
    std_f1 = np.std(gen_gap_scores[3, :])
    print('Avg. Accuracy {:.4f} (±{:.4f}), '.format(avg_acc, std_acc),
          'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc, std_prc),
          'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll, std_rcll),
          'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1, std_f1))
    print('Subject-wise results')
    for subject in range(nb_subjects):
        print('Subject ', subject + 1, ' results: ')
        acc = gen_gap_scores[0, subject]
        prc = gen_gap_scores[1, subject]
        rcll = gen_gap_scores[2, subject]
        f1 = gen_gap_scores[3, subject]
        print('Accuracy {:.4f}, '.format(acc),
              'Precision {:.4f}, '.format(prc),
              'Recall {:.4f}, '.format(rcll),
              'F1-Score {:.4f}'.format(f1))

    # create boxplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.suptitle('Average Participant Results', size=16)
    axs[0, 0].set_title('Accuracy')
    axs[0, 0].boxplot(participant_scores[0, :, :].T, labels=class_names, showmeans=True)
    axs[0, 1].set_title('Precision')
    axs[0, 1].boxplot(participant_scores[1, :, :].T, labels=class_names, showmeans=True)
    axs[1, 0].set_title('Recall')
    axs[1, 0].boxplot(participant_scores[2, :, :].T, labels=class_names, showmeans=True)
    axs[1, 1].set_title('F1-Score')
    axs[1, 1].boxplot(participant_scores[3, :, :].T, labels=class_names, showmeans=True)
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)
    fig.subplots_adjust(hspace=0.5)
    mkdir_if_missing(filepath)
    if args.name:
        plt.savefig(os.path.join(filepath, filename + '_bx_{}.png'.format(args.name)))
        plot_confusion_matrix(input_cm, class_names, normalize=False,
                              output_path=os.path.join(filepath, filename + '_cm_{}.png'.format(args.name)))
    else:
        plt.savefig(os.path.join(filepath, filename + '_bx.png'))
        plot_confusion_matrix(input_cm, class_names, normalize=False,
                              output_path=os.path.join(filepath, filename + '_cm.png'))


def evaluate_mod_participant_scores(algo_name, savings_scores, participant_scores, participant_scores_unmod, gen_gap_scores, input_cm, class_names, nb_subjects, filepath, filename, args):
    """
    Function which prints evaluation metrics of each participant, overall average and saves confusion matrix

    :param participant_scores: numpy array
        Array containing all results
    :param gen_gap_scores:
        Array containing generalization gap results
    :param input_cm: confusion matrix
        Confusion matrix of overall results
    :param class_names: list of strings
        Class names
    :param nb_subjects: int
        Number of subjects in dataset
    :param filepath: str
        Directory where to save plots to
    :param filename: str
        Name of plot
    :param args: dict
        Overall settings dict
    """
    print('\nSAVINGS RESULTS')
    print('-------------------')
    print('\nMODIFIED PREDICTION RESULTS')
    print('-------------------')
    print('Calculating average performance/saving over all activities and subjects...')
        
    print('Average class results (taking average over each run first)...')
    # calculating avg. metrics for each seed layer for each activity
    # empty list for metrics/saving seed wise
    act_acc_seed_lst = [] 
    act_prc_seed_lst = []
    act_rcll_seed_lst = []
    act_f1_seed_lst = []
    act_f1_unmod_lst = []
    act_comp_lst = []
    act_data_lst = []
    for s in range(args.nb_seeds):
        act_acc_seed_lst.append(np.mean(participant_scores[0, s, :, :], axis=-1))
        act_prc_seed_lst.append(np.mean(participant_scores[1, s, :, :], axis=-1))
        act_rcll_seed_lst.append(np.mean(participant_scores[2, s, :, :], axis=-1))
        act_f1_seed_lst.append(np.mean(participant_scores[3, s, :, :], axis=-1))
        act_f1_unmod_lst.append(np.mean(participant_scores_unmod[3, s, :, :], axis= -1))
        act_comp_lst.append(np.mean(savings_scores[0, s, :, :], axis=-1))
        act_data_lst.append(np.mean(savings_scores[1, s, :, :], axis=-1))
    
    print('Calculating average performance/saving over all activities and subjects...')
    avg_acc_all = np.mean(np.mean(act_acc_seed_lst, axis=-1))
    std_acc_all = np.std(np.mean(act_acc_seed_lst, axis=-1))
    avg_prc_all = np.mean(np.mean(act_prc_seed_lst, axis=-1))
    std_prc_all = np.std(np.mean(act_prc_seed_lst, axis=-1))
    avg_rcll_all = np.mean(np.mean(act_rcll_seed_lst, axis=-1))
    std_rcll_all = np.std(np.mean(act_rcll_seed_lst, axis=-1))
    avg_f1_all = np.mean(np.mean(act_f1_seed_lst, axis=-1))
    std_f1_all = np.std(np.mean(act_f1_seed_lst, axis=-1))
    avg_f1_unmod_all = np.mean(np.mean(act_f1_unmod_lst, axis=-1))
    std_f1_unmod_all = np.std(np.mean(act_f1_unmod_lst, axis=-1))
    
    avg_comp_saved_all = np.mean(np.mean(act_comp_lst, axis=-1))
    std_comp_saved_all = np.std(np.mean(act_comp_lst, axis=-1))
    avg_data_saved_all = np.mean(np.mean(act_data_lst, axis=-1))
    std_data_saved_all = np.std(np.mean(act_data_lst, axis=-1))

    print('Avg. Accuracy {:.4f} (±{:.4f}), '.format(avg_acc_all, std_acc_all),
          'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc_all, std_prc_all),
          'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll_all, std_rcll_all), '\n',
          'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1_all, std_f1_all),'\n',
          'Avg. F1-Score unmodified {:.4f} (±{:.4f})'.format(avg_f1_unmod_all, std_f1_unmod_all),'\n',
          'Avg. comp saved {:.4f} (±{:.4f}), '.format(avg_comp_saved_all, std_comp_saved_all),'\n',
          'Avg. data saved {:.4f} (±{:.4f})'.format(avg_data_saved_all, std_data_saved_all))
    
    # calculating mean and standard deviation class wise, averaging over several runs
    avg_acc_activity = np.mean(act_acc_seed_lst, axis=-2)
    std_acc_activity = np.std(act_acc_seed_lst, axis=-2)
    avg_prc_activity = np.mean(act_prc_seed_lst, axis=-2)
    std_prc_activity = np.std(act_prc_seed_lst, axis=-2)
    avg_rcll_activity = np.mean(act_rcll_seed_lst, axis=-2)
    std_rcll_activity = np.std(act_rcll_seed_lst, axis=-2)
    avg_f1_activity = np.mean(act_f1_seed_lst, axis=-2)
    std_f1_activity = np.std(act_f1_seed_lst, axis=-2) 
    avg_f1_unmod_activity = np.mean(act_f1_unmod_lst, axis=-2)
    std_f1_unmod_activity = np.std(act_f1_unmod_lst, axis=-2)
    avg_comp_saved_activity = np.mean(act_comp_lst, axis=-2)
    std_comp_saved_activity = np.std(act_comp_lst, axis=-2)
    avg_data_saved_activity = np.mean(act_data_lst, axis=-2)
    std_data_saved_activity = np.std(act_data_lst, axis=-2)
   
    # printing avg./stdev for metrics/saving for each activity
    for i, class_name in enumerate(class_names):
        print('averaging over 3 seeds...')
        print('Class {}: Avg. Accuracy {:.4f} (±{:.4f}), '.format(class_name, avg_acc_activity[i], std_acc_activity[i]),
              'Avg. Precision {:.4f} (±{:.4f}), '.format(avg_prc_activity[i], std_prc_activity[i]),
              'Avg. Recall {:.4f} (±{:.4f}), '.format(avg_rcll_activity[i], std_rcll_activity[i]),'\n',
              'Avg. F1-Score {:.4f} (±{:.4f})'.format(avg_f1_activity[i], std_f1_activity[i]), '\n',
              'Avg. F1-Score unmodified {:.4f} (±{:.4f})'.format(avg_f1_unmod_activity[i], std_f1_unmod_activity[i]),'\n',
              'Avg. comp saved {:.4f} (±{:.4f})'.format(avg_comp_saved_activity[i], std_comp_saved_activity[i]),'\n',
              'Avg. data saved {:.4f} (±{:.4f})'.format(avg_data_saved_activity[i], std_data_saved_activity[i]))
        
    # appending overall avg. scores
    avg_f1_activity = np.append(avg_f1_activity, avg_f1_all)
    avg_f1_unmod_activity = np.append(avg_f1_unmod_activity, avg_f1_unmod_all)
    avg_comp_saved_activity = np.append(avg_comp_saved_activity, avg_comp_saved_all)
    avg_data_saved_activity = np.append(avg_data_saved_activity, avg_data_saved_all)
    std_f1_activity = np.append(std_f1_activity, std_f1_all)
    std_f1_unmod_activity = np.append(std_f1_unmod_activity, std_f1_unmod_all)
    std_comp_saved_activity = np.append(std_comp_saved_activity, std_comp_saved_all)
    std_data_saved_activity = np.append(std_data_saved_activity, std_data_saved_all)
        
    
    # calculate evaluation criterion
    eval_criterion = np.abs(1-avg_f1_activity) + np.abs(1-avg_data_saved_activity/100) + np.abs(1-avg_comp_saved_activity/100)
    
    # save mean and std dev results for each experiment
    save_act_avg_std_results(algo_name, avg_f1_activity, std_f1_activity, avg_f1_unmod_activity, std_f1_unmod_activity, \
                            avg_comp_saved_activity, std_comp_saved_activity, avg_data_saved_activity, std_data_saved_activity,eval_criterion, filepath, args)
    
    # plot bar-graph for activity wise results
    mod_bar_plot_activity(algo_name, avg_f1_activity, avg_f1_unmod_activity, avg_comp_saved_activity, avg_data_saved_activity, filepath, args)


    # saving subj wise average results for bar graph
    avg_f1_sub_lst = []
    avg_f1_unmod_sub_lst = []
    std_f1_sub_lst = []
    std_f1_unmod_sub_lst = []
    avg_comp_saved_sub_lst = []
    std_comp_saved_sub_lst = []
    avg_data_saved_sub_lst = []
    std_data_saved_sub_lst = []
    for subject in range(nb_subjects):
        avg_f1_sub = np.mean(participant_scores[3, :, :, subject])
        avg_f1_unmod_sub = np.mean(participant_scores_unmod[3, :, :, subject])
        std_f1_sub = np.std(participant_scores[3, :, :, subject])
        std_f1_unmod_sub = np.std(participant_scores_unmod[3, :, :, subject])
        avg_comp_saved_sub = np.mean(savings_scores[0, :, :, subject])
        std_comp_saved_sub = np.std(savings_scores[0, :, :, subject])
        avg_data_saved_sub = np.mean(savings_scores[1, :, :, subject])
        std_data_saved_sub = np.std(savings_scores[1, :, :, subject])
        
        avg_f1_sub_lst.append(avg_f1_sub)
        avg_f1_unmod_sub_lst.append(avg_f1_unmod_sub)
        std_f1_sub_lst.append(std_f1_sub)
        std_f1_unmod_sub_lst.append(std_f1_unmod_sub)
        avg_comp_saved_sub_lst.append(avg_comp_saved_sub)
        std_comp_saved_sub_lst.append(std_comp_saved_sub)
        avg_data_saved_sub_lst.append(avg_data_saved_sub)
        std_data_saved_sub_lst.append(std_data_saved_sub)
       
    # save mean and std dev results subj. wise for each experiment 
    save_subj_avg_std_results(algo_name, avg_f1_sub_lst, std_f1_sub_lst, avg_f1_unmod_sub_lst, std_f1_unmod_sub_lst, \
                            avg_comp_saved_sub_lst, std_comp_saved_sub_lst, avg_data_saved_sub_lst, \
                                std_data_saved_sub_lst, filepath, args)
    #plotting bar graph sub wise
    mod_bar_plot_sbj(algo_name, avg_f1_sub_lst, avg_f1_unmod_sub_lst, avg_comp_saved_sub_lst, avg_data_saved_sub_lst, filepath, args)
        
    # create boxplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.suptitle('Modified Average Participant Results', size=16)
    activity = args.class_names
    capital_activity = [word.replace('_', ' ').title() for word in activity]
    
    act_acc = np.vstack(act_acc_seed_lst)
    act_prc = np.vstack(act_prc_seed_lst)
    act_rcll = np.vstack(act_rcll_seed_lst)
    act_f1 = np.vstack(act_f1_seed_lst)
        
    axs[0, 0].set_title('Accuracy')
    axs[0, 0].boxplot(act_acc, labels=capital_activity, showmeans=True)
    axs[0, 1].set_title('Precision')
    axs[0, 1].boxplot(act_prc, labels=capital_activity, showmeans=True)
    axs[1, 0].set_title('Recall')
    axs[1, 0].boxplot(act_rcll, labels=capital_activity, showmeans=True)
    axs[1, 1].set_title('F1-Score')
    axs[1, 1].boxplot(act_f1, labels=capital_activity, showmeans=True)
    
    # Plot confusion matrix
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)
    fig.subplots_adjust(hspace=0.5)
    mkdir_if_missing(filepath)
    if args.name:
        plt.savefig(os.path.join(filepath, filename + '_bx_mod_{}_{}_{}.png'.format(args.dataset, args.algo_name, args.name)))
        plot_confusion_matrix(input_cm, capital_activity, normalize=True, title='Mod. Confusion Matrix',
                              output_path=os.path.join(filepath, filename + '_cm_mod_{}_{}_{}.png'.format(args.dataset, args.algo_name, args.name)))
    else:
        plt.savefig(os.path.join(filepath, filename + '_bx_mod.png'))
        plot_confusion_matrix(input_cm, capital_activity, title='Mod. Confusion Matrix', normalize=True,
                              output_path=os.path.join(filepath, filename + '_cm_mod.png'))
    