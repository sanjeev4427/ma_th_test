import os
import numpy as np
from sklearn.metrics import f1_score
from misc.osutils import mkdir_if_missing
from data_processing.data_analysis import plot_sensordata_and_labels
from skip_heuristics_scripts.data_skipping import data_skipping
from skip_heuristics_scripts.skip_heuristics import skip_heuristics
def graph_gt_val_mod_val(sbj, args, log_dir, *arg):
    config = vars(args)
    data = arg[1]
    # for _, sbj in enumerate(np.unique(data[:, 0])):

    train_data = data[data[:, 0] != sbj]
    val_data = data[data[:, 0] == sbj]
    # window_threshold = best[:,0]
    # skip_windows = best[:,1]
    # tolerance_value = best[:,2]
    
    if args.dataset == 'rwhar':
        val_output = np.loadtxt(rf'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\ml_training_data\rwhar\val_pred_sbj__{int(sbj)+1}.csv', dtype=float, delimiter=',')
    elif args.dataset == 'wetlab':
        val_output = np.loadtxt(rf' ', dtype=float, delimiter=',')
    
    # f_one_skip, f_one_target, f_one_val_vs_mod_val, \
    #     comp_saved_ratio, data_saved_ratio, computations_saved,\
    #         data_saved, comp_windows, data_windows, \
    #             all_mod_val_preds= skip_heuristics(activity, args, window_threshold, 
    #                                                                skip_windows, tolerance_value, val_output)
    gt = np.copy(val_output[:,1])
    val_pred = np.copy(val_output[:,0])
    mod_val_pred = np.copy(val_output[:,0])
    computations_saved = np.zeros(args.nb_classes)
    data_saved = np.zeros(args.nb_classes)
    filename = arg[0]
    all_mod_val_preds,_,_= data_skipping(-2, mod_val_pred, config, data_saved, computations_saved, True, filename)

    # plotting graph
    mkdir_if_missing(log_dir)
    if args.name:
        plot_sensordata_and_labels(val_data, gt, args.class_names, val_pred,
                                    all_mod_val_preds,
                                    figname=os.path.join(log_dir, 'sbj_' + str(int(sbj)) + '_' + args.name + '.png'))
    else:
        plot_sensordata_and_labels(val_data, gt, args.class_names, val_pred,
                                    all_mod_val_preds,
                                    figname=os.path.join(log_dir, 'sbj_' + str(int(sbj)) + '.png'))   
            



       