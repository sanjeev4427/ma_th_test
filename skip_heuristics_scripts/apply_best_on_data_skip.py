
import numpy as np
def apply_best_on_data_skip(config, current_activity, filename):
    if config['dataset'] == 'rwhar':
        filename = filename
    elif config['dataset'] == 'wetlab':
         filename = filename
    # best = np.loadtxt(filename,skiprows=1, usecols=(1,2,3),delimiter=',').T
    # best = [[1]*8,[1]*8, [2]*8]
    # print(best)
    best = np.loadtxt(filename, skiprows=1, usecols=(1,2,3),delimiter=',').T
    best_threshold = best[0]
    best_win_skip = best[1]
    best_tolerance = best[2]
    window_threshold = best_threshold[int(current_activity)]
    skip_window = best_win_skip[int(current_activity)]
    tolerance_value = best_tolerance[int(current_activity)] 
    return int(window_threshold), int(skip_window), int(tolerance_value)

# fundtion for experiment one set of hyperparameters for all activities
def apply_best_on_data_skip_exp_one_hy(config, current_activity, filename):
    if config['dataset'] == 'rwhar':
        filename = filename
    elif config['dataset'] == 'wetlab':
         filename = filename
    # best = np.loadtxt(filename,skiprows=1, usecols=(1,2,3),delimiter=',').T
    # best = [[1]*8,[1]*8, [2]*8]
    # print(best)
    best = np.loadtxt(filename, skiprows=1, usecols=(0,1,2),delimiter=',').T
    window_threshold = best[0]
    skip_window = best[1]
    tolerance_value = best[2]
    return int(window_threshold), int(skip_window), int(tolerance_value)