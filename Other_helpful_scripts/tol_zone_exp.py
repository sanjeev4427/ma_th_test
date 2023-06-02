import math
import numpy as np
from sklearn.metrics import f1_score

def tolerance_zone(
    mod_val_preds_current,
    config,
    window_count,
    last_window,
    tolerance=0,
    tolerance_window_count=1,
    last_tolerance_window=-1
    ):
    """
    Method called when activity changes.

    Args:
        mod_val_preds_current (int): current value of validation prediction
        config (dict): General setting dictionary
        tolerance (int): tolerance threshold used to reset values. Defaults to 0.
        tolerance_window_count (int): activity window count in tolerance zone. Defaults to 0.
        last_tolerance_window (int): stores validation prediction in last loop. Defaults to -1.
        window_count (int): count activity windows for any particular predicted activity. Defaults to 0.

    Returns:
        int: activity windows count for any particular predicted activity.
        int: tolerance threshold used to reset values.
        int: activity window count in tolerance zone.
        int: validation prediction in last loop.

    """
    # increase tolerance counter
    tolerance += 1

    # if current window same as one in previous tolerance loop
    if mod_val_preds_current == last_tolerance_window:
        # increase tolerance window counter
        tolerance_window_count += 1

    else:
        # set last tolerance window to current
        last_tolerance_window = mod_val_preds_current
        # set tolerance window counter to 1
        # tolerance_window_count = 0        
        tolerance_window_count = 1

    # if tolerance window counter is greater than 2, SAVING_TOLERANCE = 2
    # then quit counting of current activity and switch to tolerance activity
    if tolerance_window_count >= (config["saving_tolerance"]):
        # overwrite current count and last window to tolerance values
        window_count = tolerance_window_count
        last_window = last_tolerance_window
        # reset tolerance, tolerance window counter and last tolerance window
        tolerance = 0
        # tolerance_window_count = 0
        tolerance_window_count = 1
        last_tolerance_window = -1
    # else if tolerance greater than tolerance threshold,
    # then quit counting of current activity and reset counting
    # elif tolerance > config["saving_tolerance"]:
    #     # reset window counter and last window
    #     window_count = 1
    #     last_window = -1
    #     # reset tolerance values
    #     tolerance = 0
    #     # tolerance_window_count = 0
    #     # tolerance_window_count = 1
    #     last_tolerance_window = -1
    # if tolerance safety measures to do not trigger, treat current window as it was of the
    else:
        window_count += 1
        last_window = last_window

    return window_count, tolerance, tolerance_window_count, last_tolerance_window, last_window

def data_skipping(mod_val_preds):
    
    config = {'saving_tolerance' : 7}
    j = 0
    tolerance = 0
    window_count = 1
    # tolerance_window_count = 0
    tolerance_window_count = 1
    last_window = -1
    last_tolerance_window = -1

    tolerance_window_count_lst = list()
    win_count_lst = list()
    last_window_lst = list()
    last_tolerance_window_lst = list()
    tol_lst = list()

    while j < (len(mod_val_preds)):        
        # if last window is same as current; increase counter
        if mod_val_preds[j] == last_window:
            window_count += 1
            mod_val_preds[j-tolerance:j] = mod_val_preds[j]
            tolerance = 0
            tolerance_window_count = 1
            last_tolerance_window = -1

        elif last_window == -1:
            last_window = mod_val_preds[j]
            
        else:
            (
                window_count,
                tolerance,
                tolerance_window_count,
                last_tolerance_window, 
                last_window
            ) = tolerance_zone(
                mod_val_preds[j],
                config,
                window_count,
                last_window,
                tolerance,
                tolerance_window_count,
                last_tolerance_window,
                
            )
        tolerance_window_count_lst .append(tolerance_window_count)
        win_count_lst .append(window_count)
        last_window_lst .append(last_window)
        last_tolerance_window_lst .append(last_tolerance_window)
        tol_lst.append(tolerance)
        j += 1
    # print('val:', mod_val_preds,'\n','WC:', win_count_lst,'\n','TWC:', tolerance_window_count_lst,
                #  '\n','last_window:', last_window_lst,'\n','last_tol_win:', last_tolerance_window_lst,'\n' 
                #  'tol:', tol_lst)
    print(tol_lst)
    return mod_val_preds

# mod_val_preds = [5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 5, 5, 3, 3, 3, 3]
gt = np.array([ 5, 5, 5, 5, 3, 3, 3, 3, 3, 5, 5, 8, 8, 8, 3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,6,5,7,8,9,3,4,5,3,3,3,3,3,3,3,3,3,3])
mod_val_preds = np.copy(gt)
mod_val_preds = data_skipping(mod_val_preds)
print(gt, '\n', mod_val_preds)
print(f1_score(gt,mod_val_preds, average='macro'))