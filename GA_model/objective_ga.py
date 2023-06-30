import numpy as np
from SA_model.simulated_annealing import loss_function
from skip_heuristics_scripts.skip_heuristics import skip_heuristics

# objective function for GA
def objective_ga(activity, args, h_param, all_mod_eval_output):
      
        window_threshold = h_param[0]
        skip_windows = h_param[1]
        tolerance_value = int(h_param[2])
        # tolerance_value = tol_value
        
        # applying skip heuristics to get modified predictions 
        f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val, f_one_gt_mod_val_avg, f_one_gt_val_avg, \
        comp_saved_ratio, data_saved_ratio,_ = skip_heuristics(activity, args, window_threshold, skip_windows, tolerance_value, all_mod_eval_output)

        # for activity wise optimization
        # f_one, comp_saved_ratio, data_saved_ratio \
        #     = skip_heuristics(activity, args, window_threshold, skip_windows, tolerance_value, all_mod_eval_output)  

        # f_alpha = 10
        # c_alpha = 1
        # d_alpha = 2
        
        f_alpha = args.f_alpha
        c_alpha = args.c_alpha
        d_alpha = args.d_alpha
        
        #set f1 target
        f_one_target = 1
        # f1 of mod. predictions wrt to GT target
        f_one = f_one_gt_mod_val


        #defining computation saved calculation
        lam = args.sw_overlap /100
        comp_saved_ratio = skip_windows/(window_threshold + skip_windows)*100
        data_saved_ratio = 100*(skip_windows - lam*(skip_windows+1))/(skip_windows - lam*(skip_windows+1) + window_threshold + (window_threshold-1)*(1 - lam))
        # defining loss
        loss = loss_function(f_alpha, c_alpha, d_alpha, f_one, f_one_target, f_one_gt_val, f_one_gt_mod_val_avg, comp_saved_ratio, data_saved_ratio)

        return loss, f_one, f_one_target, float(data_saved_ratio), float(comp_saved_ratio), f_one_gt_mod_val,  f_one_gt_val, f_one_val_mod_val,\
                                                                                        f_one_gt_mod_val_avg, f_one_gt_val_avg