import matplotlib.pyplot as plt
import numpy as np
import os
from misc.osutils import mkdir_if_missing

def bar_plot_act_f1_comp(sbj, activity_name_lst, f1_gt_mod_val_lst, f1_gt_val_lst, comp_saved_lst, log_dir, config, algo_name, f1_avg = False, *args):
    # Data
    activities = activity_name_lst
    # activities = ['null_class','cutting','inverting','peeling','pestling','pipetting','pouring','pour catalysator','stirring','transfer']

    f1_score = f1_gt_mod_val_lst
    f1_gt_val = f1_gt_val_lst
    comp_saved = np.array(comp_saved_lst)/100
    name = '_'
    if f1_avg == True:
        # Avg. f1 score when best setting applied to full data
        name = f'f1_avg_{int(sbj)+1}'
        f1_avg = args[0]
        f1_gt_val_avg = args[1]
        thr_win_sum = np.sum(args[2])
        skip_win_sum = np.sum(args[3])
        avg_comp_saved = skip_win_sum/(skip_win_sum + thr_win_sum)
        activities.append('Average')
        f1_score.append(np.array([f1_avg]))
        f1_gt_val.append(f1_gt_val_avg)
        comp_saved = np.append(comp_saved, avg_comp_saved)
    # create a list of floats from list of arrays
    f1_score = [float(x) for sublist in f1_score for x in sublist]
    # Calculate percent difference
    percent_diff = [(f1 - f1_gt) / f1_gt * 100 for f1, f1_gt in zip(f1_score,f1_gt_val)]
    # percent_diff = [float(x) for sublist in percent_diff for x in sublist]

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the title and axis labels
    ax.set_title('Activity vs. F1 Score (mod_val w.r.t GT) and F1 Score (val_pred w.r.t GT) and Computaion saved \n (Target f1 = 1)') #val_pred w.r.t GT
    ax.set_xlabel('Activity')
    ax.set_ylabel('F1 Score / Computaion saved')

    # Set the x-axis tick labels
    ax.set_xticklabels(activities, rotation=45, ha='right',fontsize = 'x-small')

    # Set bar positions and width
    x_pos = np.arange(len(activities))
    width = 0.20

    
    # Plot bars for f1 score and computation saved
    bar1 = ax.bar(x_pos, f1_score, width, label='F1 Score (mod_val_pred w.r.t GT)')
    bar2 = ax.bar(x_pos + width, f1_gt_val, width, label='F1 Score (val_pred w.r.t GT)')
    bar3 = ax.bar(x_pos + 2*width, comp_saved, width, label='Computation saved')

    # Add text on bars
    for i,bar in enumerate([bar1, bar2]):
        hloc = []
        xloc = []
        for rect in bar:
            width, height = rect.get_width(), rect.get_height()
            x, y = rect.get_xy()
            ax.text(x + width / 2, y + height/2, f"{height * 100:.2f}%", ha='center', va='center', rotation=90,fontsize = 'small')
            xloc.append(x) 
            hloc.append(height)
        if i == 0:
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            for act,rect in enumerate(bar):
                width, height = rect.get_width(), rect.get_height()
                x, y = rect.get_xy()
                ax.text(x + width / 2, y + height + 0.08, f"{percent_diff[act]:.2f}%", ha='center', va='center', rotation=90, bbox=props, color='red')

        # ax.text(np.mean(xloc), max(hloc), f"{percent_diff[act]:.2f}%", ha='center', va='center', rotation=0)

    # Add text on bar comp saved
    for bar in [bar3]:
        for rect in bar:
            width, height = rect.get_width(), rect.get_height()
            x, y = rect.get_xy()
            ax.text(x + width / 2, y + height/2, f"{height*100:.2f}%", ha='center', va='center', rotation=90, fontsize = 'small')

    # add horizontal line 
    plt.axhline(y=1,linewidth=1, color='g')

    # Set x-axis tick labels to activities
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(activities, rotation=45)
    ax.yaxis.grid(color='gray', linestyle='--', linewidth=0.5)

    # Show legend
    ax.legend(loc = 'best',fontsize = 'small')
    mkdir_if_missing(log_dir)
    if config['name'] == True:
        filename=os.path.join(log_dir, f'bar_plot_f1_{config["dataset"]}_{algo_name}' + '_'+ name + args.name + '.png')
    else: 
        filename=os.path.join(log_dir, f'bar_plot_f1_{config["dataset"]}_{algo_name}' + '_' + name + '.png')
    plt.savefig(
        filename, format="png", bbox_inches="tight"
    )
    # # Show plot
    # plt.show()

