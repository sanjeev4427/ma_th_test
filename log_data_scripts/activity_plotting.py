import matplotlib.pyplot as plt
import numpy as np
import os
from misc.osutils import mkdir_if_missing

def activity_plot_loss_anneal(loss_array, n_iter, config, ann_rate, activity_name, args, log_dir):
    plt.figure()
    plt.plot(range(n_iter), loss_array)
    plt.xlabel("Iteration")
    plt.ylabel("Loss ")
    plt.title(f"Loss vs iteration \n activity: {activity_name}")
    mkdir_if_missing(log_dir)
    filename=os.path.join(log_dir, f'loss_iter_{config["dataset"]}_{activity_name}_SA' + '_' + args.name + '.png')
    plt.savefig(
        filename, format="png", bbox_inches="tight"
    )
    #plt.show()
    return None

def activity_plot_f1_iter(fscore_array, n_iter, config, ann_rate, activity_name, args, log_dir):
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
    plt.title(f"F1 score vs iteration \n activity: {activity_name}")
    mkdir_if_missing(log_dir)
    filename=os.path.join(log_dir, f'f1score_iter_{config["dataset"]}_{activity_name}_SA' + '_' + args.name + '.png')
    plt.savefig(
        filename, format="png", bbox_inches="tight"
    )
    #plt.show()
    return None

def activity_plot_data_saved_iter(data_saved_array, n_iter, config, ann_rate, activity_name, args, log_dir):
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
    plt.title(f"Average data saved vs iteration \n activity: {activity_name}")
    mkdir_if_missing(log_dir)
    filename=os.path.join(log_dir, f'data_saved_iter_{config["dataset"]}_{activity_name}_SA' + '_' + args.name + '.png')
    plt.savefig(
        filename, format="png", bbox_inches="tight"
    )
    #plt.show()
    return None

def activity_plot_comp_saved_iter(comp_saved_array, n_iter, config, ann_rate, activity_name, args, log_dir):
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
    plt.title(f"Average computation saved vs iteration \n activity: {activity_name}")
    mkdir_if_missing(log_dir)
    filename=os.path.join(log_dir, f'comp_saved_iter_{config["dataset"]}_{activity_name}_SA' + '_' + args.name + '.png')
    plt.savefig(
        filename, format="png", bbox_inches="tight"
    )
    #plt.show()
    return None

def activity_plot_threshold_iter(threshold_array, n_iter, config, ann_rate, activity_name, args, log_dir):
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
    plt.title(f"Threshold value vs iteration \n activity: {activity_name}")
    mkdir_if_missing(log_dir)
    filename=os.path.join(log_dir, f'threshold_iter_{config["dataset"]}_{activity_name}_SA' + '_' + args.name + '.png')
    plt.savefig(
        filename, format="png", bbox_inches="tight"
    )
    #plt.show()
    return None

def activity_plot_skip_windows_iter(skip_windows_array, n_iter, config, ann_rate, activity_name, args, log_dir):
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
    plt.title(f"Skip_windows value vs iteration \n activity: {activity_name}")
    mkdir_if_missing(log_dir)
    filename=os.path.join(log_dir, f'tolerance_iter_{config["dataset"]}_{activity_name}_SA' + '_' + args.name + '.png')
    plt.savefig(
        filename, format="png", bbox_inches="tight"
    )
    #plt.show()
    return None

def activity_plot_tol_value_iter(tol_value_array, n_iter, config, ann_rate, activity_name, args, log_dir):

    plt.figure()
    plt.plot(range(n_iter), tol_value_array)
    plt.xlabel("Iteration")
    plt.ylabel("Tolerance value")
    plt.title(f"Tolerance value vs iteration \n activity: {activity_name}")
    mkdir_if_missing(log_dir)
    filename=os.path.join(log_dir, f'tolerance_iter_{config["dataset"]}_{activity_name}_SA' + '_' + args.name + '.png')
    plt.savefig(
        filename, format="png", bbox_inches="tight"
    )
    plt.savefig(
        rf'C:\Users\minio\Box\Thesis- Marius\figures\skip_windows_iter_{ann_rate}_{config["dataset"]}_{activity_name}.png', format="png", bbox_inches="tight"
    )
    #plt.show()
    return None


def activity_plot_temperature_iter(temperature_array, n_iter, config, ann_rate, activity_name, args, log_dir):

    plt.figure()
    plt.plot(range(n_iter), temperature_array)
    plt.xlabel("Iteration")
    plt.ylabel("Temperature value")
    plt.title(f"Temperature value vs iteration \n activity: {activity_name}")
    mkdir_if_missing(log_dir)
    filename=os.path.join(log_dir, f'temperature_iter_{config["dataset"]}_{activity_name}_SA' + '_' + args.name + '.png')
    plt.savefig(
        filename, format="png", bbox_inches="tight"
    )
    return None

def activity_plot_finding_initial_temp(temperature_array, config, n_iter_name,log_dir):
    
    plt.figure()
    plt.plot(range(len(temperature_array)), temperature_array)
    plt.xlabel("Warming up iterations")
    plt.ylabel("Temperature value")
    plt.title(f"Temperature value vs Warming up iterations \n final temp = {temperature_array[-1]}")
    mkdir_if_missing(log_dir)
    figname=os.path.join(log_dir, f'finding_initial_temp_iter_at_each_temp_{n_iter_name}_{config["dataset"]}_SA' + '.png')
    plt.savefig(
        figname, format="png", bbox_inches="tight"
    )
    filename=os.path.join(log_dir, f'finding_initial_temp_iter_at_each_temp_{n_iter_name}_{config["dataset"]}_SA' + '.csv')
    np.savetxt(filename,
            temperature_array, 
            fmt = '%s',
           delimiter =",")
    return None

def activity_plot_acceptance_ratio_finding_initial_temp(acceptance_ratio_array, config, log_dir, n_iter_name):
    
    plt.figure()
    plt.plot(range(len(acceptance_ratio_array)), acceptance_ratio_array)
    plt.xlabel("Warming up iterations")
    plt.ylabel("Acceptance ratio")
    plt.title(f"Acceptance ratio vs Warming up iterations \n iteration at each Temp: {n_iter_name}")
    mkdir_if_missing(log_dir)
    figname=os.path.join(log_dir, f'acc_ratio_finding_initial_temp_iter_at_each_temp_{n_iter_name}_{config["dataset"]}_SA' + '.png')
    plt.savefig(
        figname, format="png", bbox_inches="tight"
    )
    filename=os.path.join(log_dir, f'acc_ratio_finding_initial_temp_iter_at_each_temp_{n_iter_name}_{config["dataset"]}_SA' + '.csv')
    np.savetxt(filename,
           acceptance_ratio_array, 
            fmt = '%s',
           delimiter =",")
    return None

def activtity_bar_graph_f1_comp_saved():
    import matplotlib.pyplot as plt
    import numpy as np
    # bar graph for wetlab f1, comp saved vs activities
    # Data
    activities = ['null_class', 'cutting', 'inverting', 'peeling', 'pestling', 'pipetting', 'pouring', 'pour catalysator', 'stirring', 'transfer']
    f1_scores = np.array([0.96961446, 0.96466617, 0.96769492, 0.95428398, 0.95292292, 0.95299854, 0.71380014, 0.75347401, 0.95294461, 0.72483221])*100
    computation_saved = [87.85671386, 79.56397718, 62.82352941, 74.00115808, 89.59980237, 63.45304398, 88.794926, 68.37324525, 37.39009072, 97.26315789]

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set x and y axis labels and title
    ax.set_xlabel('Activity')
    ax.set_ylabel('F1 Score / Computation Saved')
    ax.set_title('Activity vs. F1 Score / Computation Saved')

    # Set bar positions and width
    x_pos = np.arange(len(activities))
    width = 0.35

    # Plot bars for f1 score and computation saved
    ax.bar(x_pos, f1_scores, width, label='F1 Score')
    ax.bar(x_pos + width, computation_saved, width, label='Computation Saved')

    # Set x-axis tick labels to activities
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(activities, rotation=45)

    # Show legend
    ax.legend()

    # Show plot
    plt.show()

    import matplotlib.pyplot as plt
    import numpy as np

    # bar graph for rwhar f1, comp saved vs activities
    # Data
    activities = ['climbing_down', 'climbing_up', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking']
    f1_scores = np.array([0.98963731,
    0.99517574,
    0.97701149,
    0.98740446,
    0.99554475,
    0.99578415,
    0.99034335,
    0.9881861])*100
    computation_saved = [98.71794872, 98.58253315, 99.12359551, 98.87305975, 98.79032258, 96.47355164, 98.19341126, 97.12750568]

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set x and y axis labels and title
    ax.set_xlabel('Activity')
    ax.set_ylabel('F1 Score / Computation Saved')
    ax.set_title('Activity vs. F1 Score / Computation Saved')

    # Set bar positions and width
    x_pos = np.arange(len(activities))
    width = 0.35

    # Plot bars for f1 score and computation saved
    ax.bar(x_pos, f1_scores, width, label='F1 Score')
    ax.bar(x_pos + width, computation_saved, width, label='Computation Saved')

    # Set x-axis tick labels to activities
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(activities, rotation=45)

    # Show legend
    ax.legend()

    # Show plot
    plt.show()
    import matplotlib.pyplot as plt
    import numpy as np

    # rwhar time on-off
    # Data
    activities = ['climbing_down', 'climbing_up', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking']
    thresholds = (np.array([1, 2, 9, 1, 1, 3, 1, 3])-1)*0.4 + 1
    skip_windows = (np.array([77,98,72,93,100,94,84,100])-1)*0.4 + 1

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set bar positions and width
    x_pos = np.arange(len(activities))
    width = 0.35

    # Plot bars for f1 score and computation saved
    ax.bar(x_pos, thresholds, width, label='Device On')
    ax.bar(x_pos + width, skip_windows, width, label='Device off')

    # Set x-axis tick labels to activities
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(activities, rotation=45)

    # Show legend
    ax.legend()
    # Add labels, title, and legend
    ax.set_xlabel('Activity')
    ax.set_ylabel('Device On and Off time (seconds)')
    ax.set_title('Device On and Off time for Different Activities')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(activities)
    ax.legend()

    # Show the plot
    plt.show()

def bar_graph_f1_score_f1_target():
    import matplotlib.pyplot as plt
    import numpy as np

    # Data
    activities = ['null_class', 'cutting', 'inverting', 'peeling', 'pestling', 'pipetting', 'pouring', 'pour catalysator', 'stirring', 'transfer']
    f1_score = [0.83830423, 0.4946472, 0.28439803, 0.17297699, 0.54387338, 0.16236611, 0.0234657, 0.07528231, 0.24683369, 0.12966601]
    target_f1 = [0.87086876, 0.53578832, 0.37027708, 0.14817518, 0.55915749, 0.18443363, 0.02313167, 0.10539683, 0.32065371, 0.01314554]

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the title and axis labels
    ax.set_title('Activity vs. F1 Score after skip and F1 score without skip \n (Target f1 = 1)')
    ax.set_xlabel('Activity')
    ax.set_ylabel('F1 Score (with skip) / F1 Score (with skip)')

    # Set the x-axis tick labels
    ax.set_xticklabels(activities, rotation=45, ha='right')

    # Set bar positions and width
    x_pos = np.arange(len(activities))
    width = 0.35

    # Plot bars for f1 score and computation saved
    bar1 = ax.bar(x_pos, f1_score, width, label='F1 Score (with skip)')
    bar2 = ax.bar(x_pos + width, target_f1, width, label='F1 Score (without skip)')

    # Add text on bars
    for bar in [bar1, bar2]:
        for rect in bar:
            width, height = rect.get_width(), rect.get_height()
            x, y = rect.get_xy()
            ax.text(x + width / 2, y + height + 0.08, f"{height * 100:.2f}%", ha='center', va='center', rotation=90)

    # add horizontal line 
    plt.axhline(y=1,linewidth=1, color='g')

    # Set x-axis tick labels to activities
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(activities, rotation=45)
    ax.yaxis.grid(color='gray', linestyle='--', linewidth=0.5)

    # Show legend
    ax.legend(loc = 'upper right')

    # Show plot
    plt.show()  