'''
Find f1 score activity wise after deep learning training 
'''
import numpy as np
from sklearn.metrics import f1_score

def get_f1_score_activity(dataset_name):
    if dataset_name == 'rwhar':
        label_name = ['climbing_down', 'climbing_up', 'jumping', 'lying',\
                       'running', 'sitting', 'standing', 'walking']
        activity_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    if dataset_name == 'wetlab':
        label_name = ['null_class', 'cutting', 'inverting', 'peeling', 'pestling',\
                       'pipetting', 'pouring', 'pour catalysator', 'stirring', 'transfer']
        all_eval_output = np.loadtxt(r'C:\Users\minio\Box\Thesis- Marius\aaimss_thesis\logs\193729-wetlab_trained\val_pred_all_sbj_wetlab.csv', dtype=float, delimiter = ',')
        activity_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    for activity in activity_labels:

        activity_name = label_name[activity]
        # set target f1
        f_one_target = f1_score(all_eval_output[:,1], all_eval_output[:,0], labels = np.array([activity]), average= None)
        print(f'f1 score for activity {activity_name}: {f_one_target}')
    avg_f1 = f1_score(all_eval_output[:,1], all_eval_output[:,0], average= 'macro')
    print(f'Average f1 score: {avg_f1}')

dataset_name = 'wetlab'
get_f1_score_activity(dataset_name)

