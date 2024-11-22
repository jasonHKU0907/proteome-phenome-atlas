
import glob
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import random
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve
from joblib import Parallel, delayed
warnings.filterwarnings('error')


def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c.replace("_","")) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

def threshold(array, cutoff):
    array1 = array.copy()
    array1[array1 < cutoff] = 0
    array1[array1 >= cutoff] = 1
    return array1

def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold'])

def get_eval(y_test, pred_prob, cutoff):
    pred_binary = threshold(pred_prob, cutoff)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_binary).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp)
    Youden = sens + spec - 1
    f1 = 2 * prec * sens / (prec + sens)
    auc = roc_auc_score(y_test, pred_prob)
    evaluations = np.round((auc, acc, sens, spec, prec, Youden, f1), 5)
    evaluations = pd.DataFrame(evaluations).T
    evaluations.columns = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'Youden-index', 'F1-score']
    return evaluations

def get_avg_output(mydf, gt_col, pred_col, cutoff, nb_iters):
    idx_lst = [ele for ele in range(len(mydf))]
    out_df = pd.DataFrame()
    for i in range(nb_iters):
        random.seed(i)
        bt_idx = [random.choice(idx_lst) for _ in range(len(idx_lst))]
        mydf_bt = mydf.copy()
        mydf_bt = mydf_bt.iloc[bt_idx, :]
        tmpout_df = get_eval(mydf_bt[gt_col], mydf_bt[pred_col], cutoff)
        out_df = pd.concat([out_df, tmpout_df], axis = 0)
    result_df = out_df.T
    result_df['Median'] = result_df.median(axis=1)
    result_df['LBD'] = result_df.quantile(0.025, axis=1)
    result_df['UBD'] = result_df.quantile(0.975, axis=1)
    output_lst = []
    for i in range(7):
        output_lst.append('{:.3f}'.format(result_df['Median'][i]) + ' [' +
                          '{:.3f}'.format(result_df['LBD'][i]) + ' - ' +
                          '{:.3f}'.format(result_df['UBD'][i]) + ']')
    result_df['output'] = output_lst
    myout = result_df.T
    return myout.iloc[-1,:]

dpath = '/home1/jiayou/Documents/Projects/ProDisAtlas/'
#dpath = '/Volumes/JasonWork/Projects/ProDisAtlas/'

tgt_dir_lst = sort_nicely(glob.glob(dpath + 'Results/Prediction/CrossSectionalAnalysis/Predictions/*.csv'))

for tgt_dir in tqdm(tgt_dir_lst[300:]):
    tgt = os.path.basename(tgt_dir)[:-4]
    tgt_pred_df = pd.read_csv(tgt_dir)
    ct_pro = Find_Optimal_Cutoff(tgt_pred_df.target_y, tgt_pred_df.y_pred_pro)[0]
    ct_cov = Find_Optimal_Cutoff(tgt_pred_df.target_y, tgt_pred_df.y_pred_cov)[0]
    ct_pro_cov = Find_Optimal_Cutoff(tgt_pred_df.target_y, tgt_pred_df.y_pred_pro_cov)[0]
    res_pro = get_avg_output(tgt_pred_df, 'target_y', 'y_pred_pro', ct_pro, nb_iters=1000)
    res_cov = get_avg_output(tgt_pred_df, 'target_y', 'y_pred_cov', ct_cov, nb_iters=1000)
    res_pro_cov = get_avg_output(tgt_pred_df, 'target_y', 'y_pred_pro_cov', ct_pro_cov, nb_iters=1000)
    res_df = pd.concat([res_pro, res_cov, res_pro_cov], axis=1)
    res_df = res_df.T
    res_df.index = ['Protein', 'Demographic', 'Protein+Demographic']
    res_df.to_csv(dpath + 'Results/Prediction/Evaluation/CrossSectional/' + tgt + '.csv', index=True)

#!/bin/bash
#SBATCH -p amd         # Queue
#SBATCH -N 1          # Node count required for the job
#SBATCH -n 2           # Number of tasks to be launched
#SBATCH --mem=48G
#SBATCH -J P_300_all           # Job name
#SBATCH -o S_P_300_all.out       # Standard output
#SBATCH -w amdnode2

cd /home1/jiayou/Documents/Projects/ProDisAtlas/ServerCode/Evaluation/
python Eval_Prevalent_300_all.py
