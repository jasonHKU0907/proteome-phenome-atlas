
import glob
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd
warnings.filterwarnings('error')


def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c.replace("_","")) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

mywidth = 5
col1, col2, col3 = 'deepskyblue', 'yellowgreen', 'red'
y_pred_col1, y_pred_col2, y_pred_col3 = 'y_pred_pro', 'y_pred_cov', 'y_pred_pro_cov'

dpath = '/Volumes/JasonWork/Projects/ProDisAtlas/'
tgt_dir_lst = sort_nicely(glob.glob(dpath + 'Results/Prediction/CrossSectionalAnalysis/Predictions/*.csv'))
tgt_dict = pd.read_csv(dpath + 'Data/Target/Targets2Analysis.csv', encoding='latin-1')

tgt_dir = tgt_dir_lst[1]

for tgt_dir in tqdm(tgt_dir_lst):
    tgt = os.path.basename(tgt_dir)[:-4]
    tgt_name = tgt_dict.loc[tgt_dict.NAME == tgt].Long_Name.iloc[0]
    output_img = dpath + 'Results/Prediction/Plot/AUC/CrossSectional/' + tgt + '.pdf'
    eval_df = pd.read_csv(dpath + 'Results/Prediction/Evaluation/CrossSectional/' + tgt + '.csv')
    legend1 = 'Protein                         : ' + eval_df.AUC.iloc[0]
    legend2 = 'Demographic               : ' + eval_df.AUC.iloc[1]
    legend3 = 'Protein+Demographic : ' + eval_df.AUC.iloc[2]
    tgt_pred_df = pd.read_csv(tgt_dir)
    fig, ax = plt.subplots(figsize = (12, 12))
    plt.rcParams["font.family"] = "Arial"
    ################################### 1  #########################################
    fpr, tpr, _ = roc_curve(tgt_pred_df.target_y, tgt_pred_df[y_pred_col1])
    plt.plot(fpr, tpr, col1, linewidth=mywidth, label=legend1)
    ################################### 2  #########################################
    fpr, tpr, _ = roc_curve(tgt_pred_df.target_y, tgt_pred_df[y_pred_col2])
    plt.plot(fpr, tpr, col2, linewidth=mywidth, label=legend2)
    ################################### 3  #########################################
    fpr, tpr, _ = roc_curve(tgt_pred_df.target_y, tgt_pred_df[y_pred_col3])
    plt.plot(fpr, tpr, col3, linewidth=mywidth, label=legend3)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.0, 1.0])
    plt.ylim([-0.0, 1.0])
    plt.ylabel('True Positive Rate', fontsize=32, family='Arial')
    plt.xlabel('False Positive Rate', fontsize=32, family='Arial')
    if len(tgt_name)<45:
        plt.title(tgt_name, fontsize = 32, weight='bold', family='Arial')
    elif (len(tgt_name)>=45)&(len(tgt_name)<60):
        plt.title(tgt_name, fontsize = 24, weight='bold', family='Arial')
    else:
        plt.title(tgt_name, fontsize = 16, weight='bold', family='Arial')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=28, family='Arial')
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=28, family='Arial')
    plt.grid(which='minor', alpha=0.2, linestyle=':')
    plt.grid(which='major', alpha=0.5, linestyle='--')
    plt.legend(loc=4, fontsize=25, labelspacing=1.5, facecolor='gainsboro')
    # ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    # ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    plt.tight_layout()
    plt.savefig(output_img)
    plt.close('all')

