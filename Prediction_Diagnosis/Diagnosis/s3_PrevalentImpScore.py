
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
import seaborn as sns
warnings.filterwarnings('error')

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c.replace("_","")) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

dpath = '/Volumes/JasonWork/Projects/ProDisAtlas/'
tgt_dir_lst = sort_nicely(glob.glob(dpath + 'Results/Prediction/CrossSectionalAnalysis/ProImportance/*.csv'))
tgt_dict = pd.read_csv(dpath + 'Data/Target/Targets2Analysis.csv', encoding='latin-1')

for tgt_dir in tqdm(tgt_dir_lst):
    tgt = os.path.basename(tgt_dir)[:-4]
    tgt_name = tgt_dict.loc[tgt_dict.NAME == tgt].Long_Name.iloc[0]
    output_img = dpath + 'Results/Prediction/Plot/Importance/CrossSectional/' + tgt + '.pdf'
    tgt_pred_df = pd.read_csv(tgt_dir)
    tgt_pred_df = tgt_pred_df.iloc[:30,:]
    fig, ax = plt.subplots(figsize=(12, 5))
    plt.rcParams["font.family"] = "Arial"
    palette = sns.color_palette("Blues", n_colors=len(tgt_pred_df))
    palette.reverse()
    sns.barplot(ax=ax, x="Pro_code", y="TotalGain_cv", palette=palette, data=tgt_pred_df.sort_values(by="TotalGain_cv", ascending=False))
    y_imp_up_lim = round(tgt_pred_df['TotalGain_cv'].max() + 0.005, 2)
    ax.set_ylim([0, y_imp_up_lim])
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xticklabels(tgt_pred_df['Pro_code'], rotation=45, fontsize=14, horizontalalignment='right')
    ax.set_ylabel('Protein Importance', weight='bold', fontsize=18)
    # ax.set_title(my_title, y=1.0, pad=-25, weight='bold', fontsize=24)
    ax.set_xlabel('')
    ax.grid(which='minor', alpha=0.2, linestyle=':')
    ax.grid(which='major', alpha=0.5, linestyle='--')
    ax.set_axisbelow(True)
    if len(tgt_name)<45:
        plt.title(tgt_name, fontsize = 32, weight='bold', family='Arial')
    elif (len(tgt_name)>=45)&(len(tgt_name)<60):
        plt.title(tgt_name, fontsize = 24, weight='bold', family='Arial')
    else:
        plt.title(tgt_name, fontsize = 16, weight='bold', family='Arial')
    plt.tight_layout()
    plt.savefig(output_img)
    plt.close('all')


