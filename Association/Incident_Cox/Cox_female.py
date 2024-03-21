

import glob
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from tqdm import tqdm
import os
import re
from statsmodels.stats.multitest import fdrcorrection
from mne.stats import bonferroni_correction
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('error')

def results_summary(tgt_out_df):
    hr_out_lst, p_out_lst = [], []
    for i in range(len(tgt_out_df)):
        hr = f'{tgt_out_df.hr.iloc[i]:.2f}'
        lbd = f'{tgt_out_df.hr_lbd.iloc[i]:.2f}'
        ubd = f'{tgt_out_df.hr_ubd.iloc[i]:.2f}'
        hr_out_lst.append(hr + ' [' + lbd + '-' + ubd + ']')
        if tgt_out_df.pval_bfi.iloc[i] < 0.001:
            p_out_lst.append('***')
        elif tgt_out_df.pval_bfi.iloc[i] < 0.01:
            p_out_lst.append('**')
        elif tgt_out_df.pval_bfi.iloc[i] < 0.05:
            p_out_lst.append('*')
        else:
            p_out_lst.append('')
    return (hr_out_lst, p_out_lst)


def process(pro_f):
    tmp_pro_df = pro_cov_df[['eid', pro_f, pro_f + '_SampAge']]
    tmp_df = pd.merge(tmp_tgt_df, tmp_pro_df, how='left', on=['eid'])
    tmp_df.rename(columns={pro_f: 'x_pro', pro_f + '_SampAge': 'x_pro_sa'}, inplace=True)
    rm_eid_idx = tmp_df.index[tmp_df.x_pro.isnull() == True]
    tmp_df.drop(rm_eid_idx, axis=0, inplace=True)
    tmp_df.reset_index(inplace=True, drop=True)
    tmp_df['x_pro_sa'].fillna(tmp_df['x_pro_sa'].median(), inplace=True)
    nb_all, nb_case = len(tmp_df), tmp_df.target_y.sum()
    prop_case = np.round(nb_case / nb_all * 100, 3)
    if nb_case >= 10:
        i, tmpout = 1, []
        while ((len(tmpout) == 0) | (i > 1e7)):
            cph = CoxPHFitter(penalizer=1e-7 * i)
            i = 10 * i
            try:
                cph.fit(tmp_df, duration_col='BL2Target_yrs', event_col='target_y', formula=my_formula)
                hr = np.round(cph.hazard_ratios_.x_pro, 5)
                lbd = np.round(np.exp(cph.confidence_intervals_).loc['x_pro'][0], 5)
                ubd = np.round(np.exp(cph.confidence_intervals_).loc['x_pro'][1], 5)
                pval = cph.summary.p.x_pro
                tmpout = [pro_f, nb_all, nb_case, prop_case, hr, lbd, ubd, pval]
            except:
                pass
        if tmpout == []:
            tmpout = [pro_f, nb_all, nb_case, prop_case, np.nan, np.nan, np.nan, np.nan]
        else:
            pass
    else:
        tmpout = [pro_f, nb_all, nb_case, prop_case, np.nan, np.nan, np.nan, np.nan]
    return tmpout


def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c.replace("_","")) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l


nb_cpus = 58

dpath = '/home1/jiayou/Documents/Projects/ProDisAtlas/'
#dpath = '/Volumes/JasonWork/Projects/ProDisAtlas/'
badoutfile = dpath + 'Results/Association/IncidentAnalysis/bad_targets_incident_female.csv'


target_file_lst = sort_nicely(glob.glob(dpath + 'Data/Target/Targets2Analysis/*.csv'))
target_info_df = pd.read_csv(dpath + 'Data/Target/TargetVsProtein.csv', usecols=['NAME', 'SEX'])

pro_cov_df = pd.read_csv(dpath + 'Data/ProteinData/ProteinData_n_Cov.csv')
pro_f_lst = pro_cov_df.columns[13:2933].tolist()
pro_cov_df['Race'].replace([1,2,3,4], [1, 0, 0, 0], inplace = True)

cov_f_lst = ['eid', 'Age', 'Sex', 'Race', 'TDI', 'BMI', 'smk', 'fastingtime', 'season']
my_formula_full = "Age + C(Sex) + C(Race) + TDI + BMI + C(smk) + fastingtime + C(season) + x_pro_sa + x_pro"
my_formula_non_sex = "Age + C(Race) + TDI + BMI + C(smk) + fastingtime + C(season) + x_pro_sa + x_pro"
my_formula = my_formula_non_sex
cov_df = pro_cov_df[cov_f_lst]

bad_tgt = []


for tgt_file in tqdm(target_file_lst):
    try:
        tgt_name = os.path.basename(tgt_file).split('.')[0]
        tmp_tgt_df = pd.read_csv(tgt_file, usecols=['eid', 'target_y', 'BL2Target_yrs'])
        rm_bl_idx = tmp_tgt_df.index[tmp_tgt_df.BL2Target_yrs <= 0]
        tmp_tgt_df.drop(rm_bl_idx, axis=0, inplace=True)
        tmp_tgt_df.reset_index(inplace=True, drop=True)
        tmp_tgt_df = pd.merge(tmp_tgt_df, cov_df, how='left', on=['eid'])
        if tmp_tgt_df.target_y.sum() > 50:
            tmp_tgt_df = tmp_tgt_df.loc[tmp_tgt_df.Sex == 0]
            tmp_tgt_df.reset_index(inplace=True, drop=True)
            if tmp_tgt_df.target_y.sum() >= 10:
                tgt_out_df = Parallel(n_jobs=nb_cpus)(delayed(process)(pro_f) for pro_f in pro_f_lst)
                tgt_out_df = pd.DataFrame(tgt_out_df)
                tgt_out_df.columns = ['Pro_code', 'nb_individuals', 'nb_case', 'prop_case(%)', 'hr', 'hr_lbd', 'hr_ubd',
                                      'pval_raw']
                _, p_f_bfi = bonferroni_correction(tgt_out_df.pval_raw.fillna(1), alpha=0.05)
                tgt_out_df['pval_bfi'] = p_f_bfi
                tgt_out_df.loc[tgt_out_df['pval_bfi'] >= 1, 'pval_bfi'] = 1
                tgt_out_df['hr_output'], tgt_out_df['pval_significant'] = results_summary(tgt_out_df)
                tgt_out_df = tgt_out_df[['Pro_code', 'nb_individuals', 'nb_case', 'prop_case(%)', 'hr', 'hr_lbd',
                                         'hr_ubd', 'pval_raw', 'pval_bfi', 'hr_output', 'pval_significant']]
                tgt_out_df.to_csv(dpath + 'Results/Association/IncidentAnalysis/Female/' + tgt_name + '.csv',
                                  index=False)
            else:
                bad_tgt.append([tgt_name, tmp_tgt_df.target_y.sum()])
        else:
            bad_tgt.append([tgt_name, tmp_tgt_df.target_y.sum()])
    except:
        pass

bad_df = pd.DataFrame(bad_tgt)
bad_df.columns = ['disease_code', 'nb_cases']
bad_df.to_csv(badoutfile, index=False)


scp -P 59813 /Users/jason/PycharmProjects/ProDisAtlas/AssociationStudy/Logistic/* jiayou@10.190.248.211:/home1/jiayou/Documents/Projects/ProDisAtlas/ServerCode/Association/CrossSectionalAnalysis/