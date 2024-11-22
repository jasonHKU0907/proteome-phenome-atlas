

import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import warnings
from lightgbm import LGBMClassifier
from joblib import Parallel, delayed
warnings.filterwarnings('error')

nb_cpus = 10
nb_params = 100
my_seed = 2024

def select_params_combo(my_dict, nb_items, my_seed):
    combo_list = [dict(zip(my_dict.keys(), v)) for v in product(*my_dict.values())]
    random.seed(my_seed)
    return random.sample(combo_list, nb_items)

def normal_imp(mydict):
    mysum = sum(mydict.values())
    mykeys = mydict.keys()
    for key in mykeys:
        mydict[key] = mydict[key]/mysum
    return mydict

def get_cov_f_lst(tgt2pred_df, tgt):
    sex_id = tgt2pred_df.loc[tgt2pred_df.Disease_code == tgt].SEX.iloc[0]
    if (sex_id == 1) | (sex_id == 2):
        cov_f_lst = ['AGE', 'RACE', 'TDI', 'BMI', 'SMK', 'ALC', 'SBP']
    else:
        cov_f_lst = ['AGE', 'Sex', 'RACE', 'TDI', 'BMI', 'SMK', 'ALC', 'SBP']
    return cov_f_lst

def get_pro_f_lst(mydf, train_idx, f_lst, my_params):
    X_train, y_train = mydf.iloc[train_idx][f_lst], mydf.iloc[train_idx].target_y
    my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=False, verbosity=1, seed=2023)
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)
    totalgain_imp = my_lgb.booster_.feature_importance(importance_type='gain')
    totalgain_imp = dict(zip(my_lgb.booster_.feature_name(), totalgain_imp.tolist()))
    tg_imp_df = pd.DataFrame({'Pro_code': list(totalgain_imp.keys()), 'TotalGain': list(totalgain_imp.values())})
    tg_imp_df.sort_values(by = 'TotalGain', inplace = True, ascending = False)
    return tg_imp_df.Pro_code.tolist()[:30]

def get_best_params(mydf, pro_f_lst, fold_id_lst, my_params_lst):
    for my_params in my_params_lst:
        auc_cv_lst = []
        my_params0 = my_params.copy()
        for fold_id in fold_id_lst:
            train_idx = mydf['Split'].index[mydf['Split'] != fold_id]
            test_idx = mydf['Split'].index[mydf['Split'] == fold_id]
            X_train, y_train = mydf.iloc[train_idx][pro_f_lst], mydf.iloc[train_idx].target_y
            X_test, y_test = mydf.iloc[test_idx][pro_f_lst], mydf.iloc[test_idx].target_y
            my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, seed=my_seed)
            my_lgb.set_params(**my_params0)
            my_lgb.fit(X_train, y_train)
            y_pred_prob = my_lgb.predict_proba(X_test)[:, 1]
            auc_cv_lst.append(roc_auc_score(y_test, y_pred_prob))
        my_params0['AUC_cv_MEAN'] = np.round(np.mean(auc_cv_lst), 5)
    my_params0.sort_values(by = 'AUC_cv_MEAN', ascending = False, inplace = True)
    best_param = get_dict(my_params0.iloc[0,:6])
    return best_param
    
def model_training(mydf, train_idx, test_idx, f_lst, my_params):
    X_train, X_test = mydf.iloc[train_idx][f_lst], mydf.iloc[test_idx][f_lst]
    y_train = mydf.iloc[train_idx].target_y
    my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=False, verbosity=1, seed=2023)
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)
    y_pred = my_lgb.predict_proba(X_test)[:, 1].tolist()
    return y_pred, my_lgb

def get_iter_predictions(mydf, full_pro_f_lst, cov_f_lst, fold_id, my_params0, my_params):
    train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    pro_f_lst = get_pro_f_lst(mydf, train_idx, full_pro_f_lst, my_params0)
    my_params_pro = get_best_params(mydf, pro_f_lst, inner_fold_id_lst, candidate_params_lst)
    my_params_cov = get_best_params(mydf, cov_f_lst, inner_fold_id_lst, candidate_params_lst)
    my_params_pro_cov = get_best_params(mydf, cov_f_lst + pro_f_lst, inner_fold_id_lst, candidate_params_lst)
    y_pred_pro, lgb_pro = model_training(mydf, train_idx, test_idx, pro_f_lst, my_params)
    y_pred_cov, lgb_cov = model_training(mydf, train_idx, test_idx, cov_f_lst, my_params)
    y_pred_pro_cov, lgb_pro_cov = model_training(mydf, train_idx, test_idx, cov_f_lst + pro_f_lst, my_params)
    y_test_lst = mydf.target_y.iloc[test_idx].tolist()
    eid_lst = mydf.eid.iloc[test_idx].tolist()
    y_pred_pro_lst = y_pred_pro
    y_pred_cov_lst = y_pred_cov
    y_pred_pro_cov_lst = y_pred_pro_cov
    totalgain_imp = lgb_pro.booster_.feature_importance(importance_type='gain')
    totalgain_imp = dict(zip(lgb_pro.booster_.feature_name(), totalgain_imp.tolist()))
    totalcover_imp = lgb_pro.booster_.feature_importance(importance_type='split')
    totalcover_imp = dict(zip(lgb_pro.booster_.feature_name(), totalcover_imp.tolist()))
    tg_imp_cv = Counter(normal_imp(totalgain_imp))
    tc_imp_cv = Counter(normal_imp(totalcover_imp))
    return (tg_imp_cv, tc_imp_cv, eid_lst, y_test_lst, y_pred_pro_lst, y_pred_cov_lst, y_pred_pro_cov_lst)


params_dict = {'n_estimators': [100, 200, 300, 400, 500],
               'max_depth': np.linspace(5, 30, 6).astype('int32').tolist(),
               'num_leaves': np.linspace(5, 30, 6).astype('int32').tolist(),
               'subsample': np.linspace(0.6, 1, 9).tolist(),
               'learning_rate': [0.1, 0.05, 0.01, 0.001],
               'colsample_bytree': np.linspace(0.6, 1, 9).tolist()}

candidate_params_lst = select_params_combo(params_dict, nb_params, my_seed)

dpath = '/home1/jiayou/Documents/Projects/ProDisAtlas/'
#dpath = '/Volumes/JasonWork/Projects/ProDisAtlas/'

tgt2pred_df = pd.read_csv(dpath + 'Data/Target/Prevalent.csv', encoding='latin-1')
tgt2pred_lst = tgt2pred_df.Disease_code.tolist()
pro_df = pd.read_csv(dpath + 'Data/ProteinData/ProteinData.csv')
full_pro_f_lst = pro_df.columns.tolist()[1:]
cov_df = pd.read_csv(dpath + 'Data/Covariates/Covariates.csv')

mydf = pd.merge(cov_df, pro_df, how = 'left', on = ['eid'])
fold_id_lst = list(set(mydf.Region_code))
fold_id_lst.sort()

bad_tgt_lst, nb_tgt_lst = [], []

for tgt in tqdm(tgt2pred_lst[125:250]):
    try:
        cov_f_lst = get_cov_f_lst(tgt2pred_df, tgt)
        tmp_tgt_df = pd.read_csv(dpath + 'Data/Target/Targets2Analysis/' + tgt + '.csv', usecols=['eid', 'target_y', 'BL2Target_yrs'])
        rm_bl_idx = tmp_tgt_df.index[(tmp_tgt_df.BL2Target_yrs > 0) & (tmp_tgt_df.target_y == 1)]
        tmp_tgt_df.drop(rm_bl_idx, axis=0, inplace=True)
        tmp_tgt_df.reset_index(inplace=True, drop=True)
        tmp_df = pd.merge(tmp_tgt_df, mydf, how='left', on=['eid'])
        eid_lst, y_test_lst, y_pred_pro_lst, y_pred_cov_lst, y_pred_pro_cov_lst = [], [], [], [], []
        tg_imp_cv, tc_imp_cv = Counter(), Counter()
        fold_results_lst = Parallel(n_jobs=nb_cpus)(delayed(get_iter_predictions)(tmp_df, full_pro_f_lst, cov_f_lst, fold_id, my_params0, my_params) for fold_id in fold_id_lst)
        for fold_results in fold_results_lst:
            tg_imp_cv += fold_results[0]
            tc_imp_cv += fold_results[1]
            eid_lst += fold_results[2]
            y_test_lst += fold_results[3]
            y_pred_pro_lst += fold_results[4]
            y_pred_cov_lst += fold_results[5]
            y_pred_pro_cov_lst += fold_results[6]
        tg_imp_cv = normal_imp(tg_imp_cv)
        tg_imp_df = pd.DataFrame({'Pro_code': list(tg_imp_cv.keys()), 'TotalGain_cv': list(tg_imp_cv.values())})
        tc_imp_cv = normal_imp(tc_imp_cv)
        tc_imp_df = pd.DataFrame({'Pro_code': list(tc_imp_cv.keys()), 'TotalCover_cv': list(tc_imp_cv.values())})
        imp_df = pd.merge(left=tc_imp_df, right=tg_imp_df, how='left', on=['Pro_code'])
        imp_df.sort_values(by='TotalGain_cv', ascending=False, inplace=True)
        pred_df = pd.DataFrame({'eid': eid_lst, 'target_y': y_test_lst, 'y_pred_pro': y_pred_pro_lst,
                                'y_pred_cov': y_pred_cov_lst, 'y_pred_pro_cov': y_pred_pro_cov_lst})
        imp_df.to_csv(dpath + 'Results/Prediction/CrossSectionalAnalysis/ProImportance/' + tgt + '.csv', index=False)
        pred_df.to_csv(dpath + 'Results/Prediction/CrossSectionalAnalysis/Predictions/' + tgt + '.csv', index=False)
    except:
        bad_tgt_lst.append(tgt)

bad_tgt_df = pd.DataFrame({'Disease_code': bad_tgt_lst})
bad_tgt_df.to_csv(dpath + 'Results/Prediction/CrossSectionalAnalysis/bad_tgt.csv', index=False)
