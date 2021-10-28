'''
Fairness Aware XGBoost - UCI Adult Income Data

Implementation of https://arxiv.org/abs/2009.01442
'''

import numpy as np
import pandas as pd
import xgboost as xgb
import dask.array as da
import dask.distributed
import dask.dataframe as dd

from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

'''
Load the data available at 
https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv

The modified objective function defined in https://arxiv.org/abs/2009.01442 
encourages more favourable outcomes to the minority group. 

In order to ensure this, one needs to follow the following encoding scheme 

If `t` represents the favourable outcome of a classifier and `1 - t`, 
the unfavourable outcome, encode minority members by setting `s = t` 
and majority members (males) by setting `s = 1 - t`

In our example, a label 1.0 represents an unfavourable outcome (likely to be \
re-arrested) and correspondingly, the minority race is represented with a value 1.0
'''

def load_data(filename):
    df = dd.read_csv(filename)
    X = df.drop('two_year_recid', axis=1).drop('race', axis=1).drop('id', axis=1)
    y = df['two_year_recid']
    s = df['race']
    return df, X, y, s

def fairbceobj(fair_reg=0.0):
    def fairbceobj_inner(preds, dtrain):
        labels = dtrain.get_label()
        sensitive_feature = dtrain.get_sensitive_feature()

        preds = 1.0 / (1.0 + np.exp(-preds))  # transform raw leaf weight

        grad = preds - labels + (fair_reg * (sensitive_feature - preds))
        hess = (1.0 - fair_reg) * preds * (1.0 - preds)

        return grad, hess
    return fairbceobj_inner

def get_metrics(eval_df, minority_indicator=0.0):
    max_acc = 0
    max_acc_thresh = 0
    for threshold in np.linspace(0.1, 1.0, 100):
        preds = (eval_df['score'] > threshold).astype('float32')
        acc = accuracy_score(eval_df['y'].to_numpy(), preds)
        if acc > max_acc:
            max_acc = acc
            max_acc_thresh = threshold
    eval_df['y_pred'] = (eval_df['score'] > max_acc_thresh).astype('float32')
    
    mTP = eval_df.loc[eval_df['s'] == minority_indicator]\
                 .loc[eval_df['y_pred'] == eval_df['y']]\
                 .loc[eval_df['y'] == 1.0]\
                 .shape[0]
    mTN = eval_df.loc[eval_df['s'] == minority_indicator]\
                 .loc[eval_df['y_pred'] == eval_df['y']]\
                 .loc[eval_df['y'] == 0.0]\
                 .shape[0]
    mFP = eval_df.loc[eval_df['s'] == minority_indicator]\
                 .loc[eval_df['y_pred'] != eval_df['y']]\
                 .loc[eval_df['y'] == 0.0]\
                 .shape[0]
    mFN = eval_df.loc[eval_df['s'] == minority_indicator]\
                 .loc[eval_df['y_pred'] != eval_df['y']]\
                 .loc[eval_df['y'] == 1.0]\
                 .shape[0]
    
    
    MTP = eval_df.loc[eval_df['s'] == (1.0 - minority_indicator)]\
                 .loc[eval_df['y_pred'] == eval_df['y']]\
                 .loc[eval_df['y'] == 1.0]\
                 .shape[0]
    MTN = eval_df.loc[eval_df['s'] == (1.0 - minority_indicator)]\
                 .loc[eval_df['y_pred'] == eval_df['y']]\
                 .loc[eval_df['y'] == 0.0]\
                 .shape[0]
    MFP = eval_df.loc[eval_df['s'] == (1.0 - minority_indicator)]\
                 .loc[eval_df['y_pred'] != eval_df['y']]\
                 .loc[eval_df['y'] == 0.0]\
                 .shape[0]
    MFN = eval_df.loc[eval_df['s'] == (1.0 - minority_indicator)]\
                 .loc[eval_df['y_pred'] != eval_df['y']]\
                 .loc[eval_df['y'] == 1.0]\
                 .shape[0]
    
    TP = mTP + MTP
    FP = mFP + MFP
    TN = mTN + MTN
    FN = mFN + MFN
    
    try:
        precision = TP / float(TP + FP)
    except ZeroDivisionError as e:
        precision = np.nan
        pass
    try:
        recall = TP / float(TP + FN)
    except ZeroDivisionError as e:
        recall = np.nan
        pass
    try:
        accuracy = (TP + TN) / float(TP + TN + FP + FN)
    except ZeroDivisionError as e:
        accuracy = np.nan
        pass
    try:
        f1_score = 1.0 / ((1.0 / precision) + (1.0 / recall))
    except ZeroDivisionError as e:
        f1_score = np.nan
        pass
    try:
        m_base_rate = (mTP + mFP) / float(mTP + mFP + mTN + mFN)
    except:
        m_base_rate = np.nan
        pass
    try:
        M_base_rate = (MTP + MFP) / float(MTP + MFP + MTN + MFN)
    except:
        M_base_rate = np.nan
        pass
        
    try:
        disparate_impact = m_base_rate / float(M_base_rate)
    except ZeroDivisionError as e:
        disparate_impact = np.nan
        pass
    try:
        mTPR = mTP/ float(mTP + mFN)
    except ZeroDivisionError as e:
        mTPR = np.nan
        pass
    try:
        MTPR = MTP / float(MTP + MFN)
    except ZeroDivisionError as e:
        MTPR = np.nan
        pass
        
    eod = mTPR - MTPR

    results = {}
    results['precision'] = precision
    results['recall'] = recall
    results['accuracy'] = accuracy
    results['f1_score'] = f1_score
    results['disparate_impact'] = disparate_impact
    results['equal_opportunity_difference'] = eod
    results['mTP'] = mTP
    results['mFP'] = mFP
    results['mTN'] = mTN
    results['mFN'] = mFN
    results['MTP'] = MTP
    results['MFP'] = MFP
    results['MTN'] = MTN
    results['MFN'] = MFN
    return results

if __name__ == '__main__':

    train_df, X_train, y_train, s_train = load_data('../data/compas.txt.train')
    test_df, X_test, y_test, s_test = load_data('../data/compas.txt.test')

    client = dask.distributed.Client()

    dtrain = xgb.dask.DaskDMatrix(client, X_train, label=y_train, sensitive_feature=s_train)
    dtest = xgb.dask.DaskDMatrix(client, X_test, label=y_test, sensitive_feature=s_test)


    param = {'max_depth': 5, 'eta': 0.1, 'objective': 'binary:logistic', 'reg_lambda' : 0}
    num_round = 500
    vanilla_params = param.copy()
    vanilla_result = xgb.dask.train(client, vanilla_params, dtrain, num_round)


    fair_results = {}
    fair_params = param.copy()

    print('Training fair models')
    for fair_reg in tqdm(np.linspace(0.0, 0.8, num=100)):
        fair_results[fair_reg] = {}
        fair_result = xgb.dask.train(client, fair_params, dtrain, num_round, obj=fairbceobj(fair_reg=fair_reg))
        fair_results[fair_reg] = fair_result


    results = {}
    print('Predictions with fair models')
    for fair_reg in tqdm(np.linspace(0.0, 0.8, num=100)):
        y = xgb.dask.predict(client, fair_results[fair_reg], dtest)
        eval_df = pd.DataFrame([y_test.to_dask_array().compute(), y.compute(), s_test.to_dask_array().compute()]).transpose()
        eval_df.columns = ["y", "score", "s"]
        results[fair_reg] = get_metrics(eval_df, minority_indicator=1.0)


    di_arr = []
    eod_arr = []
    fair_reg_arr = []
    for k,v in results.items():
        fair_reg_arr.append(k)
        di = v['disparate_impact']
        eod = v['equal_opportunity_difference']
        di_arr.append(di)
        eod_arr.append(eod)

    results_df = pd.DataFrame.from_dict(results)

    z = zip(*[(y,x) for (x,y) in zip(results_df.transpose()['disparate_impact'].values, results_df.transpose()['accuracy'].values)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle('FairXGBoost on COMPAS data')

    ax1.plot(fair_reg_arr, di_arr)
    ax1.set_xlabel('Fair Regularization Strength')
    ax1.set_ylabel('Disparate Impact')

    ax2.scatter(*z)
    ax2.set_ylabel('Disparate Impact')
    ax2.set_xlabel('Accuracy')
    plt.show()