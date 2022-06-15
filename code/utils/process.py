import numpy as np
import pandas as pd
import sys
from sklearn.metrics import ndcg_score


def cal_ndcg(predicts, labels, user_ids, k_list):

    d = {'user': np.squeeze(user_ids), 'predict':np.squeeze(predicts), 'label':np.squeeze(labels)}
    df = pd.DataFrame(d)
    user_unique = df.user.unique()

    ndcgs = [[] for _ in range(len(k_list))]
    for user_id in user_unique:
        user_srow = df.loc[df['user'] == user_id]
        upred = user_srow['predict'].tolist()
        if len(upred) < 2:
            #print('less than 2', user_id)
            continue
        #supred = [upred] if len(upred)>1 else [upred + [-1]]  # prevent error occured if only one sample for a user
        ulabel = user_srow['label'].tolist()
        #sulabel = [ulabel] if len(ulabel)>1 else [ulabel +[1]]

        for i in range(len(k_list)):
            ndcgs[i].append(ndcg_score([ulabel], [upred], k=k_list[i])) 

    ndcg_mean =  np.mean(np.array(ndcgs), axis=1)
    return ndcg_mean

def cal_recall(predicts, labels, user_ids, k):
    d = {'user': np.squeeze(user_ids), 'predict':np.squeeze(predicts), 'label':np.squeeze(labels)}
    df = pd.DataFrame(d)
    user_unique = df.user.unique()

    recall = []
    for user_id in user_unique:
        user_sdf = df[df['user'] == user_id]
        if user_sdf.shape[0] < 2:
            #print('less than 2', user_id)
            continue
        user_sdf = user_sdf.sort_values(by=['predict'], ascending=False)
        total_rel = min(user_sdf['label'].sum(), k)
        #total_rel = user_sdf['label'].sum()
        intersect_at_k = user_sdf['label'][0:k].sum()

        try:
            recall.append(float(intersect_at_k)/float(total_rel))
        except:
            continue

    return np.mean(np.array(recall))


"""
=========================================================
Evaluation function
=========================================================
"""
def eval_metrics(predictions, labels, user_ids, test=False):
    predictions = np.concatenate(predictions, 0)
    labels = np.concatenate(labels, 0)
    user_ids = np.concatenate(user_ids, 0)

    labels = labels.astype(int)
    ndcg_list = cal_ndcg(predictions, labels, user_ids, [5,10,20]) 

    if test:
        recall5 = cal_recall(predictions, labels, user_ids, 5) 
        recall10 = cal_recall(predictions, labels, user_ids, 10) 
        recall20 = cal_recall(predictions, labels, user_ids, 20) 
        return ndcg_list, (recall5, recall10, recall20)
    else:
        return ndcg_list


