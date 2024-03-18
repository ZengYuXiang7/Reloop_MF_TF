# coding : utf-8
# Author : yuxiang Zeng

import torch as t
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def ErrorMetrics(realVec, estiVec):
    if isinstance(realVec, np.ndarray):
        realVec = realVec.astype(float)
    elif isinstance(realVec, t.Tensor):
        realVec = realVec.cpu().detach().numpy().astype(float)
    if isinstance(estiVec, np.ndarray):
        estiVec = estiVec.astype(float)
    elif isinstance(estiVec, t.Tensor):
        estiVec = estiVec.cpu().detach().numpy().astype(float)

    absError = np.abs(estiVec - realVec)
    MAE = np.mean(absError)
    RMSE = np.linalg.norm(absError) / np.sqrt(np.array(absError.shape[0]))
    NMAE = np.sum(np.abs(realVec - estiVec)) / np.sum(realVec)
    NRMSE = np.sqrt(np.sum((realVec - estiVec) ** 2)) / np.sqrt(np.sum(realVec ** 2))

    Acc = []
    thresholds = [0.01, 0.05, 0.10]
    for threshold in thresholds:
        threshold = realVec * threshold  # 定义阈值为真实值的5%
        accurate_predictions = absError < threshold
        accuracy = np.mean(accurate_predictions.astype(float))
        Acc.append(accuracy)

    return {
        'MAE' : MAE,
        'RMSE' : RMSE,
        'NMAE': NMAE,
        'NRMSE': NRMSE,
        'Acc' : Acc,
    }



def class_Metrics(y_pred, y_test):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    Acc = accuracy_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, average='micro')  # 如果是多分类问题，可以修改 average 参数
    P = precision_score(y_test, y_pred, average='micro')  # 如果是多分类问题，可以修改 average 参数
    Recall = recall_score(y_test, y_pred, average='micro')  # 如果是多分类问题，可以修改 average 参数
    return {
        'Acc' : Acc,
        'F1' : F1,
        'P' : P,
        'Recall' : Recall
    }
