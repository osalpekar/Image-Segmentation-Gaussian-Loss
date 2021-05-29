import os
import sys

import torch
import sklearn.metrics

def jaccard(y_pred, y_true, thresh=0.5):
    scores = []
    y_pred = y_pred.to("cpu")
    y_true = y_true.to("cpu")

    for i in range(len(y_true)):
        binary_preds = (y_pred[i][0].flatten() > thresh).int()
        binary_trues = y_true[i].flatten()
        score = sklearn.metrics.jaccard_score(binary_trues, binary_preds, average='micro')
        scores.append(score)

    return torch.tensor(sum(scores)/len(scores))
