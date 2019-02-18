"""
Utility script for calculating the log loss
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve, confusion_matrix

def plotConfusionMatrix(yTrue, yPredict, dataName, targetNames):
    cm = confusion_matrix(yTrue, yPredict, labels=targetNames)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot = np.around(cm, 2)

    fig, ax = plt.subplots(figsize=(9,7))
    sns.heatmap(cm, xticklabels=targetNames,
    yticklabels=targetNames, cmap='Blues',
    annot=annot, lw=0.5)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_aspect('equal')
    plt.title(dataName)

    return cm

def plasticcLogLoss(y_true, y_pred, relative_class_weights=None):
    """
    Implementation of weighted log loss used for the Kaggle challenge
    """
    predictions = y_pred.copy()

    # sanitize predictions
    epsilon = sys.float_info.epsilon # this is machine dependent but essentially prevents log(0)
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    predictions = predictions / np.sum(predictions, axis=1)[:, np.newaxis]

    predictions = np.log(predictions)
    # multiplying the arrays is equivalent to a truth mask as y_true only contains zeros and ones
    class_logloss = []
    for i in range(predictions.shape[1]):
        # average column wise log loss with truth mask applied
        result = np.average(predictions[:, i][y_true[:, i] == 1])
        class_logloss.append(result)
    return -1 * np.average(class_logloss, weights=relative_class_weights)

weights = np.array([1/18, 1/9, 1/18, 1/18, 1/18, 1/18, 1/18, 1/9, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/19])
