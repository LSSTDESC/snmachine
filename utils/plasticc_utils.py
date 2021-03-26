"""
Utility script for calculating the log loss and some useful plots
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix
from snmachine import snclassifier


def plot_confusion_matrix(y_true, y_pred, title, target_names, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    annot = np.around(cm, 2)

    dict_label_to_real = {15: 'TDE', 42: 'SNII', 52: 'SNIax', 62: 'SNIbc',
                          64: 'KN', 67: 'SNIa-91bg',
                          88: 'AGN', 90: 'SNIa', 95: 'SLSN-I',
                          1: 'SNIa', 2: 'SNII', 3: 'SNIbc'}  # to erase later; This is just for de-bug
    target_names = np.vectorize(dict_label_to_real.get)(target_names)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, xticklabels=target_names,
                yticklabels=target_names, cmap='Blues',
                annot=annot, lw=0.5, vmin=0, vmax=1)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_aspect('equal')
    plt.title(title)

    return cm, fig


def plasticc_log_loss(y_true, probs):
    """Implementation of weighted log loss used for the Kaggle challenge.

    Parameters
    ----------
    y_true: np.array of shape (# samples,)
        Array of the true classes
    probs : np.array of shape (# samples, # features)
        Class probabilities for each sample. The order of the classes
        corresponds to that in the attribute `classes_` of the classifier used.

    Returns
    -------
    float
        Weighted log loss used for the Kaggle challenge
    """
    predictions = probs.copy()
    # `labels` assumes the probabilities are ordered in the same way
    labels = np.unique(y_true)
    weights_dict = {6: 1/18, 15: 1/9, 16: 1/18, 42: 1/18, 52: 1/18, 53: 1/18,
                    62: 1/18, 64: 1/9, 65: 1/18, 67: 1/18, 88: 1/18, 90: 1/18,
                    92: 1/18, 95: 1/18, 99: 1/19,
                    1: 1/18, 2: 1/18, 3: 1/18}

    # sanitize predictions
    epsilon = sys.float_info.epsilon  # machine dependent but prevents log(0)
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    predictions = predictions / np.sum(predictions, axis=1)[:, np.newaxis]

    predictions = np.log(predictions)  # logarithm because we want a log loss
    class_logloss, weights = [], []  # initialize the classes logloss and weights
    for i in range(np.shape(predictions)[1]):  # run for each class
        current_label = labels[i]
        is_right_class = y_true == current_label  # only events from that class
        result = np.average(predictions[is_right_class, i])
        class_logloss.append(result)
        weights.append(weights_dict[current_label])
    return -1 * np.average(class_logloss, weights=weights)


def plot_roc_curve(y_probs, y_true):
    """Plot a ROC curve with all the classes. It only works for PLAsTiCC SN.

    Parameters
    ----------
    y_probs : array (# obj, # classes)
        An array containing probabilities of each class for each object.
    y_true : array (N_obj, )
        An array containing the true class for each object.
    auc : float
    """
    dict_label_to_real = {15: 'TDE', 42: 'SNII', 52: 'SNIax', 62: 'SNIbc',
                          64: 'KN', 67: 'SNIa-91bg',
                          88: 'AGN', 90: 'SNIa', 95: 'SLSN-I',
                          1: 'SNIa', 2: 'SNII', 3: 'SNIbc'}  # to erase later; This is just for de-bug
    target_names = np.unique(y_true)
    target_true_names = np.vectorize(dict_label_to_real.get)(target_names)
    plt.figure(figsize=(10, 6))
    # Compute ROC curve and ROC area for each class
    fpr, tpr, auc = {}, {}, {}  # initialize dictionaries

    for i in range(len(target_true_names)):
        fpr[i], tpr[i], auc[i] = snclassifier.roc(y_probs, y_true,
                                                  which_column=i)

    linewidth = 3
    colors = {'SNIa': 'C2', 'SNII': 'C0', 'SNIbc': 'C1'}
    for i, color in zip(range(3), colors):
        true_name = target_true_names[i]
        try:  # we are working with SN Ia, SNbc and SNII
            plt.plot(fpr[i], tpr[i], color=colors[true_name], lw=linewidth,
                     label='AUC {} = {:0.3f}'.format(true_name, auc[i]))
        except KeyError:
            plt.plot(fpr[i], tpr[i], lw=linewidth,
                     label='AUC {} = {:0.3f}'.format(true_name, auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=linewidth)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC: 1 vs All')
    plt.legend(loc="lower right")