"""
Module for training and optimizing classifiers. It mostly warps sklearn and
LightGBM functionality.
"""
from __future__ import division

__all__ = []  # 'roc',


from past.builtins import basestring

import collections
import itertools
import os
import pickle
import sys
import time
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.naive_bayes  # requires a specific import
import sklearn.neural_network  # requires a specific import

from functools import partial
from multiprocessing import Pool
from scipy.integrate import trapz
from sklearn import model_selection
from sklearn.model_selection import PredefinedSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from utils import plasticc_utils

# This allows the user to easily loop through all possible classifiers
choice_of_classifiers = ['svm', 'knn', 'random_forest', 'decision_tree',
                         'boost_dt', 'boost_rf', 'nb', 'neural_network']
# boost_rf is a set of boosted random forests which Max came up with.


# Custom scoring/metrics functions ##
def logloss_score(classifier, X_features, y_true):
    """PLAsTiCC logloss classification score.

    This custom scoring method can be used in a grid search.

    Parameters
    ----------
    classifier : classifier instance `sklearn`, `LightGBM` or
                `BaseClassifier.child.classifier`
        Classifier.
    X_features : pandas.DataFrame or np.array
        Features of shape (n_samples, n_features).
    y_true : 1D array-like
        Ground truth (correct) labels of shape (n_samples,).

    Returns
    -------
    float
        Symmetric of the PLAsTiCC logloss score. We use the symmetric
        because this function is going to be maximised and the optimal
        result of the logloss is its minimum (logloss = 0).
    """
    probs = classifier.predict_proba(X_features)
    logloss = plasticc_utils.plasticc_log_loss(y_true, probs)
    return -logloss  # symmetric because we want to maximise this output


def auc_score(classifier, X_features, y_true, which_column):
    """A Area Under the ROC Curve (AUC) classification score.

    ROC stands for Receiver Operating Characteristic Curve
    This custom scoring method can be used in a grid search.

    Parameters
    ----------
    classifier : classifier instance `sklearn`, `LightGBM` or
                `BaseClassifier.child.classifier`
        Classifier.
    X_features : pandas.DataFrame or np.array
        Features of shape (n_samples, n_features).
    y_true : 1D array-like
        Ground truth (correct) labels of shape (n_samples,).
    which_column : int
        The index of the column refering to the desired class (e.g. Ias, which
        might correspond to class 1, or 90). This allows the user to optimise
        for different classes.

    Returns
    -------
    auc : float
        AUC score.
    """
    probs = classifier.predict_proba(X_features)
    fpr, tpr, auc = roc(pr=probs, Yt=y_true, which_column=which_column)
    return auc  # symmetric because we want to maximise this output


def roc(pr, Yt, true_class=0, which_column=-1):
    """Produce the false positive rate and true positive rate required to plot
    a ROC curve, and the area under that curve.

    Parameters
    ----------
    pr : array
        An array of probability scores, either a 1d array of size N_samples or
        an nd array, in which case the column corresponding to the true class
        will be used.
    Yt : array
        An array of class labels, of size (N_samples,)
    true_class : int, optional
        Which class is taken to be the "true class" (e.g. Ia vs everything
        else). If `which_column`!=-1, `true_class` is overriden. - NOTE this
        only works for sequential labels (as in SPCC). Should NOT be used for
        PLAsTiCC!
    which_column : int, optional
        Defaults to -1 where `true_class` is used instead. If
        `which_column`!=-1, `true_class` is overriden and `which_column`
        selects which column of the probabilities to take as the "true class".
        - use this instead of `true_class` for PLAsTiCC.

    Returns
    -------
    fpr : array
        An array containing the false positive rate at each probability
        threshold.
    tpr : array
        An array containing the true positive rate at each probability
        threshold.
    auc : float
        The area under the ROC curve
    """
    probs = pr.copy()
    Y_test = Yt.copy()

    # Deals with starting class assignment at 1.
    min_class = (int)(Y_test.min())

    Y_test = Y_test.squeeze()
    # sequential labels (as in SPCC) case - backwards compatibility
    if len(pr.shape) > 1 and which_column == -1:
        try:
            probs_1 = probs[:, true_class-min_class]
        except IndexError:
            raise IndexError('If `which_column` is -1, the `Yt` labels must be'
                             'sequential and `true_class` must be provided.')
    # Used by `optimised_classify`
    elif len(pr.shape) > 1 and which_column != -1:
        if which_column >= np.shape(probs)[1]:
            sys.exit(f'`which_column` must be -1 or between 0 and '
                     f'{np.shape(probs)[1]-1}.')
            # some error is happening and the raise error bellow does not stop
            # the code -> TODO: find why and fix it
            raise IndexError(f'`which_column` must be -1 or between 0 and '
                             f'{np.shape(probs)[1]-1}.')
        probs_1 = probs[:, which_column]

        # the classes are in the same order as `probs`
        unique_labels = np.unique(Yt)

        true_class = unique_labels[which_column]
    # We give a 1D array of probability so use it - no ambiguity
    else:
        probs_1 = probs

    threshold = np.linspace(0., 1., 50)  # 50 evenly spaced numbers between 0,1

    # This creates an array where each column is the prediction for each
    # threshold
    preds = np.tile(probs_1,
                    (len(threshold), 1)).T >= np.tile(threshold,
                                                      (len(probs_1), 1))
    Y_bool = (Y_test == true_class)
    Y_bool = np.tile(Y_bool, (len(threshold), 1)).T

    TP = (preds & Y_bool).sum(axis=0)
    FP = (preds & ~Y_bool).sum(axis=0)
    TN = (~preds & ~Y_bool).sum(axis=0)
    FN = (~preds & Y_bool).sum(axis=0)

    tpr = np.zeros(len(TP))
    tpr[TP != 0] = TP[TP != 0]/(TP[TP != 0] + FN[TP != 0])
    fpr = FP/(FP+TN)

    fpr = np.array(fpr)[::-1]
    tpr = np.array(tpr)[::-1]

    auc = trapz(tpr, fpr)

    return fpr, tpr, auc


def plot_roc(fpr, tpr, auc, labels=[], cols=[],  label_size=26, tick_size=18,
             line_width=3, figsize=(8, 6)):
    """Plots a ROC curve or multiple curves.

    The function can plot the results from multiple classifiers if fpr and tpr
    are arrays where each column corresponds to a different classifier.

    Parameters
    ----------
    fpr : array
        An array containing the false positive rate at each probability
        threshold
    tpr : array
        An array containing the true positive rate at each probability
        threshold
    auc : float
        The area under the ROC curve
    labels : list, optional
        Labels of each curve (e.g. ML algorithm names)
    cols : list, optional
        Colors of the line(s)
    label_size : float, optional
        Size of x and y axis labels.
    tick_size: float, optional
        Size of tick labels.
    line_width : float, optional
        Line width.
    """

    # Automatically fill in the colors if not supplied
    if not isinstance(cols, basestring) and len(cols) == 0:
        cols = ['#185aa9', '#008c48', '#ee2e2f', '#f47d23', '#662c91',
                '#a21d21', '#b43894', '#010202']

    # This should work regardless of whether it's one or many roc curves
    # fig=plt.figure(figsize=figsize)
    fig = plt.gcf()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color', cols)
    ax.plot(fpr, tpr, lw=line_width)
    # ax.plot(fpr, tpr)

    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    plt.xlabel('False positive rate (contamination)', fontsize=label_size)
    plt.ylabel('True positive rate (completeness)', fontsize=label_size)

    # Robust against the possibility of AUC being a single number instead of a
    # list
    if not isinstance(auc, collections.Sequence):
        auc = [auc]

    if len(labels) > 0:
        labs = []
        for i in range(len(labels)):
            labs.append(labels[i]+' (%.3f)' % (auc[i]))
    else:
        labs = np.array(range(len(ax.lines)), dtype='str')
        for i in range(len(labs)):
            labs[i] = (labs[i]+' (%.3f)' % (auc[i]))
    plt.legend(labs, loc='lower right',  bbox_to_anchor=(0.95, 0.05))
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, normalise=False, labels=None,
                          title='Confusion matrix'):
    """Make a plot from a pre-computed confusion matrix.

    Parameters
    ----------
    cm : np.array
       The confusion matrix, as computed by the
       snclassifier.compute_confusion_matrix
    normalise : bool, optional
       If False, we use the absolute numbers in each matrix entry. If True, we
       use the fractions within each true class
    labels : list of str
       Labels for each class that appear in the plot
    title : str
       Surprisingly, this is the title for the plot.
    """
    if labels is None:
        labels = np.arange(len(cm[:, 0])).tolist()
    plt.figure()
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def F1(pr,  Yt, true_class, full_output=False):
    """Calculate an F1 score for many probability threshold increments
    and select the best one.

    Parameters
    ----------
    pr : array
        An array of probability scores, either a 1d array of size N_samples or
        an nd array, in which case the column corresponding to the true class
        will be used.
    Yt : array
        An array of class labels, of size (N_samples,)
    true_class : int
        which class is taken to be the "true class" (e.g. Ia vs everything
        else)
    full_output : bool, optional
        If true returns two vectors corresponding to F1 as a function of
        threshold, instead of the best value.

    Returns
    -------
    best_F1 : float
        (If full_output=False) The largest F1 value.
    best_threshold : array
        (If full_output=False) The probability threshold corresponding to
        best_F1.
    f1  : array
        (If full_output=True) F1 as a function of threshold.
    threshold  : array
        (If full_output=True) Vector of thresholds (from 0 to 1)
    """
    probs = pr.copy()
    Y_test = Yt.copy()
    min_class = Y_test.min()  # deals with starting class assignment at 1.
    Y_test = Y_test.squeeze()

    if len(pr.shape) > 1:
        probs_1 = probs[:, true_class-min_class]
    else:
        probs_1 = probs

    threshold = np.arange(0, 1, 0.01)

    # This creates an array where each column is the prediction for each
    # threshold
    preds = np.tile(probs_1,
                    (len(threshold), 1)).T > np.tile(threshold,
                                                     (len(probs_1), 1))
    Y_bool = (Y_test == true_class)
    Y_bool = np.tile(Y_bool, (len(threshold), 1)).T

    TP = (preds & Y_bool).sum(axis=0)
    FP = (preds & ~Y_bool).sum(axis=0)
    FN = (~preds & Y_bool).sum(axis=0)

    f1 = np.zeros(len(TP))
    f1[TP != 0] = 2*TP[TP != 0]/(2 * TP[TP != 0] + FN[TP != 0] + FP[TP != 0])

    if full_output:
        return f1, threshold
    else:
        best_F1 = f1.max()
        best_threshold_index = np.argmax(f1)
        best_threshold = threshold[best_threshold_index]

        return best_F1, best_threshold


def FoM(pr,  Yt, which_column=-1, true_class=1, full_output=False):
    """Calculate a Kessler FoM for many probability threshold increments
    and select the best one.

    FoM is defined as:
    FoM = TP^2/((TP+FN)(TP+3*FP))

    Parameters
    ----------
    pr : array
        An array of probability scores, either a 1d array of size N_samples or
        an nd array, in which case the column corresponding to the true class
        will be used.
    Yt : array
        An array of class labels, of size (N_samples,)
    true_class : int
        which class is taken to be the "true class" (e.g. Ia vs everything
        else)
    full_output : bool, optional
        If true returns two vectors corresponding to F1 as a function of
        threshold, instead of the best value.

    Returns
    -------
    best_FoM : float
        (If full_output=False) The largest FoM value.
    best_threshold : array
        (If full_output=False) The probability threshold corresponding to
        best_FoM.
    fom  : array
        (If full_output=True) FoM as a function of threshold.
    threshold  : array
        (If full_output=True) Vector of thresholds (from 0 to 1).
    """
    weight = 3.0

    probs = pr.copy()
    Y_test = Yt.copy()
    min_class = Y_test.min()  # deals with starting class assignment at 1.
    Y_test = Y_test.squeeze()

    # sequential labels (as in SPCC) case - backwards compatibility
    if len(pr.shape) > 1 and which_column == -1:
        try:
            probs_1 = probs[:, true_class-min_class]
        except IndexError:
            raise IndexError('If `which_column` is -1, the `Yt` labels must be'
                             'sequential and `true_class` must be provided.')
    elif len(pr.shape) > 1 and which_column != -1:
        if which_column >= np.shape(probs)[1]:
            raise IndexError(f'`which_column` must be -1 or between 0 and '
                             f'{np.shape(probs)[1]-1}.')
        probs_1 = probs[:, which_column]

        # the classes are in the same order as `probs`
        unique_labels = np.unique(Yt)

        true_class = unique_labels[which_column]
    # We give a 1D array of probability so use it - no ambiguity
    else:
        probs_1 = probs

    threshold = np.arange(0, 1, 0.01)

    # This creates an array where each column is the prediction for each
    # threshold
    preds = np.tile(probs_1,
                    (len(threshold), 1)).T > np.tile(threshold,
                                                     (len(probs_1), 1))
    Y_bool = (Y_test == true_class)
    Y_bool = np.tile(Y_bool, (len(threshold), 1)).T

    TP = (preds & Y_bool).sum(axis=0)
    FP = (preds & ~Y_bool).sum(axis=0)
    FN = (~preds & Y_bool).sum(axis=0)

    fom = np.zeros(len(TP))
    fom[TP != 0] = (TP[TP != 0]**2
                    / (TP[TP != 0] + FN[TP != 0])
                    / (TP[TP != 0] + weight * FP[TP != 0]))

    if full_output:
        return fom, threshold

    else:
        best_FoM = fom.max()
        best_threshold_index = np.argmax(fom)
        best_threshold = threshold[best_threshold_index]

        return best_FoM, best_threshold


def run_several_classifiers(classifier_list, features, labels,
                            scoring, train_set, scale_features=True,
                            param_grid=None, random_seed=42, which_column=0,
                            output_root=None, **kwargs):
    """The features must be pandas DataFrame

    Parameters
    ----------
    classifier_list : list
        Which available ML classifiers to run.
    features : pandas.DataFrame
        Features of the dataset events.
    labels : pandas.DataFrame
        Labels of the dataset events.
    scoring : callable, str
        The metric used to evaluate the predictions on the test or
        validation sets. See
        `sklearn.model_selection._search.GridSearchCV` [1]_ for details on
        how to choose this input.
        `snmachine` also contains the 'logloss' and 'auc' custom scoring.
        For more details about these, see `logloss_score` and
        `auc_score`, respectively.
    train_set : {float, list-like}
        If float, it is the fraction of objects that will be used as training
        set. If list, it is the IDs of the objects to use as training set.
    scale_features : bool, optional
        If True (default and recommended), rescale features using sklearn's
        preprocessing Scalar class.
    param_grid : {None, dict}, optional
        Dictionary containing the parameters names (`str`) as keys and lists
        of their possible settings as values.
        If `None`, the default `param_grid` is used. This is defined in child
        classes of `BaseClassifier`.
    random_seed : {int, RandomState instance}, optional
        Random seed or random state instance to use. It allows reproducible
        results.
    which_column : int, optional
        The index of the column refering to the desired class (e.g. Ias, which
        might correspond to class 1, or 90). This allows the user to optimise
        for different classes.
    output_root : {None, str}, optional
        If None, don't save anything. If str, save the classifiers'
        probability, ROC and AUC there.
    **kwargs : dict, optional
        number_processes : int, optional
            Number of processors to use for parallelisation (shared memory
            only). By default `number_processes` = 1.
        plot_roc_curve : bool, optional
            Whether to plot the ROC curves for the classifiers.

    Returns
    -------
    X_train : pandas.DataFrame
        Features of the events with which to train the classifier.
    X_test : pandas.DataFrame
        Features of the events with which to test the classifier.
    y_train : pandas.core.series.Series
        Labels of the events with which to train the classifier.
    y_test : pandas.core.series.Series
        Labels of the events with which to test the classifier.
    """
    initial_time = time.time()

    # Split into training and validation sets
    X_train, X_test, y_train, y_test = _split_train_test(
        features, labels, train_set, random_seed)

    # Rescale the data (highly recommended)
    if scale_features:
        scaler = StandardScaler()
        scaler.fit(np.vstack((X_train, X_test)))
        X_train = scaler.transform(X_train)  # it is now an np.array
        X_test = scaler.transform(X_test)  # it is now an np.array

    classifier_instances = {}
    probabilities = {}
    y_pred = {}

    # Train, optimise and make predictions with the classifiers
    number_processes = kwargs.pop('number_processes', 1)
    if number_processes > 1:  # run in parallel
        partial_func = partial(_run_classifier, X_train=X_train,
                               y_train=y_train, X_test=X_test,
                               param_grid=param_grid, scoring=scoring,
                               which_column=which_column,
                               random_seed=random_seed)
        p = Pool(number_processes, maxtasksperchild=1)
        result = p.map(partial_func, classifier_list)

        for i, classifier_name in enumerate(classifier_list):
            best_classifier = result[i][1]
            probabilities[classifier_name] = result[i][0]
            classifier_instances[classifier_name] = best_classifier
            y_pred[classifier_name] = best_classifier.predict(X_test)
    else:  # run serially
        for classifier_name in classifier_list:
            probs, best_classifier = _run_classifier(
                classifier_name=classifier_name, X_train=X_train,
                y_train=y_train, X_test=X_test, param_grid=param_grid,
                scoring=scoring, which_column=which_column,
                random_seed=random_seed)

            classifier_instances[classifier_name] = best_classifier
            probabilities[classifier_name] = probs
            y_pred[classifier_name] = best_classifier.predict(X_test)

    # Calculate performance
    fpr = []  # false positive rate
    tpr = []  # true positive rate
    auc = []  # Area under the ROC Curve
    for classifier_name in classifier_list:
        best_classifier = classifier_instances[classifier_name]
        probs = probabilities[classifier_name]

        fpr_class, tpr_class, auc_class = roc(probs, y_test,
                                              which_column=which_column)
        fom_class, _ = FoM(probs, y_test, which_column=which_column,
                           full_output=False)
        print(f'Classifier {classifier_name}: AUC = {auc_class} ; FoM = '
              f'{fom_class}.')
        fpr.append(fpr_class)
        tpr.append(tpr_class)
        auc.append(auc_class)

        if output_root is not None:  # save results
            print(f'Probabilities, AUC and ROC saved on {output_root}.')
            probs_df = pd.DataFrame(probs, columns=np.unique(y_train))
            probs_df.set_index(y_test.index, inplace=True)
            probs_df.to_pickle(os.path.join(output_root,
                                            f'probs_{classifier_name}.pickle'))

            fpr_tpr = np.array([fpr_class, tpr_class]).T
            fpr_tpr = pd.DataFrame(fpr_tpr, columns=['FPR', 'TPR'])
            fpr_tpr.to_pickle(os.path.join(output_root,
                                           f'roc_{classifier_name}.pickle'))

            np.save(os.path.join(output_root, f'auc_{classifier_name}.npy'),
                    auc_class)
    fpr = np.array(fpr).T
    tpr = np.array(tpr).T

    # Make plots
    plot_roc_curve = kwargs.pop('plot_roc_curve', True)
    if plot_roc_curve:
        plot_roc(fpr, tpr, auc, labels=classifier_list, label_size=16,
                 tick_size=12, line_width=1.5)

    # Construct confusion matrices
    cms = {}
    for classifier_name in classifier_list:
        cm = sklearn.metrics.confusion_matrix(y_true=y_test,
                                              y_pred=y_pred[classifier_name])
        # also, I should use the CM that is defined in the utils instead
        cms[classifier_name] = cm

    print('Time taken to extract features: {:.2f}s.'
          ''.format(time.time()-initial_time))
    return classifier_instances, cms


def _split_train_test(features, labels, train_set, random_seed):
    """Split the dataset into training and test sets.

    Parameters
    ----------
    features : pandas.DataFrame
        Features of the dataset events.
    labels : pandas.DataFrame
        Labels of the dataset events.
    train_set : {float, list-like}
        If float, it is the fraction of objects that will be used as training
        set. If list, it is the IDs of the objects to use as training set.
    random_seed : {int, RandomState instance}
        Random seed or random state instance to use. It allows reproducible
        results.

    Returns
    -------
    X_train : pandas.DataFrame
        Features of the events with which to train the classifier.
    X_test : pandas.DataFrame
        Features of the events with which to test the classifier.
    y_train : pandas.core.series.Series
        Labels of the events with which to train the classifier.
    y_test : pandas.core.series.Series
        Labels of the events with which to test the classifier.
    """
    if np.isscalar(train_set):  # `train_set` was the size of training set
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            features, labels, train_size=train_set,
            random_state=random_seed)
    else:  # `train_set` was a list of object names
        X_train = features.loc[train_set]
        y_train = labels.loc[train_set]
        is_not_train_set = ~ features.index.isin(train_set)
        X_test = features[is_not_train_set]
        y_test = labels[is_not_train_set]
    return X_train, X_test, y_train, y_test


def _run_classifier(classifier_name, X_train, y_train, X_test,
                    param_grid, scoring, which_column, random_seed):
    """Note this does not have the same inputs as
    `run_several_classifiers`"""

    # Add to `classifier_map` any classifier implemeted in `snclassifier`
    classifier_map = {'svm': SVMClassifier,
                      'knn': KNNClassifier,
                      'neural_network': NNClassifier,
                      'random_forest': RFClassifier,
                      'decision_tree': DTClassifier,
                      'boost_dt': BoostDTClassifier,
                      'boost_rf': BoostRFClassifier,
                      'nb': NBClassifier,
                      'lgbm': LightGBMClassifier}

    # Initialise classifier instance
    classifier_instance = classifier_map[classifier_name](
        classifier_name=classifier_name, random_seed=random_seed)

    # Optimise classifier
    classifier_instance.optimise(X_train, y_train, param_grid=param_grid,
                                 scoring=scoring, number_cv_folds=5,
                                 metadata=None,
                                 **{'which_column': which_column})
    best_classifier = classifier_instance.classifier

    # Predict the class probabilities
    probs = best_classifier.predict_proba(X_test)

    # Returns the best fitting sklearn classifier object
    return probs, best_classifier


class BaseClassifier():
    """Base class to hold a classifier and its methods. The derived classes
    encapsulate specific methods, such as classifier optimization.
    """

    def __init__(self, classifier_name, random_seed=42):
        """Class constructor.

        Parameters:
        ----------
        classifier_name : str, optional
            Name of the classifier, which is used to save it. By default it is
            `lgbm_classifier`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
        """
        self.is_optimised = False  # the classifier was not yet optimised
        self.random_seed = random_seed
        self.classifier_name = classifier_name

    def classifier(self):
        """Returns the classifier instance.

        Returns
        -------
        classifier instance
            The classifier instance initialized with this class.
        """
        return self.classifier

    def optimise(self):
        """Optimise the classifier.
        """
        return NotImplementedError('This method should be defined on child '
                                   'classes.')

    def save_classifier(self, output_path):
        """Save the classifier instance.

        It saves the classifier in `pickle` format on the folder `output_path`
        with the name stored `self.classifier_name`.

        Parameters
        ----------
        output_path : str
            Path to the folder where the classifier will be saved.
        """
        classifier_name = self.classifier_name
        path_to_save = os.path.join(output_path, classifier_name+'.pck')
        with open(path_to_save, 'wb') as clf_path:
            pickle.dump(self, clf_path)
        print(f'Classifier saved in {path_to_save} .')

    def _compute_cv_iterable(self, cv_fold, metadata):
        """Computes a cross-validation iterable for augmented datasets.

        In order to avoid information leaks, all augmented events generated by
        the training set augmentation which derived from the same original
        event were placed in the same fold.

        Parameters
        ----------
        cv_fold : sklearn.model_selection._split.StratifiedKFold
            Stratified K-Folds cross-validator.
        metadata : pandas.DataFrame
            Metadata of the events with which to train the classifier.

        Returns
        -------
        predefined_split : sklearn.model_selection._split.PredefinedSplit
            Predefined split cross-validator.
        """
        print('Cross-validation for an augmented dataset.')
        aug_objs_original_obj = np.array(metadata.original_event).astype(str)
        fold_index = np.zeros(len(metadata), dtype=int) - 1

        # Retrieve the target/classes of the original events
        output = [list(metadata.original_event).index(elem)
                  for elem in set(metadata.original_event)]
        y_original = metadata.target[sorted(output)]
        y_original.index = metadata.original_event[sorted(output)]
        y_original = y_original.astype(int)

        indices_split = cv_fold.split(np.zeros_like(y_original), y_original)
        # Add augmented objects corresponding to the added originals
        for i, (train, test) in enumerate(indices_split):
            # Original events in fold i
            fold_objs = y_original.iloc[test].index
            # Find all augmented events derived from the original events in
            # fold i and add them to the cross-validation fold
            is_aug_objs_in_fold = np.in1d(aug_objs_original_obj, fold_objs)
            fold_index[is_aug_objs_in_fold] = i

        predefined_split = PredefinedSplit(fold_index)
        return predefined_split

    @staticmethod
    def load_classifier(path_saved_classifier):
        """Load a previously saved classifier instance.

        Parameters
        ----------
        path_saved_classifier : str
            Path to the file of the classifier instance. This also includes the
            classifier name.

        Returns
        -------
        classifier_instance : instance BaseClassifier or childs
            Instance of the class that holds a classifier and its methods.
        """
        with open(path_saved_classifier, 'rb') as input:
            classifier_instance = pickle.load(input)
        return classifier_instance

    def _is_classifier_optimised(self):
        """Check if the classifier was already optimised.

        Raises
        ------
        ValueError
            If the dataset has already been optimised, prevent it from
            suffereing a new optimisation.
        """
        if self.is_optimised is True:
            raise ValueError('The classifier was already optimised. Create a '
                             'new classifier to perform a new optimisation.')

    @property
    def random_seed(self):
        """Return the random state used in the classifier.

        Returns
        -------
        int
            Random seed used. Saving this seed allows reproducible results.
            If given, it must be between 0 and 2**32 - 1.
        """
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value):
        """Set the seed to the random state used in the classifier.

        It also initilizes the random state generator used in the classifier.

        Parameters
        ----------
        value: int, optional
            Random seed used. Saving this seed allows reproducible results.
            If given, it must be between 0 and 2**32 - 1.
        """
        # Initialise the random state
        self._rs = np.random.RandomState(value)
        self._random_seed = value

    @property
    def scoring(self):
        """Return scoring name or callable.

        Returns
        -------
        str, callable
            The scoring used to evaluate the performance of a model.
        """
        return self._scoring

    @scoring.setter
    def scoring(self, value):
        """Set the scoring to evaluate the performance of a model.

        Here we link the name of the custom scoring methods to their
        implementation.

        Parameters
        ----------
        value : {str, callable}, optional
            The strategy to evaluate the performance of a cross-validated
            model. By deafault it is `accuracy`, which calls
            `sklearn.metrics.accuracy_score`.

        Raises
        ------
        ValueError
            The scoring name is not recognised.
        """
        if value == 'auc':
            value = self._auc_score
        elif value == 'logloss':
            value = logloss_score
        self._scoring = value

    def _set_auc_score_kwargs(self, y_train, **kwargs):
        """Set the parameters needed for the AUC score.

        Parameters
        ----------
        y_train : pandas.core.series.Series
            Labels of the events with which to train the classifier.
        **kwargs : dict, optional
            If the scoring is the ROC curve AUC (`scoring='auc'`), include as
            `true_class` the desired class to optimise (e.g. Ias, which might
            correspond to class 1 or 90 depending on the dataset).
        """
        if 'true_class' in kwargs:
            self.true_class = kwargs['true_class']
            # Do some error checking here to avoid confusion in the roc curve
            # code when using it for optimisation
            class_labels = np.unique(y_train)
            self.which_column = np.where(class_labels == self.true_class)[0][0]
        else:
            self.true_class = 0
            self.which_column = 0
        if 'which_column' in kwargs:
            self.which_column = kwargs['which_column']

    def _auc_score(self, classifier, X_features, y_true):
        """A Area Under the ROC Curve (AUC) classification score.

        ROC stands for Receiver Operating Characteristic Curve.
        This custom scoring method can be used in a grid search.

        This function differs from the related `auc_score` because it uses the
        `which_column` value stored in the `BaseClassifier` instance.

        Parameters
        ----------
        classifier : classifier instance `sklearn`, `LightGBM` or
                    `BaseClassifier.child.classifier`
            Classifier.
        X_features : pandas.DataFrame or np.array
            Features of shape (n_samples, n_features).
        y_true : 1D array-like
            Ground truth (correct) labels of shape (n_samples,).

        Returns
        -------
        auc : float
            AUC score.
        """
        probs = classifier.predict_proba(X_features)
        fpr, tpr, auc = roc(pr=probs, Yt=y_true,
                            which_column=self.which_column)
        return auc  # symmetric because we want to maximise this output


class SklearnClassifier(BaseClassifier):
    def __init__(self, classifier_name='sklearn_classifier', random_seed=None,
                 **kwargs):
        """Class enclosing a Scikit-learn [1]_ classifier.

        Parameters
        ----------
        classifier_name : str, optional
            Name of the classifier, which is used to save it. By default it is
            `sklearn_classifier`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
        **kwargs : dict, optional
            Optional keywords to pass arguments into `sklearn` classifiers.

        References
        ----------
        .. [1] Pedregosa et al. "Scikit-learn: Machine Learning in Python",
        JMLR 12, pp. 2825-2830, 2011
        """
        super().__init__(classifier_name=classifier_name,
                         random_seed=random_seed)
        if 'probability' in kwargs:
            self.prob = kwargs.pop('probability')
        else:
            # By default we want to always return probabilities. User can
            # override this.
            self.prob = True

    def optimise(self, X_train, y_train, scoring, param_grid=None,
                 number_cv_folds=5, metadata=None, **kwargs):
        """Optimise the classifier.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Features of the events with which to train the classifier.
        y_train : pandas.core.series.Series
            Labels of the events with which to train the classifier.
        scoring : callable, str
            The metric used to evaluate the predictions on the test or
            validation sets. See
            `sklearn.model_selection._search.GridSearchCV` [1]_ for details on
            how to choose this input.
            `snmachine` also contains the 'logloss' and 'auc' custom scoring.
            For more details about these, see `logloss_score` and
            `auc_score`, respectively.
        param_grid : {None, dict}, optional
            Dictionary containing the parameters names (`str`) as keys and
            lists of their possible settings as values.
            If `None`, the default `param_grid` is used. This is defined in
            child classes of `SklearnClassifier`.
        number_cv_folds : int, optional
            Number of folds for cross-validation. By default it is 5.
        metadata : {None, pandas.DataFrame}, optional
            Metadata of the events with which to train the classifier.
        **kwargs : dict, optional
            If the scoring is the ROC curve AUC (`scoring='auc'`), include as
            `true_class` the desired class to optimise (e.g. Ias, which might
            correspond to class 1 or 90 depending on the dataset).

        References
        ----------
        .. [1] Pedregosa et al. "Scikit-learn: Machine Learning in Python",
        JMLR 12, pp. 2825-2830, 2011
        """
        self._is_classifier_optimised()
        time_begin = time.time()
        if scoring == 'auc':
            self._set_auc_score_kwargs(y_train=y_train, **kwargs)
        self.scoring = scoring

        if param_grid is None:
            param_grid = self.param_grid_default
            print('Using the default parameter grid optimisation.')

        # Standard grid search
        self._compute_grid_search(X_train=X_train, y_train=y_train,
                                  param_grid=param_grid,
                                  number_cv_folds=number_cv_folds,
                                  metadata=metadata)

        self.is_optimised = True
        print(f'The optimisation takes {time.time() - time_begin:.3f}s.')

    def _compute_grid_search(self, X_train, y_train, param_grid,
                             number_cv_folds, metadata):
        """Computes a standard grid search.

        This grid search is optimised using cross validation with
        `number_cv_folds` folds.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Features of the events with which to train the classifier.
        y_train : pandas.core.series.Series
            Labels of the events with which to train the classifier.
        param_grid : dict
            Dictionary containing the parameters names (`str`) as keys and
            lists of their possible settings as values.
        number_cv_folds : int
            Number of folds for cross-validation.
        metadata : pandas.DataFrame
            Metadata of the events with which to train the classifier.

        Raises
        ------
        AttributeError
            A grid must be provided in `param_grid` to perform a standard grid
            search. Thus, this input cannot be `None`.
        """
        cv_fold = StratifiedKFold(n_splits=number_cv_folds, shuffle=True,
                                  random_state=self._rs)

        if metadata is not None:
            # Whether the dataset is augmented
            is_aug = 'augmented' in metadata
        else:
            # If no metadata is provided, assume the dataset is not augmented
            is_aug = False

        if is_aug:
            cv = self._compute_cv_iterable(cv_fold, metadata)
        else:
            cv = cv_fold

        grid_search = model_selection.GridSearchCV(self.classifier,
                                                   param_grid=param_grid,
                                                   scoring=self.scoring, cv=cv)

        grid_search.fit(X_train, y_train)  # this searches through the grid

        # Save the grid search and update the saved classifier with the best
        # estimator obtained on the grid search
        self.grid_search = grid_search
        self.classifier = grid_search.best_estimator_

        # Warn if the hyperparameters are outside the default range
        best_params = grid_search.best_params_
        for param in best_params.keys():
            try:
                float(best_params[param])  # check if it is a number
                if best_params[param] <= min(param_grid[param]):
                    warnings.warn(f'Lower boundary on parameter {param} may be'
                                  f' too high. Optimum may not have been '
                                  f'reached.', Warning)
                elif best_params[param] >= max(param_grid[param]):
                    warnings.warn(f'Upper boundary on parameter {param} may be'
                                  f' too low. Optimum may not have been '
                                  f'reached.', Warning)
            except (ValueError, TypeError):
                pass  # Ignore a parameter that isn't numerical


class SVMClassifier(SklearnClassifier):
    """Uses Support vector machine (SVM) for classification.
    """
    def __init__(self, classifier_name='svm_classifier', random_seed=None,
                 **svm_params):
        """Class enclosing a Support vector machine classifier.

        This class uses the Support vector machine (SVM) implementation of
        Scikit-learn [1]_.

        Parameters
        ----------
        classifier_name : str, optional
            Name of the classifier, which is used to save it. By default it is
            `svm_classifier`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
        **svm_params : dict, optional
            Optional keywords to pass arguments into `sklearn.svm.SVC`.

        References
        ----------
        .. [1] Pedregosa et al. "Scikit-learn: Machine Learning in Python",
        JMLR 12, pp. 2825-2830, 2011
        """
        super().__init__(classifier_name=classifier_name,
                         random_seed=random_seed, **svm_params)

        # Important and necessary SVM parameters
        kernel = svm_params.pop('kernel', 'rbf')

        unoptimised_classifier = sklearn.svm.SVC(kernel=kernel,
                                                 probability=self.prob,
                                                 random_state=self._rs,
                                                 **svm_params)
        self.classifier = unoptimised_classifier
        # Store the unoptimised classifier
        self.unoptimised_classifier = unoptimised_classifier
        print(f'Created classifier of type: {self.classifier}.\n')

        # Good defaulf ranges for these parameters
        self.param_grid_default = {'C': np.logspace(-2, 5, 5),
                                   'gamma': np.logspace(-8, 3, 5)}


class KNNClassifier(SklearnClassifier):
    """Uses k-nearest neighbors vote (KNN) for classification.
    """
    def __init__(self, classifier_name='knn_classifier', random_seed=None,
                 **knn_params):
        """Class enclosing a k-nearest neighbors vote classifier.

        This class uses the k-nearest neighbors vote (KNN) implementation of
        Scikit-learn [1]_.

        Parameters
        ----------
        classifier_name : str, optional
            Name of the classifier, which is used to save it. By default it is
            `knn_classifier`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
        **knn_params : dict, optional
            Optional keywords to pass arguments into
            `sklearn.neighbors.KNeighborsClassifier`.

        References
        ----------
        .. [1] Pedregosa et al. "Scikit-learn: Machine Learning in Python",
        JMLR 12, pp. 2825-2830, 2011
        """
        super().__init__(classifier_name=classifier_name,
                         random_seed=random_seed, **knn_params)

        # Important and necessary KNN parameters
        n_neighbors = knn_params.pop('n_neighbors', 5)
        weights = knn_params.pop('weights', 'distance')

        unoptimised_classifier = sklearn.neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, **knn_params)
        self.classifier = unoptimised_classifier
        # Store the unoptimised classifier
        self.unoptimised_classifier = unoptimised_classifier
        print(f'Created classifier of type: {self.classifier}.\n')

        # Good defaulf ranges for these parameters
        self.param_grid_default = {'n_neighbors': list(range(1, 180, 5)),
                                   'weights': ['distance']}


class NNClassifier(SklearnClassifier):
    """Uses Multi-layer Perceptron for classification.
    """
    def __init__(self, classifier_name='nn_classifier',
                 random_seed=None, **nn_params):
        """Class enclosing a Multi-layer Perceptron classifier.

        This class uses the Multi-layer Perceptron classifier of a Neural
        Network (NN) implementation of Scikit-learn [1]_.

        Parameters
        ----------
        classifier_name : str, optional
            Name of the classifier, which is used to save it. By default it is
            `nn_classifier`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
        **nn_params : dict, optional
            Optional keywords to pass arguments into
            `sklearn.neural_network.MLPClassifier`.

        References
        ----------
        .. [1] Pedregosa et al. "Scikit-learn: Machine Learning in Python",
        JMLR 12, pp. 2825-2830, 2011
        """
        super().__init__(classifier_name=classifier_name,
                         random_seed=random_seed, **nn_params)

        # Important and necessary NN parameters
        layer_sizes = nn_params.pop('hidden_layer_sizes', (5, 2))
        algo = nn_params.pop('algorithm', 'adam')
        activation = nn_params.pop('activation', 'tanh')

        unoptimised_classifier = sklearn.neural_network.MLPClassifier(
            solver=algo, hidden_layer_sizes=layer_sizes, activation=activation,
            random_state=self._rs, **nn_params)
        self.classifier = unoptimised_classifier
        # Store the unoptimised classifier
        self.unoptimised_classifier = unoptimised_classifier
        print(f'Created classifier of type: {self.classifier}.\n')

        # Good defaulf ranges for these parameters
        self.param_grid_default = {
            'hidden_layer_sizes': [(layer,) for layer in range(80, 120, 5)]}


class RFClassifier(SklearnClassifier):
    """Uses random forest (RF) for classification.
    """
    def __init__(self, classifier_name='rf_classifier',
                 random_seed=None, **rf_params):
        """Class enclosing a random forest classifier.

        This class uses the random forest classifier (RF) implementation of
        Scikit-learn [1]_.

        Parameters
        ----------
        classifier_name : str, optional
            Name of the classifier, which is used to save it. By default it is
            `rf_classifier`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
        **rf_params : dict, optional
            Optional keywords to pass arguments into
            `sklearn.ensemble.RandomForestClassifier`.

        References
        ----------
        .. [1] Pedregosa et al. "Scikit-learn: Machine Learning in Python",
        JMLR 12, pp. 2825-2830, 2011
        """
        super().__init__(classifier_name=classifier_name,
                         random_seed=random_seed, **rf_params)
        unoptimised_classifier = sklearn.ensemble.RandomForestClassifier(
            random_state=self._rs, **rf_params)
        self.classifier = unoptimised_classifier
        # Store the unoptimised classifier
        self.unoptimised_classifier = unoptimised_classifier
        print(f'Created classifier of type: {self.classifier}.\n')

        # Good defaulf ranges for these parameters
        self.param_grid_default = {'n_estimators': list(range(200, 900, 100)),
                                   'criterion': ['gini', 'entropy']}


class DTClassifier(SklearnClassifier):
    """Uses a decision tree (DT) for classification.
    """
    def __init__(self, classifier_name='dt_classifier',
                 random_seed=None, **dt_params):
        """Class enclosing a decision tree classifier.

        This class uses the decision tree classifier (DT) implementation of
        Scikit-learn [1]_.

        Parameters
        ----------
        classifier_name : str, optional
            Name of the classifier, which is used to save it. By default it is
            `dt_classifier`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
        **dt_params : dict, optional
            Optional keywords to pass arguments into
            `sklearn.tree.DecisionTreeClassifier`.

        References
        ----------
        .. [1] Pedregosa et al. "Scikit-learn: Machine Learning in Python",
        JMLR 12, pp. 2825-2830, 2011
        """
        super().__init__(classifier_name=classifier_name,
                         random_seed=random_seed, **dt_params)
        unoptimised_classifier = sklearn.tree.DecisionTreeClassifier(
            random_state=self._rs, **dt_params)
        self.classifier = unoptimised_classifier
        # Store the unoptimised classifier
        self.unoptimised_classifier = unoptimised_classifier
        print(f'Created classifier of type: {self.classifier}.\n')

        # Good defaulf ranges for these parameters
        self.param_grid_default = {'criterion': ['gini', 'entropy'],
                                   'min_samples_leaf': list(range(1, 400, 25))}


class BoostDTClassifier(SklearnClassifier):
    """Uses boosted decision trees for classification.
    """
    def __init__(self, classifier_name='boost_dt_classifier',
                 random_seed=None, **boost_dt_params):
        """Class enclosing a boosted decision tree classifier.

        This class uses decision trees as the base estimator of the boosted
        ensemble AdaBoost classifier. This class implements the algorithm
        AdaBoost-SAMME [1]_ through Scikit-learn [2]_.

        Parameters
        ----------
        classifier_name : str, optional
            Name of the classifier, which is used to save it. By default it is
            `boost_dt_classifier`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
        **boost_dt_params : dict, optional
            Optional keywords to pass arguments into
            `sklearn.ensemble.AdaBoostClassifier`.

        References
        ----------
        .. [1] Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009
        .. [2] Pedregosa et al. "Scikit-learn: Machine Learning in Python",
        JMLR 12, pp. 2825-2830, 2011
        """
        super().__init__(classifier_name=classifier_name,
                         random_seed=random_seed, **boost_dt_params)
        unoptimised_classifier = sklearn.ensemble.AdaBoostClassifier(
            random_state=self._rs, **boost_dt_params)
        self.classifier = unoptimised_classifier
        # Store the unoptimised classifier
        self.unoptimised_classifier = unoptimised_classifier
        print(f'Created classifier of type: {self.classifier}.\n')

        # Good defaulf ranges for these parameters
        base_estimators = [sklearn.tree.DecisionTreeClassifier(
            criterion='entropy',
            min_samples_leaf=leafs) for leafs in range(5, 55, 10)]
        self.param_grid_default = {'base_estimator': base_estimators,
                                   'n_estimators': list(range(5, 85, 10))}


class BoostRFClassifier(SklearnClassifier):
    """Uses boosted random forests for classification.
    """
    def __init__(self, classifier_name='boost_rf_classifier',
                 random_seed=None, **boost_rf_params):
        """Class enclosing a boosted random forest classifier.

        This class uses random forests as the base estimator of the boosted
        ensemble AdaBoost classifier. This class implements the algorithm
        AdaBoost-SAMME [1]_ through Scikit-learn [2]_.

        Parameters
        ----------
        classifier_name : str, optional
            Name of the classifier, which is used to save it. By default it is
            `boost_rf_classifier`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
        **boost_rf_params : dict, optional
            Optional keywords to pass arguments into
            `sklearn.ensemble.AdaBoostClassifier`.

        References
        ----------
        .. [1] Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009
        .. [2] Pedregosa et al. "Scikit-learn: Machine Learning in Python",
        JMLR 12, pp. 2825-2830, 2011
        """
        super().__init__(classifier_name=classifier_name,
                         random_seed=random_seed, **boost_rf_params)
        unoptimised_classifier = sklearn.ensemble.AdaBoostClassifier(
            random_state=self._rs, **boost_rf_params)
        self.classifier = unoptimised_classifier
        # Store the unoptimised classifier
        self.unoptimised_classifier = unoptimised_classifier
        print(f'Created classifier of type: {self.classifier}.\n')

        # Good defaulf ranges for these parameters
        # This is a strange boosted random forest classifier that Max came up
        # that works quite well, but is likely biased in general
        base_estimators = [
            sklearn.ensemble.RandomForestClassifier(400, 'entropy'),
            sklearn.ensemble.RandomForestClassifier(600, 'entropy')]
        self.param_grid_default = {'base_estimator': base_estimators,
                                   'n_estimators': list([2, 3, 5, 10])}


class NBClassifier(SklearnClassifier):
    """Uses Gaussian Naive Bayes for classification.
    """
    def __init__(self, classifier_name='nb_classifier',
                 random_seed=None, **nb_params):
        """Class enclosing a Gaussian Naive Bayes classifier.

        This class uses the Gaussian Naive Bayes implementation of
        Scikit-learn [1]_.

        Parameters
        ----------
        classifier_name : str, optional
            Name of the classifier, which is used to save it. By default it is
            `nb_classifier`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
        **nb_params : dict, optional
            Optional keywords to pass arguments into
            `sklearn.naive_bayes.GaussianNB`.

        References
        ----------
        .. [1] Pedregosa et al. "Scikit-learn: Machine Learning in Python",
        JMLR 12, pp. 2825-2830, 2011
        """
        super().__init__(classifier_name=classifier_name,
                         random_seed=random_seed, **nb_params)
        unoptimised_classifier = sklearn.naive_bayes.GaussianNB(**nb_params)
        self.classifier = unoptimised_classifier
        # Store the unoptimised classifier
        self.unoptimised_classifier = unoptimised_classifier
        print(f'Created classifier of type: {self.classifier}.\n')

        # Good defaulf ranges for these parameters
        print('This class has no default hyperparameter range.')
        self.param_grid_default = {}


class LightGBMClassifier(BaseClassifier):
    """Uses a tree based learning algorithm for classification from LightGBM.
    """

    def __init__(self, classifier_name='lgbm_classifier', random_seed=None,
                 **lgb_params):
        """Class enclosing a `LightGBM` classifier.

        This class uses the tree based learning algorithms implemented in
        LightGBM [1]_.

        Parameters
        ----------
        classifier_name : str, optional
            Name of the classifier, which is used to save it. By default it is
            `lgbm_classifier`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
        **lgb_params : dict, optional
            Optional keywords to pass arguments into `lgb.LGBMClassifier`.

        References
        ----------
        .. [1] Guolin Ke et al. “LightGBM: A Highly Efficient Gradient
        Boosting Decision Tree.” Advances in Neural Information Processing
        Systems 30 (NIPS 2017), pp. 3149-3157.
        """
        super().__init__(classifier_name=classifier_name,
                         random_seed=random_seed)
        unoptimised_classifier = lgb.LGBMClassifier(
            random_state=self._random_seed, **lgb_params)
        self.classifier = unoptimised_classifier
        # Store the unoptimised classifier
        self.unoptimised_classifier = unoptimised_classifier
        print(f'Created classifier of type: {self.classifier}.\n')

    def optimise(self, X_train, y_train, scoring, param_grid=None,
                 number_cv_folds=5, metadata=None, **kwargs):
        """Optimise the classifier.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Features of the events with which to train the classifier.
        y_train : pandas.core.series.Series
            Labels of the events with which to train the classifier.
        scoring : callable, str
            The metric used to evaluate the predictions on the test or
            validation sets. See
            `sklearn.model_selection._search.GridSearchCV` [1]_ for details on
            how to choose this input.
            `snmachine` also contains the 'logloss' and 'auc' custom scoring.
            For more details about these, see `logloss_score` and
            `auc_score`, respectively.
        param_grid : {None, dict}, optional
            Dictionary containing the parameters names (`str`) as keys and
            lists of their possible settings as values.
            If `None`, it performs a specific hyperparameter optimisation that
            is faster than optimising a high dimensional grid through a
            standard grid search. See Notes for the details of this
            hyperparameter optimisation.
        number_cv_folds : int, optional
            Number of folds for cross-validation. By default it is 5.
        metadata : {None, pandas.DataFrame}, optional
            Metadata of the events with which to train the classifier.
        **kwargs : dict, optional
            If the scoring is the ROC curve AUC (`scoring='auc'`), include as
            `true_class` the desired class to optimise (e.g. Ias, which might
            correspond to class 1 or 90 depending on the dataset).

        Notes
        -----
        The hyperparameter optimisation used as deafult is: First, optimise
        each hyperparameter individually using a 1D grid, keeping the other
        hyperparameters at default values. Then, construct a higher
        dimensional grid containing all the hyperparameters with three
        possible values for each hyperparameter informed by the earlier 1D
        optimization. Finally, optimise this higher dimensional grid through a
        standard grid search.
        The values of the hyperparameters in this optimisation were fine-tuned
        to PLAsTiCC. Thus, this combination will probably not work for a
        general problem.

        References
        ----------
        .. [1] Pedregosa et al. "Scikit-learn: Machine Learning in Python",
        JMLR 12, pp. 2825-2830, 2011
        """
        self._is_classifier_optimised()
        time_begin = time.time()
        if scoring == 'auc':
            self._set_auc_score_kwargs(y_train=y_train, **kwargs)
        self.scoring = scoring

        use_fast_optimisation = param_grid is None

        # The hyperparameter optimisation of `use_fast_optimisation` is
        # described in the Notes of the docstring
        if use_fast_optimisation is True:
            self._compute_fast_optimisation(X_train=X_train, y_train=y_train,
                                            number_cv_folds=number_cv_folds,
                                            metadata=metadata)
        # Standard grid search
        else:
            self._compute_grid_search(X_train=X_train, y_train=y_train,
                                      param_grid=param_grid,
                                      number_cv_folds=number_cv_folds,
                                      metadata=metadata)

        self.is_optimised = True
        print(f'The optimisation takes {time.time() - time_begin:.3f}s.')

    def _compute_grid_search(self, X_train, y_train, param_grid,
                             number_cv_folds, metadata):
        """Computes a standard grid search.

        This grid search is optimised using cross validation with
        `number_cv_folds` folds.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Features of the events with which to train the classifier.
        y_train : pandas.core.series.Series
            Labels of the events with which to train the classifier.
        param_grid : dict
            Dictionary containing the parameters names (`str`) as keys and
            lists of their possible settings as values.
        number_cv_folds : int
            Number of folds for cross-validation.
        metadata : pandas.DataFrame
            Metadata of the events with which to train the classifier.

        Raises
        ------
        AttributeError
            A grid must be provided in `param_grid` to perform a standard grid
            search. Thus, this input cannot be `None`.
        """
        if param_grid is None:
            raise AttributeError('To perform a standard grid search, you must '
                                 'provide a grid in `param_grid`.')

        cv_fold = StratifiedKFold(n_splits=number_cv_folds, shuffle=True,
                                  random_state=self._rs)

        if metadata is not None:
            # Whether the dataset is augmented
            is_aug = 'augmented' in metadata
        else:
            # If no metadata is provided, assume the dataset is not augmented
            is_aug = False

        if is_aug:
            cv = self._compute_cv_iterable(cv_fold, metadata)
        else:
            cv = cv_fold

        grid_search = model_selection.GridSearchCV(self.classifier,
                                                   param_grid=param_grid,
                                                   scoring=self.scoring, cv=cv)
        grid_search.fit(X_train, y_train)  # this searches through the grid

        # Save the grid search and update the saved classifier with the best
        # estimator obtained on the grid search
        self.grid_search = grid_search
        self.classifier = grid_search.best_estimator_

    def _compute_fast_optimisation(self, X_train, y_train, number_cv_folds,
                                   metadata):
        """Optimises each parameter individually and then uses a grid search.

        First, optimise each hyperparameter individually using a 1D grid,
        keeping the other hyperparameters at default values. Then, construct
        a higher dimensional grid containing all the hyperparameters with
        three possible values for each hyperparameter informed by the
        earlier 1D optimization. Finally, optimise this higher dimensional
        grid through a standard grid search.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Features of the events with which to train the classifier.
        y_train : pandas.core.series.Series
            Labels of the events with which to train the classifier.
        scoring : callable, str
            The metric used to evaluate the predictions on the test or
            validation sets. See
            `sklearn.model_selection._search.GridSearchCV` [1]_ for details on
            how to choose this parameter.
        number_cv_folds : int
            Number of folds for cross-validation.
        metadata : pandas.DataFrame
            Metadata of the events with which to train the classifier.

        Notes
        -----
        The values of the hyperparameters in this optimisation were fine-tuned
        to PLAsTiCC. Thus, this combination will probably not work for a
        general problem.

        References
        ----------
        .. [1] Pedregosa et al. "Scikit-learn: Machine Learning in Python",
        JMLR 12, pp. 2825-2830, 2011
        """
        print('Using the default parameter grid optimisation.')
        # This is the grid used to optimise each hyperparameter individually
        param_grid = {'num_leaves': np.arange(10, 55, 5),
                      'learning_rate': np.logspace(-3, -.01, 50),
                      'n_estimators': np.arange(25, 120, 10),
                      'min_child_samples': np.arange(20, 80, 10),
                      'max_depth': np.arange(1, 20, 3),
                      'min_split_gain': np.linspace(0., 2., 21)}

        best_param = {}  # to refister the best value of the 1D optimisation
        for param in param_grid.keys():
            new_param_grid = {param: param_grid[param]}
            print(f'Optimise parameter {param}.')

            # Optimise `param` with the other hyperparameters at default values
            self._compute_grid_search(X_train=X_train, y_train=y_train,
                                      param_grid=new_param_grid,
                                      number_cv_folds=number_cv_folds,
                                      metadata=metadata)
            # Register the best value
            best_param[param] = self.grid_search.best_params_[param]

        # New grid to optimise all the hyperparameters simultaneously
        param_grid = self._construct_6d_grid(best_param)
        # Optimise `param` with the other hyperparameters at default values
        print(f'Final optimisation - grid \n{param_grid}')
        self._compute_grid_search(X_train=X_train, y_train=y_train,
                                  param_grid=param_grid,
                                  number_cv_folds=number_cv_folds,
                                  metadata=metadata)

    @staticmethod
    def _construct_6d_grid(best_param):
        """Constructs a 6D grid containing all the hyperparameters.

        This function constructs a six-dimensional grid containing the LightGBM
        hyperparameters `num_leaves`, `learning_rate`, `n_estimators`,
        `min_child_samples`, `max_depth`, and `min_split_gain` with three
        possible values for each hyperparameter informed by the 1D
        optimisation performed in
        `LightGBMClassifier._compute_fast_optimisation`.

        Parameters
        ----------
        best_param : dict
            Dictionary containing the values of the hyperparameters
            `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`,
            `max_depth`, and `min_split_gain`, obtained after the 1D
            optimisation performed in
            `LightGBMClassifier._compute_fast_optimisation`.

        Notes
        -----
        The values of the hyperparameters in this optimisation were fine-tuned
        to PLAsTiCC. Thus, this combination will probably not work for a
        general problem.
        """
        param_grid = {}

        # The bellow values were informed by exploring the optmisation of
        # several classifiers. For a more detailed optimisation, run
        # `LightGBMClassifier._compute_grid_search`
        param = 'num_leaves'
        param_best_value = best_param[param]
        min_value, max_value = 10, 50
        if param_best_value <= min_value:
            param_grid[param] = [param_best_value, 15, max_value]
        elif param_best_value >= max_value:
            param_grid[param] = [min_value, param_best_value,
                                 param_best_value + 5]
        elif (param_best_value > min_value) and (param_best_value < max_value):
            param_grid[param] = [min_value, param_best_value, max_value]

        param = 'learning_rate'
        param_best_value = best_param[param]
        min_value, max_value = .04, .24
        if abs(param_best_value - min_value) < 1E-3:
            param_grid[param] = [.005, param_best_value, max_value]
        elif param_best_value < min_value:
            param_grid[param] = [max(param_best_value - .005, 0.001),
                                 param_best_value, max_value]
        elif abs(param_best_value - max_value) < 1E-3:
            param_grid[param] = [min_value, param_best_value, .7]
        elif param_best_value > max_value:
            param_grid[param] = [min_value, param_best_value,
                                 param_best_value + .1]
        elif (param_best_value > min_value) and (param_best_value < max_value):
            param_grid[param] = [min_value, param_best_value, max_value]

        param = 'n_estimators'
        param_best_value = best_param[param]
        min_value, max_value = 25, 115
        if abs(param_best_value - min_value) <= 5:
            param_grid[param] = [param_best_value, 45, max_value]
        elif param_best_value < min_value:
            param_grid[param] = [param_best_value, 20, max_value]
        elif param_best_value >= max_value:
            param_grid[param] = [min_value, 45, param_best_value]
        elif (param_best_value > min_value) and (param_best_value < max_value):
            param_grid[param] = [min_value, param_best_value, max_value]

        param = 'min_child_samples'
        param_best_value = best_param[param]
        min_value, max_value = 20, 70
        if abs(param_best_value - min_value) <= 5:
            param_grid[param] = [10, param_best_value, max_value]
        elif param_best_value < min_value:
            param_grid[param] = [param_best_value, 20, max_value]
        elif param_best_value == max_value:
            param_grid[param] = [min_value, 45, param_best_value]
        elif (param_best_value > min_value) and (param_best_value < max_value):
            param_grid[param] = [min_value, param_best_value, max_value]

        param = 'max_depth'
        param_best_value = best_param[param]
        min_value, max_value = 3, 19
        if param_best_value <= min_value:
            param_grid[param] = [param_best_value, 5, max_value]
        elif param_best_value >= max_value:
            param_grid[param] = [min_value, 16, param_best_value]
        elif (param_best_value > min_value) and (param_best_value < max_value):
            param_grid[param] = [min_value, param_best_value, max_value]

        param = 'min_split_gain'
        param_best_value = best_param[param]
        min_value, max_value = 0., .7
        if abs(param_best_value - min_value) < 1E-3:
            param_grid[param] = [param_best_value, .1, max_value]
        elif abs(param_best_value - max_value) < 1E-3:
            param_grid[param] = [min_value, param_best_value, .8]
        elif param_best_value > max_value:
            param_grid[param] = [min_value, param_best_value,
                                 param_best_value + .1]
        elif (param_best_value > min_value) and (param_best_value < max_value):
            param_grid[param] = [min_value, param_best_value, max_value]

        return param_grid
