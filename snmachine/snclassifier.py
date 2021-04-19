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
import numpy as np
import sklearn

from astropy.table import Table, join, unique
from functools import partial
from multiprocessing import Pool
from scipy.integrate import trapz
from sklearn import model_selection
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix as sklearn_cm
from sklearn.model_selection import PredefinedSplit, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from utils import plasticc_utils

if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
else:
    import matplotlib.pyplot as plt

# This allows the user to easily loop through all possible classifiers
choice_of_classifiers = ['svm', 'knn', 'random_forest', 'decision_tree',
                         'boost_dt', 'boost_rf', 'nb']
# boost_rf is a set of boosted random forests which Max came up with.

try:
    from sklearn.neural_network import MLPClassifier
    choice_of_classifiers.append('neural_network')
except ImportError:
    print('Neural networks not available in this version of scikit-learn. '
          'Neural networks are available from development version 0.18.')


# Costum scoring/metrics functions ##
def logloss_score(classifier, X_features, y_true):
    """PLAsTiCC logloss classification score.

    This costum scoring method can be used in a grid search.

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
    This costum scoring method can be used in a grid search.

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
        probs_1 = probs[:, true_class-min_class]
    # Used by `optimised_classify`
    elif len(pr.shape) > 1 and which_column != -1:
        if which_column >= np.shape(probs)[1]:
            sys.exit(f'`which_column` must be -1 or between 0 and '
                     f'{np.shape(probs)[1]-1}.')  # the error was not working
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
        Line width
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


def compute_confusion_matrix(Yfit, Ytrue):
    """Computes the confusion matrix.

    Wraps the scikit-learn routine to compute the confusion matrix.

    Parameters
    ----------
    Yfit : list
         predicted classes for the test set
    Ytrue : list
         true classes for the test set

    Returns
    -------
    confusion_matrix: numpy.array
    """
    return sklearn_cm(y_true=Ytrue, y_pred=Yfit)


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
    """
    Calculate an F1 score for many probability threshold increments
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
    # TN = (~preds & ~Y_bool).sum(axis=0)  # not used in this function
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


def FoM(pr,  Yt, true_class=1, full_output=False):
    """
    Calculate a Kessler FoM for many probability threshold increments
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
    # TN = (~preds & ~Y_bool).sum(axis=0)  # not used in this function
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


class OptimisedClassifier():
    """Implements an optimised classifier (although it can be run without
    optimisation). Equipped with interfaces to several sklearn classes and
    functions.
    """

    NB_param_dict = {}
    KNN_param_dict = {'n_neighbors': list(range(1, 180, 5)),
                      'weights': ['distance']}
    SVM_param_dict = {'C': np.logspace(-2, 5, 5),
                      'gamma': np.logspace(-8, 3, 5)}
    NN_param_dict = {'hidden_layer_sizes': [(l,) for l in range(80, 120, 5)]}

    DT_param_dict = {'criterion': ['gini', 'entropy'],
                     'min_samples_leaf': list(range(1, 400, 25))}
    RF_param_dict = {'n_estimators': list(range(200, 900, 100)),
                     'criterion': ['gini', 'entropy']}
    ests = [
        DecisionTreeClassifier(criterion='entropy',
                               min_samples_leaf=l) for l in range(5, 55, 10)]
    Boost_param_dict = {'base_estimator': ests,
                        'n_estimators': list(range(5, 85, 10))}
    # This is a strange boosted random forest classifier that Max came up that
    # works quite well, but is likely biased in general
    Boost_RF_param_dict = {
        'base_estimator': [RandomForestClassifier(400, 'entropy'),
                           RandomForestClassifier(600, 'entropy')],
        'n_estimators': list([2, 3, 5, 10])}

    # Dictionary to hold good default ranges for parameters for each
    # classifier.
    params = {'svm': SVM_param_dict, 'knn': KNN_param_dict,
              'decision_tree': DT_param_dict, 'random_forest': RF_param_dict,
              'boost_dt': Boost_param_dict, 'boost_rf': Boost_RF_param_dict,
              'nb': NB_param_dict, 'neural_network': NN_param_dict}

    def __init__(self, classifier, optimise=True,  **kwargs):
        """Wrapper around sklearn classifiers

        Parameters
        ----------
        classifier : str or sklearn.Classifier
            Either a string (one of choice_of_classifiers) or sklearn
            classifier object.
        optimise : bool, optional
            Whether or not to optimise the parameters.
        kwargs : optional
            Keyword arguments passed directly to the classifier.
        """

        self.classifier_name = classifier
        if isinstance(classifier, basestring):
            # Choose from available classifiers, with default parameters best
            # for SNe (unless overridden)
            if 'probability' in kwargs:
                self.prob = kwargs.pop('probability')
            else:
                # By default we want to always return probabilities. User can
                # override this.
                self.prob = True

            try:
                if classifier == 'svm':
                    if 'kernel' in kwargs:
                        kern = kwargs.pop('kernel')  # Removes this from kwargs
                    else:
                        kern = 'rbf'
                    self.clf = svm.SVC(kernel=kern, probability=self.prob,
                                       **kwargs)

                elif classifier == 'knn':
                    if 'n_neighbours' in kwargs:
                        n = kwargs.pop('n_neighbours')
                    else:
                        n = 5
                    if 'weights' in kwargs:
                        wgts = kwargs.pop('weights')
                    else:
                        wgts = 'distance'
                    self.clf = neighbors.KNeighborsClassifier(n_neighbors=n,
                                                              weights=wgts,
                                                              **kwargs)

                elif classifier == 'random_forest':
                    self.clf = RandomForestClassifier(**kwargs)
                elif classifier == 'decision_tree':
                    self.clf = DecisionTreeClassifier(**kwargs)
                elif classifier == 'boost_dt' or classifier == 'boost_rf':
                    self.clf = AdaBoostClassifier(**kwargs)
                elif classifier == 'nb':
                    self.clf = GaussianNB(**kwargs)
                elif classifier == 'neural_network':
                    if 'hidden_layer_sizes' in kwargs:
                        layer_sizes = kwargs.pop('hidden_layer_sizes')
                    else:
                        layer_sizes = (5, 2)
                    if 'algorithm' in kwargs:
                        algo = kwargs.pop('algorithm')
                    else:
                        algo = 'adam'
                    if 'activation' in kwargs:
                        activation = kwargs.pop('activation')
                    else:
                        activation = 'tanh'
                    self.clf = MLPClassifier(solver=algo,
                                             hidden_layer_sizes=layer_sizes,
                                             activation=activation,  **kwargs)
                elif classifier == 'multiple_classifier':
                    pass
                else:
                    print(f'Requested classifier not recognised.\nChoice of '
                          f'built-in classifiers: {choice_of_classifiers}.')
                    sys.exit(0)

            except TypeError:
                # Gracefully catch errors of sending the wrong kwargs to the
                # classifiers
                raise AttributeError(f'One of the kwargs \n{kwargs.keys()}\n'
                                     f'does not belong to classifier '
                                     f'{classifier}.')
                sys.exit(0)

        else:
            # This is already some sklearn classifier or an object that
            # behaves like one
            self.clf = classifier
        print(f'Created classifier of type: {self.clf}.\n')

    def classify(self, X_train, y_train, X_test):
        """Run unoptimised classifier with initial parameters.

        Parameters
        ----------
        X_train : array
            Array of training features of shape (n_train,n_features).
        y_train : array
            Array of known classes of shape (n_train).
        X_test : array
            Array of validation features of shape (n_test,n_features).

        Returns
        -------
        Yfit : array
            Predicted classes for X_test
        probs : array
            (If self.prob=True) Probability for each object to belong to each
            class.
        """
        self.clf.fit(X_train, y_train)
        if self.prob:  # Probabilities requested
            probs = self.clf.predict_proba(X_test)

            # Each object is assigned the class with highest probability
            Yfit = probs.argmax(axis=1)

            classes = np.unique(y_train)
            classes.sort()
            Yfit = classes[Yfit]
            return Yfit, probs
        else:
            Yfit = self.clf.predict(X_test)
            # scr=self.clf.score(X_test, y_test)
            return Yfit

    def _custom_auc_score(self, estimator, X, Y):
        """Custom scoring method for use with GridSearchCV.

        Parameters
        ----------
        estimator : sklearn.Classifier
            The current classifier (used by GridSearchCV).
        X : array
            Array of training features of shape (n_train,n_features).
        Y : array
            Array of known classes of shape (n_train)

        Returns
        -------
        auc : float
            AUC score

        """
        probs = estimator.predict_proba(X)

        fpr, tpr, auc = roc(probs, Y, which_column=self.which_column)
        return auc

    def _custom_logloss_score(self, estimator, X, Y):
        """Custom scoring method for use with GridSearchCV.

        Parameters
        ----------
        estimator : sklearn.Classifier
            The current classifier (used by GridSearchCV).
        X : array
            Array of training features of shape (n_train,n_features).
        Y : array
            Array of known classes of shape (n_train).

        Returns
        -------
        float
            Symmetric of the PLASTICC logloss score. The symmetric is returned
            because we want a funtion to maximise and the optimal result of
            the logloss is its minimum (logloss=0).
        """
        probs = estimator.predict_proba(X)
        logloss = plasticc_utils.plasticc_log_loss(Y, probs)
        return -logloss  # symmetric because we want to maximise this output

    def optimised_classify(self, X_train, y_train, X_test,
                           scoring_func='accuracy', **kwargs):
        """Run optimised classifier using grid search with cross validation to
        choose optimal classifier parameters.

        Parameters
        ----------
        X_train : array
            Array of training features of shape (n_train,n_features)
        y_train : array
            Array of known classes of shape (n_train)
        X_test : array
            Array of validation features of shape (n_test,n_features)
        scoring_func : string, optional
            Choice of which function to optimise with when optimising
            hyperparameters. Currently implemented are "auc" for the ROC AUC
            score (then "true_class" should be given as well as a kwarg),
            "logloss" for the PLASTICC logloss function or accuracy from
            sklearn (default).
        params : dict, optional
            Allows the user to specify which parameters and over what ranges
            to optimise. If not set, defaults will be used.
        true_class : int, optional
            The class determined to be the desired class (e.g. Ias, which
            might correspond to class 1). This allows the user to optimise for
            different classes (based on ROC curve AUC).

        Returns
        -------
        Yfit : array
            Predicted classes for X_test
        probs : array
            (If self.prob=True) Probability for each object to belong to each
            class.
        """
        if 'params' in kwargs:
            params = kwargs['params']
        else:
            params = self.params[self.classifier_name]

        if 'true_class' in kwargs:
            self.true_class = kwargs['true_class']
            # Do some error checking here to avoid confusion in the roc curve
            # code when using it for optimisation
            class_labels = np.unique(y_train)
            self.which_column = np.where(class_labels == self.true_class)[0][0]
        else:
            self.true_class = 0
            self.which_column = 0

        if scoring_func == "auc":
            scoring = self._custom_auc_score
        elif scoring_func == "logloss":
            scoring = self._custom_logloss_score
        else:
            scoring = "accuracy"

        self.clf = model_selection.GridSearchCV(self.clf, params,
                                                scoring=scoring, cv=5)

        self.clf.fit(X_train, y_train)  # This actually does the grid search
        best_params = self.clf.best_params_
        print('Optimised parameters:', best_params)

        for k in best_params.keys():
            # This is the safest way to check if something is a number
            try:
                float(best_params[k])
                if best_params[k] <= min(params[k]):
                    print()
                    print(f'WARNING: Lower boundary on parameter {k} may be '
                          f'too high. Optimum may not have been reached.')
                    print()
                elif best_params[k] >= max(params[k]):
                    print()
                    print(f'WARNING: Upper boundary on parameter {k} may be '
                          f'too low. Optimum may not have been reached.')
                    print()

            except (ValueError, TypeError):
                pass  # Ignore a parameter that isn't numerical

        if self.prob:  # Probabilities requested
            probs = self.clf.predict_proba(X_test)

            # Each object is assigned the class with highest probability
            Yfit = probs.argmax(axis=1).tolist()

            classes = np.unique(y_train)
            classes.sort()
            Yfit = classes[Yfit]
            return Yfit, probs
        else:
            Yfit = self.clf.predict(X_test)
            return Yfit


def __call_classifier(classifier, X_train, y_train, X_test, param_dict,
                      return_classifier):
    """Specifically designed to run with multiprocessing."""

    c = OptimisedClassifier(classifier)
    if classifier in param_dict.keys():
        y_fit, probs = c.optimised_classify(X_train, y_train, X_test,
                                            params=param_dict[classifier])
    else:
        y_fit, probs = c.optimised_classify(X_train, y_train, X_test)

    if return_classifier:
        # Returns the best fitting sklearn classifier object
        return probs, c.clf
    else:
        return probs


def run_pipeline(features, types, output_name='', columns=[],
                 classifiers=['nb', 'knn', 'svm', 'neural_network',
                              'boost_dt'],
                 training_set=0.3, param_dict={}, number_processes=1,
                 scale=True, plot_roc_curve=True, return_classifier=False,
                 classifiers_for_cm_plots=[], type_dict=None, seed=1234):
    """
    Utility function to classify a dataset with a number of classification
    methods. This does assume your test set has known values to compare
    against. Returns, if requested, the classifier objects to run on future
    test sets.

    Parameters
    ----------
    features : astropy.table.Table or array
        Features either in the form of a table or array
    types : astropy.table.Table or array
        Classes, either in the form of a table or array
    output_name : str, optional
        Full root path and name for output (e.g. '<output_path>/salt2-run-')
    columns : list, optional
        If you want to run a subset of columns
    classifiers : list, optional
        Which available ML classifiers to use
    training_set : float or list, optional
        If a float, this is the fraction of objects that will be used as
        training set. If a list, it's assumed these are the ID's of the
        objects to be used.
    param_dict : dict, optional
        Use to run different ranges of hyperparameters for the classifiers
        when optimising.
    number_processes : int, optional
        Number of processors for multiprocessing (shared memory only). Each
        classifier will then be run in parallel.
    scale : bool, optional
        Rescale features using sklearn's preprocessing Scalar class (highly
        recommended this is True).
    plot_roc_curve : bool, optional
        Whether or not to plot the ROC curve at the end
    return_classifier : bool, optional
        Whether or not to return the actual classifier objects (due to the
        limitations of multiprocessing, this can't be done in parallel at the
        moment).

    Returns
    -------
    dict
        (If return_classifier=True) Dictionary of fitted sklearn Classifier
        objects.

    """
    t1 = time.time()

    if type_dict is None:
        type_dict = {value: value for value in range(len(unique(types,
                                                                keys='Type')))}

    if isinstance(features, Table):
        # The features are in astropy table format and must be converted to a
        # numpy array before passing to sklearn

        # We need to make sure we match the correct Y values to X values. The
        # safest way to do this is to make types an astropy table as well.

        if not isinstance(types, Table):
            types = Table(data=[features['Object'], types],
                          names=['Object', 'Type'])
        feats = join(features, types, 'Object')

        if len(columns) == 0:
            columns = feats.columns[1:-1]

        # Split into training and validation sets
        if np.isscalar(training_set):
            objs = feats['Object']
            objs = np.random.RandomState(seed=seed).permutation(objs)
            training_set = objs[:(int)(training_set*len(objs))]

        # Otherwise a training set has already been provided as a list of
        # object names and we can continue
        feats_train = feats[np.in1d(feats['Object'], training_set)]
        feats_test = feats[~np.in1d(feats['Object'], training_set)]

        X_train = np.array([feats_train[c] for c in columns]).T
        y_train = np.array(feats_train['Type'])
        X_test = np.array([feats_test[c] for c in columns]).T
        y_test = np.array(feats_test['Type'])

    else:
        # Otherwise the features are already in the form of a numpy array
        if np.isscalar(training_set):
            inds = np.random.RandomState(seed=seed).permutation(
                range(len(features)))
            train_inds = inds[:(int)(len(inds)*training_set)]
            test_inds = inds[(int)(len(inds)*training_set):]
        else:
            # We assume the training set has been provided as indices
            train_inds = training_set
            test_inds = range(len(types))[~np.in1d(range(
                len(types)), training_set)]

        X_train = features[train_inds]
        y_train = types[train_inds]
        X_test = features[test_inds]
        y_test = types[test_inds]

    # Rescale the data (highly recommended)
    if scale:
        scaler = StandardScaler()
        scaler.fit(np.vstack((X_train, X_test)))
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    probabilities = {}
    classifier_objects = {}

    if number_processes > 1 and return_classifier:
        print("Due to limitations with python's multiprocessing module, "
              "classifier objects cannot be returned if multiple processors "
              "are used. Continuing serially...")
        print()

    if number_processes > 1 and not return_classifier:
        partial_func = partial(__call_classifier, X_train=X_train,
                               y_train=y_train, X_test=X_test,
                               param_dict=param_dict, return_classifier=False)
        p = Pool(number_processes, maxtasksperchild=1)
        result = p.map(partial_func, classifiers)

        for i in range(len(result)):
            cls = classifiers[i]
            probabilities[cls] = result[i]
    else:
        for cls in classifiers:
            retval = __call_classifier(cls, X_train, y_train, X_test,
                                       param_dict, return_classifier)
            if return_classifier:
                probabilities[cls] = retval[0]
                classifier_objects[cls] = retval[1]
            else:
                probabilities[cls] = retval[0]

    for i in range(len(classifiers)):
        cls = classifiers[i]
        probs = probabilities[cls]
        fpr, tpr, auc = roc(probs, y_test, true_class=1)
        fom, thresh_fom = FoM(probs, y_test, true_class=1, full_output=False)

        print(f'Classifier {cls}: AUC = {auc} ; FoM = {fom}.')

        if i == 0:
            FPR = fpr
            TPR = tpr
            AUC = [auc]
        else:
            FPR = np.column_stack((FPR, fpr))
            TPR = np.column_stack((TPR, tpr))
            AUC.append(auc)

        # Only save if an output directory is supplied
        if len(output_name) != 0:
            typs = np.unique(y_train)
            typs.sort()
            typs = np.array(typs, dtype='str').tolist()

            if isinstance(features, Table):
                index_column = feats_test['Object']
            else:
                index_column = np.array(test_inds, dtype='str')
            dat = np.column_stack((index_column, probs))
            nms = ['Object'] + typs
            tab = Table(dat, dtype=['S64'] + ['f'] * probs.shape[1], names=nms)
            flname = output_name + cls + '.probs'
            tab.write(flname, format='ascii')

            tab = Table(np.column_stack((fpr, tpr)), names=['FPR', 'TPR'])
            tab.write(output_name + cls + '.roc', format='ascii')

            np.savetxt(output_name + cls + '.auc', [auc])

    print(f'\nTime taken {(time.time()-t1)/60.} minutes.')

    labels = []
    for tp_row in unique(types, keys='Type'):
        labels.append(tp_row['Type'])

    if plot_roc_curve:
        plot_roc(FPR, TPR, AUC, labels=classifiers, label_size=16,
                 tick_size=12, line_width=1.5)
        plt.show()

    if classifiers_for_cm_plots is None:
        classifiers_for_cm_plots = []

    if classifiers_for_cm_plots == 'all':
        classifiers_for_cm_plots = classifiers

    cms = []
    for cls in classifiers_for_cm_plots:
        if cls not in classifiers:
            print(f'{cls} not in our choice of classifiers!')
            continue
        y_fit = (probabilities[cls].argmax(axis=1)).tolist()
        cm = compute_confusion_matrix(y_fit, y_test)
        cms.append(cm)

    if return_classifier:
        return classifier_objects, cms


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
        print(f'Classifier saved in {path_to_save}.')

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

        Here we link the name of the costum scoring methods to their
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
        This costum scoring method can be used in a grid search.

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
            `snmachine` also contains the 'logloss' and 'auc' costum scoring.
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
            param_grid = self.param_grid_default(y_train, **kwargs)

        if 'true_class' in kwargs:
            self.true_class = kwargs['true_class']
            # Do some error checking here to avoid confusion in the roc curve
            # code when using it for optimisation
            class_labels = np.unique(y_train)
            self.which_column = np.where(class_labels == self.true_class)[0][0]
        else:
            self.true_class = 0
            self.which_column = 0

        # Standard grid search
        self.compute_grid_search(X_train=X_train, y_train=y_train,
                                 scoring=scoring, param_grid=param_grid,
                                 number_cv_folds=number_cv_folds,
                                 metadata=metadata)

        self.is_optimised = True
        print(f'The optimisation takes {time.time() - time_begin:.3f}s.')

    def compute_grid_search(self, X_train, y_train, scoring, param_grid,
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
        scoring : callable, str
            The metric used to evaluate the predictions on the test or
            validation sets. See
            `sklearn.model_selection._search.GridSearchCV` [1]_ for details on
            how to choose this parameter.
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

        References
        ----------
        .. [1] Pedregosa et al. "Scikit-learn: Machine Learning in Python",
        JMLR 12, pp. 2825-2830, 2011
        """
        if param_grid is None:
            param_grid = self.param_grid_default

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
                                                   scoring=scoring, cv=cv)

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
        if 'kernel' in svm_params:
            kernel = svm_params.pop('kernel')  # Removes this from kwargs
        else:
            kernel = 'rbf'
        unoptimised_classifier = sklearn.svm.SVC(kernel=kernel,
                                                 probability=self.prob,
                                                 **svm_params)
        self.classifier = unoptimised_classifier
        # Store the unoptimised classifier
        self.unoptimised_classifier = unoptimised_classifier
        print(f'Created classifier of type: {self.classifier}.')

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
        if 'n_neighbors' in knn_params:
            n_neighbors = knn_params.pop('n_neighbors')
        else:
            n_neighbors = 5
        if 'weights' in knn_params:
            weights = knn_params.pop('weights')
        else:
            weights = 'distance'
        unoptimised_classifier = sklearn.neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, **knn_params)
        self.classifier = unoptimised_classifier
        # Store the unoptimised classifier
        self.unoptimised_classifier = unoptimised_classifier
        print(f'Created classifier of type: {self.classifier}.')

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
        if 'hidden_layer_sizes' in nn_params:
            layer_sizes = nn_params.pop('hidden_layer_sizes')
        else:
            layer_sizes = (5, 2)
        if 'algorithm' in nn_params:
            algo = nn_params.pop('algorithm')
        else:
            algo = 'adam'
        if 'activation' in nn_params:
            activation = nn_params.pop('activation')
        else:
            activation = 'tanh'
        unoptimised_classifier = sklearn.neural_network.MLPClassifier(
            solver=algo, hidden_layer_sizes=layer_sizes, activation=activation,
            random_state=self._rs, **nn_params)
        self.classifier = unoptimised_classifier
        # Store the unoptimised classifier
        self.unoptimised_classifier = unoptimised_classifier
        print(f'Created classifier of type: {self.classifier}.')

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
        print(f'Created classifier of type: {self.classifier}.')

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
        print(f'Created classifier of type: {self.classifier}.')

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
        print(f'Created classifier of type: {self.classifier}.')

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
        print(f'Created classifier of type: {self.classifier}.')

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
        print(f'Created classifier of type: {self.classifier}.')

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
        print(f'Created classifier of type: {self.classifier}.')

    def optimise(self, X_train, y_train, scoring,
                 use_fast_optimisation=False, param_grid=None,
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
            `snmachine` also contains the 'logloss' and 'auc' costum scoring.
            For more details about these, see `logloss_score` and
            `auc_score`, respectively.
        use_fast_optimisation : bool, optional
            Whether to perform a specific hyperparameter optimisation that is
            faster than optimising a high dimensional grid through a standard
            grid search. By default it is `False`.
            See Notes for the details of this hyperparameter optimisation.
        param_grid : {None, dict}, optional
            Dictionary containing the parameters names (`str`) as keys and
            lists of their possible settings as values.
            If `use_fast_optimisation = True`, this input is ignored.
        number_cv_folds : int, optional
            Number of folds for cross-validation. By default it is 5.
        metadata : {None, pandas.DataFrame}, optional
            Metadata of the events with which to train the classifier.

        Notes
        -----
        The hyperparameter optimisation used for `use_fast_optimisation` is:
        First, optimise each hyperparameter individually using a 1D grid,
        keeping the other hyperparameters at default values. Then, construct a
        higher dimensional grid containing all the hyperparameters with three
        possible values for each hyperparameter informed by the earlier 1D
        optimization. Finally, optimise this higher dimensional grid through a
        standard grid search.

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

        # The hyperparameter optimisation of `use_fast_optimisation` is
        # described in the Notes of the docstring
        if use_fast_optimisation is True:
            self._compute_fast_optimisation(X_train=X_train, y_train=y_train,
                                            number_cv_folds=number_cv_folds,
                                            metadata=metadata)
        # Standard grid search
        else:
            self.compute_grid_search(X_train=X_train, y_train=y_train,
                                     scoring=scoring, param_grid=param_grid,
                                     number_cv_folds=number_cv_folds,
                                     metadata=metadata)

        self.is_optimised = True
        print(f'The optimisation takes {time.time() - time_begin:.3f}s.')

    def compute_grid_search(self, X_train, y_train, scoring, param_grid,
                            number_cv_folds, metadata, **kwargs):
        """Computes a standard grid search.

        This grid search is optimised using cross validation with
        `number_cv_folds` folds.

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
            `snmachine` also contains the 'logloss' and 'auc' costum scoring.
            For more details about these, see `logloss_score` and
            `auc_score`, respectively.
        param_grid : dict
            Dictionary containing the parameters names (`str`) as keys and
            lists of their possible settings as values.
        number_cv_folds : int
            Number of folds for cross-validation.
        metadata : pandas.DataFrame
            Metadata of the events with which to train the classifier.
        **kwargs : dict, optional
            If the scoring is the ROC curve AUC (`scoring='auc'`), include as
            `true_class` the desired class to optimise (e.g. Ias, which might
            correspond to class 1 or 90 depending on the dataset).

        Raises
        ------
        AttributeError
            A grid must be provided in `param_grid` to perform a standard grid
            search. Thus, this input cannot be `None`.

        References
        ----------
        .. [1] Pedregosa et al. "Scikit-learn: Machine Learning in Python",
        JMLR 12, pp. 2825-2830, 2011
        """
        if scoring == 'auc':
            self._set_auc_score_kwargs(y_train=y_train, **kwargs)
        self.scoring = scoring

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
                                                   scoring=scoring, cv=cv)
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

        References
        ----------
        .. [1] Pedregosa et al. "Scikit-learn: Machine Learning in Python",
        JMLR 12, pp. 2825-2830, 2011
        """
        # This is the grid used to optimise each hyperparameter individually
        param_grid = {'num_leaves': np.arange(10, 55, 5),
                      'learning_rate': np.logspace(-3, -.01, 50),
                      'n_estimators': np.arange(25, 120, 10),
                      'min_child_samples': np.arange(20, 80, 10),
                      'max_depth': np.arange(1, 20, 3),
                      'min_split_gain': np.linspace(0., 2., 21)}
        # For testing purposes v TODO ; it is to see after testing saving
        # classifier
        param_grid = {'num_leaves': np.arange(10, 25, 5),
                      'learning_rate': np.logspace(-3, -.01, 3),
                      'n_estimators': np.arange(25, 40, 10),
                      'min_child_samples': np.arange(20, 25, 10),
                      'max_depth': np.arange(1, 6, 3),
                      'min_split_gain': np.linspace(0., 2., 3)}

        best_param = {}  # to refister the best value of the 1D optimisation
        for param in param_grid.keys():
            new_param_grid = {param: param_grid[param]}

            # Optimise `param` with the other hyperparameters at default values
            self.compute_grid_search(X_train=X_train, y_train=y_train,
                                     scoring=self.scoring,
                                     param_grid=new_param_grid,
                                     number_cv_folds=number_cv_folds,
                                     metadata=metadata)
            # Register the best value
            best_param[param] = self.grid_search.best_params_[param]

        # New grid to optimise all the hyperparameters simultaneously
        param_grid = self._construct_6d_grid(best_param)
        # Optimise `param` with the other hyperparameters at default values
        self.compute_grid_search(X_train=X_train, y_train=y_train,
                                 scoring=self.scoring, param_grid=param_grid,
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
        """
        param_grid = {}

        # The bellow values were informed by exploring the optmisation of
        # several classifiers. For a more detailed optimisation, run
        # `LightGBMClassifier.compute_grid_search`
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
