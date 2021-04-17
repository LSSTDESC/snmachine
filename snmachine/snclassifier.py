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
import sys
import time

import lightgbm as lgb
import numpy as np

# Solve imblearn problems introduced with sklearn version 0.24
import sklearn
import sklearn.neighbors
import sklearn.utils
import sklearn.ensemble
from sklearn.utils._testing import ignore_warnings
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
sys.modules['sklearn.utils.safe_indexing'] = sklearn.utils._safe_indexing
sys.modules['sklearn.utils.testing'] = sklearn.utils._testing
sys.modules['sklearn.ensemble.bagging'] = sklearn.ensemble._bagging
sys.modules['sklearn.ensemble.base'] = sklearn.ensemble._base
sys.modules['sklearn.ensemble.forest'] = sklearn.ensemble._forest
sys.modules['sklearn.metrics.classification'] = sklearn.metrics._classification

from astropy.table import Table, join, unique
from functools import partial
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
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
                           scoring_func='accuracy', balance_classes=False,
                           **kwargs):
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
        balance_classes : bool, optional
            If True, balances the classes using SMOTE. Otherwise, it runs
            without balancing. Default is False.

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

        if balance_classes is True:
            self.clf = make_pipeline(SMOTE(sampling_strategy='not majority'),
                                     self.clf)  # balance dataset
            new_params = {}
            for key in params:
                new_params['randomforestclassifier__'+key] = params[key]
            params = new_params

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
        classifier_name : TODO It is really needed?
            TODO
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
        """
        self.is_optimised = False  # the classifier was not yet optimised
        self.random_seed = random_seed

    def classifier(self):
        """Returns the classifier instance."""
        return self.classifier

    def optimise(self):
        """Optimise the classifier.
        """
        if self.is_optimised is True:
            print('Raise error/ ask for confirmation because the classifier '
                  'was already optimised')

        self.is_optimised = True

    def predict(self, features):
        """"Predict the classes of a dataset.
        """

    def predict_proba(self):
        """d"""

    def save_classifier(self, output_path):
        """Save the classifier.
        """

    @classmethod
    def load_classifier():
        """Load a previously saved classifier.
        """

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


class SklearnClassifier(BaseClassifier):
    3


class SVMClassifier(SklearnClassifier):
    """Uses Support vector machine (SVM) for classification.
    """

    def __init__(self,
                 random_seed=None, **kwargs):
        """Class enclosing the SVM classifier.

        Parameters
        ----------
        dataset : Dataset object (sndata class)
            Dataset to augment.
        **kwargs : dict, optional
            Optional keywords to pass arguments into `choose_z` and into
            `snamchine.gps.compute_gps`.
        """


class LightGBMClassifier(BaseClassifier):
    """Uses a tree based learning algorithm for classification from LightGBM.
    """

    def __init__(self, classifier_name, random_seed=None, **lgb_params):
        """Class enclosing a LightGBM classifier.

        Parameters
        ----------
        dataset : Dataset object (sndata class)
            Dataset to augment.
        **lgb_params : dict, optional
            Optional keywords to pass arguments into `lgb.LGBMClassifier`.
        """
        super().__init__(classifier_name, random_seed)
        unoptimised_classifier = lgb.LGBMClassifier(
            random_state=self._random_seed, **lgb_params)
        self.classifier = unoptimised_classifier
        # Store the unoptimised classifier
        self.unoptimised_classifier = unoptimised_classifier

    def optimise(self, X_train, y_train, scoring,
                 use_fast_optimisation=False, param_grid=None,
                 random_state=None, number_cv_folds=5, metadata=None):
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
            how to choose this parameter.
        use_fast_optimisation : bool, optional
            TODO
        param_grid : dict
            Dictionary containing the parameters names (`str`) as keys and
            lists of their possible settings as values.
        random_state : int, RandomState instance or None, optional
            The `random_state` affects the ordering of the indices, which
            controls which events are in each fold of the cross-validation.
            TODO
        number_cv_folds : int, optional
            Number of folds for cross-validation. By default it is 5.
        metadata : {None, pandas.DataFrame}, optional
            Metadata of the events with which to train the classifier.
            TODO

        References
        ----------
        .. [1] Pedregosa et al. "Scikit-learn: Machine Learning in Python",
        JMLR 12, pp. 2825-2830, 2011
        """
        self._is_classifier_optimised()

        if random_state is None:
            random_state = self._rs

        # First, optimise each hyperparameter individually using a 1D grid,
        # keeping the other hyperparameters at default values. Then, construct
        # a higher dimensional grid containing all the hyperparameters with
        # three possible values for each hyperparameter informed by the
        # earlier 1D optimization. Finally, optimise this higher dimensional
        # grid through a standard grid search.
        if use_fast_optimisation is True:
            self._compute_fast_optimisation(X_train=X_train, y_train=y_train,
                                            scoring=scoring,
                                            param_grid=param_grid,
                                            random_state=random_state,
                                            number_cv_folds=number_cv_folds,
                                            metadata=metadata)
        # Standard grid search
        else:
            self.compute_grid_search(X_train=X_train, y_train=y_train,
                                     scoring=scoring, param_grid=param_grid,
                                     random_state=random_state,
                                     number_cv_folds=number_cv_folds,
                                     metadata=metadata)

        self.is_optimised = True

    def compute_grid_search(self, X_train, y_train, scoring, param_grid,
                            random_state, number_cv_folds, metadata):
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
        random_state : int, RandomState instance or None
            The `random_state` affects the ordering of the indices, which
            controls which events are in each fold of the cross-validation.
        number_cv_folds : int
            Number of folds for cross-validation.
        metadata : pandas.DataFrame
            Metadata of the events with which to train the classifier.

        References
        ----------
        .. [1] Pedregosa et al. "Scikit-learn: Machine Learning in Python",
        JMLR 12, pp. 2825-2830, 2011
        """
        time_begin = time.time()

        cv_fold = StratifiedKFold(n_splits=number_cv_folds, shuffle=True,
                                  random_state=random_state)

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
        print(f'The optimisation takes {time.time() - time_begin:.3f}s.')

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

    def _compute_fast_optimisation(self, X_train, y_train, param_grid, scoring,
                                   random_state, number_cv_folds, metadata):
        return 'Not yet implemented'
