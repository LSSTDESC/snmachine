"""
Module for analysis tools, plotting and associated functions.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix
from snmachine import snclassifier

# Built-in dictionaries of class labels to their real name
dict_label_to_real_spcc = {1: 'SNIa', 2: 'SNII', 3: 'SNIbc'}
dict_label_to_real_plasticc = {15: 'TDE', 42: 'SNII', 52: 'SNIax', 62: 'SNIbc',
                               64: 'KN', 67: 'SNIa-91bg', 88: 'AGN',
                               90: 'SNIa', 95: 'SLSN-I'}


# Plotting functions
def plot_confusion_matrix(y_true, y_pred, title=None, normalise=None,
                          dict_label_to_real=None, figsize=None):
    """Plot a confusion matrix.

    Uses the true and predicted class labels to compute a confusion matrix.
    This can be non-normalised, normalised by true class/row (the diagonals
    show the accuracy of each class), and by predicted class/column (the
    diagonals show the precision).

    Parameters
    ----------
    y_true : 1D array-like
        Ground truth (correct) labels of shape (n_samples,).
    y_true : 1D array-like
        Predicted class labels of shape (n_samples,).
    title : {None, str}, optional
        Title of the plot.
    normalise : {None, str}, optional
       If `None`, use the absolute numbers in each matrix entry. If 'accuracy',
       normalise per true class. If 'precision', normalise per predicted class.
    dict_label_to_real : dict, optional
        Dictionary containing the class labels as key and its real name as
        values. E.g. for PLAsTiCC
        `dict_label_to_real = {42: 'SNII', 62: 'SNIbc', 90: 'SNIa'}`.
        If `None`, the default class labels are used.
    figsize : {None, tuple}
        If `None`, use the default `figsize` of the plot. Otherwise, create a
        figure with the given size.

    Returns
    -------
    cm : np.array
       The confusion matrix, as computed by `sklearn.metrics.confusion_matrix`.
    """
    # Make and normalise the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalise == 'accuracy':
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        kwargs = {'vmin': 0, 'vmax': 1}
        print("Confusion matrix normalised by true class.")
    elif normalise == 'precision':
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        kwargs = {'vmin': 0, 'vmax': 1}
        print("Confusion matrix normalised by predicted class.")
    else:
        print('Confusion matrix without normalisation')

    # Classes in the dataset
    target_names = np.unique(y_true)
    if dict_label_to_real is not None:
        target_names = np.vectorize(dict_label_to_real.get)(target_names)

    # Plot the confusion matrix
    if figsize is not None:
        _, ax = plt.subplots(figsize=figsize)  # good values: (9, 7)
    else:
        _, ax = plt.subplots()
    sns.heatmap(cm, xticklabels=target_names,
                yticklabels=target_names, cmap='Blues',
                annot=True, fmt='.2f', lw=0.5,
                cbar_kws={'label': 'Fraction of events',
                          'shrink': .82}, **kwargs)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_aspect('equal')
    if title is not None:
        plt.title(title)

    return cm


def plot_classifier_roc_curve(y_true, y_probs, title=None,
                              dict_label_to_real=None, figsize=None, **kwargs):
    """Plot ROC curves of each class vs other classes.

    Parameters
    ----------
    y_true : array (N_obj, )
        An array containing the true class for each object.
    y_probs : array (# obj, # classes)
        An array containing probabilities of each class for each object.
    title : {None, str}, optional
        Title of the plot.
    dict_label_to_real : dict, optional
        Dictionary containing the class labels as key and its real name as
        values. E.g. for PLAsTiCC
        `dict_label_to_real = {42: 'SNII', 62: 'SNIbc', 90: 'SNIa'}`.
        If `None`, the default class labels are used.
    figsize : {None, tuple}
        If `None`, use the default `figsize` of the plot. Otherwise, create a
        figure with the given size.
    **kwargs : dict, optional
        colors : dict, default = None
            Dictionary containing the classes names (`str`) as key and its
            colour as values. If `None`, it uses the `seaborn` default colours.
        lines_width: int, default = 3
            Lines width to print the ROC curves.
        xlabel : str, deafult = 'False positive rate (contamination)'
            The x label text.
        ylabel : str, default = 'True positive rate (completeness)'
            The y label text.
    """
    # Classes in the dataset
    target_names = np.unique(y_true)
    if dict_label_to_real is not None:
        target_names = np.vectorize(dict_label_to_real.get)(target_names)

    # Compute ROC curve and AUC for each class
    fpr, tpr, auc = {}, {}, {}  # initialize dictionaries

    for i in range(len(target_names)):
        fpr[i], tpr[i], auc[i] = snclassifier.compute_roc_values(
            probs=y_probs, y_test=y_true, which_column=i)

    # Plot the ROC curves
    if figsize is not None:
        plt.figure(figsize=figsize)  # good values: (10, 6)
    else:
        plt.figure()

    linewidth = kwargs.pop('lines_width', 3)
    colors = kwargs.pop('colors', None)
    number_classes = len(list(auc.keys()))
    for i in np.arange(number_classes):
        true_name = target_names[i]
        if colors is not None:
            plt.plot(fpr[i], tpr[i], color=colors[true_name], lw=linewidth,
                     label='AUC {} = {:0.3f}'.format(true_name, auc[i]))
        else:
            plt.plot(fpr[i], tpr[i], lw=linewidth,
                     label='AUC {} = {:0.3f}'.format(true_name, auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=linewidth)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    xlabel = kwargs.pop('xlabel', 'False positive rate (contamination)')
    ylabel = kwargs.pop('ylabel', 'True positive rate (completeness)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    if title is not None:
        plt.title(title)
    else:
        plt.title('Multi-Class ROC: 1 vs All')


# X^2/number of datapoints plot
def plot_chisq_over_pts_per_label(dataset, dict_label_to_real=None,
                                  output_root=None,
                                  file_name='chisq_over_pts_plots.pdf'):
    """Plots the X^2/number of datapoints histogram for each class label.

    The plot can be saved as `.pdf`, `.png` or other file types accepted by
    `matplotlib.pyplot.savefig`.

    Parameters
    ----------
    dataset : Dataset object (sndata class)
        Dataset
    dict_label_to_real : dict, optional
        Dictionary containing the class labels as key and its real name as
        values. E.g. for PLAsTiCC
        `dict_label_to_real = {42: 'SNII', 62: 'SNIbc', 90: 'SNIa'}`.
        If `None`, the default class labels are used.
    output_root : {None, str}, optional
        If `None`, don't save the plots. If `str`, it is the output directory,
        so save the plots there.
    file_name : str, optional
        Name with which we want to save the file. Default is
        `chisq_over_pts_plots.pdf`.
        `output_root` can't be `None` for this parameter to be considered.
    """
    if output_root is None and file_name != 'chisq_over_pts_plots.pdf':
        print(f'`output_root` is None so the plot has not been saved. '
              f'`file_name` = {file_name} ignored.')
    dict_chisq_over_pts_per_label = compute_dict_chisq_over_pts_per_label(
        dataset)
    unique_labels = np.unique(dataset.labels)
    number_cols_plot = 3
    # Calculate the necessary number of rows
    number_rows_plot = (len(unique_labels)-1)//number_cols_plot + 1
    fig, ax = plt.subplots(nrows=number_rows_plot, ncols=number_cols_plot,
                           figsize=(20, number_rows_plot*3))
    for i in np.arange(len(unique_labels)):
        plt.subplot(number_rows_plot, 3, i+1)
        make_chisq_over_pts_plot_of_label(
            dict_chisq_over_pts_per_label, unique_labels[i],
            dict_label_to_real=dict_label_to_real)

    # Common x and y labels
    fig.text(0.5, 0.04, '$X^2$/datapoints', ha='center')
    fig.text(0.08, 0.5, 'Number of objects', va='center', rotation='vertical')

    if output_root is not None:
        plt.savefig(os.path.join(output_root, file_name), bbox_inches='tight')
        print('Plot saved in '+str(os.path.join(output_root, file_name)))


def compute_dict_chisq_over_pts_per_label(dataset):
    """Computes dictionary of the X^2/number of datapoints per label.

    The dictionary associates each label with the X^2/number of datapoints of
    its objects.

    Parameters
    ----------
    dataset : Dataset object (sndata class)
        Dataset

    Returns
    -------
    dict_chisq_over_pts_per_label : dict
        A dictionary whose keys are the labels and whose values are the
        X^2/number of datapoints values of all the objects with that label.

    Raises
    ------
    AttributeError
        `dataset` needs to contain labels, so it can't be a test dataset.
    """
    chisq_over_pts_per_obj = get_chisq_over_pts_per_obj(dataset)
    labels = dataset.labels
    if labels is None:
        raise AttributeError('This dataset does not contain labels so we can '
                             'not get anything per label.')
    unique_labels = np.unique(labels)
    dict_chisq_over_pts_per_label = {}
    for label in unique_labels:
        is_right = labels.values == label
        chisq_over_pts_label = chisq_over_pts_per_obj[is_right]
        chisq_over_pts_label = chisq_over_pts_label.values.flatten()
        dict_chisq_over_pts_per_label[label] = chisq_over_pts_label
    return dict_chisq_over_pts_per_label


def get_chisq_over_pts_per_obj(dataset):
    """Calculates the reduced X^2 of each object and outputs it into a DataFrame.

    The X^2/number of datapoints of each object is by default returned as a
    dictionary. Here, after that step, the dictionary is transformed into a
    pandas DataFrame so it is easier to manipulate.

    Parameters
    ----------
    dataset : Dataset object (sndata class)
        Dataset

    Returns
    -------
    pandas.core.frame.DataFrame
        DataFrame with the object names and their X^2/number of datapoints.
    """
    dict_chisq_over_pts_per_obj = dataset.compute_chisq_over_pts()
    return pd.DataFrame.from_dict(dict_chisq_over_pts_per_obj, orient='index')


def make_chisq_over_pts_plot_of_label(dict_chisq_over_pts_per_label, label,
                                      dict_label_to_real=None):
    """Plots X^2/number of datapoints histogram of the objects of a specific label.

    Parameters
    ----------
    dict_chisq_over_pts_per_label : dict
        A dictionary whose keys are the labels and whose values are the
        X^2/number of datapoints values of all the objects with that label.
    label : int, str or float
        The label needs to be the same type as the one in the keys of
        `dict_chisq_over_pts_per_label`.
    dict_label_to_real : dict, optional
        Dictionary containing the class labels as key and its real name as
        values. E.g. for PLAsTiCC
        `dict_label_to_real = {42: 'SNII', 62: 'SNIbc', 90: 'SNIa'}`.
        If `None`, the default class labels are used.

    Raises
    ------
    KeyError
        `label` needs to be a key in `dict_chisq_over_pts_per_label`.
    KeyError
        If `dict_label_to_real` is not None, `label` needs to be a key in it.
    """
    try:
        chisq_over_pts_this_label = dict_chisq_over_pts_per_label[label]
    except KeyError:
        dict_keys = list(dict_chisq_over_pts_per_label.keys())
        raise KeyError(f'`label` needs to be a key of '
                       f'`dict_chisq_over_pts_per_label`. These are : '
                       f'{dict_keys}')
    min_chisq_over_pts = np.min(chisq_over_pts_this_label)
    max_chisq_over_pts = np.max(chisq_over_pts_this_label)
    mean_chisq_over_pts = np.mean(chisq_over_pts_this_label)

    if dict_label_to_real is not None:
        try:
            label = dict_label_to_real[label]
        except KeyError:
            raise KeyError(f'{label} must be a key in the dictionary '
                           f'`dict_label_to_real`. Alternativelly, set this '
                           f'dictionary to `None`.')

    number_objs = len(chisq_over_pts_this_label)
    plot_label = (f'Label {label} ; {number_objs} objs ; <reduced $X^2$> = '
                  f'{mean_chisq_over_pts:.3f}')
    bins = np.logspace(np.log10(min_chisq_over_pts),
                       np.log10(max_chisq_over_pts), 50)
    plt.hist(x=chisq_over_pts_this_label, bins=bins, label=plot_label)
    plt.xscale('log')
    plt.legend()


# Recall and precision plots
def compute_recall_values(quantity, bins, is_pred_right, is_true_type_list,
                          extra_subset=True, use_mid_bins=True, seed=42):
    """Computes the recall and bootstrapped confidence intervals.

    The bins can either reference a single value (`use_mid_bins = False`) or a
    range (`use_mid_bins = True`).

    Parameters
    ----------
    quantity : numpy.ndarray
        Values of the events in the quantity to analyse.
    bins : numpy.ndarray
        Bins to divide the `quantity` on.
    is_pred_right : list-like
        Mask of the events correctly predicted.
    is_true_type_list : list
        List of lists where each masks a different *true* class of events.
    extra_subset : {True, list-like}, optional
        If `True`, all events are considered. If it is a list, it contain a
        mask for the events to consider (`True` if it should be included).
    use_mid_bins : bool, optional
        If `True`, it considers the events with the quantity in a range
        (default). Otherwise, it only considers events with fixed quantity
        values.
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    recall_s : numpy.ndarray
        Recall of events of each class in each quantity bin.
    boot_recall_has_something_ci : list
        Value of the lower and upper limits of the confidence interval for
        each class, and in each quantity bin.
    number_in_bin_s : numpy.ndarray
        Number of events of each class in each quantity bin.
    """
    initial_time = time.time()
    recall_s, number_in_bin_s = compute_recall_with_bins(
        quantity=quantity, bins=bins, is_pred_right=is_pred_right,
        is_true_type_list=is_true_type_list, use_mid_bins=use_mid_bins,
        extra_subset=extra_subset)

    has_something = compute_has_something(quantity=quantity, bins=bins,
                                          extra_subset=extra_subset,
                                          use_mid_bins=use_mid_bins)

    boot_recall_has_something = compute_recall_has_something(
        has_something, is_pred_right, is_true_type_list=is_true_type_list,
        extra_subset=extra_subset, seed=42)

    boot_recall_has_something_ci = compute_boot_ci(boot_recall_has_something)
    print('Time taken to compute the recall values: {:.2f}s.'
          ''.format(time.time()-initial_time))
    return recall_s, boot_recall_has_something_ci, number_in_bin_s


def compute_precision_values(quantity, bins, is_pred_right, is_pred_type_list,
                             extra_subset=True, use_mid_bins=True, seed=42):
    """Computes the precision and bootstrapped confidence intervals.

    The bins can either reference a single value (`use_mid_bins = False`) or a
    range (`use_mid_bins = True`).

    Parameters
    ----------
    quantity : numpy.ndarray
        Values of the events in the quantity to analyse.
    bins : numpy.ndarray
        Bins to divide the `quantity` on.
    is_pred_right : list-like
        Mask of the events correctly predicted.
    is_pred_type_list : list
        List of lists where each masks a different *predictec* class of events.
    extra_subset : {True, list-like}, optional
        If `True`, all events are considered. If it is a list, it contain a
        mask for the events to consider (`True` if it should be included).
    use_mid_bins : bool, optional
        If `True`, it considers the events with the quantity in a range
        (default). Otherwise, it only considers events with fixed quantity
        values.
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    precision_s : numpy.ndarray
        Precision of events of each class in each quantity bin.
    boot_recall_has_something_ci : list
        Value of the lower and upper limits of the confidence interval for
        each class, and in each quantity bin.
    number_in_bin_s : numpy.ndarray
        Number of events of each class in each quantity bin.
    """
    initial_time = time.time()
    precision_s, number_in_bin_s = compute_precision_with_bins(
        quantity=quantity, bins=bins, is_pred_right=is_pred_right,
        is_pred_type_list=is_pred_type_list, use_mid_bins=use_mid_bins,
        extra_subset=extra_subset)

    has_something = compute_has_something(quantity=quantity, bins=bins,
                                          extra_subset=extra_subset,
                                          use_mid_bins=use_mid_bins)

    boot_precision_has_something = compute_precision_has_something(
        has_something, is_pred_right, is_pred_type_list=is_pred_type_list,
        extra_subset=extra_subset, seed=seed)

    boot_precion_has_something_ci = compute_boot_ci(
        boot_precision_has_something)
    print('Time taken to compute the precision values: {:.2f}s.'
          ''.format(time.time()-initial_time))
    return precision_s, boot_precion_has_something_ci, number_in_bin_s


def compute_recall_with_bins(quantity, bins, is_pred_right, is_true_type_list,
                             extra_subset=True, use_mid_bins=True):
    """Computes the recall per quantity bin.

    The bins can either reference a single value (`use_mid_bins = False`) or a
    range (`use_mid_bins = True`).

    Parameters
    ----------
    quantity : numpy.ndarray
        Values of the events in the quantity to analyse.
    bins : numpy.ndarray
        Bins to divide the `quantity` on.
    is_pred_right : list-like
        Mask of the events correctly predicted.
    is_true_type_list : list
        List of lists where each masks a different *true* class of events.
    extra_subset : {True, list-like}, optional
        If `True`, all events are considered. If it is a list, it contain a
        mask for the events to consider (`True` if it should be included).
    use_mid_bins : bool, optional
        If `True`, it considers the events with the quantity in a range
        (default). Otherwise, it only considers events with fixed quantity
        values.

    Returns
    -------
    recall_s : numpy.ndarray
        Recall of events of each class in each quantity bin.
    number_in_bin_s : numpy.ndarray
        Number of events of each class in each quantity bin.
    """
    # `quantity` is continuous or discrete
    if use_mid_bins:  # continuous
        mid_bins = (bins[:-1]+bins[1:])/2
        bins_number = len(mid_bins)
    else:  # discrete
        bins_number = len(bins)

    recall_s = []
    number_in_bin_s = []
    for i in np.arange(bins_number):  # for each bin
        if use_mid_bins:
            is_in_bin_i = (quantity >= bins[i]) & (quantity < bins[i+1])
        else:
            is_in_bin_i = (quantity == bins[i])

        number_sne = len(is_true_type_list)
        recall_js = np.zeros(number_sne)
        number_in_bin_js = np.zeros(number_sne)
        for j in np.arange(number_sne):  # for each class
            is_sne_in_j = (is_in_bin_i & is_true_type_list[j]
                           & extra_subset)  # all SNe truly SN[j]
            y = is_pred_right[is_sne_in_j]  # correctly classifyied SN
            recall_js[j] = np.sum(y) / np.sum(is_sne_in_j)  # TP / (TP+FN)
            number_in_bin_js[j] = np.sum(is_sne_in_j)
        recall_s.append(recall_js)
        number_in_bin_s.append(number_in_bin_js)

    # Reshape to obtain the bin values in the rows and classes in the columns
    recall_s = np.concatenate(recall_s).reshape(np.shape(recall_s))
    number_in_bin_s = np.concatenate(number_in_bin_s).reshape(
        np.shape(number_in_bin_s))
    return recall_s, number_in_bin_s


def compute_precision_with_bins(quantity, bins, is_pred_right,
                                is_pred_type_list, extra_subset=True,
                                use_mid_bins=True):
    """Computes the precision per quantity bin.

    The bins can either reference a single value (`use_mid_bins = False`) or a
    range (`use_mid_bins = True`).

    Parameters
    ----------
    quantity : numpy.ndarray
        Values of the events in the quantity to analyse.
    bins : numpy.ndarray
        Bins to divide the `quantity` on.
    is_pred_right : list-like
        Mask of the events correctly predicted.
    is_pred_type_list : list
        List of lists where each masks a different *predictec* class of events.
    extra_subset : {True, list-like}, optional
        If `True`, all events are considered. If it is a list, it contain a
        mask for the events to consider (`True` if it should be included).
    use_mid_bins : bool, optional
        If `True`, it considers the events with the quantity in a range
        (default). Otherwise, it only considers events with fixed quantity
        values.

    Returns
    -------
    precision_s : numpy.ndarray
        Precision of events of each class in each quantity bin.
    number_in_bin_s : numpy.ndarray
        Number of events of each class in each quantity bin.
    """
    # `quantity` is continuous or discrete
    if use_mid_bins:  # continuous
        mid_bins = (bins[:-1]+bins[1:])/2
        bins_number = len(mid_bins)
    else:  # discrete
        bins_number = len(bins)

    precision_s = []
    number_in_bin_s = []
    for i in np.arange(bins_number):  # for each bin
        if use_mid_bins:
            is_in_bin_i = (quantity >= bins[i]) & (quantity < bins[i+1])
        else:
            is_in_bin_i = (quantity == bins[i])

        number_sne = len(is_pred_type_list)
        precision_js = np.zeros(number_sne)
        number_in_bin_js = np.zeros(number_sne)
        for j in np.arange(number_sne):  # for each class
            is_sne_in_j = (is_in_bin_i & is_pred_type_list[j]
                           & extra_subset)  # all SNe predicted to be SN[j]
            y = is_pred_right[is_sne_in_j]  # correctly classifyied SN
            precision_js[j] = np.sum(y) / np.sum(is_sne_in_j)  # TP / (TP+FP)
            number_in_bin_js[j] = np.sum(is_sne_in_j)
        precision_s.append(precision_js)
        number_in_bin_s.append(number_in_bin_js)

    precision_s = np.concatenate(precision_s).reshape(np.shape(precision_s))
    number_in_bin_s = np.concatenate(number_in_bin_s).reshape(np.shape(
        number_in_bin_s))
    return precision_s, number_in_bin_s


def compute_has_something(quantity, bins, extra_subset=True,
                          use_mid_bins=True):
    """Computes masks for events with quanitity in each bin.

    The bins can either reference a single value (`use_mid_bins = False`) or a
    range (`use_mid_bins = True`).

    Parameters
    ----------
    quantity : numpy.ndarray
        Values of the events in the quantity to analyse.
    bins : numpy.ndarray
        Bins to divide the `quantity` on.
    extra_subset : {True, list-like}, optional
        If `True`, all events are considered. If it is a list, it contain a
        mask for the events to consider (`True` if it should be included).
    use_mid_bins : bool, optional
        If `True`, it considers the events with the quantity in a range
        (default). Otherwise, it only considers events with fixed quantity
        values.

    Returns
    -------
    has_something : list
        List of lists where each masks the events in a different bin.
    """
    has_something = []

    if use_mid_bins:
        bins_number = len(bins)-1
    else:
        bins_number = len(bins)

    if not isinstance(extra_subset, bool):  # only subset of events
        quantity = quantity[extra_subset]

    # Create masks for the events in each bin
    for i in np.arange(bins_number):
        if use_mid_bins:
            is_in_bin_i = ((quantity >= bins[i]) & (quantity < bins[i+1]))
        else:
            is_in_bin_i = (quantity == bins[i])
        has_something.append(is_in_bin_i)

    return has_something


def compute_recall_has_something(has_something, is_pred_right,
                                 is_true_type_list, extra_subset=None,
                                 seed=42):
    """Computes the recall of boostrapped events with a given property.

    Parameters
    ----------
    has_something : list
        List of lists where each masks the events in a different bin.
    is_pred_right : list-like
        Mask of the events correctly predicted.
    is_true_type_list : list
        List of lists where each masks a different *true* class of events.
    extra_subset : {True, list-like}, optional
        If `True`, all events are considered. If it is a list, it contain a
        mask for the events to consider (`True` if it should be included).
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    prop_pred_right_s : list
        Recall of 300 boostrapped events in each class, and in each quantity
        bin.
    """
    np.random.seed(seed)  # reproducible results

    prop_pred_right_s = []
    for j in np.arange(len(is_true_type_list)):  # choose a class
        is_to_consider = (
            (extra_subset is not None)
            & (extra_subset is not isinstance(extra_subset, bool)))
        if is_to_consider:
            is_subset = is_true_type_list[j][extra_subset]
            y = is_pred_right[extra_subset]
        else:
            is_subset = (is_true_type_list[j])
            y = is_pred_right

        prop_pred_right_i = []
        for i in np.arange(len(has_something)):  # choose a something
            is_in_bin = is_subset & has_something[i]  # our population

            number_in_bin = np.sum(is_in_bin)  # # objs in our population

            is_pred_right_in_bin = y[is_in_bin]  # size = number_in_bin_i
            if number_in_bin != 0:  # there are events in this bin
                prop_pred_right_ks = []
                for k in range(300):  # 300 for bootstrapping the values
                    is_pred_right_in_k = np.random.choice(is_pred_right_in_bin,
                                                          size=number_in_bin,
                                                          replace=True)
                    sum_pred_right_k = np.sum(is_pred_right_in_k)
                    prop_pred_right_ks.append(sum_pred_right_k/number_in_bin)
            else:  # no events in this bin
                prop_pred_right_ks = [None]
            prop_pred_right_i.append(prop_pred_right_ks)
        prop_pred_right_s.append(prop_pred_right_i)
    return prop_pred_right_s


def compute_precision_has_something(has_something, is_pred_right,
                                    is_pred_type_list, extra_subset=None,
                                    seed=42):
    """Computes the precision of boostrapped events with a given property.

    Parameters
    ----------
    has_something : list
        List of lists where each masks the events in a different bin.
    is_pred_right : list-like
        Mask of the events correctly predicted.
    is_pred_type_list : list
        List of lists where each masks a different *predicted* class of events.
    extra_subset : {True, list-like}, optional
        If `True`, all events are considered. If it is a list, it contain a
        mask for the events to consider (`True` if it should be included).
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    precision_s : list
        Precision of 300 boostrapped events in each class, and in each
        quantity bin.
    """
    np.random.seed(seed)  # reproducible results

    precision_s = []
    for j in np.arange(len(is_pred_type_list)):  # choose a SN class
        is_to_consider = (
            (extra_subset is not None)
            & (extra_subset is not isinstance(extra_subset, bool)))
        if is_to_consider:
            is_subset = is_pred_type_list[j][extra_subset]
            y = is_pred_right[extra_subset]
        else:
            is_subset = (is_pred_type_list[j])
            y = is_pred_right

        precision_i = []
        number_in_bin_i = []
        for i in np.arange(len(has_something)):  # choose something
            is_in_bin = is_subset & has_something[i]  # our population

            number_in_bin = np.sum(is_in_bin)  # # objs in our population
            number_in_bin_i.append(number_in_bin)

            is_pred_right_in_bin = y[is_in_bin]  # size = number_in_bin_i
            precision_ks = []
            for k in range(300):  # 300 for bootstrapping the values
                is_pred_right_in_k = np.random.choice(is_pred_right_in_bin,
                                                      size=number_in_bin,
                                                      replace=True)
                precision_ks.append(np.sum(is_pred_right_in_k)/number_in_bin)
            precision_i.append(precision_ks)
        precision_s.append(precision_i)
    return precision_s


def compute_boot_ci(boot_data):
    """Computes bootstrapped 2.5% and 97.5% confidence intervals.

    It uses the boostrapped recall or precision values previously calculated.

    Parameters
    ----------
    boot_data :  list
        Value of 300 boostrapped events in each class, and in each quantity
        bin.

    Returns
    -------
    boot_data_ci_s : list
        Value of the lower and upper limits of the confidence interval for
        each class, and in each quantity bin.
    """
    boot_data_ci_s = []
    shape_data = np.shape(boot_data)
    for i in np.arange(shape_data[0]):  # SN class
        boot_data_ci = []
        for j in np.arange(shape_data[1]):  # some property
            boot_data_ij = boot_data[i][j]
            try:
                percentil_025 = np.percentile(boot_data_ij, 2.5)
                percentil_975 = np.percentile(boot_data_ij, 97.5)
                boot_data_ci.append([percentil_025, percentil_975])
            except TypeError:  # no events in this class and bin
                boot_data_ci.append([None])
        boot_data_ci_s.append(np.array(boot_data_ci))
    return boot_data_ci_s


def plot_sne_has_something(something_s, boot_has_something_ci,
                           bins, is_true_type_list, sn_order, **kwargs):
    """Plots the recall or precision and confidence interval for each class.

    Parameters
    ----------
    something_s : numpy.ndarray
        Recall or precision of events of each class in each quantity bin.
    boot_has_something_ci : list
        Value of the lower and upper limits of the confidence interval for
        each class, and in each quantity bin.
    bins : numpy.ndarray
        Bins used to compute `something_s` and `boot_has_something_ci`.
    is_true_type_list : list
        List of lists where each masks a different *true* class of events.
    sn_order : list
        Ordered list of the names of the classes. The fist name should
        correspond to the first column of `something_s`.
    kwargs : dict, optional
        colors : list, default = None
            Ordered list of the colours with which to plot the classes results.
            If `None`, it uses the `seaborn` default colours.
        linewidth: int, default = 3
            Lines width to print the plots.
    """
    colors = kwargs.pop('colors', None)

    for j in np.arange(len(is_true_type_list)):
        sn_type = sn_order[j]
        # Remove NaN values
        y_vals = something_s[:, j]
        index_not_none = ~np.isnan(y_vals)
        y_vals = y_vals[index_not_none]
        y_ci = boot_has_something_ci[j]
        y_ci = y_ci[index_not_none]
        y_ci = np.array(list(y_ci))
        bins_j = bins[index_not_none]

        linewidth = kwargs.pop('linewidth', 3)
        if colors is not None:  # use inputed colors
            plt.plot(bins_j, y_vals, label=sn_type, color=colors[j],
                     linewidth=linewidth)
            plt.fill_between(bins_j, y1=y_ci[:, 0], y2=y_ci[:, 1],
                             color=colors[j], alpha=.3)
        else:  # use `seaborn` default colors
            plt.plot(bins_j, y_vals, label=sn_type, linewidth=linewidth)
            plt.fill_between(bins_j, y1=y_ci[:, 0], y2=y_ci[:, 1], alpha=.3)


# Recall and precision tools
def compute_lc_length(dataset):
    """Computes the length of the light curves.

    Computes the length of each individual light curve in `dataset`.

    Parameters
    ----------
    dataset : Dataset object (sndata class)
        Dataset.

    Returns
    -------
    lc_length : numpy.ndarray
        Length of each individual light curve.
    """
    obj_names = dataset.object_names

    lc_length = np.zeros(len(obj_names))
    for i in np.arange(len(obj_names)):
        obj = obj_names[i]
        obj_data = dataset.data[obj].to_pandas()
        obj_data = obj_data.sort_values(by='mjd')
        mjd_times = np.array(obj_data.mjd)
        time_diff = mjd_times[-1] - mjd_times[0]
        lc_length[i] = np.median(time_diff)
    return lc_length
