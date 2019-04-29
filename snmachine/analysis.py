"""
Module containing most of the analysis tools needed in snmachine
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from snmachine import sndata

def plot_chisq_over_datapoints_per_label(dataset, output_root=None, file_name='chisq_over_datapoints_plots.pdf'):
    """Produces a plot containing the X^2/number of datapoints histogram for each label.

    The plot can be saved as `.pdf`, `.png` or other file types accepted by `matplotlib.pyplot.savefig`.

    Parameters
    ----------
    dataset : Dataset object (sndata class)
        Dataset
    output_root : {None, str}, optional
        If None, don't save the plots. If str, it is the output directory, so save them there.
    file_name : str, optional
        Name with which we want to save the file. Default is `chisq_over_datapoints_plots.pdf`.
        `output_root` can't be None for this parameter to be considered.
    """
    if output_root == None and file_name != 'chisq_over_datapoints_plots.pdf':
        print('`output_root` is None so the plot has not been saved. `file_name` ignored.')
    dict_chisq_over_datapoints_per_label = get_dict_chisq_over_datapoints_per_label(dataset)
    unique_labels = np.unique(dataset.labels)
    number_cols_plot = 3
    number_rows_plot = (len(unique_labels)-1)//number_cols_plot + 1 # Formula to calculate the necessary number of rows
    fig, ax = plt.subplots(nrows=number_rows_plot, ncols=number_cols_plot, figsize=(20, number_rows_plot*3))
    for i in np.arange(len(unique_labels)):
        plt.subplot(number_rows_plot, 3, i+1)
        make_chisq_over_datapoints_plot_of_label(dict_chisq_over_datapoints_per_label, unique_labels[i])
    fig.text(0.5, 0.04, '$X^2$/datapoints', ha='center') # Add common x label
    fig.text(0.08, 0.5, 'Number of objects', va='center', rotation='vertical') # Add common y label
    if output_root != None:
        plt.savefig(os.path.join(output_root, file_name), bbox_inches='tight')
        print('Plot saved in '+str(os.path.join(output_root, file_name)))


def get_dict_chisq_over_datapoints_per_label(dataset):
    """Produce a dictionary that associates each label with the X^2/number of datapoints of its objects.

    Parameters
    ----------
    dataset : Dataset object (sndata class)
        Dataset

    Returns
    -------
    dict_chisq_over_datapoints_per_label : dict
        A dictionary whose keys are the labels and whose values are the
        X^2/number of datapoints values of all the objects with that label.
    """
    chisq_over_datapoints_per_obj = get_chisq_over_datapoints_per_obj(dataset)
    labels = dataset.labels
    unique_labels = np.unique(labels)
    dict_chisq_over_datapoints_per_label = {}
    for label in unique_labels:
        is_the_right_label = labels.values == label
        dict_chisq_over_datapoints_per_label[label] = chisq_over_datapoints_per_obj[is_the_right_label].values.flatten()
    return dict_chisq_over_datapoints_per_label


def get_chisq_over_datapoints_per_obj(dataset):
    """Calculates the reduced X^2 of each object and outputs it into a DataFrame.

    The X^2/number of datapoints of each object is by default returned as a dictionary. Here, after that step,
    the dictionary is transformed into a pandas DataFrame so it is easier to manipulate.

    Parameters
    ----------
    dataset : Dataset object (sndata class)
        Dataset

    Returns
    -------
    pandas.core.frame.DataFrame
        DataFrame with the object names and their X^2/number of datapoints.
    """
    dict_chisq_over_datapoints_per_obj = dataset.compute_chisq_over_datapoints()
    return pd.DataFrame.from_dict(dict_chisq_over_datapoints_per_obj, orient='index')


def make_chisq_over_datapoints_plot_of_label(dict_chisq_over_datapoints_per_label, label):
    """Plots a X^2/number of datapoints histogram of the objects of a specific label.

    Parameters
    ----------
    dict_chisq_over_datapoints_per_label : dict
        A dictionary whose keys are the labels and whose values are the
        X^2/number of datapoints values of all the objects with that label.
    label : int, str or float
        The label needs to be the same type as the one in the keys of
        `dict_chisq_over_datapoints_per_label`.

    Raises
    ------
    KeyError
        `label` needs to be a key in `dict_chisq_over_datapoints_per_label`.
    """
    try:
        chisq_over_datapoints_this_label = dict_chisq_over_datapoints_per_label[label]
    except KeyError:
        dict_keys = list(dict_chisq_over_datapoints_per_label.keys())
        raise KeyError("`label` needs to be a key of `dict_chisq_over_datapoints_per_label`. These are : {}".format(dict_keys))
    min_chisq_over_datapoints = np.min(chisq_over_datapoints_this_label)
    max_chisq_over_datapoints = np.max(chisq_over_datapoints_this_label)
    mean_chisq_over_datapoints = np.mean(chisq_over_datapoints_this_label)

    dict_label_to_real = {15:'TDE', 42:'SNII', 52:'SNIax', 62:'SNIbc', 64:'KN', 67:'SNIa-91bg',
                          88:'AGN', 90:'SNIa', 95:'SLSN-I'} # to erase later; This is just for de-bug
    label = dict_label_to_real[label]

    plot_label = 'Label {} ; {} objs ; <reduced $X^2$> = {:.3f}'.format(label,
                                                                        len(chisq_over_datapoints_this_label),
                                                                        mean_chisq_over_datapoints)
    bins = np.logspace(np.log10(min_chisq_over_datapoints), np.log10(max_chisq_over_datapoints), 50)
    plt.hist(x=chisq_over_datapoints_this_label, bins=bins, label=plot_label)
    plt.xscale('log')
    plt.legend()