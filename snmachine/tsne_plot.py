"""
Utility script for making nice t-SNE plots (https://lvdmaaten.github.io/tsne/).
"""

from __future__ import division

__all__ = []

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from sklearn.manifold import TSNE


def get_tsne(feats, objs, perplexity=100, seed=-1):
    """Return the transformed features running the sklearn t-SNE code.

    Parameters
    ----------
    feats : astropy.table.Table
        Input features.
    objs : list
        Subset of objects to run on (t-SNE is slow for large numbers, 2000
        randomly selected objects is a good compromise).
    perplexity : float, optional
        t-SNE parameter which controls (roughly speaking) how sensitive the
        t-SNE plot is to small details.

    Returns
    -------
    Xfit : array
        Transformed, embedded 2-d features

    """
    if seed != -1:
        np.random.seed(seed)
    manifold = TSNE(perplexity=perplexity)
    short_inds = np.in1d(feats['Object'], objs)
    X = feats[short_inds]
    X = np.array([X[c] for c in X.colnames[1:]]).T
    Xfit = manifold.fit_transform(X)
    return Xfit


def plot_tsne(Xfit, types, loc="upper left", type_dict=None):
    """Plot the resulting t-SNE embedded features.

    Parameters
    ----------
    Xfit : array
        Transformed, embedded 2-d features.
    types : array
        Types of the supernovae (to colour the points appropriately).
    loc : str, optional
        Location of the legend in the plot.
    """
    unique_types = np.unique(types)
    colors = iter(cm.rainbow(np.linspace(0, 1, len(unique_types))))

    markers = ['o', '^', 's']
    legs = []
    for i in range(len(unique_types))[::-1]:
        inds = np.where(types == unique_types[i])[0]
        if type_dict is not None:
            label = type_dict.get(unique_types[i])
        else:
            label = unique_types[i]
        leg = plt.scatter(Xfit[inds, 0], Xfit[inds, 1], color=next(colors),
                          alpha=0.5, marker=markers[0], s=16.0, linewidths=0.3,
                          rasterized=True, label=label)
        legs.append(leg)
    plt.xlabel('Embedded feature 1')
    plt.ylabel('Embedded feature 2')
    plt.gcf().tight_layout()
    plt.legend(loc=loc)
    plt.plot()


def plot(feats, types, objs=[], seed=-1, type_dict=None):
    """Convenience function to run t-SNE and plot

    Parameters
    ----------
    feats : astropy.table.Table
        Input features.
    types : array
        Types of the supernovae (to colour the points appropriately).
    objs : list
        Subset of objects to run on (t-SNE is slow for large numbers, 2000
        randomly selected objects is a good compromise).
    """
    if len(objs) == 0:
        objs = feats['Object']
    Xfit = get_tsne(feats, objs, seed=seed)
    plot_tsne(Xfit, types, type_dict=type_dict)
