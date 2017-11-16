"""
Utility script for making nice t-SNE plots (https://lvdmaaten.github.io/tsne/)
"""

from __future__ import division, print_function
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def get_tsne(feats,objs,perplexity=100, seed=-1):
    """
    Return the transformed features running the sklearn t-SNE code.

    Parameters
    ----------
    feats : astropy.table.Table
        Input features
    objs : list
        Subset of objects to run on (t-SNE is slow for large numbers, 2000 randomly selected objects is a good compromise)
    perplexity : float, optional
        t-SNE parameter which controls (roughly speaking) how sensitive the t-SNE plot is to small details

    Returns
    -------
    Xfit : array
        Transformed, embedded 2-d features

    """
    if seed!=-1:
        np.random.seed(seed)
    manifold=TSNE(perplexity=perplexity)
    short_inds=np.in1d(feats['Object'],objs)
    X=feats[short_inds]
    X=np.array([X[c] for c in X.colnames[1:]]).T
    Xfit=manifold.fit_transform(X)
    return Xfit

def plot_tsne(Xfit,types, loc='best', seed=-1):
    """
    Plot the resulting t-SNE embedded features.

    Parameters
    ----------
    Xfit : array
        Transformed, embedded 2-d features
    types : array
        Types of the supernovae (to colour the points appropriately)
    loc : str, optional
        Location of the legend in the plot
    """
    colours=['#1b9e77','#7570b3','#d95f02']
    legend_names=['Ia','II','Ibc']
    unique_types=np.unique(types)
    markers=['o','^','s']
    legs=[]
    for i in range(len(unique_types))[::-1]:
        inds=np.where(types==unique_types[i])[0]
        l=plt.scatter(Xfit[inds,0],Xfit[inds,1],color=colours[i],alpha=0.5,
                      marker=markers[i],s=16.0,linewidths=0.3,rasterized=True)
        legs.append(l)
    fntsize=10
    plt.legend(legs[::-1],legend_names,scatterpoints=1,loc=loc)
    plt.gca().get_legend().get_frame().set_lw(0.2)
    plt.xlabel('Embedded feature 1')
    plt.ylabel('Embedded feature 2')
    plt.gcf().tight_layout()
    plt.plot()

def plot(feats, types,objs=[], seed=-1):
    """
    Convenience function to run t-SNE and plot

    Parameters
    ----------
    feats : astropy.table.Table
        Input features
    types : array
        Types of the supernovae (to colour the points appropriately)
    objs : list
        Subset of objects to run on (t-SNE is slow for large numbers, 2000 randomly selected objects is a good compromise)
    """
    if len(objs)==0:
        objs=feats['Object']
    Xfit=get_tsne(feats,objs, seed=seed)
    plot_tsne(Xfit,types)


