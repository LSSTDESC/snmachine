from __future__ import division
from snmachine import sndata, snfeatures, snclassifier, tsne_plot, example_data
from argparse import ArgumentParser
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
import time, os, pywt, subprocess
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from astropy.table import Table,join,vstack
from astropy.io import fits,ascii
import sklearn.metrics
import sncosmo
import yaml
import pandas as pd

def process():
    parser = ArgumentParser(description="Configuration for cadence comparisons")

    parser.add_argument('--config', '-c')
    parser.add_argument('--jobid', '-j', type=int)

    arguments = parser.parse_args()
    return vars(arguments)

def plot_roc_curve(fpr, tpr, label=None):

    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel(r"False Positive Rate (FPR)", fontsize=16)
    plt.ylabel(r"True Positive Rate (TPR)", fontsize=16)

def plot_tsne():

    # TODO::
    pass

def plot_confusion_matrix(cm,normalise=False,labels=['Ia','II','Ibc', 'Others'],title='Confusion matrix'):
    '''
    Make a plot from a pre-computed confusion matrix.

    Parameters
    ----------
    cm : np.array
    ¦  The confusion matrix, as computed by the snclassifier.compute_confusion_matrix
    normalise : boolean, optional
    ¦  If False, we use the absolute numbers in each matrix entry. If True, we use the
    ¦  fractions within each true class
    labels : list of str
    ¦  Labels for each class that appear in the plot
    title : str
    ¦  Surprisingly, this is the title for the plot.
    '''
    plt.figure()
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == "__main__":

    args = process()
    jobid = args['jobid']

    try:
        with open(args['config']) as f:
            params = yaml.load(f)
    except IOError:
        print("Invalid yaml file provided")
        exit()

    print("The PARAMS are:\n {}".format(params))

    # jobid = params.get("jobid", None)
    # print("The JOBID is:\n {}".format(jobid))

    # dataset = params.get("dataset", None)
    # print("The dataset is:\n{}".format(dataset))

    rocs = params.get("rocs", None)
    print("The ROC data being used is listed as:\n{}".format(rocs))

    # final_outdir=os.path.join('output_data', 'output_%s_no_z' %dataset,'')
    final_outdir=os.path.join('output_data', '')
    print("FINAL OUTDIR = {}".format(final_outdir))
    # out_class=os.path.join(final_outdir, 'classifications_{}'.format(jobid), '')
    # print("CLASSIFY OUTDIR = {}".format(out_class))
    out_plots=os.path.join(final_outdir, 'plots', '')
    print("PLOTS = {}".format(out_plots))

    try:
        subprocess.call(['mkdir', '-p', out_plots])
    except IOError:
        print("Already exists, use another name...")
        # exit()

    # plt.figure(figsize=(12,6))
    for i in range(len(rocs)):

        sim = rocs[i]['sim']

        y_true = pd.read_csv(rocs[i]['norm_types'], engine='python') #colossus_2665_ddf_Y1_types_normalised.csv
        y_pred = pd.read_csv(rocs[i]['probs'], sep='\s+', engine='python') #waveletrandom_forest2000.probs

        results = pd.merge(y_pred, y_true, left_on = 'Object', right_on = 'Object')
        print(results)

        # new_true = results.filter(['Object','Type'], axis=1)
        new_true = results.filter(['Type'], axis=1)
        print(new_true)

        # new_pred = results.filter(['Object','1', '2', '3', '4'], axis=1)
        new_pred = results.filter(['1', '2', '3', '4'], axis=1)
        print(new_pred)
        # new_pred = new_pred.idxmax(axis=1)

        new_true = new_true.values
        print(type(new_true))
        new_pred = new_pred.values
        print(type(new_pred))

        y_preds = np.argmax(new_pred, axis=1)
        y_preds = y_preds+1
        print(y_preds)
        print(y_preds.shape)
        print(type(y_preds))
        # y_trues = np.argmax(new_true, axis=1)
        y_trues = new_true.ravel()
        print(y_trues)
        print(y_trues.shape)
        print(len(y_trues))
        print(type(y_trues))

        cm = confusion_matrix(y_trues, y_preds)
        print("Confusion Matrix:\n{}".format(cm))
        # plot_confusion_matrix(cm, title)
        row_sums = cm.sum(axis=1, keepdims=True)
        norm_conf_mx = cm / row_sums

        print(classification_report(y_trues, y_preds))

        plt.figure(figsize=(9,9))
        sns.heatmap(norm_conf_mx, square=True, annot=True, cbar=False, fmt='.2f')
        plt.xlabel('predicted value')
        plt.ylabel('true value')
        # title = "Confusion Matrix for : {}".format(sim)
        plt.title(r"Confusion Matrix for : {{}}".format(sim))
        plt.savefig("{}wavelets_rf_confusionmatrix_{}{}.png".format(out_plots, jobid, sim))

    plt.close('all')
    ########################################

    plt.figure(figsize=(12,6))
    for i in range(len(rocs)):

        df = pd.read_csv(rocs[i]['file'], sep='\s+', engine='python')
        fpr = df['FPR']
        tpr = df['TPR']

        auc = pd.read_csv(rocs[i]['auc'], sep='\s+', header=None, engine='python')
        auc = auc.values[0][0]

        total_num_sne = pd.read_csv(rocs[i]['types'], sep='\s+', engine='python')
        # num_sne = rocs[i]['ratio']*total_num_sne.shape[0]
        # num_sne = int(num_sne)
        num_sne = 2000

        sim = rocs[i]['sim']

        # plot_roc_curve(fpr, tpr, label="{}. Training on {} SNe, AUC = {:.3f}".format(sim, num_sne, auc))
        plot_roc_curve(fpr, tpr, label="{}. AUC = {:.3f}".format(sim, auc))
        plt.legend(loc="lower right")

    title = r"ROC Wavelet Features & Random Forest Algorithm Trained on 2000 Objects"
    plt.title(title)
    plt.savefig("{}wavelets_rf_ROC_{}.png".format(out_plots, jobid))
    plt.close('all')
