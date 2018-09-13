from __future__ import division
from snmachine import sndata, snfeatures, snclassifier, tsne_plot, example_data
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import rc
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
import time, os, pywt, subprocess
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

    # dataset.replace("_", "")

    plt.figure(figsize=(12,6))
    for i in range(len(rocs)):

        df = pd.read_csv(rocs[i]['file'], sep='\s+')
        fpr = df['FPR']
        tpr = df['TPR']

        auc = pd.read_csv(rocs[i]['auc'], sep='\s+', header=None)
        auc = auc.values[0][0]

        total_num_sne = pd.read_csv(rocs[i]['types'], sep='\s+')
        num_sne = rocs[i]['ratio']*total_num_sne.shape[0]
        num_sne = int(num_sne)

        sim = rocs[i]['sim']

        plot_roc_curve(fpr, tpr, label="{}. Training on {} SNe, AUC = {:.2f}".format(sim, num_sne, auc))
        plt.legend(loc="lower right")

    title = r"ROC Wavelet Features & Random Forest Algorithm"
    plt.title(title)
    plt.savefig("{}Wavelets_RF_ROC_{}.png".format(out_plots, jobid))
    plt.close('all')
