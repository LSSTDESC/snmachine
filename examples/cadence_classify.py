from __future__ import division
from snmachine import sndata, snfeatures, snclassifier, tsne_plot, example_data
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import time, os, pywt, subprocess
from sklearn.decomposition import PCA
from astropy.table import Table,join,vstack
from astropy.io import fits,ascii
import sklearn.metrics
import sncosmo
import yaml

def process():
    parser = ArgumentParser(description="Configuration for cadence comparisons")

    parser.add_argument('--config', '-c')
    parser.add_argument('--jobid', '-j', type=int)

    arguments = parser.parse_args()
    return vars(arguments)

if __name__ == "__main__":

# jobid=8740
    args = process()
    jobid = args['jobid']

    try:
        with open(args['config']) as f:
            params = yaml.load(f)
    except IOError:
        print("Invalid yaml file provided")
        exit()

    print("The JOBID is:\n {}".format(jobid))
    print("The PARAMS are:\n {}".format(params))

    dataset = params.get("dataset", None)
    print("The dataset is:\n{}".format(dataset))

    rt = params.get("dataroot", None)
    print("The dataroot is:\n{}".format(rt))

    prefix_Ia = params.get("prefix_Ia", None)
    print("The data prefix for Ia's is:\n{}".format(prefix_Ia))

    prefix_NONIa = params.get("prefix_NONIa", None)
    print("The data prefix for NON Ia's is:\n{}".format(prefix_NONIa))

    final_outdir=os.path.join('output_data', 'output_%s_no_z' %dataset,'')

    ## READ IN TYPES SINGLE FILE
    dat=sndata.LSSTCadenceSimulations(rt, prefix_Ia, prefix_NONIa, indices=range(1,21))
    # dat=sndata.LSSTCadenceSimulations(rt, prefix_Ia, prefix_NONIa)

    #For now we restrict ourselves to three supernova types: Ia (1), II (2) and Ibc (3)
    types=dat.get_types()
    print(type(types))

    # If we write this table to file and inpsect the format of supernova types we find there are 6 variants:
    # ascii.write(types, 'types.csv', format='csv', fast_writer=True)
    # awk_command = "awk -F ',' '{print $2}' ../examples/types.csv | uniq -c"
    # subprocess.call(awk_command, shell=True)

    # Like for SPCC example notebook where we restrict ourselves to three supernova types: Ia (1), II (2) and Ibc (3) by carrying out the following pre-proccessing steps
    types['Type'] = types['Type']-100

    types['Type'][np.floor(types['Type']/10)==2]=2
    types['Type'][np.floor(types['Type']/10)==3]=3

    dat.data[dat.object_names[0]]

    # READ IN WAVELET DATA
    # Copy int to finaldir
    wavelet_feats=snfeatures.WaveletFeatures(wavelet='sym2', ngp=100)

    # waveints='/share/data1/tallam/'

    wave_raw, wave_err=wavelet_feats.restart_from_wavelets(dat, os.path.join(final_outdir, 'int', ''))

    wavelet_features,vals,vec,means=wavelet_feats.extract_pca(dat.object_names.copy(), wave_raw)

    # wave_features=Table.read("/share/data1/tallam/{}_all_wavelets.dat".format(jobid), format='ascii')

    # plt.figure(1)
    # tsne_plot.plot(wavelet_features,join(wavelet_features,types)['Type'])
    # plt.savefig("plots/{}_Wavelets_RF_tSNE_{}.png".format(dataset, jobid))
    # plt.close(1)

    ## Classify

    nproc=4

    # Print available classifiers
    print(snclassifier.choice_of_classifiers)

    # ### Wavelet features
    out_class=os.path.join(final_outdir, 'classifications', '')

    subprocess.call(['mkdir',out_class])

    plt.figure(2)
    clss=snclassifier.run_pipeline(wavelet_features,types,output_name=os.path.join(out_class,'wavelets'),
                              classifiers=['random_forest'], nprocesses=nproc)
    plt.savefig("plots/{}_Wavelets_RF_ROC_{}.png".format(dataset, jobid))
    plt.close(2)

    plt.close('all')

