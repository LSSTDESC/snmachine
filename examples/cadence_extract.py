from __future__ import division
from mpi4py import MPI
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

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    print("RANK {} REPORTING FOR DUTY!".format(rank))
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

    homedir = os.environ['HOME']
    username = os.environ.get('USER')

    final_outdir=os.path.join('share', 'hypatia', 'snmachine_resources', 'data', 'LSST_Cadence_WhitePaperClassResults', 'output_data', 'output_%s_no_z' %dataset,'')

    outdir=os.path.join(os.path.sep, 'share','data1', username, '{}_cadencetmp_{}_{}'.format(dataset, jobid, rank), '')

    out_features=os.path.join(outdir, 'features', '')
    out_class=os.path.join(outdir, 'classifications', '')
    out_int=os.path.join(outdir, 'int', '')

    print('temp outdir '+outdir)
    print('final outdir '+final_outdir)

    if rank == 0:

        print("MASTER NODE::{}".format(rank))

        print("I'M WAITING ...")

        comm.barrier()

        print("I WAITED")

        for i in range(size-1):
            LIST_WAVELETS="ls /share/data1/tallam/{}_cadencetmp_{}_{}/int/wavelet_*".format(dataset, jobid, i+1)
            subprocess.call(LIST_WAVELETS, shell=True)

            final_int=os.path.join(final_outdir, 'int', '')
            RSYNC_FILES="rsync -ravh /share/data1/tallam/{}_cadencetmp_{}_{}/int/wavelet_* {}".format(dataset, jobid, i+1, final_int)
            subprocess.call(RSYNC_FILES, shell=True)

        print("FINISHED")
        comm.Abort()

    elif rank > 0:

        print("WORKER NODE::{}".format(rank))

        print("TESTING WRITE TO SCRATCHDIR:: {}".format(rank))
        somefile="/state/partition1/somefile.txt"
        subprocess.call(['touch', somefile])
        print("LOOKING FOR FILE IN SCRATCH:\n{}".format(os.path.isfile(somefile)))

        if not os.path.exists(final_outdir):
            os.makedirs(final_outdir)
            os.makedirs(os.path.join(final_outdir, 'features', ''))
            os.makedirs(os.path.join(final_outdir, 'int', ''))
            os.makedirs(os.path.join(final_outdir, 'classifications', ''))

        if os.path.isdir(outdir):

            print("Removing old output directory")
            subprocess.call(['rm', '-r', outdir])

            print("Creating new output directory")
            subprocess.call(['mkdir',outdir])
            subprocess.call(['mkdir',out_features])
            subprocess.call(['mkdir',out_class])
            subprocess.call(['mkdir',out_int])

        else:

            print("Creating new output directory from scratch")
            subprocess.call(['mkdir',outdir])
            subprocess.call(['mkdir',out_features])
            subprocess.call(['mkdir',out_class])
            subprocess.call(['mkdir',out_int])

        print(outdir)
        #Data root
        print(rt)

        dat=sndata.LSSTCadenceSimulations(rt, prefix_Ia, prefix_NONIa, indices=[rank])

        dat.data[dat.object_names[0]]
        # ## Extract features for the data
        print("Rank : {} has data {} : ".format(rank, dat))

        # The next step is to extract useful features from the data. This can often take a long time, depending on the feature extraction method, so it's a good idea to save these to file (`snmachine` by default saves to astropy tables)

        read_from_file=False #We can use this flag to quickly rerun from saved features

        run_name=os.path.join(out_features,'%s_all' %dataset)
        # run_name=os.path.join(example_data, out_features,'%s_all' %dataset)
        print("RUN NAME:\n{}".format(run_name))

        ### Wavelet features

        # The wavelet feature extraction process is quite complicated, although it is fairly fast. Remember to save the PCA eigenvalues, vectors and mean for later reconstruction!

        waveFeats=snfeatures.WaveletFeatures()

        if read_from_file:
            wave_features=Table.read('%s_wavelets.dat' %run_name, format='ascii')
            #Crucial for this format of id's
            blah=wave_features['Object'].astype(str)
            wave_features.replace_column('Object', blah)
            PCA_vals=np.loadtxt('%s_wavelets_PCA_vals.dat' %run_name)
            PCA_vec=np.loadtxt('%s_wavelets_PCA_vec.dat' %run_name)
            PCA_mean=np.loadtxt('%s_wavelets_PCA_mean.dat' %run_name)
        else:
            wave_features=waveFeats.extract_features(dat,nprocesses=40,output_root=out_int,save_output='all')
            wave_features.write('{}_wavelets_rank_{}.dat'.format(run_name, rank), format='ascii')
            np.savetxt('%s_wavelets_PCA_vals.dat' %run_name,waveFeats.PCA_eigenvals)
            np.savetxt('%s_wavelets_PCA_vec.dat' %run_name,waveFeats.PCA_eigenvectors)
            np.savetxt('%s_wavelets_PCA_mean.dat' %run_name,waveFeats.PCA_mean)
            PCA_vals=waveFeats.PCA_eigenvals
            PCA_vec=waveFeats.PCA_eigenvectors
            PCA_mean=waveFeats.PCA_mean
