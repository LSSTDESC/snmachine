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

# TODO

def process():
    parser = ArgumentParser(description="Configuration for cadence comparisons")

    parser.add_argument('--config', '-c')
    parser.add_argument('--jobid', '-j', type=int)

    arguments = parser.parse_args()
    return vars(arguments)

def copy_files(delete=True):
    print('Copying files ...')
    t1=time.time()
    if not os.path.exists(final_outdir):
        os.makedirs(final_outdir)
        os.makedirs(os.path.join(final_outdir, 'features', ''))
        os.makedirs(os.path.join(final_outdir, 'int', ''))
        os.makedirs(os.path.join(final_outdir, 'classifications', ''))

    if delete:
        os.system('rsync -avq --exclude %s --remove-source-files %s* %s'%(out_int, outdir, final_outdir))
        os.system('rm -r %s'%(outdir))
    else:
        os.system('rsync -avq %s* %s'%(outdir, final_outdir))

    print('Time taken for file copying '+str(time.time()-t1))

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

    final_outdir=os.path.join('output_data', 'output_%s_no_z' %dataset,'')

    outdir=os.path.join(os.path.sep, 'share','data1', username, 'cadencetmp_{}_{}'.format(jobid, rank), '')

    out_features=os.path.join(outdir, 'features', '')
    out_class=os.path.join(outdir, 'classifications', '')
    out_int=os.path.join(outdir, 'int', '')

    print('temp outdir '+outdir)
    print('final outdir '+final_outdir)

    if rank == 0:
        print("MASTER NODE::{}".format(rank))

        # print("I'M WAITING ...")

        # comm.barrier()

        # print("I WAITED")

        # remove_excess_headers = "sed '1!{{/^Object/d}}' /state/parition1/{}_wavelets.dat > /share/data1/tallam/{}_wavelets.dat".format(jobid)
        # subprocess.call(remove_excess_headers, shell=True)
    # # if rank == 0:
    # #     run_name=os.path.join(out_features,'%s_all' %dataset)
        # wave_features=Table.read("/state/partition1/{}_wavelets.dat".format(jobid), format='ascii')

        # ## READ IN TYPES SINGLE FILE

        # plt.figure(1)
        # tsne_plot.plot(wave_features,join(wave_features,types)['Type'])
        # plt.savefig("plots/{}_Wavelets_RF_tSNE_{}.png".format(dataset, jobid))
        # plt.close(1)

        # ## Classify

        # nproc=4

        # # Print available classifiers
        # print(snclassifier.choice_of_classifiers)

        # # ### Wavelet features

        # subprocess.call(['mkdir',out_class])

        # plt.figure(2)
        # clss=snclassifier.run_pipeline(wave_features,types,output_name=os.path.join(out_class,'wavelets'),
        #                           classifiers=['random_forest'], nprocesses=nproc)
        # plt.savefig("plots/{}_Wavelets_RF_ROC_{}.png".format(dataset, jobid))
        # plt.close(2)

        # plt.close('all')

        # copy_files()

    elif rank > 0:

        print("WORKER NODE::{}".format(rank))

        print("TESTING WRITE TO SCRATCHDIR:: {}".format(rank))
        somefile="/state/partition1/somefile.txt"
        subprocess.call(['touch', somefile])
        print("LOOKING FOR FILE IN SCRATCH:\n{}".format(os.path.isfile(somefile)))

        # This is where to save intermediate output for the feature extraction method.
        # In some cases (such as the wavelets), these  files can be quite large.
        # out_features=os.path.join(outdir, 'features', '')
        # out_class=os.path.join(outdir, 'classifications', '')
        # out_int=os.path.join(outdir, 'int', '')

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
        # rt="/share/hypatia/snmachine_resources/RBTEST_DDF_IaCC_Y10_G10/"
        print(rt)

        # fits file prefix
        # prefix_Ia = 'RBTEST_DDF_MIXED_Y10_G10_Ia-'
        # prefix_NONIa = 'RBTEST_DDF_MIXED_Y10_G10_NONIa-'

        # dat=sndata.LSSTCadenceSimulations(rt, prefix_Ia, prefix_NONIa, indices=range(1,2))
        dat=sndata.LSSTCadenceSimulations(rt, prefix_Ia, prefix_NONIa, indices=[8])
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
            wave_features=waveFeats.extract_features(dat,nprocesses=6,output_root=out_int,save_output='all')
            wave_features.write('{}_wavelets_rank_{}.dat'.format(run_name, rank), format='ascii')
            np.savetxt('%s_wavelets_PCA_vals.dat' %run_name,waveFeats.PCA_eigenvals)
            np.savetxt('%s_wavelets_PCA_vec.dat' %run_name,waveFeats.PCA_eigenvectors)
            np.savetxt('%s_wavelets_PCA_mean.dat' %run_name,waveFeats.PCA_mean)
            PCA_vals=waveFeats.PCA_eigenvals
            PCA_vec=waveFeats.PCA_eigenvectors
            PCA_mean=waveFeats.PCA_mean


        # Then bring back together. Barrier after they all done. Node 0.
        # BARRIER
        # wait for all files to be written
        # comm.barrier() -------------

        # GATHER

        # Combine all rank_wavelets.dat files together in a single file called
        # combine_wavelet_features = "cat {0}_wavelets_rank_{1}.dat >> /state/partition1/{2}_wavelets.dat".format(run_name, rank, jobid)
        # subprocess.call(combine_wavelet_features, shell=True)

        # comm.barrier()

        ## WRITE TYPES ------- Send to parition1
        ## COMBINE TYPES ------- Send to parition1

        # remove_excess_headers = "sed '1!{{/^Object/d}}' /state/parition1/{1}_wavelets.dat > /state/partition1/{1}_wavelets.dat".format(run_name, jobid)
        # subprocess.call(remove_excess_headers, shell=True)
    # # if rank == 0:
    # #     run_name=os.path.join(out_features,'%s_all' %dataset)
        # wave_features=Table.read("/state/partition1/{0}_wavelets.dat".format(jobid, run_name), format='ascii')

        # plt.figure(1)
        # tsne_plot.plot(wave_features,join(wave_features,types)['Type'])
        # plt.savefig("plots/{}_Wavelets_RF_tSNE_{}.png".format(dataset, jobid))
        # plt.close(1)

        # ## Classify

        # nproc=4

        # # Print available classifiers
        # print(snclassifier.choice_of_classifiers)

        # # ### Wavelet features

        # plt.figure(2)
        # clss=snclassifier.run_pipeline(wave_features,types,output_name=os.path.join(out_class,'wavelets'),
        #                           classifiers=['random_forest'], nprocesses=nproc)
        # plt.savefig("plots/{}_Wavelets_RF_ROC_{}.png".format(dataset, jobid))
        # plt.close(2)

        # plt.close('all')

        # copy_files()
