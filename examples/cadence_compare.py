from __future__ import division
from snmachine import sndata, snfeatures, snclassifier, tsne_plot, example_data
# from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import time, os, pywt,subprocess
from sklearn.decomposition import PCA
from astropy.table import Table,join,vstack
from astropy.io import fits,ascii
import sklearn.metrics
import sncosmo

# TODO
# Set up argparse general file that asks for:
#    1. jobid -- get from PBS
#    2. name for dataset -- yaml
#    3. location of OpSim dataset -- yaml
#    4. prefix_Ia and prefix_NONIa information -- yaml

dataset='kraken_2026_ddf'

timeid=time.strftime("%Y%m%d%H%M")

jobid=9999


#####################################################################################################################
############################################### READ DATA ###########################################################
#####################################################################################################################

homedir = os.environ['HOME']
username = os.environ.get('USER')

final_outdir=os.path.join('output_data', 'output_%s_no_z' %dataset,'')
# final_outdir=os.path.join(os.path.sep, homedir, 'data_sets', 'sne', 'sn_output','output_%s' %(run_name), '')
outdir=os.path.join(os.path.sep, 'share','data1', username, 'cadencetmp_n1', '')

print('temp outdir '+outdir)
print('final outdir '+final_outdir)

#This is where to save intermediate output for the feature extraction method. In some cases (such as the wavelets), these
#files can be quite large.
out_features=os.path.join(outdir, 'features', '')
out_class=os.path.join(outdir, 'classifications', '')
out_int=os.path.join(outdir, 'int', '')

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
# if not os.path.exists(out_features):
#     print("OUT FEATURES :\n"+out_class)
#     os.makedirs(out_features)
# if not os.path.exists(out_class):
#     print("OUT CLASS :\n"+out_class)
#     os.makedirs(out_class)
# if not os.path.exists(out_int):
#     print("OUT INT :\n"+out_int)
#     os.makedirs(out_int)

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
#    os.system('rsync %s* %s' %(os.path.join(outdir, 'features', ''), os.path.join(final_outdir, 'features', '')))
#    print 'rsync %s* %s' %(os.path.join(outdir, 'features', ''), os.path.join(final_outdir, 'features', ''))
#    os.system('rsync %s* %s' %(os.path.join(outdir, 'int', ''), os.path.join(final_outdir, 'int', '')))
#    os.system('rsync %s* %s' %(os.path.join(outdir, 'classifications', ''), os.path.join(final_outdir, 'classifications', '')))

    print('Time taken for file copying '+str(time.time()-t1))


###################################################################################
#Change outdir to somewhere on your computer if you like
# outdir=os.path.join('args_output_%s_no_z' %dataset,'')
# out_features=os.path.join(outdir,'features') #Where we save the extracted features to
# out_class=os.path.join(outdir,'classifications') #Where we save the classification probabilities and ROC curves
# out_int=os.path.join(outdir,'int') #Any intermediate files (such as multinest chains or GP fits)

# if os.path.isdir(outdir):

#     print("Removing old output directory")
#     subprocess.call(['rm', '-r', outdir])

#     print("Creating new output directory")
#     subprocess.call(['mkdir',outdir])
#     subprocess.call(['mkdir',out_features])
#     subprocess.call(['mkdir',out_class])
#     subprocess.call(['mkdir',out_int])

# else:

#     subprocess.call(['mkdir',outdir])
#     subprocess.call(['mkdir',out_features])
#     subprocess.call(['mkdir',out_class])
#     subprocess.call(['mkdir',out_int])

# print(outdir)

#Data root
rt="/share/hypatia/snmachine_resources/RBTEST_DDF_IaCC_Y10_G10/"
print(rt)

# fits file prefix
prefix_Ia = 'RBTEST_DDF_MIXED_Y10_G10_Ia-'
prefix_NONIa = 'RBTEST_DDF_MIXED_Y10_G10_NONIa-'
dat=sndata.LSSTCadenceSimulations(rt, prefix_Ia, prefix_NONIa, indices=range(1,2))
# dat=sndata.LSSTCadenceSimulations(rt, prefix_Ia, prefix_NONIa)

# SCATTER

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

# The next step is to extract useful features from the data. This can often take a long time, depending on the feature extraction method, so it's a good idea to save these to file (`snmachine` by default saves to astropy tables)

read_from_file=False #We can use this flag to quickly rerun from saved features

# out_features_dat=os.path.join(example_data, out_features,'features') #Where we save the extracted features to
# subprocess.call(['mkdir', '-p',out_features_dat])

## Not used ------------------------
# print("EXAMPLE DATA NAME:\n{}".format(example_data))

run_name=os.path.join(out_features,'%s_all' %dataset)
# run_name=os.path.join(example_data, out_features,'%s_all' %dataset)
print("RUN NAME:\n{}".format(run_name))


# t-SNE plots:
# These are useful visualisation plots which embed high dimensional features into a lower dimensional space to indicate how well the features separate between classes (see https://lvdmaaten.github.io/tsne/)

# ### Wavelet features

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
    wave_features.write('%s_wavelets.dat' %run_name, format='ascii')
    np.savetxt('%s_wavelets_PCA_vals.dat' %run_name,waveFeats.PCA_eigenvals)
    np.savetxt('%s_wavelets_PCA_vec.dat' %run_name,waveFeats.PCA_eigenvectors)
    np.savetxt('%s_wavelets_PCA_mean.dat' %run_name,waveFeats.PCA_mean)
    PCA_vals=waveFeats.PCA_eigenvals
    PCA_vec=waveFeats.PCA_eigenvectors
    PCA_mean=waveFeats.PCA_mean

# BARRIER
# wait for all files to be written

# GATHER

# Combine all rank_wavelets.dat files together in a single file called
# run_name_wavelets_all.data, then:

#     wave_features=Table.read('%s_wavelets.dat' %run_name, format='ascii')

#This code takes the fitted parameters and generates the model light curve for
# plotting purposes.
# dat.set_model(waveFeats.fit_sn,wave_features,PCA_vec,PCA_mean,0,dat.get_max_length(),dat.filter_set)

plt.figure(1)
tsne_plot.plot(wave_features,join(wave_features,types)['Type'])
plt.savefig("plots/{}_Wavelets_RF_tSNE_{}.png".format(dataset, timeid))
plt.close(1)

# ## Classify

# Finally, we're ready to run the machine learning algorithm. There's a utility function in the `snclassifier` library to make it easy to run all the algorithms available, including converting features to `numpy` arrays and rescaling them and automatically generating ROC curves and metrics. Hyperparameters are automatically selected using a grid search combined with cross-validation. All functionality can also be individually run from `snclassifier`.

# Classifiers can be run in parallel, change this parameter to the number of processors on your machine (we're only running 4 algorithms so it won't help to set this any higher than 4).

nproc=4

# Print available classifiers
print(snclassifier.choice_of_classifiers)

# ### Wavelet features

plt.figure(2)
clss=snclassifier.run_pipeline(wave_features,types,output_name=os.path.join(out_class,'wavelets'),
                          classifiers=['random_forest'], nprocesses=nproc)
plt.savefig("plots/{}_Wavelets_RF_ROC_{}.png".format(dataset, timeid))
plt.close(2)

plt.close('all')

copy_files()
