
# coding: utf-8

# In[1]:


# get_ipython().system('uname -a')


# # Notebook for comparing observing stratergies and their effect on Supernova classification

# This notebook illustrates the use of the `snmachine` supernova classification package for comparing cadence runs for the LSST.
#
# See Lochner et al. (2016) http://arxiv.org/abs/1603.00882

# <img src="pipeline.png" width=600>

# This image illustrates the how the pipeline works. As the user, you can choose what feature extraction method you want to use.
#
# For this analysis we have chosen to use wavelets for feature extraction and then to use random forests algorithm to complete the classification.

# In[2]:


# get_ipython().run_cell_magic('capture', '--no-stdout ', '#I use this to supress unnecessary warnings for clarity\n%load_ext autoreload\n%autoreload #Use this to reload modules if they are changed on disk while the notebook is running\nfrom __future__ import division\nfrom snmachine import sndata, snfeatures, snclassifier, tsne_plot\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport time, os, pywt,subprocess\nfrom sklearn.decomposition import PCA\nfrom astropy.table import Table,join,vstack\nfrom astropy.io import fits,ascii\nimport sklearn.metrics \nimport sncosmo\n%matplotlib nbagg')
from __future__ import division
from snmachine import sndata, snfeatures, snclassifier, tsne_plot, example_data
import numpy as np
import matplotlib.pyplot as plt
import time, os, pywt,subprocess
from sklearn.decomposition import PCA
from astropy.table import Table,join,vstack
from astropy.io import fits,ascii
import sklearn.metrics
import sncosmo


# ## Set up output structure

# We make lots of output files so it makes sense to put them in one place. This is the recommended output file structure.

# In[3]:


dataset='lsst'


# In[4]:


# WARNING...
#Multinest uses a hardcoded character limit for the output file names. I believe it's a limit of 100 characters
#so avoid making this file path to lengthy if using nested sampling or multinest output file names will be truncated

#Change outdir to somewhere on your computer if you like
outdir=os.path.join('output_%s_no_z' %dataset,'')
out_features=os.path.join(outdir,'features') #Where we save the extracted features to
out_class=os.path.join(outdir,'classifications') #Where we save the classification probabilities and ROC curves
out_int=os.path.join(outdir,'int') #Any intermediate files (such as multinest chains or GP fits)

subprocess.call(['mkdir',outdir])
subprocess.call(['mkdir',out_features])
subprocess.call(['mkdir',out_class])
subprocess.call(['mkdir',out_int])


# In[5]:


print(outdir)


# ## Initialise dataset object

# Load cadence simulation data

# In[6]:


# get_ipython().system('pwd')


# In[7]:


#Data root
rt="/Users/tallamjr/PhD/project/data/CadenceCompare/RBTEST_DDF_IaCC_Y10_G10/"
# rt="/share/hypatia/snmachine_resources/RBTEST_DDF_IaCC_Y10_G10/"
print(rt)


# In[8]:


#We can automatically untar the data from here
# if not os.path.exists(rt):
#    subprocess.call(['tar', '-zxvf', 'SPCC_SUBSET.tar.gz'])


# In[9]:


# fits file prefix
prefix_Ia = 'RBTEST_DDF_MIXED_Y10_G10_Ia-'
prefix_NONIa = 'RBTEST_DDF_MIXED_Y10_G10_NONIa-'
dat=sndata.LSSTCadenceSimulations(rt, prefix_Ia, prefix_NONIa, indices=range(1,2))


# In[ ]:


#dat=sndata.LSSTCadenceSimulations(rt, indices=range(1,2))


# In[ ]:


#For now we restrict ourselves to three supernova types: Ia (1), II (2) and Ibc (3)
types=dat.get_types()


# In[ ]:


print(type(types))


# If we write this table to file and inpsect the format of supernova types we find there are 6 variants:

# In[ ]:


# ascii.write(types, 'types.csv', format='csv', fast_writer=True)
# get_ipython().system("awk -F ',' '{print $2}' ../examples/types.csv | uniq -c")


# Like for SPCC example notebook where we restrict ourselves to three supernova types: Ia (1), II (2) and Ibc (3) by carrying out the following pre-proccessing steps

# In[ ]:


types['Type'] = types['Type']-100


# In[ ]:


# types


# Final pre-processing step.

# In[ ]:


types['Type'][np.floor(types['Type']/10)==2]=2
types['Type'][np.floor(types['Type']/10)==3]=3


# In[ ]:


# types


# Now we can plot all the data and cycle through it (left and right arrows on your keyboard)

# In[ ]:


# dat.plot_all()


# Each light curve is represented in the Dataset object as an astropy table, compatible with `sncosmo`:

# In[ ]:


dat.data[dat.object_names[0]]


# ## Extract features for the data

# The next step is to extract useful features from the data. This can often take a long time, depending on the feature extraction method, so it's a good idea to save these to file (`snmachine` by default saves to astropy tables)

# In[ ]:


read_from_file=False #We can use this flag to quickly rerun from saved features

out_features_dat=os.path.join(example_data, out_features,'features') #Where we save the extracted features to
subprocess.call(['mkdir', '-p',out_features_dat])

run_name=os.path.join(example_data, out_features,'%s_all' %dataset)
print(run_name)


# t-SNE plots:
# These are useful visualisation plots which embed high dimensional features into a lower dimensional space to indicate how well the features separate between classes (see https://lvdmaaten.github.io/tsne/)

# ### Wavelet features

# The wavelet feature extraction process is quite complicated, although it is fairly fast. Remember to save the PCA eigenvalues, vectors and mean for later reconstruction!

# In[ ]:


waveFeats=snfeatures.WaveletFeatures()


# In[ ]:


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


# In[ ]:


dat.set_model(waveFeats.fit_sn,wave_features,PCA_vec,PCA_mean,0,dat.get_max_length(),dat.filter_set)


# In[ ]:


# dat.plot_all()


# In[ ]:


plt.figure(1)
tsne_plot.plot(wave_features,join(wave_features,types)['Type'])
plt.savefig("kraken_2026_DDF_RF_Wavelets_tSNE.png")
plt.close(1)

# ## Classify

# Finally, we're ready to run the machine learning algorithm. There's a utility function in the `snclassifier` library to make it easy to run all the algorithms available, including converting features to `numpy` arrays and rescaling them and automatically generating ROC curves and metrics. Hyperparameters are automatically selected using a grid search combined with cross-validation. All functionality can also be individually run from `snclassifier`.

# Classifiers can be run in parallel, change this parameter to the number of processors on your machine (we're only running 4 algorithms so it won't help to set this any higher than 4).

# In[ ]:


nproc=4


# In[ ]:


#Available classifiers
print(snclassifier.choice_of_classifiers)


# ### Wavelet features

# In[ ]:


plt.figure(2)
clss=snclassifier.run_pipeline(wave_features,types,output_name=os.path.join(out_class,'wavelets'),
                          classifiers=['random_forest'], nprocesses=nproc)
plt.savefig("kraken_2026_DDF_RF_Wavelets_ROC.png")
plt.close(2)

plt.close('all')
