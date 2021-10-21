from snmachine import sndata,snfeatures
import numpy as np
import pandas
from astropy.table import Table
import pickle
import os,sys

'''
print('starting readin of monster files')
#raw_data=pandas.read_csv('/share/hypatia/snmachine_resources/data/plasticc/test_set.csv')
raw_data=pandas.read_csv('/share/hypatia/snmachine_resources/data/plasticc/training_set.csv')
print('read in data set')
#raw_metadata=pandas.read_csv('/share/hypatia/snmachine_resources/data/plasticc/test_set_metadata.csv')
raw_metadata=pandas.read_csv('/share/hypatia/snmachine_resources/data/plasticc/training_set_metadata.csv')
print('read in metadata')
sys.stdout.flush()

#objects=np.unique(raw_data['object_id'])
#filters=np.unique(raw_data['passband']).astype('str')
'''

index=int(sys.argv[1])
print('Performing feature extraction on batch %d'%index)

out_folder='/share/hypatia/snmachine_resources/data/plasticc/data_products/plasticc_test/with_nondetection_cutting/fullset/data/'

print('loading data')
sys.stdout.flush()
with open(os.path.join(out_folder,'dataset_%d.pickle'%index),'rb') as f:
	d=pickle.load(f)
int_folder=os.path.join(out_folder,'int')
feats_folder=os.path.join(out_folder,'features')

print('data loaded')
sys.stdout.flush()
#d=sndata.EmptyDataset(filter_set=filters,survey_name='plasticc',folder=out_folder)

#d.object_names=d.object_names[:10]

print('nobj: '+str(len(d.object_names)))
print('extracting features')
sys.stdout.flush()
wf=snfeatures.WaveletFeatures(wavelet='sym2',ngp=1100)

pca_folder='/share/hypatia/snmachine_resources/data/plasticc/dummy_pca/'

feats=wf.extract_features(d,nprocesses=1,save_output='all',output_root=int_folder, recompute_pca=False, pca_path=pca_folder,xmax=1100)
feats.write(os.path.join(feats_folder, 'wavelet_features_%d.fits'%index),overwrite=True)
'''
with open(os.path.join(feats_folder,'PCA_mean.pickle'),'wb') as f1:
	pickle.dump(wf.PCA_mean,f1)
with open(os.path.join(feats_folder,'PCA_eigenvals.pickle'),'wb') as f2:
        pickle.dump(wf.PCA_eigenvals,f2)
with open(os.path.join(feats_folder,'PCA_eigenvectors.pickle'),'wb') as f3:
        pickle.dump(wf.PCA_eigenvectors,f3)


np.savetxt(os.path.join(feats_folder,'PCA_mean.txt'),wf.PCA_mean)
np.savetxt(os.path.join(feats_folder,'PCA_eigenvals.txt'),wf.PCA_eigenvals)
np.savetxt(os.path.join(feats_folder,'PCA_eigenvectors.txt'),wf.PCA_eigenvectors)
'''
