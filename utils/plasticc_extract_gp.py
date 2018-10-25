from snmachine import sndata,snfeatures
import numpy as np
import pandas
from astropy.table import Table
import pickle
import os,sys


print('starting readin of monster files')
#raw_data=pandas.read_csv('/share/hypatia/snmachine_resources/data/plasticc/test_set.csv')
raw_data=pandas.read_csv('/share/hypatia/snmachine_resources/data/plasticc/training_set.csv')
print('read in data set')
#raw_metadata=pandas.read_csv('/share/hypatia/snmachine_resources/data/plasticc/test_set_metadata.csv')
raw_metadata=pandas.read_csv('/share/hypatia/snmachine_resources/data/plasticc/training_set_metadata.csv')
print('read in metadata')
sys.stdout.flush()

objects=np.unique(raw_data['object_id'])
filters=np.unique(raw_data['passband']).astype('str')


out_folder='/home/roberts/data_sets/sne/plasticc/plasticc_training/'
int_folder=os.path.join(out_folder,'int')
feats_folder=os.path.join(out_folder,'features')

d=sndata.EmptyDataset(filter_set=filters,survey_name='plasticc',folder=out_folder)

for o,counter in zip(objects,range(len(objects))):
	print('obj #%d: %s'%(counter,o))
	sys.stdout.flush()
	#slice data frame into one object/rest WITHOUT querying the entire frame
	
	linecounter=0
	while linecounter < len(raw_data) and raw_data.iloc[linecounter]['object_id']==o:
		linecounter+=1

	fr=raw_data.iloc[:linecounter]
	raw_data=raw_data.iloc[linecounter:]
	

	#assemble lightcurve table pertaining to one object
	#fr=raw_data[raw_data['object_id']==o]

	tab=Table([fr['mjd'],fr['passband'].astype('str'),fr['flux'],fr['flux_err']],names=['mjd','filter','flux','flux_error'])

	tab['mjd']-=tab['mjd'].min()

	tab.meta['name']=o.astype('str')
	tab.meta['z']=raw_metadata['hostgal_specz'][raw_metadata['object_id']==o]

	#insert into data set
	d.insert_lightcurve(tab)

with open(os.path.join(out_folder,'dataset.pickle'),'wb') as f:
        pickle.dump(d,f)
'''

print('loading data')
sys.stdout.flush()
with open('/home/roberts/data_sets/sne/plasticc/plasticc_test/dataset_full.pickle','rb') as f:
	d=pickle.load(f)
'''

print('nobj: '+str(len(objects)))
print('extracting features')
sys.stdout.flush() 
wf=snfeatures.WaveletFeatures()
feats=wf.extract_features(d,nprocesses=80,save_output='all',output_root=int_folder)
feats.write(os.path.join(feats_folder, 'wavelet_features.fits'),overwrite=True)

with open(os.path.join(feats_folder,'PCA_mean.pickle'),'wb') as f1:
	pickle.dump(wf.PCA_mean,f1)
with open(os.path.join(feats_folder,'PCA_eigenvals.pickle'),'wb') as f2:
        pickle.dump(wf.PCA_eigenvals,f2)
with open(os.path.join(feats_folder,'PCA_eigenvectors.pickle'),'wb') as f3:
        pickle.dump(wf.PCA_eigenvectors,f3)

'''
np.savetxt(os.path.join(feats_folder,'PCA_mean.txt'),wf.PCA_mean)
np.savetxt(os.path.join(feats_folder,'PCA_eigenvals.txt'),wf.PCA_eigenvals)
np.savetxt(os.path.join(feats_folder,'PCA_eigenvectors.txt'),wf.PCA_eigenvectors)
'''

