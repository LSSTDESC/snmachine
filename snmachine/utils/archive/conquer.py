from snmachine import sndata,snfeatures
import numpy as np
import pandas
from astropy.table import Table
import pickle
import os,sys

index=int(sys.argv[1])
print(index)
[start_index,stop_index]=np.loadtxt('/share/hypatia/snmachine_resources/data/plasticc/data_products/plasticc_test/with_nondetection_cutting/fullset/data/objs_%d.txt'%index,dtype='i8')

print('starting readin of monster files')
raw_data=pandas.read_csv('/share/hypatia/snmachine_resources/data/plasticc/test_set.csv',skiprows=range(1,start_index+1),nrows=stop_index-start_index)
#raw_data=pandas.read_csv('/home/z56693rs/data_sets/sne/LSST_plasticc/training_set.csv',nrows=1e5)
print('read in data set')
raw_metadata=pandas.read_csv('/share/hypatia/snmachine_resources/data/plasticc/test_set_metadata.csv')
#raw_metadata=pandas.read_csv('/home/z56693rs/data_sets/sne/LSST_plasticc/training_set_metadata.csv')
print('read in metadata')
raw_data=raw_data[raw_data.detected==1]
sys.stdout.flush()

if 'target' in raw_metadata.columns:
	wehavetypes=True
else:
	wehavetypes=False
print('Do we have types?'+str(wehavetypes))

print('chopped out chunk from line %d to line %d'%(start_index,stop_index))
sys.stdout.flush()

print(raw_data.columns)
objects=np.unique(raw_data['object_id'])
filters=np.unique(raw_data['passband']).astype('str')

print('nobj: '+str(len(objects)))

out_folder='/share/hypatia/snmachine_resources/data/plasticc/data_products/plasticc_test/with_nondetection_cutting/fullset/data/'
int_folder=os.path.join(out_folder,'int')
feats_folder=os.path.join(out_folder,'features')

filter_names={0:'lsstu',1:'lsstg',2:'lsstr',3:'lssti',4:'lsstz',5:'lsstY'}

d=sndata.PlasticcData(filter_set=list(filter_names.values()),survey_name='plasticc',folder=out_folder)

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

	fnames=[filter_names[f] for f in fr['passband']]
	print('fnames: '+str(fnames))
	tab=Table([fr['mjd'],fnames,fr['flux'],fr['flux_err']],names=['mjd','filter','flux','flux_error'])

	if len(tab)>0:
		tab['mjd']-=tab['mjd'].min()

	tab.meta['name']=o.astype('str')

	meta_line=raw_metadata[raw_metadata['object_id']==o].iloc[0]
#	print(type(meta_line))
#	print(type(meta_line['target']))
	tab.meta['ra']=meta_line['ra']
	tab.meta['decl']=meta_line['decl']
	tab.meta['gal_l']=meta_line['gal_l']
	tab.meta['gal_b']=meta_line['gal_b']
	tab.meta['ddf']=meta_line['ddf']
	tab.meta['hostgal_specz']=meta_line['hostgal_specz']
	tab.meta['hostgal_photoz']=meta_line['hostgal_photoz']
	tab.meta['hostgal_photoz_err']=meta_line['hostgal_photoz_err']
	tab.meta['distmod']=meta_line['distmod']
	tab.meta['mwebv']=meta_line['mwebv']

	if wehavetypes:
		tab.meta['type']=int(meta_line['target'])

	#judgment call on what redshift to use in snmachine
	tab.meta['z']=tab.meta['hostgal_specz']

	#insert into data set
        d.insert_lightcurve(tab)


with open(os.path.join(out_folder,'dataset_%d.pickle'%index),'wb') as f:
    pickle.dump(d,f)

'''
wf=snfeatures.WaveletFeatures()
feats=wf.extract_features(d,nprocesses=80,save_output='all',output_root=int_folder)
feats.write(os.path.join(feats_folder, 'wavelet_features.fits'),overwrite=True)

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
