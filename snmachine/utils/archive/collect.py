import pickle
from snmachine import sndata
import sys,os

nproc=200

outfile='/share/hypatia/snmachine_resources/data/plasticc/data_products/plasticc_test/with_nondetection_cutting/fullset/data/'
d=sndata.PlasticcData(folder=outfile)
objs=[]


for chunk in range(nproc):
	print('reading in chunk %d'%chunk)
	sys.stdout.flush()
	with open(os.path.join(outfile,'dataset_%d.pickle'%chunk),'rb') as f:
		newdata=pickle.load(f)
	d.data.update(newdata.data)
	objs+=list(newdata.object_names)

d.filter_set=newdata.filter_set
d.object_names=list(objs)

print('writing data set')
sys.stdout.flush()
with open(os.path.join(outfile,'dataset_full.pickle'),'wb') as f:
	pickle.dump(d,f)
