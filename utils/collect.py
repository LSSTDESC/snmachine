import pickle
from snmachine import sndata
import sys

nproc=1

d=sndata.EmptyDataset()
objs=[]


for chunk in range(nproc):
	print('reading in chunk %d'%chunk)
	sys.stdout.flush()
	with open('/home/roberts/data_sets/sne/plasticc/plasticc_training/plasticc_aux/dataset_%d.pickle'%chunk,'rb') as f:
		newdata=pickle.load(f)
	d.data.update(newdata.data)
	objs+=list(newdata.object_names)

d.filter_set=newdata.filter_set
d.object_names=list(objs)

print('writing data set')
sys.stdout.flush()
with open('/home/roberts/data_sets/sne/plasticc/plasticc_training/dataset_full.pickle','wb') as f:
	pickle.dump(d,f)
