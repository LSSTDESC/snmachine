import pandas
import numpy as plt
import sys

data=pandas.read_csv('/share/hypatia/snmachine_resources/data/plasticc/test_set.csv')
print('length pre detection cuts : '+str(len(data)))
#data=data.loc[data.detected==1]
print('length post detection cuts: '+str(len(data)))

nproc=200
lines_per_proc=int(len(data)/nproc)

linecounter=0

for i in range(nproc-1):
	print(i)
	print('linecounter: '+str(linecounter))
	start_counter=linecounter
	linecounter+=lines_per_proc

	lastobj=data.iloc[linecounter]['object_id']
	while linecounter<len(data) and data.iloc[linecounter]['object_id']==lastobj:
		linecounter+=1
	stop_counter=linecounter
	print('chunk %d: cutting lines %d to %d'%(i,start_counter,stop_counter))
	sys.stdout.flush()
	plt.savetxt('/share/hypatia/snmachine_resources/data/plasticc/data_products/plasticc_test/with_nondetection_cutting/fullset/data/objs_%d.txt'%i,plt.array([start_counter,stop_counter]),fmt='%d')

plt.savetxt('/share/hypatia/snmachine_resources/data/plasticc/data_products/plasticc_test/with_nondetection_cutting/fullset/data/objs_%d.txt'%(nproc-1),plt.array([linecounter,len(data)]),fmt='%d')
