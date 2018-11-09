import pandas
import numpy as plt
import sys

data=pandas.read_csv('/share/hypatia/snmachine_resources/data/plasticc/training_set.csv')
print(len(data))

nproc=1
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
	plt.savetxt('/home/roberts/data_sets/sne/plasticc/plasticc_training/plasticc_aux/objs_%d.txt'%i,plt.array([start_counter,stop_counter]),fmt='%d')

plt.savetxt('/home/roberts/data_sets/sne/plasticc/plasticc_training/plasticc_aux/objs_%d.txt'%(nproc-1),plt.array([linecounter,len(data)]),fmt='%d')
