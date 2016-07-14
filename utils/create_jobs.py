"""
Utility script for creating job scripts for running on a cluster.
This is specific to a cluster at UCL but can be adapted to any TORQUE cluster.
"""

from __future__ import division
import os
import numpy as np

#good_nodes=range(13, 18)+range(19,22) #We want to avoid node 18 (and any nodes that aren't free of course)
good_nodes=range(13,16)
#good_nodes=[]
node_ind=0

job_dir='/home/mlochner/sn/jobs/'
#job_dir='jobs/'
if not os.path.exists(job_dir):
    os.makedirs(job_dir)
    
n12=10 #How many cores12 nodes requesting
n24=len(good_nodes) #How many cores24 nodes requesting

proc12=n12*12 #Total number of cores12 processors
proc24=n24*24

dataset='lsst_ddf'
subset='none'

use_redshift=False

if use_redshift:
	reds='redshift'
else:
	reds=''

subset_name=dataset+'_subset_%d'

def make_job_script(ppn, subset_name):
    global node_ind
    queue='cores'+(str)(ppn)
    if ppn==12:
        node_string='#PBS -l nodes=1:ppn=%d' %ppn
    else:
        node_string='#PBS -l nodes=compute-0-%d:ppn=%d' %(good_nodes[node_ind],ppn)
        node_ind+=1
    fl=open(job_dir+subset_name+'.pbs', 'w')
    fl.write('#!/bin/tcsh -f\n \
#PBS -V\n \
%s\n \
#PBS -r n\n \
#PBS -S /bin/tcsh\n \
#PBS -q %s\n \
#PBS -l walltime=99:00:00\n' %(node_string,queue))
    fl.write('source .tcshrc\n')
    fl.write('cd /home/mlochner/sn\n')

    fl.write('python /home/mlochner/snmachine/utils/run_pipeline.py %s%s.txt %d %s\n' %(job_dir, subset_name, ppn,reds))
    #fl.write('python /home/michelle/SN_Class/snmachine/run_pipeline.py %s%s.txt\n' %(job_dir, subset_name))
    fl.close()
    
def make_job_spawner(n12, n24, job_dir, subset_root):
    #Create a simple bash script to qsub all the jobs
    fl=open(os.path.join(job_dir, 'run_all.sh'), 'w')
    for i in range(n12):
        subs=subset_root %i
        fl.write('qsub %s\n' %os.path.join(job_dir, subs+'.pbs')) 
    for j in range(n12, n12+n24):
        subs=subset_root %j
        fl.write('qsub %s\n' %os.path.join(job_dir, subs+'.pbs'))
    fl.close()



if dataset=='des':
    survey_name='SIMGEN_PUBLIC_DES'
    #rootdir='/home/michelle/SN_Class/Simulations/'+survey_name+'/'
    rootdir='/home/mlochner/sn/'+survey_name+'/'
    if subset=='spectro':
        objects=np.loadtxt('DES_spectro.list', dtype='str')
    else:
        objects=np.loadtxt(rootdir+survey_name+'.LIST', dtype='str') #Our list of objects to split up

elif dataset=='sdss':
    survey_name='SMP_Data'
    #rootdir='/home/michelle/SN_Class/Simulations/'+survey_name+'/'
    rootdir='/home/mlochner/sn/'+survey_name+'/'
    if subset=='spectro':
        objects=np.loadtxt(rootdir+'spectro.list', dtype='str')
    else:
        objects=np.loadtxt(rootdir+'sdss_classes.list', dtype='str',skiprows=1)[:,0] #Our list of objects to split up

elif 'lsst' in dataset:
    if dataset=='lsst_main':
        survey_name='ENIGMA_1189_10YR_MAIN'
    else:
        survey_name='ENIGMA_1189_10YR_DDF'
    rootdir='/home/mlochner/sn/'+survey_name+'/'
    objects=np.loadtxt(rootdir+'high_SNR_snids.txt', dtype='str')
    #objects=np.loadtxt('missing_objects.txt',dtype='str')
    
nobj=len(objects) #Total number of objects

#Figure out how many objects go onto cores24 processors and how many on cores12
nobj12=(int)(nobj/(proc12+proc24)*proc12)
nobj24=nobj-nobj12

#Split the data as evenly as possible amongst processors
obj12=np.array_split(objects[:nobj12], n12)
if n24>0:
	obj24=np.array_split(objects[nobj12:], n24)

#We write each set of objects to file and use it as an argument to run_pipeline.py to find the subset
for i in range(n12):
    subs=subset_name %i
    np.savetxt(job_dir+subs+'.txt', obj12[i], fmt='%s')
    make_job_script(12, subs)
    
for j in range(n24):
    subs=subset_name %(j+len(obj12))
    np.savetxt(job_dir+subs+'.txt', obj24[j], fmt='%s')
    make_job_script(24,subs)
    
make_job_spawner(n12, n24, job_dir, subset_name)
