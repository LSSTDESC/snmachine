"""
Utility script for creating job scripts for running on a cluster.
This is specific to a cluster at UCL but can be adapted to any TORQUE cluster.
"""

from __future__ import division
import os
import numpy as np

#good_nodes=range(13, 18)+range(19,24) #We want to avoid node 18 (and any nodes that aren't free of course)
#good_nodes=[15, 19, 21, 22, 23, 24]
good_nodes=[]
node_ind=0

job_dir='/home/roberts/data_sets/sne/des/jobs/'
#job_dir='jobs/'
if not os.path.exists(job_dir):
    os.makedirs(job_dir)

n12=12 #How many cores12 nodes requesting
n24=len(good_nodes) #How many cores24 nodes requesting

proc12=n12*12 #Total number of cores12 processors
proc24=n24*24

dataset='cadence_rolling'
subset='none'
train_choice='repr'

use_redshift=False

if use_redshift:
	reds='redshift'
else:
	reds=''

subset_name=dataset+'_subset_%d'
fullset_name=dataset+'_fullset'

def make_job_script(ppn, subset_name, extra_flags=''):
    global node_ind
    queue='cores'+(str)(ppn)
    if ppn==12:
        node_string='#PBS -l nodes=1:ppn=%d' %ppn
    else:
        if 'no-xtr' in extra_flags or 'preprocess' in extra_flags:#this will be called only once, for the final script
            node_ind=0
        node_string='#PBS -l nodes=compute-0-%d:ppn=%d' %(good_nodes[node_ind],ppn)
        node_ind+=1
    if 'preprocess' in extra_flags:
        fl=open(job_dir+subset_name+'_preprocess.pbs', 'w')
    else:
        fl=open(job_dir+subset_name+'.pbs', 'w')
    fl.write('#!/bin/tcsh -f\n \
#PBS -V\n \
%s\n \
#PBS -r n\n \
#PBS -S /bin/tcsh\n \
#PBS -q %s\n \
#PBS -l walltime=99:00:00\n' %(node_string,queue))
    fl.write('source .tcshrc\n')
    fl.write('cd /home/roberts/sne/snmachine/	\n')

    fl.write('python /home/roberts/sne/snmachine/utils/run_pipeline.py %s%s.txt %d %s %s %s\n' %(job_dir, subset_name, ppn, reds, train_choice, extra_flags))
    #fl.write('python /home/michelle/SN_Class/snmachine/run_pipeline.py %s%s.txt\n' %(job_dir, subset_name))
    fl.close()

def make_job_spawner(n12, n24, job_dir, subset_root):
    #Create a simple bash script to qsub all the jobs
    fl=open(os.path.join(job_dir, 'run_all.sh'), 'w')
    fl.write('job_pre=$(qsub %s)\n'%(os.path.join(job_dir, fullset_name+'_preprocess.pbs')) )
    joblist=''
    for i in range(n12):
        subs=subset_root %i
        fl.write('job%d=$(qsub -W depend=afterok:$job_pre %s)\n' %(i, os.path.join(job_dir, subs+'.pbs')))
        joblist+=':$job%d' %i
    for j in range(n12, n12+n24):
        subs=subset_root %j
        fl.write('job%d=$(qsub -W depend=afterok:$job_pre %s)\n' %(j, os.path.join(job_dir, subs+'.pbs')))
        joblist+=':$job%d' %j
    fl.write('qsub -W depend=afterok%s %s\n'%(joblist, os.path.join(job_dir, fullset_name+'.pbs')) )
    fl.close()



if dataset=='des':
    survey_name='SIMGEN_PUBLIC_DES'
    #rootdir='/home/michelle/SN_Class/Simulations/'+survey_name+'/'
    rootdir='/home/roberts/data_sets/sne/des/'+survey_name+'/'
    if subset=='spectro':
        objects=np.genfromtxt('DES_spectro.list', dtype='str')
    else:
        objects=np.genfromtxt(rootdir+survey_name+'.LIST', dtype='str') #Our list of objects to split up
    #print(objects[0])
    #print(type(objects[0]))

elif dataset=='sdss':
    survey_name='SMP_Data'
    #rootdir='/home/michelle/SN_Class/Simulations/'+survey_name+'/'
    rootdir='/home/roberts/data_sets/sne/sdss/'+survey_name+'/'
    if subset=='spectro':
        objects=np.genfromtxt(rootdir+'spectro.list', dtype='str')
    else:
        objects=np.genfromtxt(rootdir+survey_name+'.list', dtype='str') #Our list of objects to split up

elif 'lsst' in dataset:
    if dataset=='lsst_main':
        survey_name='ENIGMA_1189_10YR_MAIN'
    else:
        survey_name='ENIGMA_1189_10YR_DDF'
    rootdir='/home/mlochner/sn/'+survey_name+'/'
    objects=np.genfromtxt(rootdir+'high_SNR_snids.txt', dtype='str')
    #objects=np.loadtxt('missing_objects.txt',dtype='str')

elif 'cadence' in dataset:
    if dataset=='cadence_rolling':
        survey_name='Rolling_3_80_reshuffled_WFD'
    elif dataset=='cadence_minion':
        survey_name='minion_1016_WFD'
    rootdir='/share/hypatia/snmachine_resources/data/LSST_cadence_sims/'+survey_name+'/FullLightCurveFitsFiles/RH_LSST_SNMIX_WFD/'
    objects=np.genfromtxt(rootdir+'.LIST',dtype='str')


nobj=len(objects) #Total number of objects

#Figure out how many objects go onto cores24 processors and how many on cores12
nobj12=(int)(nobj/(proc12+proc24)*proc12)
nobj24=nobj-nobj12

#Split the data as evenly as possible amongst processors
if n12>0:
    obj12=np.array_split(objects[:nobj12], n12)
else:
    obj12=[]
if n24>0:
    obj24=np.array_split(objects[nobj12:], n24)
else:
    obj24=[]

#We write each set of objects to file and use it as an argument to run_pipeline.py to find the subset
for i in range(n12):
    subs=subset_name %i
    np.savetxt(job_dir+subs+'.txt', obj12[i], fmt='%s')
    make_job_script(12, subs, extra_flags='no-class')

for j in range(n24):
    subs=subset_name %(j+len(obj12))
    np.savetxt(job_dir+subs+'.txt', obj24[j], fmt='%s')
    make_job_script(24,subs, extra_flags='no-class')

#This is the job that does the classification, and is executed on one single node only
np.savetxt(job_dir+fullset_name+'.txt', objects, fmt='%s')
if n24>0:
    make_job_script(24, fullset_name, extra_flags='no-xtr')
    make_job_script(24, fullset_name, extra_flags='preprocess')
elif n12>0:
    make_job_script(12, fullset_name, extra_flags='no-xtr')
    make_job_script(12, fullset_name, extra_flags='preprocess')
else:
    print('You gave me no nodes to work on. Do not do that again.')

make_job_spawner(n12, n24, job_dir, subset_name)
