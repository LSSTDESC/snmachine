"""
Utility script for creating job scripts for running on a cluster.
This is specific to a cluster at UCL but can be adapted to any TORQUE cluster.
"""

from __future__ import division
from argparse import ArgumentParser
import os
import sys
import numpy as np


def make_job_script(ppn, subset_name, train_choice, extra_flags=''):
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
    fl.write('#!/bin/bash --norc\n \
#PBS -V\n \
%s\n \
#PBS -r n\n \
#PBS -S /bin/tcsh\n \
#PBS -q %s\n \
#PBS -l walltime=99:00:00\n' %(node_string,queue))
    # fl.write('source .tcshrc\n')
    fl.write('source ' + homedir + '/snmachine/install/setup.sh\n')
    fl.write('cd ' + homedir + '/snmachine/	\n')

    fl.write('python ' + homedir + '/snmachine/utils/run_pipeline.py %s%s.txt %d %s %s %s\n' %(job_dir, subset_name, ppn, reds, train_choice, extra_flags))
    #fl.write('python /home/michelle/SN_Class/snmachine/run_pipeline.py %s%s.txt\n' %(job_dir, subset_name))
    fl.close()

def make_job_spawner(job_dir, n12, n24, subset_root):
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

# if dataset=='des':
#     survey_name='SIMGEN_PUBLIC_DES'
#     #rootdir='/home/michelle/SN_Class/Simulations/'+survey_name+'/'
#     rootdir=homedir + '/data_sets/sne/des/'+survey_name+'/'
#     if subset=='spectro':
#         objects=np.genfromtxt('DES_spectro.list', dtype='str')
#     else:
#         objects=np.genfromtxt(rootdir+survey_name+'.LIST', dtype='str') #Our list of objects to split up
    #print(objects[0])
    #print(type(objects[0]))

# elif dataset=='sdss':
#     survey_name='SMP_Data'
#     #rootdir='/home/michelle/SN_Class/Simulations/'+survey_name+'/'
#     rootdir=homedir + '/data_sets/sne/sdss/'+survey_name+'/'
#     if subset=='spectro':
#         objects=np.genfromtxt(rootdir+'spectro.list', dtype='str')
#     else:
#         objects=np.genfromtxt(rootdir+survey_name+'.list', dtype='str') #Our list of objects to split up

# elif 'lsst' in dataset:
#     if dataset=='lsst_main':
#         survey_name='ENIGMA_1189_10YR_MAIN'
#     else:
#         survey_name='ENIGMA_1189_10YR_DDF'
#     rootdir='/home/mlochner/sn/'+survey_name+'/'
#     objects=np.genfromtxt(rootdir+'high_SNR_snids.txt', dtype='str')
    #objects=np.loadtxt('missing_objects.txt',dtype='str')

if __name__ == "__main__":

    # good_nodes=range(13, 18)+range(19,24) #We want to avoid node 18 (and any nodes that aren't free of course)
    #good_nodes=[15, 19, 21, 22, 23, 24]
    # good_nodes=[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    good_nodes=[]
    # good_nodes=[25, 26, 27, 28]
    node_ind=0

    homedir = os.environ['HOME']

    # job_dir= homedir + '/data_sets/sne/des/jobs/'
    utils_dir = os.environ.get('PWD')
    job_dir = utils_dir + '/jobs/'
    # job_dir = homedir + '/snmachine/jobs/'
    #job_dir='jobs/'
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    # n12=12 #How many cores12 nodes requesting
    # n24=len(good_nodes) #How many cores24 nodes requesting

    parser = ArgumentParser(description="Provide path to dataset one wishes to"
                            "create jobs for")

    parser.add_argument('-j', '--job_dir', help='Path to where job scripts will'
                        'be written', type=str, default=job_dir)

    parser.add_argument('-d', '--dataset', help='What dataset is being used',
                        type=str, default='des')

    parser.add_argument('-op', '--path_to_object_list', help='Path to list of'
                        'objects', type=str,
                        default='/share/hypatia/snmachine_resources/data/DES_spcc/SIMGEN_PUBLIC_DES/SIMGEN_PUBLIC_DES.LIST')

    parser.add_argument('-r', '--redshift', help='Use redshift or not',
                        type=bool, default=False)

    parser.add_argument('-t', '--train_choice', help='Representivitie or'
                        'non-Representivitie training data',
                        type=str, default='repr')

    parser.add_argument('-n12', '--num12cores', help='Number of 12 cores'
            'avaiable', type=int, default=12)

    parser.add_argument('-n24', '--num24cores', help='Number of 24 cores'
            'avaiable', type=int, default=len(good_nodes))

    try:
        arguments = parser.parse_args()

        print(arguments)
        # arguemts.dataset='des'
        subset='none'
        # arguments.train_choice='repr'

        # arguments.redshift=False
        if arguments.redshift:
            reds='redshift'
        else:
            reds=''

        # elif 'cadence' in dataset:
        #     if dataset=='cadence_rolling':
        #         survey_name='Rolling_3_80_reshuffled_WFD'
        #     elif dataset=='cadence_minion':
        #         survey_name='minion_1016_WFD'
        #     rootdir='/share/hypatia/snmachine_resources/data/LSST_cadence_sims/'+survey_name+'/FullLightCurveFitsFiles/RH_LSST_SNMIX_WFD/'
        #     objects=np.genfromtxt(rootdir+'.LIST',dtype='str')


        subset_name=arguments.dataset+'_subset_%d'
        fullset_name=arguments.dataset+'_fullset'

        print('path to list of objects ', arguments.path_to_object_list)

        objects=np.genfromtxt(arguments.path_to_object_list, dtype='str') #Our list of objects to split up

        print('Done reading the csv to an array\n')
        # if arguments.dataset=='des':
        #     survey_name='SIMGEN_PUBLIC_DES'
        #     #rootdir='/home/michelle/SN_Class/Simulations/'+survey_name+'/'
        #     rootdir=homedir + '/data_sets/sne/des/'+survey_name+'/'
        #     if subset=='spectro':
        #         objects=np.genfromtxt('DES_spectro.list', dtype='str')
        #     else:
        #         objects=np.genfromtxt(rootdir+survey_name+'.LIST', dtype='str') #Our list of objects to split up

         # elif dataset=='sdss':
         #     survey_name='SMP_Data'
         #     #rootdir='/home/michelle/SN_Class/Simulations/'+survey_name+'/'
         #     rootdir=homedir + '/data_sets/sne/sdss/'+survey_name+'/'
         #     if subset=='spectro':
         #         objects=np.genfromtxt(rootdir+'spectro.list', dtype='str')
         #     else:
         #         objects=np.genfromtxt(rootdir+survey_name+'.list', dtype='str') #Our list of objects to split up

         # elif 'lsst' in dataset:
         #     if dataset=='lsst_main':
         #         survey_name='ENIGMA_1189_10YR_MAIN'
         #     else:
         #         survey_name='ENIGMA_1189_10YR_DDF'
         #     rootdir='/home/mlochner/sn/'+survey_name+'/'
         #     objects=np.genfromtxt(rootdir+'high_SNR_snids.txt', dtype='str')
         #    objects=np.loadtxt('missing_objects.txt',dtype='str') -- not used

        nobj=len(objects) #Total number of objects

        print('Now figure out distribution\n')
        #Figure out how many objects go onto cores24 processors and how many on cores12
        proc12=arguments.num12cores*12 #Total number of cores12 processors
        proc24=arguments.num24cores*24
        nobj12=(int)(nobj/(proc12+proc24)*proc12)
        nobj24=nobj-nobj12

        #Split the data as evenly as possible amongst processors
        if arguments.num12cores>0:
            obj12=np.array_split(objects[:nobj12], arguments.num12cores)
        else:
            obj12=[]
        if arguments.num24cores>0:
            obj24=np.array_split(objects[nobj12:], arguments.num24cores)
        else:
            obj24=[]

        print('split files for data parallelization\n')
        #We write each set of objects to file and use it as an argument to run_pipeline.py to find the subset
        for i in range(arguments.num12cores):
            subs=subset_name %i
            np.savetxt(arguments.job_dir+subs+'.txt', obj12[i], fmt='%s')
            make_job_script(12, subs, arguments.train_choice, extra_flags='no-class')

        for j in range(arguments.num24cores):
            subs=subset_name %(j+len(obj12))
            np.savetxt(arguments.job_dir+subs+'.txt', obj24[j], fmt='%s')
            make_job_script(24,subs, arguments.train_choice, extra_flags='no-class')

        #This is the job that does the classification, and is executed on one single node only
        np.savetxt(job_dir+fullset_name+'.txt', objects, fmt='%s')
        if arguments.num24cores>0:
            make_job_script(24, fullset_name, arguments.train_choice, extra_flags='no-xtr')
            make_job_script(24, fullset_name, arguments.train_choice, extra_flags='preprocess')
        elif arguments.num12cores>0:
            make_job_script(12, fullset_name, arguments.train_choice, extra_flags='no-xtr')
            make_job_script(12, fullset_name, arguments.train_choice, extra_flags='preprocess')
        else:
            print('You gave me no nodes to work on. Do not do that again.')

        make_job_spawner(arguments.job_dir, arguments.num12cores, arguments.num24cores, subset_name)
        # make_job_spawner(arguments.n12, arguments.n24, arguments.jobsdir, subset_name)
    # except SystemExit:
    except parser.error(""):
        print("Invalid argument inputs")
        sys.exit()

# nobj=len(objects) #Total number of objects

# #Figure out how many objects go onto cores24 processors and how many on cores12
# nobj12=(int)(nobj/(proc12+proc24)*proc12)
# nobj24=nobj-nobj12

# #Split the data as evenly as possible amongst processors
# if n12>0:
#     obj12=np.array_split(objects[:nobj12], n12)
# else:
#     obj12=[]
# if n24>0:
#     obj24=np.array_split(objects[nobj12:], n24)
# else:
#     obj24=[]

# #We write each set of objects to file and use it as an argument to run_pipeline.py to find the subset
# for i in range(n12):
#     subs=subset_name %i
#     np.savetxt(job_dir+subs+'.txt', obj12[i], fmt='%s')
#     make_job_script(12, subs, extra_flags='no-class')

# for j in range(n24):
#     subs=subset_name %(j+len(obj12))
#     np.savetxt(job_dir+subs+'.txt', obj24[j], fmt='%s')
#     make_job_script(24,subs, extra_flags='no-class')

# #This is the job that does the classification, and is executed on one single node only
# np.savetxt(job_dir+fullset_name+'.txt', objects, fmt='%s')
# if n24>0:
#     make_job_script(24, fullset_name, extra_flags='no-xtr')
#     make_job_script(24, fullset_name, extra_flags='preprocess')
# elif n12>0:
#     make_job_script(12, fullset_name, extra_flags='no-xtr')
#     make_job_script(12, fullset_name, extra_flags='preprocess')
# else:
#     print('You gave me no nodes to work on. Do not do that again.')

# make_job_spawner(n12, n24, job_dir, subset_name)
# >>>>>>> augment
