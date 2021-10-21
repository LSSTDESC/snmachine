"""
A somewhat messy example of how to run the pipeline end to end with choices of feature sets and classifiers.
"""
from __future__ import division
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from snmachine import sndata, snfeatures, snclassifier
import time, os, pywt,subprocess, sys#, StringIO
from sklearn.decomposition import PCA
from astropy.table import Table,join,vstack
import sklearn.metrics
import sncosmo
from functools import partial
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

dataset='cadence_rolling'
laptop=False
template_model='Ia'

nproc=2#number of processors
lim=10000000 #only use these many objects

#feature_sets=['templates','newling', 'karpenka', 'wavelets']
#feature_sets=['templates','newling', 'karpenka']
#feature_sets=['templates']
feature_sets=['wavelets']
#feature_sets=['templates', 'wavelets']

#cls=['knn', 'nb', 'neural_network','svm','boost_dt','random_forest']
#cls=['knn', 'nb', 'neural_network','svm','random_forest']
cls=['nb', 'boost_dt', 'knn', 'svm', 'random_forest']

use_redshift=False

sampler='nested' #Which sampling method are we using for parametric and template features

restart_from_features=True
restart_from_chains=False #If we want to change the way to compute best fit parameters
read_from_output=False
plot_data=False
select_training_data=True #If true, this will draw a new training set

if 'non-repr' in sys.argv:
    train_choice='non-repr'
else:
    train_choice='repr' #Do we create a representative training set



if dataset=='des':
    if len(sys.argv)>1:

        if 'redshift' in sys.argv:
            use_redshift=True
            run_name='des_z_augment'
        else:
            use_redshift=False
            run_name='des_no_z_augment'

        if 'non-repr' in sys.argv:
            train_choice='non-repr'
            repr=False
        else:
            train_choice='repr'
            repr=True
        if train_choice is not 'repr':
            run_name+='_non_repr'
        fl=sys.argv[1]
        subset=np.genfromtxt(fl, dtype='str')[:lim]

        subset_name=fl.split('/')[-1].split('.')[0]
        if len(sys.argv)>2:
            nproc=(int)(sys.argv[2])

    else:
        spec=np.genfromtxt('DES_spectro.list',dtype='str')
        subset=spec[:lim]
        run_name='des_spectro'
        subset_name='des_spectro'
        if use_redshift:
            subset_name+='_z'

elif 'lsst' in dataset:
    #run_name='lsst_10'
    #subset=range(18)
    if 'MAIN' in sys.argv[1]:
        dataset='lsst_main'
    else:
        dataset='lsst_ddf'
    run_name=dataset

    if 'redshift' in sys.argv:
        use_redshift=True
        run_name+='_z'
    else:
        use_redshift=False
        run_name+='_no_z'

    if 'non-repr' in sys.argv:
        train_choice='non-repr'
        repr=False
    else:
        train_choice='repr'
        repr=True
    if train_choice is not 'repr':
        run_name+='_non_repr'

    if len(sys.argv)>1:
        fl=sys.argv[1]
        subset=np.genfromtxt(fl, dtype='str')[:lim]

        #subset_name=fl.split('/')[-1].split('.')[0]
        subset_name=dataset+'_all'
        if len(sys.argv)>2:
            nproc=(int)(sys.argv[2])
    else:
        subset='none'
        subset_name=dataset

elif 'cadence' in dataset:
    run_name=
    subset=
    subset_name=

    if 'redshift' in sys.argv:
        use_redshift=True
        run_name+='_z'
    else:
        use_redshift=False
        run_name+='_no_z'
    train_choice=

elif dataset=='sdss':
    #run_name='lsst_10'
    #subset=range(18)
    run_name='sdss'
    subset='none'
    subset_name='sdss'

    if 'redshift' in sys.argv:
            use_redshift=True
            run_name='sdss_z'
    else:
        use_redshift=False
        run_name='sdss_no_z'

    if 'non-repr' in sys.argv:
        train_choice='non-repr'
        repr=False
    else:
        train_choice='repr'
        repr=True
    if train_choice is not 'repr':
        run_name+='_non_repr'

    fl=sys.argv[1]
    subset=np.genfromtxt(fl, dtype='str')[:lim]

    subset_name=fl.split('/')[-1].split('.')[0]
    if len(sys.argv)>2:
        nproc=(int)(sys.argv[2])

print('Using redshift: '+str(use_redshift)+'; training choice '+str(train_choice))


#####################################################################################################################
############################################### READ DATA ###########################################################
#####################################################################################################################


if laptop:
    outdir=os.path.join(os.path.sep, 'home', 'robert', 'data_sets', 'sne','sn_output','output_%s' %(run_name),'')
    final_outdir=outdir
    #outdir=os.path.join(os.path.sep, 'home', 'michelle', 'output_%s' %(run_name),'')
else:
    final_outdir=os.path.join(os.path.sep, 'home', 'roberts','data_sets', 'sne', 'sn_output','output_%s' %(run_name), '')
    outdir=os.path.join(os.path.sep, 'state','partition1', 'roberts', '')
print('temp outdir '+outdir)
print('final outdir '+final_outdir)

#This is where to save intermediate output for the feature extraction method. In some cases (such as the wavelets), these
#files can be quite large.
out_features=os.path.join(outdir, 'features', '')
out_class=os.path.join(outdir, 'classifications', '')
out_inter=os.path.join(outdir, 'int', '')

if not os.path.exists(final_outdir):
    os.makedirs(final_outdir)
    os.makedirs(os.path.join(final_outdir, 'features', ''))
    os.makedirs(os.path.join(final_outdir, 'int', ''))
    os.makedirs(os.path.join(final_outdir, 'classifications', ''))

if not os.path.exists(out_features):
    os.makedirs(out_features)
if not os.path.exists(out_class):
    os.makedirs(out_class)
if not os.path.exists(out_inter):
    os.makedirs(out_inter)

def copy_files(delete=True):
    print('Copying files ...')
    t1=time.time()
    if not os.path.exists(final_outdir):
        os.makedirs(final_outdir)
        os.makedirs(os.path.join(final_outdir, 'features', ''))
        os.makedirs(os.path.join(final_outdir, 'int', ''))
        os.makedirs(os.path.join(final_outdir, 'classifications', ''))

    if delete:
        os.system('rsync -avq --remove-source-files %s* %s'%(outdir, final_outdir))
        os.system('rm -r %s'%(outdir))
    else:
        os.system('rsync -avq %s* %s'%(outdir, final_outdir))
#    os.system('rsync %s* %s' %(os.path.join(outdir, 'features', ''), os.path.join(final_outdir, 'features', '')))
#    print 'rsync %s* %s' %(os.path.join(outdir, 'features', ''), os.path.join(final_outdir, 'features', ''))
#    os.system('rsync %s* %s' %(os.path.join(outdir, 'int', ''), os.path.join(final_outdir, 'int', '')))
#    os.system('rsync %s* %s' %(os.path.join(outdir, 'classifications', ''), os.path.join(final_outdir, 'classifications', '')))

    print('Time taken for file copying '+str(time.time()-t1))

def select_training(obj_names, out_features_dir, repr=True, training_names=None, num_training=0):
    print('Select training set ...')
    if repr:
        if num_training==0: #Number of objects in the training set
            num_training=(int)(len(obj_names)*1104/21319)
        indices=np.random.permutation(len(obj_names))
        train_inds=indices[:num_training]
        test_inds=indices[num_training:]
        training_set=obj_names[train_inds]
        test_set=obj_names[test_inds]


    else: #If a non-representative training set, the training set names must be supplied
        if training_names is None:
            print('Non-representative training set requested but training_names==None')
            sys.exit(0)
        training_set=training_names
        mask=np.in1d(obj_names, training_names)
        test_set=obj_names[np.where(~mask)[0]]
    print('Length of training and test sets: '+str(len(training_set))+' vs '+str(len(test_set)))
    np.savetxt(out_features_dir+'train-%s.txt' %train_choice, training_set, fmt='%s')
    np.savetxt(out_features_dir+'test-%s.txt' %train_choice, test_set, fmt='%s')
    print('Written training set to '+out_features_dir+'train-%s.txt' %train_choice)


if dataset=='des':
    #Root directory for simulated data. Replace with your own path.
    #Note: the use of os.path.join means you don't have to worry about operating system-specific things like which way the slashes go
    if laptop:
        rt=os.path.join(os.path.sep, 'home', 'robert', 'data_sets', 'sne', 'spcc', 'SIMGEN_PUBLIC_DES_AUGM',  '')
    else:
        rt=os.path.join(os.path.sep, 'home', 'roberts','data_sets', 'sne','des', 'SIMGEN_PUBLIC_DES_AUGM','')

    #Subset tells the initialiser to only load a subset of the data, in this case only the spectroscopic data
#    d=sndata.Dataset(rt,subset=subset)
    d=sndata.EmptyDataset(folder=rt)
    for obj in subset:
        lc=sncosmo.read_lc(rt+obj)
#        print(lc.meta)
        d.insert_lightcurve(lc)
#        print('Inserted light curve of object '+str(obj))

    #Extract the types for all objects. This is specific to this example
    #types=np.zeros(len(d.object_names),dtype=int)
    #for i in range(len(d.object_names)):
    #    obj=d.object_names[i]
    #    types[i]=d.data[obj].meta['type']


    types=d.get_types()

    #The spectroscopic subsample is quite small so we can't subtype. We replace 21,22 etc. with 2 and 31,32 etc. with 3
    types['Type'][np.floor(types['Type']/10)==2]=2
    types['Type'][np.floor(types['Type']/10)==3]=3



    # #### Choose training and validation samples
    if len(cls)>0:

        #We choose the indices for the training and validation samples here, to be used for all different feature methods.
        #For reproducibility, it is worth saving these indices for later. Deleting the file will result in a new sample being drawn.
        #Note: These must be a random selection for the spectro subsample, otherwise the training set will be non-representative
        print('out_features '+out_features)

        if select_training_data and 'no-class' not in sys.argv:
            spec_objs=np.genfromtxt(rt+'DES_spectro.list', dtype='str')
            if repr:
                training_names=None
            else:
                training_names=spec_objs
            if 'spectro' in run_name:
                select_training(d.object_names, out_features, repr=repr, training_names=training_names)
            else:
                select_training(d.object_names, out_features, repr=repr, training_names=training_names, num_training=len(spec_objs))
            #select_training(d.object_names, out_features, repr=repr, training_names=training_names, num_training=300)

            training_set=np.genfromtxt(out_features+'train-%s.txt' %train_choice, dtype='str')
            test_set=np.genfromtxt(out_features+'test-%s.txt' %train_choice, dtype='str')


        #train_inds=[np.where(d.object_names==x)[0][0] for x in training_set]
        #val_inds=[np.where(d.object_names==x)[0][0] for x in test_set]

        #Separate out the training and validation types. The separation of the features will come later, after feature extraction.
        #Ytrain=types[train_inds]
        #Yval=types[val_inds]

        #print np.unique(Ytrain),np.unique(Yval)

elif 'lsst' in dataset or dataset=='sdss':
    if 'lsst' in dataset:
        if dataset=='lsst_main':
            rt_name='ENIGMA_1189_10YR_MAIN'
        else:
            rt_name='ENIGMA_1189_10YR_DDF'
        if laptop:
            rt=os.path.join(os.path.sep, 'home','michelle','BigData','SN_Sims',rt_name,'')
        else:
            rt=os.path.join(os.path.sep, 'home', 'mlochner','sn',rt_name,'')
        d=sndata.OpsimDataset(rt,subset=subset,mix=False)

    else:
        if laptop:
            #rt=os.path.join(os.path.sep, 'home','michelle','BigData','SN_Sims','SMP_Data','')
            rt=os.path.join(os.path.sep, 'home','robert','data_sets', 'sne', 'sdss', 'full_dataset', 'SMP_Data','')
        else:
            rt=os.path.join(os.path.sep, 'home', 'roberts','data_sets', 'sne','sdss', 'augmented_SMP','')
#        d=sndata.SDSS_Data(rt,subset=subset)
        d=sndata.EmptyDataset()
        objs=np.genfromtxt(os.path.join(rt, 'augmented_SMP.list'), dtype='str')
        for o in objs:
            lc=Table.read(os.path.join(rt, o)+'.hdf5')
            d.insert_lightcurve(lc)

        for obj in d.object_names:
            d.data[obj].remove_rows(np.where(d.data[obj]['filter']=='sdssu')[0])

        if train_choice=='non-repr':
            training_names=np.genfromtxt(rt+'spectro.list', dtype='str')
        else:
            training_names=None
    types=d.get_types()

    types['Type'][np.floor(types['Type']/10)==2]=2
    types['Type'][np.floor(types['Type']/10)==3]=3

    flname=rt+'%s_indices.txt' %run_name
        #os.remove(flname)
    if os.path.isfile(flname):
        indices=np.genfromtxt(flname,dtype=int).tolist()
    else:
        indices=np.random.permutation(len(types))
        np.savetxt(flname, indices, fmt='%d')

if 'no-class' not in sys.argv:
    select_training(d.object_names, out_features, repr=repr, training_names=training_names)
    copy_files(delete=False)
    training_set=np.genfromtxt(out_features+'train-%s.txt' %train_choice, dtype='str')
    test_set=np.genfromtxt(out_features+'test-%s.txt' %train_choice, dtype='str')

    training_set.sort()
    test_set.sort()

    training_set=np.array(training_set,dtype='str')
    test_set=np.array(test_set,dtype='str')

    train_inds=np.in1d(d.object_names,training_set).nonzero()[0]
    val_inds=np.in1d(d.object_names, test_set).nonzero()[0]

    Ytrain=types[train_inds]
    Yval=types[val_inds]

#    copy_files()
#    print('sanity check: length of training and test set: '+str(len(train_inds))+' / '+str(len(val_inds)))
#    print(train_inds)
#    print
#    print(val_inds)
#    if len(np.intersect1d(d.object_names, training_set))>0:
#        train_inds=[np.where(d.object_names==x)[0][0] for x in training_set]
#    else:
#        train_inds=[]
#    if len(np.intersect1d(d.object_names, test_set))>0:
#        val_inds=[np.where(d.object_names==x)[0][0] for x in test_set]
#    else:
#        val_inds=[]

    #
    # Ytrain=types[train_inds]
    # Yval=types[val_inds]
    #
    # print np.unique(Ytrain['Type']),np.unique(Yval['Type'])



if 'preprocess' in sys.argv:
    if not os.path.exists(os.path.join(final_outdir, 'features', '')):
        os.makedirs(os.path.join(final_outdir, 'features', ''))
    maxfile=open(os.path.join(final_outdir, 'features', 'maximum_lightcurve_length.txt') , 'w')
    maxfile.write(str(d.get_max_length())+'\n')
    maxfile.close()
    sys.exit(0)

################################################################################
####### FEATURE EXTRACTION #####################################################
################################################################################

def append_line(flname):
    #Append one line to the logfile that contains all file names of successfully extracted features.
    #Rerunning the pipeline for classification will collate all feature files listed in there.
    logfile=open(os.path.join(final_outdir, 'features', 'extracted_feature_filenames.txt'), 'a')
    logfile.write(os.path.basename(flname)+'\n')
    logfile.close()


if 'no-xtr' not in sys.argv:

    # ### Templates feature extraction ### #

    if 'templates' in feature_sets:
        #Create a Features object

        if 'lsst' in dataset:
            if laptop:
                lsst_dir='/home/michelle/Project/SN_Class/snmachine/lsst_bands/'
            else:
                lsst_dir='/home/mlochner/snmachine/lsst_bands/'
            tempFeat=snfeatures.TemplateFeatures(model=[template_model], sampler=sampler,lsst_bands=True,lsst_dir=lsst_dir)
        else:
            tempFeat=snfeatures.TemplateFeatures(model=[template_model], sampler=sampler)
        flname=os.path.join(out_features, subset_name+'_templates.dat')
    #flname=os.path.join(out_features, run_name+'_templates.dat')
        print('features '+flname)
        if restart_from_features and os.path.exists(flname):
            print('Restarting from '+flname)
            template_features=Table.read(flname, format='ascii')[:lim]
        else:
            #Run feature extraction
            template_features=tempFeat.extract_features(d, chain_directory=out_inter, nprocesses=nproc,
                                                    use_redshift=use_redshift, restart=restart_from_chains)
            template_features.write(flname, format='ascii')
        blah=template_features['Object'].astype(str)
        template_features.replace_column('Object', blah)

        if plot_data:
            d.set_model(tempFeat.fit_sn,template_features)
            d.plot_all()

        #Copy across intermediate files from temp directory
        if laptop==False and outdir != final_outdir:
            copy_files()
        append_line(flname)

    # ### Parametric feature extraction ### #

    # Newling parameterisation

    if 'newling' in feature_sets:
        newlingFeat=snfeatures.ParametricFeatures('newling', sampler=sampler)
        print('out_features '+out_features)
        flname=os.path.join(out_features, subset_name+'_newling.dat')

        if restart_from_features and os.path.exists(flname):
            print('Restarting from '+flname)
            newling_features=Table.read(flname, format='ascii')
        else:
            newling_features=newlingFeat.extract_features(d,chain_directory=out_inter, nprocesses=nproc,
                                                      convert_to_binary=True, restart=restart_from_chains)
            print('flname '+flname)
            newling_features.write(flname,format='ascii')

        blah=newling_features['Object'].astype(str)
        newling_features.replace_column('Object', blah)

        if plot_data:
            d.set_model(newlingFeat.fit_sn,newling_features)
            d.plot_all()
        #Copy across intermediate files from temp directory
        if laptop==False and outdir != final_outdir:
            copy_files()
        append_line(flname)

    # Karpenka parameterisation

    if 'karpenka' in feature_sets:
        karpenkaFeat=snfeatures.ParametricFeatures('karpenka', sampler=sampler)
        flname=os.path.join(out_features, subset_name+'_karpenka.dat')

        if restart_from_features and os.path.exists(flname):
            print('Restarting from '+flname)
            karpenka_features=Table.read(flname, format='ascii')
        else:
            karpenka_features=karpenkaFeat.extract_features(d,chain_directory=out_inter,
                                                        nprocesses=nproc, restart=restart_from_chains)
            karpenka_features.write(flname,format='ascii')

        blah=karpenka_features['Object'].astype(str)
        karpenka_features.replace_column('Object', blah)

        if plot_data:
            d.set_model(karpenkaFeat.fit_sn,karpenka_features)
            d.plot_all()
        #Copy across intermediate files from temp directory
        if laptop==False and outdir != final_outdir:
            copy_files()
        append_line(flname)


    # ### Wavelet feature extraction ### #

    if 'wavelets' in feature_sets:
        waveletFeat=snfeatures.WaveletFeatures(wavelet='sym2', ngp=100)
        flname=os.path.join(out_features, subset_name+'_wavelets.dat')

        xmin=0
        maxfilepath=os.path.join(final_outdir, 'features', 'maximum_lightcurve_length.txt')
        if 'no-class' in sys.argv and os.path.exists(maxfilepath):
            maxfile=open(maxfilepath , 'r')
            xmax=float(maxfile.read())
            maxfile.close()
        else:
            xmax=d.get_max_length()

        if restart_from_features and os.path.exists(flname):
            print('Restarting from '+flname)
            wavelet_features=Table.read(flname, format='ascii')[:lim]

        else:
            wavelet_features=waveletFeat.extract_features(d, save_output='all',restart='none', output_root=out_inter, nprocesses=nproc, xmax=xmax)
            wavelet_features.write(flname,format='ascii')

#            subprocess.call(['cp',out_features+'PCA_vals.txt',out_features+'%s_%s_PCA_vals.txt' %(run_name, subset_name)])
 #           subprocess.call(['cp',out_features+'PCA_vec.txt',out_features+'%s_%s_PCA_vec.txt' %(run_name, subset_name)])
  #          subprocess.call(['cp',out_features+'PCA_mean.txt',out_features+'%s_%s_PCA_mean.txt' %(run_name, subset_name)])

        blah=wavelet_features['Object'].astype(str)
        wavelet_features.replace_column('Object', blah)


        if plot_data:
            vals=np.genfromtxt(out_features+'%s_%s_PCA_vals.txt' %(run_name, subset_name))
            vec=np.genfromtxt(out_features+'%s_%s_PCA_vec.txt' %(run_name, subset_name))
            mn=np.genfromtxt(out_features+'%s_%s_PCA_mean.txt' %(run_name, subset_name))

            d.set_model(waveletFeat.fit_sn,wavelet_features,vec,  mn, xmin, xmax,d.filter_set)
            d.plot_all()
        #Copy across intermediate files from temp directory
        if laptop==False and outdir != final_outdir:
            copy_files()
        append_line(flname)


else:
    #This part gets executed if we run the pipeline with the flag 'no-extr'. We read in the feature files that
    #are listed in the logfile, insert the extracted features, and move on to classification.
    #NB: The feature extraction can be distributed on nodes. The classification is to be on a single node only!

    logfile=open(os.path.join(final_outdir, 'features/extracted_feature_filenames.txt'), 'r')

    if 'templates' in feature_sets:
        template_features=[]
    if 'newling' in feature_sets:
        newling_features=[]
    if 'karpenka' in feature_sets:
        karpenka_features=[]

    for filename in logfile.readlines():
        new_feat_subset=Table.read(os.path.join(final_outdir, 'features', filename.strip('\n')), format='ascii')[:lim]
        if 'templates' in feature_sets and 'templates' in filename:
            if len(template_features)==0:
                template_features=new_feat_subset
            else:
                template_features=vstack([template_features,new_feat_subset])
        if 'newling' in feature_sets and 'newling' in filename:
            if len(newling_features)==0:
                newling_features=new_feat_subset
            else:
                newling_features=vstack([newling_features,new_feat_subset])
        if 'karpenka' in feature_sets and 'karpenka' in filename:
            if len(karpenka_features)==0:
                karpenka_features=new_feat_subset
            else:
                karpenka_features=vstack([karpenka_features,new_feat_subset])

    if 'wavelets' in feature_sets:
        wavelet_feats=snfeatures.WaveletFeatures(wavelet='sym2', ngp=100)
        wave_raw, wave_err=wavelet_feats.restart_from_wavelets(d, os.path.join(final_outdir, 'int', ''))
        wavelet_features,vals,vec,means=wavelet_feats.extract_pca(d.object_names.copy(), wave_raw)


    #we still want to write all features to file, incl pca components and suchlike

    if 'templates' in feature_sets:
        print('Read in template features for '+str(len(template_features))+' lightcurves.')
        flname=os.path.join(final_outdir, 'features', subset_name+'_wavelets.dat')
        template_features.write(flname,format='ascii')
    if 'newling' in feature_sets:
        print('Read in Newling features for '+str(len(newling_features))+' lightcurves.')
        flname=os.path.join(final_outdir, 'features', subset_name+'_wavelets.dat')
        newling_features.write(flname,format='ascii')
    if 'karpenka' in feature_sets:
        print('Read in Karpenka features for '+str(len(karpenka_features))+' lightcurves.')
        flname=os.path.join(final_outdir, 'features', subset_name+'_wavelets.dat')
        karpenka_features.write(flname,format='ascii')
    if 'wavelets' in feature_sets:
        print('Read in wavelet features for '+str(len(wavelet_features))+' lightcurves.')
        flname=os.path.join(final_outdir, 'features', subset_name+'_wavelets.dat')
        wavelet_features.write(flname,format='ascii')
    logfile.close()



################################################################################
####### CLASSIFICATION #########################################################
################################################################################



def do_classification(classifier, param_dict, Xtrain, Ytrain, Xtest, read_from_output, save_output, feature_set):
    flname='%s%s_%s.dat' %(out_class, feature_set, classifier)
    if read_from_output and os.path.exists(flname):
        print('reading from file')
        tab=Table.read(flname,format='ascii')
        probs=np.array([tab[c] for c in tab.columns[1:]]).T
        probs=probs[1:,:]
    else:
        c=snclassifier.OptimisedClassifier(classifier)

        if classifier in param_dict.keys():
            Yfit, probs=c.optimised_classify(Xtrain, Ytrain, Xtest,params=param_dict[classifier])
        else:
            Yfit, probs=c.optimised_classify(Xtrain, Ytrain, Xtest) #Use defaults

        if save_output:
            fil=open('%shyper_params_%s_%s.txt' %(out_class,feature_set,classifier),'w')
            fil.write((str)(c.clf.best_params_))
            fil.close()
            ####################################################
            #print(np.shape(val_inds))
            #print(np.shape(probs))
            dat=np.column_stack((d.object_names[val_inds],probs))
            typs=np.unique(Ytrain)
            typs.sort()
            typs=np.array(typs,dtype='str').tolist()
            nms=['Object']+typs
            tab=Table(dat,dtype=['S64']+['f']*probs.shape[1],names=nms)
            tab.write(flname,format='ascii')

            if classifier=='boost_forest':
                np.savetxt('%s%s_%s.importances' %(out_class, feature_set, classifier), c.clf.best_estimator_.feature_importances_)

    return probs

def run_classifier(Xtrain,Ytrain,Xtest,Ytest,save_output=True,feature_set='templates',out_root='',read_from_output=False, nprocesses=1):
    #Note the feature_set and the out_root kwargs are only used for naming output files and so are only relevant if save_output==True
    #These are the current possible choices of classifier.
    #read_from_output will bypass the actual classification and read from files instead (if you want to replot a ROC curve)


    #You set classifier-specific parameter ranges here. If the classifier is not in this dictionary, it will use the
    #defaults to search for optimum parameters. If your parameter range is restrictive, it will print a warning and you
    #should then increase your range to ensure you get an optimum. SVM is particularly sensitive to this.
    param_dict={}

    FOM={}
    F1={}
    n_features=Xtrain.shape[1]

    #Rescale the data
    scaler = StandardScaler()
    scaler.fit(np.vstack((Xtrain, Xtest)))
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    t1=time.time()

    if nprocesses<2:
        for i in range(len(cls)):
            flname='%s_%s_%s.dat' %(run_name,feature_set,cls[i])
            probs=do_classification(cls[i], param_dict, Xtrain, Ytrain, Xtest, read_from_output, save_output, feature_set)

            fpr, tpr, auc=snclassifier.roc(probs, Ytest, true_class=1)
            f1, thresh_f1=snclassifier.F1(probs, Ytest, 1, full_output=False)
            fom, thresh_fom=snclassifier.FoM(probs, Ytest, 1, full_output=False)
            if i==0:
                FPR=fpr
                TPR=tpr
                AUC=[auc]
            else:
                FPR=np.column_stack((FPR, fpr))
                TPR=np.column_stack((TPR, tpr))
                AUC.append(auc)
            FOM[cls[i]]=fom
            F1[cls[i]]=f1

    else:
        partial_func=partial(do_classification, param_dict=param_dict, Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest,
                            read_from_output=read_from_output, save_output=save_output, feature_set=feature_set)
        p=Pool(nprocesses)
        probs_all=p.map(partial_func, cls)

        for i in range(len(cls)):
            probs=probs_all[i]
            fpr, tpr, auc=snclassifier.roc(probs, Ytest, true_class=1)
            f1, thresh_f1=snclassifier.F1(probs, Ytest, 1, full_output=False)
            fom, thresh_fom=snclassifier.FoM(probs, Ytest, 1, full_output=False)
            flname='%s%s_%s.' %(out_class, feature_set, cls[i])
            np.savetxt(flname+'roc', np.column_stack((fpr, tpr)))
            np.savetxt(flname+'auc', [auc])
            if i==0:
                FPR=fpr
                TPR=tpr
                AUC=[auc]
            else:
                FPR=np.column_stack((FPR, fpr))
                TPR=np.column_stack((TPR, tpr))
                AUC.append(auc)
            FOM[cls[i]]=fom
            F1[cls[i]]=f1

    print('Kessler figure of merit')
    print(FOM)
    print()
    print('F1')
    print(F1)
    print()
    print('AUC')
    print(AUC)
    print()
    print('Time taken '+str((time.time()-t1)/60.)+' min')
    #Plot the roc curve for these classifiers, for this feature set.
    if save_output:
        snclassifier.plot_roc(FPR, TPR, AUC, labels=cls,label_size=16,tick_size=12)
        plt.savefig(os.path.join(os.path.sep, out_root, 'new_roc_%s_%s.png' %(run_name,feature_set)))
        plt.close()



if len(cls)>0 and 'no-class' not in sys.argv:

    if 'templates' in feature_sets:
        flname='output_%s_%s.txt' %('templates', run_name)
        print()
        print('TEMPLATES')
        print()
        # Xtrain=template_features[np.in1d(template_features['Object'],training_set)]
        # Xval=template_features[np.in1d(template_features['Object'],test_set)]
        #
        # Xtrain=np.array([Xtrain[c] for c in Xtrain.columns[1:]]).T
        # Xval=np.array([Xval[c] for c in Xval.columns[1:]]).T
        #
        # Xtrain=Xtrain[:,2:]
        # Xval=Xval[:,2:]

        # np.savetxt('Xtrain',Xtrain)
        # np.savetxt('Ytrain', Ytrain)
        # np.savetxt('Xtest', Xval)
        # np.savetxt('Ytest', Yval)


        # inds=np.random.permutation(range(len(Xtrain)))
        # Xtrain=Xtrain[inds]
        # Ytrain=Ytrain[inds]
        #
        # inds = np.random.permutation(range(len(Xval)))
        # Xval = Xval[inds]
        # Yval = Yval[inds]

        new_feats = join(template_features, types, keys='Object')
        #print(len(new_feats))
        #new_feats=new_feats['Object', 'z','t0', 'x0', 'x1','c', 'Type']
        #new_feats = new_feats['Object', 'z','t0', 'Type']
#        new_feats.write('/state/partition1/roberts/thesearethefeaturesafterreadin', format='ascii')

#        if np.sort(d.object_names) != np.sort(np.array(new_feats['Object'])):
#            print('Alarm!')


        Xtrain = new_feats[np.in1d(new_feats['Object'], training_set)]
        Ytrain = np.array(Xtrain['Type'], dtype='int')
        Xval = new_feats[np.in1d(new_feats['Object'], test_set)]
        Yval = np.array(Xval['Type'], dtype='int')

        print(len(Xtrain))
        print(len(Ytrain))
        print(len(Xval))
        print(len(Yval))

        Xtrain = np.array([Xtrain[c] for c in Xtrain.columns[1:-1]]).T
        print(Xtrain.shape)
        Xval = np.array([Xval[c] for c in Xval.columns[1:-1]]).T

        run_classifier(Xtrain,Ytrain,Xval,
                           Yval,feature_set='templates',read_from_output=read_from_output, nprocesses=nproc, out_root=out_class)
        copy_files()

    if 'newling' in feature_sets:
        flname='output_%s_%s.txt' %('newling', run_name)
        print()
        print('NEWLING')
        print()

        Xtrain=newling_features[np.in1d(newling_features['Object'],training_set)]
        Xval=newling_features[np.in1d(newling_features['Object'],test_set)]
        Xtrain=np.array([Xtrain[c] for c in Xtrain.columns[1:]]).T
        Xval=np.array([Xval[c] for c in Xval.columns[1:]]).T
        #Some parameters are very badly fit so we replace them with zeros
        Xtrain=np.nan_to_num(Xtrain)
        Xval=np.nan_to_num(Xval)

        run_classifier(Xtrain,Ytrain,Xval,
                           Yval,feature_set='newling',read_from_output=read_from_output, nprocesses=nproc, out_root=out_class)
        copy_files()

    if 'karpenka' in feature_sets:
        flname='output_%s_%s.txt' %('karpenka', run_name)
        print()
        print('KARPENKA')
        print()
        Xtrain=karpenka_features[np.in1d(karpenka_features['Object'],training_set)]
        Xval=karpenka_features[np.in1d(karpenka_features['Object'],test_set)]
        Xtrain=np.array([Xtrain[c] for c in Xtrain.columns[1:]]).T
        Xval=np.array([Xval[c] for c in Xval.columns[1:]]).T
        #Some parameters are very badly fit so we replace them with zeros
        Xtrain=np.nan_to_num(Xtrain)
        Xval=np.nan_to_num(Xval)

        run_classifier(Xtrain,Ytrain,Xval,
                           Yval,feature_set='karpenka',read_from_output=read_from_output, nprocesses=nproc, out_root=out_class)
        copy_files()

    if 'wavelets' in feature_sets:
        flname='output_%s_%s.txt' %('wavelets', run_name)
        print()
        print('WAVELETS')
        print()

        new_feats=join(wavelet_features,types,keys='Object')


        Xtrain=new_feats[np.in1d(new_feats['Object'],training_set)]
        Ytrain=np.array(Xtrain['Type'],dtype='int')
        Xval=new_feats[np.in1d(new_feats['Object'],test_set)]
        Yval = np.array(Xval['Type'], dtype='int')

        Xtrain=np.array([Xtrain[c] for c in Xtrain.columns[1:-1]]).T
        Xval=np.array([Xval[c] for c in Xval.columns[1:-1]]).T

        run_classifier(Xtrain,Ytrain,Xval,
                           Yval,feature_set='wavelets',read_from_output=read_from_output, nprocesses=nproc, out_root=out_class)
        copy_files()
