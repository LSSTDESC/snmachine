"""
Module containing Dataset classes. These read in data from various sources and turns the light curves into astropy tables that
can be read by the rest of the code.
"""
from __future__ import division
from past.builtins import basestring
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from astropy.io import fits
import sys, sncosmo, os
from random import shuffle,sample
from scipy import interpolate
import sncosmo, math

#Colours for graphs
colours={'sdssu':'#6614de','sdssg':'#007718','sdssr':'#b30100','sdssi':'#d35c00','sdssz':'k','desg':'#007718','desr':'#b30100','desi':'#d35c00','desz':'k',
            'lssty':'#2727c1','lsstu':'#6614de','lsstg':'#007718','lsstr':'#b30100','lssti':'#d35c00','lsstz':'k','lsstY':'#2727c1'}
sntypes={1:'Ia',2:'II',21:'IIn',22:'IIP',23:'IIL',3:'Ibc',32:'Ib',33:'Ic',66:'other'}
markers={'desg':'^', 'desr':'o', 'desi':'s', 'desz':'*'}
labels={'desg':'g', 'desr':'r', 'desi':'i', 'desz':'z'}

def plot_lc(lc):
    """
    External function to plot light curves.

    Parameters
    ----------
    lc : astropy.table.Table
        Light curve

    """
    """
        @param fname The filename of the supernova (relative to data_root)
    """
    
    #This selects the filters from the possible set that this object has measurements in and maintains the order
    filts=np.unique(lc['filter'])
    
    #Keep track of the min and max values on the plot for resizing axes
    min_x=np.inf
    max_x=-np.inf
    lines=[]
    for j in range(len(filts)):
        inds=np.where(lc['filter']==filts[j])[0]
        if 'flux_error' in lc.keys():
            t, F, F_err=lc['mjd'][inds], lc['flux'][inds], lc['flux_error'][inds]
            error=True
        else:
            t, F=lc['mjd'][inds], lc['flux'][inds]
            error=False
        #tdelt=t-t.min()
        tdelt=t
        if filts[j] in markers.keys():
            mkr=markers[filts[j]]
        else:
            mkr='o'
        if error:
            l=plt.errorbar(tdelt, F,yerr=F_err,  marker=mkr,linestyle='none',  color=colours[filts[j]], markersize=4)
        else:
            l=plt.plot(tdelt, F,lw=2,  marker=mkr,color=colours[filts[j]])
        lines.append(l)
        if tdelt.min()<min_x:
            min_x=tdelt.min()
        if tdelt.max()>max_x:
            max_x=tdelt.max()
            
    
    ext=0.05*(max_x-min_x)
    plt.xlim([min_x-ext, max_x+ext])
    plt.xlabel('Time (days)',  fontsize=16)
    plt.ylabel('Flux',  fontsize=16)
    
    plt.legend(lines, filts, numpoints=1,loc='best')

class Dataset:
    """
    Class to manage the files from a single dataset. The base class works with data from the SPCC.
    This class can be inherited and overridden to work a completely different kind of dataset.
    All this class really needs is a list of object names and a method, get_lightcurve, which takes an individual object name and returns a light curve.
    Other functions provided here are for plotting and convenience.
    """
    
    
    def __init__(self, folder, subset='none',  filter_set=['desg', 'desr', 'desi', 'desz']):
        """
        Initialisation.

        Parameters
        ----------
        folder : str
            Root folder containing the data
        subset : str or list-like, optional
            Something you pass to get_object_names to specify which objects you want
        filter_set : list, optional
            List of possible filters used

        """

        self.filter_set=filter_set
        self.rootdir=folder
        self.survey_name=folder.split('/')[-2]

        self.object_names=self.get_object_names(subset=subset)
        #Get all the data as a list of astropy tables (this should not be memory intensive, even for large numbers of light curves)
        self.data={}
        invalid=0 #Some objects may have empty data
        print ('Reading data...')
        for i in range(len(self.object_names)):
            lc=self.get_lightcurve(self.object_names[i])
            if len(lc['mjd']>0):
                self.data[self.object_names[i]]=lc
            else:
                invalid+=1
        if invalid>0:
            print ('%d objects were invalid and not added to the dataset.' %invalid)
        print ('%d objects read into memory.' %len(self.data))
        #We create an optional model set which can be set by whatever feature class used
        self.models={}
        
    def get_object_names(self, subset='none'):
        """
        Gets a list of the names of the files within the dataset.

        Parameters
        ----------
        subset : str or list-like, optional
            Used to specify which files you want. Current setup is get_object_names will accept a list of
            indices, a list of actual object names as a subset or the keyword 'spectro'.

        """
        if isinstance(subset,basestring):
            if subset=='spectro':
                object_names= np.genfromtxt(self.rootdir+'spectro.list', dtype='str').flatten()
            else:
                object_names= np.genfromtxt(self.rootdir+self.survey_name+'.LIST', dtype='str')
        elif all(isinstance(l,basestring) for l in subset):
            #We assume subset is a list of strings containing object names
            object_names= subset
        else:
            #Otherwise it must be a list of indices. Otherwise raise an error.
            names=np.genfromtxt(self.rootdir+self.survey_name+'.LIST', dtype='str')
            try:
                object_names= names[subset]
            except IndexError:
                print ('Invalid subset provided')
                sys.exit()
                
        return np.sort(object_names)

    
    
    def get_max_length(self):
        """Gets the length (in days) of the longest observation in the dataset.
        """
        max_obs=0
        for n in self.object_names:
            times=self.data[n]['mjd']
            dif=times.max()-times.min()
            if dif>max_obs:
                max_obs=dif
        return max_obs
    
    def get_lightcurve(self, flname):
        """
        Given a filename, returns a light curve astropy table that conforms with sncosmo requirements

        Parameters
        ----------
        flname : str
            The filename of the supernova (relative to data_root)

        Returns
        -------
        astropy.table.Table
            Light curve
        """
        fl=open(self.rootdir+flname,'r')
        mjd=[]
        flt=[]
        flux=[]
        fluxerr=[]
        z=-9
        z_err=-9
        type=-9
        for line in fl:
            s=line.split()
            if len(s)>0:
                if s[0]=='HOST_GALAXY_PHOTO-Z:':
                    z=(float)(s[1])
                    z_err=(float)(s[3])
                elif s[0]=='OBS:':
                    mjd.append((float)(s[1]))
                    flt.append('des'+s[2])
                    flux.append((float)(s[4]))
                    fluxerr.append((float)(s[5]))
                elif s[0]=='SIM_COMMENT:':
                    for k in sntypes.keys():
                        if sntypes[k] in s:
                            type=(int)(k)
                    
        
        #Zeropoint
        zp=np.array([27.5]*len(mjd))
        zpsys=['ab']*len(mjd)
        
        #Make everything arrays
        mjd=np.array(mjd)
        flt=np.array(flt, dtype='str')
        flux=np.array(flux)
        fluxerr=np.array(fluxerr)
        start_mjd=mjd.min()
        mjd=mjd-start_mjd #We shift the times of the observations to all start at zero. If required, the mjd of the initial observation is stored in the metadata.
        #Note: obviously this will only zero the observations in one filter band, the others have to be zeroed if fitting functions.
        tab = Table([mjd, flt, flux, fluxerr, zp, zpsys], names=('mjd', 'filter', 'flux', 'flux_error', 'zp', 'zpsys'), meta={'name':flname,'z':z, 'z_err':z_err, 'type':type, 'initial_observation_time':start_mjd})
        
        return tab

    
    def __plot_this(self, fname, title=True, loc='best'):
        """
        Internal function used by other functions to plot light curves.

        Parameters
        ----------
        fname : str
            The filename of the supernova (relative to data_root)
        title : str, optional
            Put a title on the plot
        loc : str, optional
            Location of legend
        """

        lc=self.data[fname]

        #This selects the filters from the possible set that this object has measurements in and maintains the order
        filts=sorted(set(self.filter_set) & set(np.unique(lc['filter'])), key = self.filter_set.index)
        
        #Keep track of the min and max values on the plot for resizing axes
        min_x=np.inf
        max_x=-np.inf
        lines=[]
        for j in range(len(filts)):
            inds=np.where(lc['filter']==filts[j])[0]
            t, F, F_err=lc['mjd'][inds], lc['flux'][inds], lc['flux_error'][inds]
            #tdelt=t-t.min()
            tdelt=t
            if filts[j] in markers.keys():
                mkr=markers[filts[j]]
            else:
                mkr='o'
            
            #Plot the model, if it has been set
            if self.plot_model:
                if fname in self.models.keys():
                    mod=self.models[fname]
                    if mod is not None:
                        inds=np.where(mod['filter']==filts[j])[0]
                        t_mod, F_mod=mod['mjd'][inds], mod['flux'][inds]
                        plt.plot(t_mod, F_mod, color=colours[filts[j]])

            l=plt.errorbar(tdelt, F,yerr=F_err,  marker=mkr, linestyle='none',  color=colours[filts[j]], markersize=4)
            lines.append(l)
            if tdelt.min()<min_x:
                min_x=tdelt.min()
            if tdelt.max()>max_x:
                max_x=tdelt.max()
                

        ext=0.05*(max_x-min_x)
        plt.xlim([min_x-ext, max_x+ext])
        plt.xlabel('Time (days)')
        plt.ylabel('Flux')
        #plt.gca().tick_params(labelsize=8)
        if title:
            plt.title('Object: %s, z:%0.2f,  Type:%s' %(fname, lc.meta['z'], lc.meta['type']))
        labs=[]
        for f in filts:
            if f in labels.keys():
                labs.append(labels[f])
            else:
                labs.append(f)
        plt.legend(lines, labs, numpoints=1,loc=loc)
        #plt.subplots_adjust(left=0.3)
        
    
    def set_model(self, fit_sn, *args):
        """
        Can use any function to set the model for all objects in the data.

        Parameters
        ----------
        fit_sn : function
            A function which can take a light curve (astropy table) argument and a list of arguments and returns an astropy table
        args : list, optional
            Whatever arguments fit_sn requires
        """
        print ('Fitting supernova models...')
        for obj in self.object_names:
            self.models[obj]=fit_sn(self.data[obj], *args)
        print ('Models fitted.')
    
    def plot_lc(self, fname, plot_model=True, title=True, loc='best'):
        """Public function to plot a single light curve.

        Parameters
        ----------
        fname : str
            The filename of the supernova (relative to data_root)
        plot_model : bool, optional
            Whether or not to overplot the model
        title : str, optional
            Put a title on the plot
        loc : str, optional
            Location of the legend
        """
        self.plot_model=plot_model
        self.__plot_this(fname, title=title, loc=loc)
        plt.show()

    def __on_press(self, event):
        """
        Allows one to cycle through the supernovae in the dataset by hitting the left or right arrow keys.

        Parameters
        ----------
        event : keyboard event object
            Keyboard event (i.e. the left or right arrow button has been pressed)
        """

        event.canvas.figure.clear()
        if event.key=='right' and self.__ind<len(self.object_names)-1:
            self.__ind+=1
        elif event.key=='left' and self.__ind>0:
            self.__ind-=1
        self.__plot_this(self.object_names[self.__ind])
        event.canvas.draw()
    
    def plot_all(self, plot_model=True):
        """
        Plots all the supernovae in the dataset and allows the user to cycle through them with the left and
        right arrow keys.

        Parameters
        ----------
        plot_model : bool, optional
            Whether or not to overplot the model.
        """
        self.plot_model=plot_model #We use a class variable because this can't be passed directly to __on_press
        fig = plt.figure()
        self.__ind=-1
        fig.canvas.mpl_connect('key_press_event', self.__on_press)
        plt.plot([0, 0])
        #subplots_adjust(right=0.95, top=0.95)
        plt.show()
        
    def get_types(self):
        """
        Returns a list of the types of the entire dataset.

        Returns
        -------
        `~numpy.ndarray`
            Array of types
        """
        typs=[]
        for o in self.object_names:
            typs.append(self.data[o].meta['type'])
        tab=Table(data=[self.object_names,typs],names=['Object','Type'])
        return tab

    def get_redshift(self):
        """
        Returns a list of the redshifts of the entire dataset.

        Returns
        -------
        `~numpy.ndarray`
            Array of redshifts
        """
        z=[]
        for o in self.object_names:
            if 'z' in self.data[o].meta:
                z.append(self.data[o].meta['z'])
            else:
                z.append(-1)
        return np.array(z)
        
    def sim_stats(self, **kwargs):
        """
        Prints information about the survey/simulation.

        Parameters
        ----------
        indices : list-like, optional
            List of indices to indicate which objects to consider. This allows you to, for example,
        see the statistics of a training subsample.
        plot_redshift : bool, optional
            Plots a histogram of the redshift distribution
        """
        if 'indices' in kwargs:
            indices=kwargs['indices']
        else:
            indices=np.arange(len(self.object_names))
        if 'plot_redshifts' in kwargs:
            plot_redshifts=kwargs['plot_redshifts']
        else:
            plot_redshifts=True
        
        N=len(indices)
        redshifts=np.zeros(N)
        types=np.zeros(N)
        for i in np.arange(N):
            ind=indices[i]
            lc=self.get_lightcurve(self.object_names[ind])
            redshifts[i]=lc.meta['z']
            types[i]=lc.meta['type']
            
        print()
        print ('Total number of SNe: %d' %(N))
        print()
        ks=self.sntypes.keys()
        ks.sort()
        for k in ks:
            nk=len(np.where(types==k)[0])
            print ('Number of %s: %d (%0.2f%%)' %(self.sntypes[k],nk ,nk/N*100))
        nk=len(np.where(types==-9)[0])
        print ('Number of unknown: %d (%0.2f%%)' %(nk ,nk/N*100))
        
        if plot_redshifts==True:
            plt.hist(redshifts[redshifts!=-9], 30, facecolor='#0057f6')
            plt.xlabel('Redshift', fontsize=16)
            plt.show()

    def reduced_chi_squared(self, subset = 'none'):
        """
        Returns the reduced chi squared for each object, once a model has been set.

        Parameters
        ----------
        subset : str or list-like, optional
            List of a subset of object names. If not supplied, the full dataset will be used

        Returns
        -------
        dict
            Dictionary of reduced chi^2 for each object
        """

        if subset == 'none':
            data_list = self.object_names
        else:
            data_list = subset
        chi_dict ={}
        for name in data_list:
            lc=self.data[name]

            #This selects the filters from the possible set that this object has measurements in and maintains the order
            filts=sorted(set(self.filter_set) & set(np.unique(lc['filter'])), key = self.filter_set.index)
            chi_2 = 0
            N = 0 # counts total number of points
            for j in range(len(filts)):
                inds=np.where(lc['filter']==filts[j])[0]
                t, F, F_err = np.array(lc['mjd'][inds]), np.array(lc['flux'][inds]), np.array(lc['flux_error'][inds])
                #tdelt=t-t.min()
                tdelt=t
                N = N + len(tdelt)
                if name in self.models:
                    mod=self.models[name]
                    if mod is not None:
                        inds=np.where(mod['filter']==filts[j])[0]
                        t_mod, F_mod = np.array(mod['mjd'][inds]), np.array(mod['flux'][inds])

                        #interpolate the model so that we can compare the model with the data
                        int_f = interpolate.interp1d(t_mod, F_mod, kind='cubic')
                        F_mod_mjd = np.array(int_f(tdelt))
                        summand=(F-F_mod_mjd)**2/(F_err**2)
                        chi_2 = np.sum(summand) + chi_2

            chi_dict[name] = chi_2 / N

        return chi_dict



class OpsimDataset(Dataset):
    """
    Class to read in an LSST simulated dataset, based on OpSim runs and SNANA simulations.
    """
    def __init__(self, folder, subset='none', mix=False, filter_set=['lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty']):
        """
        Initialisation.

        Parameters
        ----------
        folder : str
            Folder where simulations are located
        subset : str or list-like, optional
            List of a subset of object names. If not supplied, the full dataset will be used
        mix : bool, optional
            The output of the simulations is often highly ordered, this randomly permutes the objects when they're read in
        filter_set : list-like, optional
            Set of possible filters
        """
        self.survey_name='LSST'
        self.filter_set=filter_set
        self.get_data(folder, subset=subset)
        self.models={}
        
        if mix==True:
            shuffle(self.object_names)
            
        
    def get_data(self, folder, subset='none'):
        """
        Reads in the simulated data

        Parameters
        ----------
        folder : str
            Folder where simulations are located
        subset : str or list-like, optional
            List of a subset of object names. If not supplied, the full dataset will be used

        """
        if ~isinstance(subset,basestring) and (all(isinstance(l,basestring) for l in subset)):
            #We have to deal with separate Ia and nIa fits files
            Ia_head=os.path.join(folder,'LSST_Ia_HEAD.FITS')
            nIa_head=os.path.join(folder,'LSST_NONIa_HEAD.FITS')

            df = fits.open(Ia_head)[1].data
            subset=np.char.strip(subset) #If these are read from the header they have to be stripped of white space

            Ia_ids=subset[np.in1d(subset,np.char.strip(df['SNID']))]

            df = fits.open(nIa_head)[1].data
            nIa_ids=subset[np.in1d(subset,np.char.strip(df['SNID']))]

            data_Ia =  sncosmo.read_snana_fits(Ia_head, os.path.join(folder,'LSST_Ia_PHOT.FITS'),snids=Ia_ids)
            data_nIa =  sncosmo.read_snana_fits(nIa_head, os.path.join(folder,'LSST_NONIa_PHOT.FITS'),snids=nIa_ids)
        else:
            data_Ia =  sncosmo.read_snana_fits(os.path.join(folder,'LSST_Ia_HEAD.FITS'), os.path.join(folder,'LSST_Ia_PHOT.FITS'))
            data_nIa =  sncosmo.read_snana_fits(os.path.join(folder,'LSST_NONIa_HEAD.FITS'), os.path.join(folder,'LSST_NONIa_PHOT.FITS'))


        if isinstance(subset,basestring)and subset=='Ia':
            all_data=data_Ia
        elif isinstance(subset,basestring) and subset=='nIa':
            all_data=data_nIa
        else:
            all_data=data_Ia+data_nIa
        self.data={}
        self.object_names=[]
        
        invalid=0 #Some objects may have empty data
        print ('Reading data...')
        
        for i in range(len(all_data)):
            snid=all_data[i].meta['SNID']
            if isinstance(subset,basestring) or ((snid in subset) or (i in subset)):
                self.object_names.append((str)(snid))
                lc=self.get_lightcurve(all_data[i])
                if len(lc['mjd']>0):
                    self.data[snid]=lc
                else:
                    invalid+=1
        if invalid>0:
            print ('%d objects were invalid and not added to the dataset.' %invalid)
        self.object_names=np.array(self.object_names, dtype='str')
        print ('%d objects read into memory.' %len(self.data))
        
        
    def get_lightcurve(self, tab):
        """
        Converts the sncosmo convention for the astropy tables to snmachine's.

        Parameters
        ----------
        tab : astropy.table.Table
            Light curve

        """
        tab_new=tab['MJD', 'FLUXCAL', 'FLUXCALERR', 'FLT']
        tab_new.rename_column('MJD','mjd')
        start_mjd=(tab_new['mjd']).min()
        tab_new['mjd']=tab_new['mjd']-start_mjd
        tab_new.rename_column('FLUXCAL','flux')
        tab_new.rename_column('FLUXCALERR','flux_error')
        tab_new.rename_column('FLT','filter')
        tab_new=Table(tab_new, dtype=['f', 'f', 'f', 'S64'])
        old_filts=['u', 'g', 'r', 'i', 'z', 'Y']
        for f in range(len(old_filts)):
            tab_new['filter'][tab_new['filter']==old_filts[f]]=self.filter_set[f]
        zp=Column(name='zp', data=np.array([27.5]*len(tab_new['mjd'])))
        zpsys=Column(name='zpsys', data=['ab']*len(tab_new['mjd']))
        tab_new.add_column(zp)
        tab_new.add_column(zpsys)
        
        tab_new.meta={'name':tab.meta['SNID'], 'z':tab.meta['REDSHIFT_FINAL'], 'z_err':tab.meta['REDSHIFT_FINAL_ERR'], 'type':tab.meta['SNTYPE'], 
        'initial_observation_time':start_mjd}
        return tab_new
        

class SDSS_Data(Dataset):    
    """
    Class to read in the SDSS supernovae dataset
    """

    def __init__(self, folder, subset='none', training_only=False, filter_set=['sdssu','sdssg', 'sdssr', 'sdssi', 'sdssz'], subset_length = False, classification = 'none'):
        """
        Initialisation
 
        Parameters
        ----------
        folder : str
            Folder where simulations are located
        subset : str or list-like, optional
            List of a subset of object names. If not supplied, the full dataset will be used
        filter_set : list-like, optional
            Set of possible filters
        subset_length : bool or int, optional
            If supplied, will return this many random objects (can be used in conjunction with subset="spectro")
        classification : str, optional
            Can return one specific type of supernova.
        """
        self.filter_set=filter_set
        self.rootdir=folder
        self.survey_name=folder.split(os.path.sep)[-2] # second to last / / /..
        self.object_names=self.get_object_names(subset=subset,subset_length=subset_length,classification=classification)
        #Get all the data as a list of astropy tables (this should not be memory intensive, even for large numbers of light curves)
        self.data={}
        invalid=0 #Some objects may have empty data
        print ('Reading data...')
        for i in range(len(self.object_names)):
            lc=self.get_lightcurve(self.object_names[i])
            if len(lc['mjd']>0):
                self.data[self.object_names[i]]=lc
            else:
                invalid+=1
        if invalid>0:
            print ('%d objects were invalid and not added to the dataset.' %invalid)
        print ('%d objects read into memory.' %len(self.data))
        #We create an optional model set which can be set by whatever feature class used
        self.models={}

    def get_SNe(self,subset_length):
        """
        Function to take all supernovae from Master SDSS data file  and return a random sample of SNe of user-specified length
        if requested
 
        Parameters
        ----------
        subset_length : int
            Number of objects to return

        Returns
        -------
        list-like
            List of object names
 
        """
 
        fl = open(self.rootdir+self.survey_name+'.LIST')
        SN = []
        for line in fl:
            s = line.split()
            if (s[5] == "SNIa" or s[5] == "SNIb" or s[5] == "SNIc" or s[5] == "SNII" or s[5] == "SNIa?" or s[5] == "pSNIa" or s[5] == "pSNIbc" or s[5] == "pSNII" or s[5] == "zSNIa" or s[5] == "zSNIbc" or s[5] == "zSNII"):
                if len(str(s[0])) == 3:
                    SN.append("SMP_000%s.dat" % s[0])
                elif len(str(s[0])) == 4:
                    SN.append("SMP_00%s.dat" % s[0])
                elif len(str(s[0])) == 5:
                    SN.append("SMP_0%s.dat" % s[0])
    #SN now contains all file names for supernovae
        if subset_length != False:
            SN = [SN[i] for i in sorted(sample(range(len(SN)), subset_length)) ]

        return SN

    def get_spectro(self,subset_length,classification):
        """
        Function to take all spectroscopically confirmed supernovae from Master file and return a random sample of SNe of user-specified length
        if requested
 
        Parameters
        ----------
        subset_length : bool or int
            Number of objects to return (False to return all)
        classification : str
            Can specify a particular type of supernova to return ('none' for all types)
 
        Returns
        -------
        list-like
            List of object names
        """
        fl = open(self.rootdir+self.survey_name+'.LIST')
        SN = []
        classes = []
        for line in fl:
            s = line.split()
            if (s[5] == "SNIa" or s[5] == "SNIb" or s[5] == "SNIc" or s[5] == "SNII"):
                classes.append(s[5])
                if len(str(s[0])) == 3:
                    SN.append("SMP_000%s.dat" % s[0])
                elif len(str(s[0])) == 4:
                    SN.append("SMP_00%s.dat" % s[0])
                elif len(str(s[0])) == 5:
                    SN.append("SMP_0%s.dat" % s[0])
         #SN now contains all file names for spectroscopically confirmed supernovae

        if subset_length != False:
            x = sorted(sample(range(len(SN)), subset_length))
            SN = [SN[i] for i in x ]
            classes = [classes[i] for i in x]

        if classification != 'none': # can specify classification of supernova if user-requested
            if classification == 'Ia' or classification == 'SNIa':
                SN = [SN[i] for i in range(len(SN)) if classes[i] =='SNIa']
            elif classification == 'Ib' or classification == 'SNIb':
                SN = [SN[i] for i in range(len(SN)) if classes[i] =='SNIb']
            elif classification == 'Ic' or classification == 'SNIc':
                SN = [SN[i] for i in range(len(SN)) if classes[i] =='SNIc']
            elif classification == 'Ibc' or classification == 'SNIbc':
                SN = [SN[i] for i in range(len(SN)) if classes[i] == 'SNIb' or classes[i] == 'SNIc']
            elif classification == 'II' or classification == 'SNII':
                SN = [SN[i] for i in range(len(SN)) if classes[i] == 'SNII']
            else:
                print ('Invalid classification requested.')
                sys.exit()

        return SN

    def get_photo(self,subset_length,classification):
        """
        Function to take all purely photometric supernovae from Master file and return a   random sample of SNe of user-specified length
        if requested
        Parameters
        ----------
        subset_length : bool or int
            Number of objects to return (False to return all)
        classification : str
            Can specify a particular type of supernova to return ('none' for all types)
        Returns
        -------
        list-like
            List of object names
        """
        fl = open(self.rootdir+self.survey_name+'.LIST')
        SN = []
        classes = []
        for line in fl:
            s = line.split()
            if (s[5] == "pSNIa" or s[5] == "pSNIbc" or s[5] == "pSNII" or s[5] == "zSNIa" or s[5] == 'zSNIbc' or s[5] == 'zSNII' or s[5] == 'SNIa?'):
                classes.append(s[5])
                if len(str(s[0])) == 3:
                    SN.append("SMP_000%s.dat" % s[0])
                elif len(str(s[0])) == 4:
                    SN.append("SMP_00%s.dat" % s[0])
                elif len(str(s[0])) == 5:
                    SN.append("SMP_0%s.dat" % s[0])
         #SN now contains all file names for spectroscopically confirmed supernovae

        if subset_length != False:
            x = sorted(sample(range(len(SN)), subset_length))
            SN = [SN[i] for i in x ]
            classes = [classes[i] for i in x]

        if classification != 'none': # can specify classification of supernova if user-requested
            if classification =='Ia' or classification == 'SNIa':
                SN = [SN[i] for i in range(len(SN)) if classes[i] == 'SNIa?' or classes[i] == 'pSNIa' or classes[i] == 'zSNIa']
            elif classification == 'Ibc' or classification == 'SNIbc':
                SN = [SN[i] for i in range(len(SN)) if classes[i] == 'pSNIbc' or classes[i] == 'zSNIbc']
            elif classification == 'II' or classification == 'SNII':
                SN = [SN[i] for i in range(len(SN)) if classes[i] == 'pSNII' or classes[i] == 'zSNII']
            else:
                print ('Invalid classification requested.')
                sys.exit()

        return SN

    def get_object_names(self, subset='none',subset_length=False,classification='none'):
        """
        Gets a list of the names of the files within the dataset.
        Parameters
        ----------
        subset : str or list-like, optional
            List of a subset of object names. If not supplied, the full dataset will be used
        subset_length : bool or int, optional
            Number of objects to return (False to return all)
        classification : str, optional
            Can specify a particular type of supernova to return ('none' for all types)
        Returns
        -------
        list-like
            Object names
        """
        if isinstance(subset,basestring):
            if subset=='spectro':
                return np.array(self.get_spectro(subset_length,classification)) # have to convert to numpy array for wavelet features to work
                #loads random sample of the spec confirmed data defaulted to whole
            elif subset == 'photo': ##only photometric data
                return np.array(self.get_photo(subset_length,classification))
            elif subset=='none':
                return np.array(self.get_SNe(subset_length))
                # loads random sample of SNe from whole master file - sample defaulted to whole
        elif all(isinstance(l,basestring) for l in subset):
            #We assume subset is a list of strings containing object names
            return subset
        else:
            #Otherwise it must be a list of indices. Otherwise raise an error.
            names=np.genfromtxt(self.rootdir+self.survey_name+'.LIST', dtype='str')
            try:
                return names[subset]
            except IndexError:
                print ('Invalid subset provided')


    def get_info(self,flname):
        """
        Function which takes file name of supernova and returns dictionary of spectroscopic and photometric redshifts and their errors when available as
           well as the type of the supernova
        Parameters
        ----------
        flname : str
            Name of object
        Returns
        -------
        list-like
            Redshift, redshift error, type
        """
        fl = open(self.rootdir+self.survey_name+'.LIST')
        z = {'z_hel':float('nan'),'z_phot':float('nan')}# z_hel is spectroscopic heliocentric redshift and z_psnid uses zspec as prior but has many more??
        z_err = {'z_hel_err':float('nan'),'z_phot_err':float('nan')}
        t = -9
        for line in fl:
            s=line.split()
            if "SMP_000%s.dat" % s[0] == flname or "SMP_00%s.dat" % s[0] == flname or "SMP_0%s.dat" % s[0] == flname: # is this a bit slow?
                if s[103] != "\\N":
                    z['z_phot']=float(s[103])
                else:
                    z['z_phot']= -9
                if s[11] != "\\N":
                    z['z_hel']=float(s[11])
                else:
                    z['z_hel']= -9
                if s[104] != "\\N":
                    z_err['z_phot_err']=float(s[104])
                else:
                    z_err['z_phot_err']= -9
                if s[12] != "\\N":
                    z_err['z_hel_err']=float(s[12])
                else:
                    z_err['z_hel_err']= -9
                if s[5] == 'SNIa' or s[5] == 'pSNIa' or s[5] == 'zSNIa' or s[5] =='SNIa?': #all classifications of type Ia SNe - includes probable SNeIa
                    t = 1
                elif s[5] == 'SNIb' or s[5] == 'SNIc' or s[5] == 'pSNIbc' or s[5] == 'zSNIbc':
                    t = 3
                elif s[5] == 'SNII' or s[5] == 'pSNII' or s[5] == 'zSNII':
                    t = 2
        return z , z_err, t

    def get_lightcurve(self, flname):
        """
        Given a filename, returns a light curve astropy table that conforms with sncosmo requirements
        Parameters
        ----------
        flname : str
            The filename of the supernova (relative to data_root)
        Returns
        -------
        astropy.table.Table
            Light curve
        """
        fl=open(self.rootdir+flname,'r')
        mjd=[]
        flt=[] # band
        flux=[]
        fluxerr=[] # error in flux
        mag = []
        magerr = []
        line_count = 0
        for line in fl:
            s=line.split()
            line_count = line_count + 1
            if len(s)>0 and line_count > 4 and float(s[0]) < 1024: # s[0] is flag - a flag of 1024 or greater indicates bad data ( http://arxiv.org/pdf/0908.4277v1.pdf page 18)
                mjd.append(s[1])
                flux.append(s[7])
                fluxerr.append(s[8])
                mag.append(s[3])
                magerr.append(s[4])
                if s[2] == '0':
                    flt.append('sdssu')
                elif s[2] == '1':
                    flt.append('sdssg')
                elif s[2] == '2':
                    flt.append('sdssr')
                elif s[2] == '3':
                    flt.append('sdssi')
                elif s[2] == '4':
                    flt.append('sdssz')
        mjd = [float(x) for x in mjd] # contains strings - convert to floats
        flux = [float(x) for x in flux]
        fluxerr = [float(x) for x in fluxerr]
        mag = [float(x) for x in mag]
        magerr = [float(x) for x in magerr]
        (Z, Z_err, type) = self.get_info(flname)
        if Z['z_hel'] != -9:
            z = Z['z_hel'] #use spectroscopic redshifts if available
            z_err = Z_err['z_hel_err']
        elif Z['z_phot'] != -9:
            z = Z['z_phot'] #photometric redshift
            z_err = Z_err['z_phot_err']
        else:
            z = -9
            z_err = -9 # returns values of -9 if no redshift is available

        #Zeropoint
        zp=np.array([27.5]*len(mjd))
        zpsys=['ab']*len(mjd)

        #Make everything arrays
        mjd=np.array(mjd)
        flt=np.array(flt, dtype='str')
        flux=np.array(flux)
        fluxerr=np.array(fluxerr)
        mag = np.array(mag)
        magerr = np.array(magerr)
        start_mjd=mjd.min()
        r_flux = np.array([flux[i] for i in range(len(flux)) if flt[i]=='sdssr'])
        if len(r_flux) > 0:
            peak_flux = r_flux.max()
        else:
            peak_flux = -9
        mjd=mjd-start_mjd #We shift the times of the observations to all start at zero. If required, the mjd of the initial observation is stored in the metadata.
        #Note: obviously this will only zero the observations in one filter band, the others have to be zeroed if fitting functions.
        tab = Table([mjd, flt, flux, fluxerr, zp, zpsys, mag, magerr], names=('mjd', 'filter', 'flux', 'flux_error', 'zp', 'zpsys', 'mag', 'mag_error'), meta={'name':flname,'z':z, 'z_err':z_err, 'type':type, 'initial_observation_time':start_mjd, 'peak flux':peak_flux })

        return tab
    
class SDSS_Simulations(Dataset):
    """
    Class to read in the SDSS simulations dataset
    """

    def __init__(self, folder, subset='none', training_only=False, filter_set=['sdssu','sdssg', 'sdssr', 'sdssi', 'sdssz'], subset_length = False, classification = 'none'):
        """
        Initialisation
        Parameters
        ----------
        folder : str
            Folder where simulations are located
        subset : str or list-like, optional
            List of a subset of object names. If not supplied, the full dataset will be used
        filter_set : list-like, optional
            Set of possible filters
        subset_length : bool or int, optional

            If supplied, will return this many random objects (can be used in conjunction with subset="spectro")
        classification : str, optional
            Can return one specific type of supernova.
        """
        self.filter_set=filter_set
        self.rootdir=folder
        self.survey_name=folder.split(os.path.sep)[-2] # second to last / / /..
        #Get all the data as a list of astropy tables (this should not be memory intensive, even for large numbers of light curves)
        self.data={}
        invalid=0 #Some objects may have empty data
        print ('Reading data..')
        (self.data, invalid) = self.get_data(subset, subset_length, classification)
        if invalid>0:
            print ('%d objects were invalid and not added to the dataset.' %invalid)
        print ('%d objects read into memory.' %len(self.data))
        self.object_names = self.data.keys()
        #We create an optional model set which can be set by whatever feature class used
        self.models={}


    def get_data(self, subset='none', subset_length=False, classification = 'none'):
        """
        Function to get all data in same form as SDSS Data
        """
        # read in data as snana files
        d_Ia = sncosmo.read_snana_fits(self.rootdir + "SDSS_Ia_HEAD.FITS",self.rootdir + "SDSS_Ia_PHOT.FITS")
        d_nIa =  sncosmo.read_snana_fits(self.rootdir + "SDSS_NONIa_HEAD.FITS", self.rootdir + "SDSS_NONIa_PHOT.FITS")
        invalid = 0 # number of invalid LCs
        data = {}

        # allow user to specify dataset of entirely Ia, Ibc, non-Ia or II
        if classification == 'none':
            for i in range(len(d_Ia)):
                SN = self.get_lightcurve(d_Ia[i])
                if len(SN['mjd']) > 0 :  
                    data['%s'%SN.meta['snid']] = SN
                else:
                    invalid+=1
            for i in range(len(d_nIa)):
                SN = self.get_lightcurve(d_nIa[i])
                if len(SN['mjd']) > 0:
                    data['%s'%SN.meta['snid']] = SN
                else:
                    invalid+=1
        elif classification == 'Ia' or classification == 'SNIa':
            for i in range(len(d_Ia)):
                SN = self.get_lightcurve(d_Ia[i])
                if len(SN['mjd']) > 0 :     
                    data['%s'%SN.meta['snid']] = SN
                else:
                    invalid+=1
        elif classification == 'nIa' or classification == 'non-SNIa':
            for i in range(len(d_nIa)):
                SN = self.get_lightcurve(d_nIa[i])
                if len(SN['mjd']) > 0:
                    data['%s'%SN.meta['snid']] = SN
                else:
                    invalid+=1
        elif classification == 'Ibc' or classification == 'SNIbc':
            for i in range(len(d_nIa)):
                SN = self.get_lightcurve(d_nIa[i])
                if len(SN['mjd']) > 0 :  
                    if SN.meta['type'] == 3:
                        data['%s'%SN.meta['snid']] = SN
                else:
                    invalid+=1
        elif classification == 'II' or classification == 'SNII':
            for i in range(len(d_nIa)):
                SN = self.get_lightcurve(d_Ia[i])
                if len(SN['mjd']) > 0:
                    if SN.meta['type'] == 2:
                        data['%s'%SN.meta['snid']] = SN
                else:
                    invalid+=1
        else:
            print ('Invalid classification requested')
            sys.exit()
        
        # allow user to request entirely spectroscopic data or photometric data
        if subset == 'spectro':
            data = dict((key,value) for (key,value) in data.items() if value.meta['data type'] == 'spec')
        if subset == 'photo':
            data = dict((key,value) for (key,value) in data.items() if value.meta['data type'] == 'phot')
        
        # allow user to request a shortened subset of the dataset
        if subset_length != False:
            data = dict(sample(data.items(), subset_length))

        return data, invalid

    def get_lightcurve(self, lc):
        
        mjd = np.array([lc['MJD'][i] for i in range(len(lc['MJD'])) if lc['FLUXCAL'][i] > 0])
        flt = np.array(['sdss' + lc['FLT'][i] for i in range(len(lc['FLT'])) if lc['FLUXCAL'][i] > 0])
        flux = np.array( [(lc['FLUXCAL'][i]*math.pow(10,-1.44)) for i in range(len(lc['FLUXCAL'])) if lc['FLUXCAL'][i] > 0]) # ignore negative flux values
#FLUX: MULTIPLY BY 10^-1.44 (FLUX IN SDSS IS CALCULATED AS 10^(-0.4MAG +9.56) WHEREAS SIMULATED FLUXES ARE CALCULATED AS 10^(-0.4MAG + 11)- WE USE THE SDSS CONVENTION). 
        fluxerr = np.array([(lc['FLUXCALERR'][i]*math.pow(10,-1.44)) for i in range(len(lc['FLUXCALERR'])) if lc['FLUXCAL'][i] > 0])
        mag = np.array([lc['MAG'][i] for i in range(len(lc['MAGERR'])) if lc['FLUXCAL'][i] > 0])
        magerr = np.array([lc['MAGERR'][i] for i in range(len(lc['MAGERR'])) if lc['FLUXCAL'][i] > 0])
        start_mjd = mjd.min()
        r_flux = np.array([flux[i] for i in range(len(flux)) if flt[i]=='sdssr'])
        if len(r_flux) > 0:
            peak_flux = r_flux.max()
        else:
            peak_flux = -9
        mjd=mjd-start_mjd #We shift the times of the observations to all start at zero. If required, the mjd of the initial observation is stored in the metadata.
        #Note: obviously this will only zero the observations in one filter band, the others have to be zeroed if fitting functions.
        # find supernova classification and whether it is spectroscopically classified or photometrically       
        sntype = -9
        dtype = -9
        if lc.meta['SNTYPE'] == 120  :
            sntype = 1
            dtype = 'spec'
        elif lc.meta['SNTYPE'] == 106:
            sntype =1
            dtype = 'phot'
        elif lc.meta['SNTYPE'] == 32 or lc.meta['SNTYPE'] == 33:
            sntype =3
            dtype = 'spec'
        elif lc.meta['SNTYPE'] == 132 or lc.meta['SNTYPE'] == 133:
            sntype = 3
            dtype = 'phot'
        elif lc.meta['SNTYPE'] == 22:
            sntype = 2
            dtype = 'spec'
        elif lc.meta['SNTYPE'] == 122:
            sntype = 2
            dtype = 'phot'
        # get redshift - heliocentric is used where possible, otherwise a simulated heliocentric redshift is used
        z = -9
        z_err = -9
        if lc.meta['REDSHIFT_HELIO'] != -9 and lc.meta['REDSHIFT_HELIO_ERR'] != -9:
            z = lc.meta['REDSHIFT_HELIO']
            z_err = lc.meta['REDSHIFT_HELIO_ERR']
        else:
            z = lc.meta['SIM_REDSHIFT_HELIO']
            # no simulated error in redshift available

        #supernova identifier (used as object name)
        snid = lc.meta['SNID']

        #Zeropoint
        zp=np.array([27.5]*len(mjd))
        zpsys=['ab']*len(mjd)
        
        # form astropy table
        tab = Table([mjd, flt, flux, fluxerr, zp, zpsys, mag, magerr], names=('mjd', 'filter', 'flux', 'flux_error', 'zp', 'zpsys', 'mag', 'mag_error'), meta={'snid': snid,'z':z, 'z_err':z_err, 'type':sntype, 'initial_observation_time':start_mjd, 'peak flux':peak_flux , 'data type':dtype })

        return tab


class EmptyDataset(Dataset):
    """
    Empty data set, to fill up with light curves (of format astropy.table.Table) in your memory.
    """

    def __init__(self, folder=None, survey_name=None, filter_set=None):

        """
        Initialisation.
        Parameters
        ----------
        folder : str
            Root folder containing the data
        survey_name : str
            Specifies the name of the survey; needed for output folder name
        filter_set : list, optional
            List of possible filters used

        """
        if filter_set is None:
            self.filter_set=[]
        else:
            self.filter_set=filter_set
        self.rootdir=folder
        self.survey_name=survey_name
        self.data={}
        self.object_names=[]
        self.models={}

    def set_filters(self, filter_set):
        self.filter_set=filter_set

    def set_rootdir(folder):
        self.rootdir=folder
        self.survey_name=folder.splot(os.path.sep)[-2]

    def insert_lightcurve(self, lc):
        name=lc.meta['name']
        self.object_names=np.append(self.object_names,name)
        self.data[name]=lc
        for flt in np.unique(lc['filter']):
            if not flt in self.filter_set:
                print('Adding filter '+flt+' ...')
                self.filter_set.append(flt)
