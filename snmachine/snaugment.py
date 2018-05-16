import george
import scipy.optimize as op
import numpy as np
from astropy.table import Table,vstack

"""
Module handling the data augmentation of supernova data sets
"""

class SNAugment:
    """
    Skeletal base class outlining the structure for the augmentation of a 
    sndata instance. Classes that encapsulate a specific data augmentation 
    procedure should be derived from this class.
    """

    def __init__(self, d):
        """
	class constructor.

        Parameters: (why would you call this constructor in the first place?)
        ----------
        d: sndata object
            the supernova data set we want to augment

        """
        self.dataset=d
        self.meta={}#This can contain any metadata that the augmentation 
                    #process produces, and we want to keep track of.
        self.algorithm=None
        self.original=d.get_object_names()#this is a list of object names that
                                          #were in the data set prior to 
                                          #augmenting. 
                                          #DO NOT TOUCH FOR SELFISH PURPOSES

    def augment(self):
        pass

    def remove(self, obj=None):
        """
        reverts the augmentation step by fully or partially removing those 
        light curves that have been added in the augmentation procedure from 
        the data set.

        Parameters:
        ----------
        obj: list of strings
            These are the objects we will remove. If None is given, then we
            remove all augmented objects that have not been in the data set
            when we created the SNAugment object.
            NB: If obj contains object names that are in the original data set
            then we do not throw an error, but follow through on what you tell
            us to do. 
            
        """
        if obj is None:
            obj=list(set(self.dataset.object_names())-set(self.original))

        for o in obj:
            assert(o in self.dataset.object_names)
            self.dataset.data.pop(o)
            self.dataset.object_names=[x for x in self.dataset.object_names if x!=o]


class GPAugment(SNAugment):
    """
    Derived class that encapsulates data augmentation via Gaussian Processes
    """

    def __init__(self, d, templates=None):
        """
        class constructor. 

        Parameters:
        ----------
        d: sndata object
            the supernova data set we want to augment
        templates: list of strings
            If the templates argument is given (as a list of object names
             that are in the data set), then the augmentation step will take 
            these light curves to train the GPs on. If not, then every object 
            in the data set is considered fair game.
        """

        self.dataset=d
        self.meta={}
        self.meta['trained_gp']={}
        self.algorithm='GP augmentation'
        if templates is None:
            templates=d.get_object_names()
        self.meta['trained_gp']={}
        self.meta['random_state']=np.random.RandomState()
        self.meta['random_seed_state']=self.meta['random_state'].get_state()

        self.original=d.get_object_names()

    def train_filter(self,x,y,yerr,initheta=[100,20]):
        """
        Train one Gaussian process on the data from one band. We use the squared-exponential
        kernel, and we optimise its hyperparameters

        Parameters:
        -----------
        x: numpy array
            mjd values for the cadence
        y: numpy array
            flux values
        yerr: numpy array
            errors on the flux
        initheta: list, optional
            initial values for the hyperparameters. They should roughly correspond to the
            spread in y and x direction.

        Returns:
        -------
        g: george.GP
            the trained GP object
        """
        def nll(p):
            g.set_parameter_vector(p)
            ll=g.log_likelihood(y,quiet=True)
            return -ll if np.isfinite(ll) else 1.e25
        def grad_nll(p):
            g.set_parameter_vector(p)
            return -g.grad_log_likelihood(y,quiet=True)
        g=george.GP(initheta[0]**2*george.kernels.ExpSquaredKernel(metric=initheta[1]**2))
        g.compute(x,yerr)
        p0 = g.get_parameter_vector()
        results=op.minimize(nll,p0,jac=grad_nll,method='L-BFGS-B')
#        print(results.x)
        g.set_parameter_vector(results.x)
        return g

    def sample_cadence_filter(self,g,cadence,y):
        """
        Given a trained GP, and a cadence of mjd values, produce a sample from the distribution
        defined by the GP, on that cadence. The error bars are set to the spread of the GP distribution
        at the given mjd value.

        Parameters:
        -----------
        g: george.GP
            the trained Gaussian process object
        cadence: dict of type {string:numpy.array}
            the cadence defined by {filter1:mjds1, filter2:mjd2, ...}.
        y: numpy array
            the flux values of the data that the GP has been trained on.

        Returns:
        --------
        flux: numpy array
            flux values for the new sample
        fluxerr: numpy array
            error bars on the flux for the new sample
        """
        mu,cov=g.predict(y,cadence)
        flux=self.meta['random_state'].multivariate_normal(mu,cov)
        fluxerr=np.sqrt(np.diag(cov))
        return flux,fluxerr

    def produce_new_lc(self,obj,cadence=None,savegp=True,samplez=True,name='dummy'):
        """
        Assemble a new light curve from a template. If the template already has been used
        and the resulting GPs have been saved, then we use those. If not, we train a new GP.

        Parameters:
        -----------
        obj: str
           name of the object that we use as a template to train the GP on.
        cadence: dict of type {string:numpy.array}, optional.
           the cadence for the new light curve, defined by {filter:mjds}. If none is given, 
           then we pull the cadence of the template.
        savegp: bool, optional
           Do we save the trained GP in self.meta? This results in a speedup, but costs memory.
        samplez: bool, optional
           Do we give the new light curve a random redshift value drawn from a Gaussian of location
           and width defined by the template? If not, we just take the value of the template.
        name: str, optional
           object name of the new light curve. 

        Returns:
        --------
        new_lc: astropy.table.Table
           The new light curve
        """
        obj_table=self.dataset.data[obj]
        if cadence is None:
            cadence=self.extract_cadence(obj)

	#Either train a new set of GP on the template obj, or pull from metadata
        if obj in self.meta['trained_gp'].keys():
            print('fetching')
            all_g=self.meta['trained_gp'][obj]
        else:
            print('training')
            self.meta['trained_gp'][obj]={}
            all_g={}
            for f in self.dataset.filter_set:
                obj_f=obj_table[obj_table['filter']==f]
                x=np.array(obj_f['mjd'])
                y=np.array(obj_f['flux'])
                yerr=np.array(obj_f['flux_error'])
                g=self.train_filter(x,y,yerr)
                all_g[f]=g
            if savegp:
                self.meta['trained_gp'][obj]=all_g

        #Produce new LC based on the set of GP
        if samplez and 'z_err' in obj_table.meta.keys():
            newz=obj_table.meta['z']+obj_table.meta['z_err']*self.meta['random_state'].randn()
        else:
            newz=obj_table.meta['z']
        new_lc_meta={'name':name,'z':newz,'type':obj_table.meta['type'], 'template': obj, 'augment_algo': self.algorithm}
        new_lc=Table(names=['mjd','filter','flux','flux_error'],dtype=['f','S64','f','f'],meta=new_lc_meta)
        for f in self.dataset.filter_set:
            obj_f=obj_table[obj_table['filter']==f]
            y=obj_f['flux']
            flux,fluxerr=self.sample_cadence_filter(all_g[f],cadence[f],y)
            filter_col=[f]*len(cadence[f])
            dummy_table=Table((cadence[f],filter_col,flux,fluxerr),names=['mjd','filter','flux','flux_error'],dtype=['f','S64','f','f'])
            new_lc=vstack([new_lc,dummy_table])

	#Sort by cadence, for cosmetics
        new_lc.sort('mjd')
        return new_lc

    def extract_cadence(self, obj):
        """
        Given a light curve, we extract the cadence in a format that we can insert into produce_lc and sample_cadence_filter.

        Parameters:
        -----------
        obj: str
            name of the object

        Returns:
        --------
        cadence: dict of type {str:numpy.array}
            the cadence, in the format {filter1:mjd1, filter2:mjd2, ...}
        """
        table=self.dataset.data[obj]
        cadence={flt:np.array(table[table['filter']==flt]['mjd']) for flt in self.dataset.filter_set}
        return cadence

    def augment(self, fractions):
        """
        HIC SUNT DRACONES
        """
        pass
