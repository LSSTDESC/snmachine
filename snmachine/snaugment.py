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

    def train_filter(self,x,y,yerr,initheta=[100,20]):
        def nll(p):
            g.set_parameter_vector(p)
            ll=g.log_likelihood(y,quiet=True)
            return -ll if np.isfinite(ll) else 1e25
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
        mu,cov=g.predict(y,cadence)
        std=np.sqrt(np.diag(cov))
        return mu,std

    def produce_new_lc(self,obj,cadence=None,savegp=True):

        obj_table=self.dataset.data[obj]
        if cadence is None:
            cadence={}
            for f in self.dataset.filter_set:
                obj_table=self.dataset.data[obj]
                cadence[f]=obj_table[obj_table['filter']==f]['mjd']

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
        new_lc_meta={'name':'dummy','z':obj_table.meta['z'],'type':obj_table.meta['type'], 'template': obj, 'augment_algo': self.algorithm}
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
        table=self.dataset.data[obj]
        cadence={flt:np.array(table[table['filter']==flt]['mjd']) for flt in self.dataset.filter_set}
        return cadence

    def augment(self, fractions):
        """
        HIC SUNT DRACONES
        """
        pass
