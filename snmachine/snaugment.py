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
        d : sndata object
            the supernova data set we want to augment

        """
        self.dataset=d
        self.meta={}#This can contain any metadata that the augmentation 
                    #process produces, and we want to keep track of.
        self.algorithm=None
        self.original=d.object_names.copy()#this is a list of object names that
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
        obj : list of strings
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

    def __init__(self, d, templates=None, cadence_templates=None):
        """
        class constructor. 

        Parameters:
        ----------
        d : sndata object
            the supernova data set we want to augment
        templates : list of strings
            If the templates argument is given (as a list of object names
             that are in the data set), then the augmentation step will take 
            these light curves to train the GPs on. If not, then every object 
            in the data set is considered fair game.
        cadence_templates : list of strings
            If given, the augmentation will sample the cadence for the new light
            curves from these objects. If not, every object is fair game.
        """

        self.dataset=d
        self.meta={}
        self.meta['trained_gp']={}
        self.algorithm='GP augmentation'
        if templates is None:
            self.templates=d.object_names.copy()
        if cadence_templates is None:
            self.cadence_templates=d.object_names.copy()

        self.rng=np.random.RandomState()
        self.random_seed=self.rng.get_state()

        self.original=d.object_names.copy()

    def train_filter(self,x,y,yerr,initheta=[100,20]):
        """
        Train one Gaussian process on the data from one band. We use the squared-exponential
        kernel, and we optimise its hyperparameters

        Parameters:
        -----------
        x : numpy array
            mjd values for the cadence
        y : numpy array
            flux values
        yerr : numpy array
            errors on the flux
        initheta : list, optional
            initial values for the hyperparameters. They should roughly correspond to the
            spread in y and x direction.

        Returns:
        -------
        g : george.GP
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
        if len(x)==0:
            return g
        g.compute(x,yerr)
        p0 = g.get_parameter_vector()
        results=op.minimize(nll,p0,jac=grad_nll,method='L-BFGS-B')
#        print(results.x)
        g.set_parameter_vector(results.x)
        return g

    def sample_cadence_filter(self,g,cadence,y,yerr,add_measurement_noise=True):
        """
        Given a trained GP, and a cadence of mjd values, produce a sample from the distribution
        defined by the GP, on that cadence. The error bars are set to the spread of the GP distribution
        at the given mjd value.

        Parameters:
        -----------
        g : george.GP
            the trained Gaussian process object
        cadence : numpy.array
            the cadence of mjd values.
        y : numpy array
            the flux values of the data that the GP has been trained on.

        Returns:
        --------
        flux : numpy array
            flux values for the new sample
        fluxerr : numpy array
            error bars on the flux for the new sample
        """
        if len(cadence)==0:
            flux=np.array([])
            fluxerr=np.array([])
        else:
            mu,cov=g.predict(y,cadence)
            flux=self.rng.multivariate_normal(mu,cov)
            fluxerr=np.sqrt(np.diag(cov))
        #adding measurement error
        if add_measurement_noise:
            flux+=self.rng.randn(len(y))*yerr
            fluxerr=np.sqrt(fluxerr**2+yerr**2)
        return flux,fluxerr

    def produce_new_lc(self,obj,cadence=None,savegp=True,samplez=True,name='dummy'):
        """
        Assemble a new light curve from a template. If the template already has been used
        and the resulting GPs have been saved, then we use those. If not, we train a new GP.

        Parameters:
        -----------
        obj : str or astropy.table.Table
           the object (or name thereof) that we use as a template to train the GP on.
        cadence : str or dict of type {string:numpy.array}, optional.
           the cadence for the new light curve, defined either by object name or by {filter:mjds}. If none is given, 
           then we pull the cadence of the template.
        savegp : bool, optional
           Do we save the trained GP in self.meta? This results in a speedup, but costs memory.
        samplez : bool, optional
           Do we give the new light curve a random redshift value drawn from a Gaussian of location
           and width defined by the template? If not, we just take the value of the template.
        name : str, optional
           object name of the new light curve. 

        Returns:
        --------
        new_lc: astropy.table.Table
           The new light curve
        """

        if type(obj) is Table:
            obj_table=obj
            obj_name=obj.meta['name']
        elif type(obj) in [str,np.str_]:
            obj_table=self.dataset.data[obj]
            obj_name=str(obj)
        else:
            print('obj: type %s not recognised in produce_new_lc()!'%type(obj))
            #todo: actually throw an error

        if cadence is None:
            cadence_dict=self.extract_cadence(obj)
            add_measurement_noise=True
        else:
            print('warning: GP sampling does NOT include measurement noise, since sampling is performed on a different cadence!')
            add_measurement_noise=False
            if type(cadence) in [str,np.str_]:
                cadence_dict=self.extract_cadence(cadence)
            elif type(cadence) is dict:
                cadence_dict=cadence
            else:
                print('cadence: type %s not recognised in produce_new_lc()!'%type(cadence))
                #todo: actually throw an error            

	#Either train a new set of GP on the template obj, or pull from metadata
        if obj_name in self.meta['trained_gp'].keys():
            print('fetching')
#            print(obj_name)
#            print(self.dataset.filter_set)
#            print(type(self.dataset.filter_set[0]))
            all_g=self.meta['trained_gp'][obj_name]
        else:
            print('training')
            all_g={}
            for f in self.dataset.filter_set:
                obj_f=obj_table[obj_table['filter']==f]
                x=np.array(obj_f['mjd'])
                y=np.array(obj_f['flux'])
                yerr=np.array(obj_f['flux_error'])
                g=self.train_filter(x,y,yerr)
                all_g[f]=g
            if savegp:
                self.meta['trained_gp'][obj_name]=all_g

        #Produce new LC based on the set of GP
        if samplez and 'z_err' in obj_table.meta.keys():
            newz=obj_table.meta['z']+obj_table.meta['z_err']*self.rng.randn()
        else:
            newz=obj_table.meta['z']
        new_lc_meta={'name':name,'z':newz,'type':obj_table.meta['type'], 'template': obj_name, 'augment_algo': self.algorithm}
        new_lc=Table(names=['mjd','filter','flux','flux_error'],dtype=['f','U','f','f'],meta=new_lc_meta)
        for f in self.dataset.filter_set:
            obj_f=obj_table[obj_table['filter']==f]
            y=obj_f['flux']
            yerr=obj_f['flux_error']
            flux,fluxerr=self.sample_cadence_filter(all_g[f],cadence_dict[f],y,yerr,add_measurement_noise=add_measurement_noise)
            filter_col=[str(f)]*len(cadence_dict[f])
            dummy_table=Table((cadence_dict[f],filter_col,flux,fluxerr),names=['mjd','filter','flux','flux_error'],dtype=['f','U','f','f'])
            new_lc=vstack([new_lc,dummy_table])

	#Sort by cadence, for cosmetics
        new_lc.sort('mjd')
        return new_lc

    def extract_cadence(self, obj):
        """
        Given a light curve, we extract the cadence in a format that we can insert into produce_lc and sample_cadence_filter.

        Parameters:
        -----------
        obj : str or astropy.table.Table
            (name of) the object

        Returns:
        --------
        cadence : dict of type {str:numpy.array}
            the cadence, in the format {filter1:mjd1, filter2:mjd2, ...}
        """
        if type(obj) in [str, np.str_]:
            table=self.dataset.data[obj]
            objname=obj
        elif type(obj) is Table:
            table=obj
            objname=table.meta['name']
        else:
            print('obj: type %s not recognised in extract_cadence()!'%type(obj))
        cadence={flt:np.array(table[table['filter']==flt]['mjd']) for flt in self.dataset.filter_set}
        return cadence

    def augment(self, numbers, return_names=False, **kwargs):
        """
        High-level wrapper of GP augmentation: The dataset will be augmented to the numbers by type. 
        Parameters:
        ----------
        numbers : dict of type int:int
            The numbers to which the data set will be augmented, by type. Keys are the types, 
            values are the numbers of light curves pertaining to a type after augmentation.
            Types that do not appear in this dict will not be touched.
        return_names : bool
            If True, we return a list of names of the objects that have been added into the data set
        kwargs :
            Additional arguments that will be handed verbatim to produce_new_lc. Interesting choices
            include savegp=False and samplez=True
        Returns:
        --------
        newobjects : list of str
            The new object names that have been created by augmentation.
        """
        dataset_types=self.dataset.get_types()
        dataset_types['Type'][dataset_types['Type']>10]=dataset_types['Type'][dataset_types['Type']>10]//10#NB: this is specific to a particular remapping of indices!!

        types=np.unique(dataset_types['Type'])

        newnumbers=dict()
        newobjects=[]
        for t in types:
            thistype_oldnumbers=len(dataset_types['Type'][dataset_types['Type']==t])
            newnumbers[t]=numbers[t]-thistype_oldnumbers
            thistype_templates=[dataset_types['Object'][i] for i in range(len(dataset_types)) if dataset_types['Object'][i] in self.templates and dataset_types['Type'][i]==t]


            if newnumbers[t]<0:
                print('There are already %d objects of type %d in the original data set, cannot augment to %d!'%(thistype_oldnumbers,t,numbers[t]))
                continue
            elif newnumbers[t]==0:
                continue
            else:
#                print('now dealing with type: '+str(t))
#                print('templates: '+str(thistype_templates))
                for i in range(newnumbers[t]):
                    #pick random template
                    thistemplate=thistype_templates[self.rng.randint(0,len(thistype_templates))]
                    #pick random cadence
#                    thiscadence_template=self.cadence_templates[self.rng.randint(0,len(self.cadence_templates))]
#                    thiscadence_template=thistemplate

#                    cadence=self.extract_cadence(thiscadence_template)
#                    new_name='augm_t%d_%d_'%(t,i) + thistemplate + '_' + thiscadence_template + '.DAT'
                    new_name='augm_t%d_%d_'%(t,i) + thistemplate# + '.DAT'
                    new_lightcurve=self.produce_new_lc(obj=thistemplate,name=new_name,**kwargs)#,cadence=cadence)
                    self.dataset.insert_lightcurve(new_lightcurve)
#                    print('types: '+str(new_lightcurve.meta['type'])+' '+str(self.dataset.data[thistemplate].meta['type']))
                    newobjects=np.append(newobjects,new_name)
        return newobjects











