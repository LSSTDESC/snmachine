"""
Module for feature extraction on supernova light curves.
"""

from __future__ import division, print_function
import numpy as np
from . import parametric_models
import sys, pywt, time, subprocess, os, sncosmo
from scipy.interpolate import interp1d
import george
from astropy.table import Table, vstack, hstack, join
from multiprocessing import Pool
from functools import partial
from scipy import stats
import scipy.optimize as op
from iminuit import Minuit, describe
import traceback

try:
    import pymultinest
    from pymultinest.analyse import Analyzer
    has_multinest=True
except ImportError:
    has_multinest=False
    
try:
    import emcee
    has_emcee=True
except ImportError:
    has_emcee=False

try:
    import george
    has_george=True
except ImportError:
    has_george=False

try:
    from gapp import dgp
    has_gapp=True
except ImportError:
    has_gapp=False

def _GP(obj, d, ngp, xmin, xmax, initheta, save_output, output_root, gpalgo='george'):
    """
    Fit a Gaussian process curve at specific evenly spaced points along a light curve.

    Parameters
    ----------
    obj : str
        Name of the object
    d : Dataset-like object
        Dataset
    ngp : int
        Number of points to evaluate Gaussian Process at
    xmin : float
        Minimim time to evaluate at
    xmax : float
        Maximum time to evaluate at
    initheta : list-like
        Initial values for theta parameters. These should be roughly the scale length in the y & x directions.
    save_output : bool
        Whether or not to save the output
    output_root : str
        Output directory
    gpalgo : str
        which gp package is used for the Gaussian Process Regression, GaPP or george

    Returns
    -------
    astropy.table.Table
        Table with evaluated Gaussian process curve and errors

    """

    if gpalgo=='gapp' and not has_gapp:
        print('No GP module gapp. Defaulting to george instead.')
        gpalgo='george'

    lc=d.data[obj]
    filters=np.unique(lc['filter'])
    #Store the output in another astropy table
    output=[]
    for fil in d.filter_set:
        if fil in filters:
            x=lc['mjd'][lc['filter']==fil]
            y=lc['flux'][lc['filter']==fil]
            err=lc['flux_error'][lc['filter']==fil]
            sys.stdout = open(os.devnull, "w")
            if gpalgo=='gapp':
                sys.stdout = open(os.devnull, "w")
                g=dgp.DGaussianProcess(x, y, err, cXstar=(xmin, xmax, ngp))
                sys.stdout=sys.__stdout__
                rec, theta=g.gp(theta=initheta)
            elif gpalgo=='george':
                # Define the objective function (negative log-likelihood in this case).
                def nll(p):
                    g.set_parameter_vector(p)
                    ll = g.log_likelihood(y, quiet=True)
                    return -ll if np.isfinite(ll) else 1e25

                # And the gradient of the objective function.
                def grad_nll(p):
                    g.set_parameter_vector(p)
                    return -g.grad_log_likelihood(y, quiet=True)

      #          sys.stdout = open(os.devnull, "w")
                g=george.GP(initheta[0]**2*george.kernels.ExpSquaredKernel(metric=initheta[1]**2))
                g.compute(x,err)
                p0 = g.get_parameter_vector()
                results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
                g.set_parameter_vector(results.x)
       #         sys.stdout=sys.__stdout__
                xstar=np.linspace(xmin,xmax,ngp)
                mu,cov=g.predict(y,xstar)
                std=np.sqrt(np.diag(cov))
                rec=np.column_stack((xstar,mu,std))
        else:
            rec=np.zeros([ngp, 3])
        newtable=Table([rec[:, 0], rec[:, 1], rec[:, 2], [fil]*ngp], names=['mjd', 'flux', 'flux_error', 'filter'])
        if len(output)==0:
            output=newtable
        else:
            output=vstack((output, newtable))
    if save_output=='gp' or save_output=='all':
        output.write(os.path.join(output_root, 'gp_'+obj), format='ascii')        
    return output



def _run_leastsq(obj, d, model, n_attempts, seed=-1):
    """
    Minimises the chi2 on all the filter bands of a given light curve, fitting the model to each one and extracting
    the best fitting parameters

    Parameters
    ----------
    obj : str
        Object name
    d : Dataset
        Dataset object
    model : parametric model object
        Parametric model
    n_attempts : int
        We run this multiple times to try to avoid local minima and take the best fitting values.

    Returns
    -------
    astropy.table.Table
        Table of best fitting parameters for all filters

    """
    lc=d.data[obj]
    filts=np.unique(lc['filter'])

    n_params=len(model.param_names)

    #How many times to keep trying to fit each object to obtain a good fit (defined as reduced chi2 of less than 2)
    if n_attempts<1:
        n_attempts=1

    labels=['Object']
    for f in d.filter_set:
        pams=model.param_names
        for p in pams:
            labels.append(f+'-'+p)
    output=Table(names=labels, dtype=['U32']+['f']*(len(labels)-1))

    if seed!=-1:
        np.random.seed(seed)

    t1=time.time()
    row=[obj]
    for f in d.filter_set:
        if f in filts:
            x=np.array(lc['mjd'][lc['filter']==f])
            y=np.array(lc['flux'][lc['filter']==f])
            err=np.array(lc['flux_error'][lc['filter']==f])

            def mini_func(*params):
                #This function changes with each object so is necessary to redefine (since you can't pass arguments through Minuit)
                for i in range(len(params)):
                    p=params[i]
                    if p<model.limits[model.param_names[i]][0] or p>model.limits[model.param_names[i]][1]:
                        return np.inf
                ynew=model.evaluate(x, params)
                chi2=np.sum((y-ynew)*(y-ynew)/err/err)
                return chi2

            #For the sake of speed, we stop as soon as we get to reduced chi2<2, failing that we use the
            #best fit found so far
            fmin=np.inf
            min_params=[]

            for i in range(n_attempts):
                if i==0:
                    #Try the default starting point
                    input_args=model.initial.copy()
                else:
                    #Pick a new, random starting point
                    input_args={}
                    for p in model.param_names:
                        val=np.random.uniform(model.limits[p][0], model.limits[p][1])
                        input_args[p]=val
                for p in model.param_names:
                    input_args['limit_'+p]=model.limits[p]

                m=Minuit(mini_func, pedantic=False, print_level=0,forced_parameters=model.param_names, **input_args)

                m.migrad()
                parm=[]
                for p in model.param_names:
                    parm.append(m.values[p])

                rchi2=m.fval/len(x)
                if rchi2<2:
                    fmin=m.fval
                    min_params=parm
                    break
                elif m.fval<fmin:
                    fmin=m.fval
                    min_params=parm

            outfl=open('out', 'a')
            outfl.write('%s\t%s\t%f\t%d\n' %(obj, f, fmin/len(x), i))
            outfl.close()
            row+=min_params
        else:
            row+=[0]*len(model.param_names) #Fill missing values with zeros
    output.add_row(row)
    #print 'Time per filter', (time.time()-t1)/4
    return output





def _run_multinest(obj, d, model, chain_directory,  nlp, convert_to_binary, n_iter, restart=False, seed=-1):
    """
    Runs multinest on all the filter bands of a given light curve, fitting the model to each one and
    extracting the best fitting parameters.

    Parameters
    ----------
    obj : str
        Object name
    d : Dataset
        Dataset object
    model : parametric model object
        Parametric model
    chain_directory : str
        Path to where the output files go
    nlp : int
        Number of live points
    convert_to_binary : bool
        Whether or not to convert the ascii output files to binary
    n_iter : int
        Maximum number of iterations
    restart : bool
        Whether to restart from existing chain files.

    Returns
    -------
    astropy.table.Table
        Table of best fitting parameters and their errors for all filters

    """

    try:
        def prior_multinest(cube, ndim, nparams):
            """Prior function specifically for multinest. This would be called for one filter, for one object.
            @param cube A ctypes pointer to the parameter cube (actually just the current parameter values).
            @param ndim Number of dimensions
            @param nparams Number of parameters. Usually the same as ndim unless you have unsampled (e.g. calculated) parameters. These
            are assumed to be the first (ndim-nparams) parameters.
            """
            up=model.upper_limit
            low=model.lower_limit
            
            for i in range(nparams):
                cube[i]=cube[i]*(up[i]-low[i])+low[i]
            return cube
            
            
        lc=d.data[obj]
        filts=np.unique(lc['filter'])
        
        n_params=len(model.param_names)
        
        #err_plus=[pname+'_err+' for pname in self.model.param_names]
        #err_minus=[pname+'_err-' for pname in self.model.param_names]
        labels=['Object']
        for f in d.filter_set:
            pams=model.param_names
            for p in pams:
                labels.append(f+'-'+p)

        output=Table(names=labels, dtype=['U32']+['f']*(len(labels)-1))
    
        
        row=[obj]
        for f in d.filter_set:
            t1=time.time()
            if f in filts:
                #current values for x,y and err are set each time multinest is called
                x, y, err=np.column_stack((lc['mjd'][lc['filter']==f], lc['flux'][lc['filter']==f], lc['flux_error'][lc['filter']==f])).T
                
                def loglike_multinest(cube, ndim, nparams):
                    """Loglikelihood function specifically for multinest. This would be called for one filter, for one object.
                    @param cube A ctypes pointer to the parameter cube (actually just the current parameter values).
                    @param ndim Number of dimensions
                    @param nparams Number of parameters. Usually the same as ndim unless you have unsampled (e.g. calculated) parameters. These
                    are assumed to be the first (ndim-nparams) parameters.
                    """
                    params=np.zeros(nparams)
                    #This is the only obvious way to convert a ctypes pointer to a numpy array
                    for i in range(nparams):
                        params[i]=cube[i]
                    #params=[ 26.97634888,   45.13123322,    2.59183478,    0.12057552,    7.65392637]
                    ynew=model.evaluate(x, params)
                    
                    chi2=np.sum(((y-ynew)*(y-ynew))/err/err)
                    #print 'likelihood', -chi2/2.
                    return -chi2/2.
                
                chain_name=os.path.join(chain_directory, '%s-%s-%s-' %(obj.split('.')[0], f, model.model_name))
                
                if not restart or not os.path.exists(chain_name+'stats.dat'):
                    #Gives the ability to restart from existing chains if they exist
                    pymultinest.run(loglike_multinest, prior_multinest, n_params, importance_nested_sampling = False, init_MPI=False, 
                    resume = False, verbose = False, sampling_efficiency = 'parameter', n_live_points = nlp, outputfiles_basename=chain_name, 
                    multimodal=False, max_iter=n_iter, seed=seed)
                    
                #An=Analyzer(n_params, chain_name)
                #best_params=An.get_best_fit()['parameters']
                
#                chain=np.loadtxt(chain_name+'.txt')
#                best_params=np.median(chain, axis=0)[2:]
                best_params=get_MAP(chain_name)

                if convert_to_binary and not restart:
                    s='%s-%s-%s-' %(obj.split('.')[0], f, model.model_name)
                    ext=['ev.dat', 'phys_live.points', 'live.points', '.txt', 'post_equal_weights.dat'] #These are the files we can convert
                    for e in ext:
                        infile=os.path.join(chain_directory,'%s-%s-%s-%s' %(obj.split('.')[0], f, model.model_name, e))
                        outfile=infile+'.npy'
                        try:
                            x=np.loadtxt(infile)
                            np.save(outfile, x)
                            os.system('rm %s' %infile)
                        except:
                            print ('ERROR reading file', infile)
                            print ('File unconverted')
    #            stats=An.get_stats()['marginals']
    #            sig_upper=[stats[p]['1sigma'][1]-best_params[p] for p in range(len(stats))]
    #            sig_lower=[best_params[p]-stats[p]['1sigma'][0] for p in range(len(stats))]
    #            best=best_params+sig_lower+sig_upper
                #Expects first the parameters, then -sigma then +sigma

                row+=best_params
            else:
                row+=[0]*len(model.param_names) #I'm not sure if it makes the most sense to fill in missing values with zeroes...
            #print 'Time for object', obj, 'filter', f,':', (time.time()-t1)
            np.savetxt(os.path.join(chain_directory,'%s-%s-%s-.time' %(obj.split('.')[0], f, model.model_name)), [time.time()-t1])
            
        output.add_row(row)
            
        return output
    
    except:
        #Sometimes things just break
        print ('ERROR in', obj)
        print (sys.exc_info()[0])
        print (sys.exc_info()[1])
        traceback.print_tb(sys.exc_info()[2])
        
        return None
        

    
    
def _run_leastsq_templates(obj, d, model_name, use_redshift, bounds, seed=-1):
    """
    Fit template-based supernova models using least squares.

    Parameters
    ----------
    obj : str
        Object name
    d : Dataset
        Dataset object
    model_name : str
        Name of model to use (to be passed to sncosmo)
    use_redshift : bool
        Whether or not to use provided redshift information
    bounds : dict
        Bounds on parameters

    Returns
    -------
    astropy.table.Table
        Table of best fitting parameters
    """

    lc=d.data[obj]

    if model_name=='mlcs2k2':
        dust=sncosmo.CCM89Dust()
        model=sncosmo.Model(model_name,effects=[dust],effect_names=['host'], effect_frames=['rest'])
    else:
        model=sncosmo.Model(model_name)


    #labels=['Object']+model.param_names+['Chisq']
    #output=Table(names=labels, dtype=['S32']+['f']*(len(model.param_names))+['f'])
    labels = ['Object'] + model.param_names
    output=Table(names=labels, dtype=['U32']+['f']*(len(model.param_names)))

    t1=time.time()
    row=[obj]

    if use_redshift:
        model.set(z=lc.meta['z'])
        prms=model.param_names
        prms=prms[1:]
        bnds=bounds.copy()
        bnds.pop('z', None)
        res, fitted_model=sncosmo.fit_lc(lc, model, vparam_names=prms,
            bounds=bnds, minsnr=0)
    else:
        res, fitted_model=sncosmo.fit_lc(lc, model, vparam_names=model.param_names,
            bounds=bounds, minsnr=0)
    best=res['parameters']
    best=best.tolist()
    row+=best
    #row+=[res['chisq']]

    output.add_row(row)

    return output
        

def _run_multinest_templates(obj, d, model_name, bounds, chain_directory='./',  nlp=1000, convert_to_binary=True, use_redshift=False, short_name='', restart=False, seed=-1):
    """

    Fit template-based supernova models using multinest.

    Parameters
    ----------
    obj : str
        Object name
    d : Dataset
        Dataset object
    model_name : str
        Name of model to use (to be passed to sncosmo)
    use_redshift : bool
        Whether or not to use provided redshift information
    bounds : dict
        Bounds on parameters

    Returns
    -------
    astropy.table.Table
        Table of best fitting parameters

    Parameters
    ----------
    obj : str
        Object name
    d : Dataset
        Dataset object
    model_name : str
        Name of model to use (to be passed to sncosmo)
    bounds : dict
        Bounds on parameters
    chain_directory : str
        Path to output directory for chains
    nlp : int
        Number of live points
    convert_to_binary : bool
        Whether or not to convert ascii Multinest files to binary
    use_redshift : bool
        Whether or not to use provided redshift information
    short_name : str
        A shorter name for the chains (to overcome Multinest's character limitation)
    restart : bool
        Whether or not to restart from existing chains

    Returns
    -------
    array-like
        List of best-fitting parameters
    """
    try:
        def prior_multinest(cube, ndim, nparams):
            """Prior function specifically for multinest. This would be called for one filter, for one object.
            @param cube A ctypes pointer to the parameter cube (actually just the current parameter values).
            @param ndim Number of dimensions
            @param nparams Number of parameters. Usually the same as ndim unless you have unsampled (e.g. calculated) parameters. These
            are assumed to be the first (ndim-nparams) parameters.
            """

            params=model.param_names
            if use_redshift:
                params=params[1:]
            for i in range(ndim):
                p=params[i]
                cube[i]=cube[i]*(bounds[p][1]-bounds[p][0])+bounds[p][0]

            return cube

        def loglike_multinest(cube, ndim, nparams):
            """Loglikelihood function specifically for multinest. This would be called for one filter, for one object.
            @param cube A ctypes pointer to the parameter cube (actually just the current parameter values).
            @param ndim Number of dimensions
            @param nparams Number of parameters. Usually the same as ndim unless you have unsampled (e.g. calculated) parameters. These
            are assumed to be the first (ndim-nparams) parameters.
            """
            dic = {}
            # This is the only obvious way to convert a ctypes pointer to a numpy array
            params = model.param_names
            if use_redshift:
                params = params[1:]
                dic['z'] = lc.meta['z']
            for i in range(nparams):
                dic[params[i]] = cube[i]

            model.set(**dic)
            chi2 = 0
            for filt in filts:
                yfit = model.bandflux(filt, X[filt], zp=27.5, zpsys='ab')
                chi2 += np.sum(((Y[filt] - yfit) / E[filt]) ** 2)
            return -chi2 / 2.

        lc=d.data[obj]
        filts=np.unique(lc['filter'])
        if model_name=='mlcs2k2':
            dust=sncosmo.CCM89Dust()
            model=sncosmo.Model(model_name,effects=[dust],effect_names=['host'], effect_frames=['rest'])
        else:
            model=sncosmo.Model(model_name)
        
        n_params=len(model.param_names)
        
        #err_plus=[pname+'_err+' for pname in self.model.param_names]
        #err_minus=[pname+'_err-' for pname in self.model.param_names]
        labels=['Object']+model.param_names
        
        t1=time.time()
        
        #Convert the astropy table to numpy array outside the likelihood function to avoid repeated calls
        X={}
        Y={}
        E={}
        for filt in filts:
            x, y, err=np.column_stack((lc['mjd'][lc['filter']==filt], lc['flux'][lc['filter']==filt], lc['flux_error'][lc['filter']==filt])).T
            X[filt]=x
            Y[filt]=y
            E[filt]=err
        

        
        chain_name=os.path.join(chain_directory, '%s-%s-' %(obj.split('.')[0], short_name))
        
        if use_redshift:
            ndim=len(model.param_names)-1
        else:
            ndim=len(model.param_names)
        
        
        if not restart or not os.path.exists(chain_name+'stats.dat'):
            pymultinest.run(loglike_multinest, prior_multinest, ndim, importance_nested_sampling = False, init_MPI=False,
            resume = False, verbose = False, sampling_efficiency = 'parameter', n_live_points = nlp, outputfiles_basename=chain_name, 
            multimodal=False, seed=seed)
        
        best_params=get_MAP(chain_name)
        
        if use_redshift:
            best_params=[lc.meta['z']]+best_params
            

        if convert_to_binary and not restart:
            s='%s-%s-' %(obj.split('.')[0], short_name)
            ext=['ev.dat', 'phys_live.points', 'live.points', '.txt', 'post_equal_weights.dat'] #These are the files we can convert
            for e in ext:
                infile=os.path.join(chain_directory,'%s-%s-%s' %(obj.split('.')[0], short_name,  e))
                outfile=infile+'.npy'
                x=np.loadtxt(infile)
                np.save(outfile, x)
                os.system('rm %s' %infile)
                

        np.savetxt(os.path.join(chain_directory,'%s-%s-.time' %(obj.split('.')[0], short_name)), [time.time()-t1])

        return np.array(best_params)
    
    except:
        #Sometimes things just break
        print ('ERROR in', obj)
        print (sys.exc_info()[0])
        print (sys.exc_info()[1])
        traceback.print_tb(sys.exc_info()[2])
        
        return None


def output_time(tm):
    """
    Simple function to output the time nicely formatted.

    Parameters
    ----------
    tm : Input time in seconds.
    """

    hrs = tm / (60 * 60)
    mins = tm / 60
    secs = tm
    if hrs >= 1:
        out = '%.2f hours' % (hrs)
    elif mins >= 1:
        out = '%.2f minutes' % (mins)
    else:
        out = '%.2f seconds' % (secs)
    print ('Time taken is', out)


def get_MAP(chain_name):
    """
    Read maximum posterior parameters from a stats file of multinest.

    Parameters
    ----------
    chain_name : str
        Root for the chain files

    Returns
    -------
    list-like
        Best-fitting parameters

    """
    stats_file = chain_name + 'stats.dat'
    fl = open(stats_file, 'r')
    lines = fl.readlines()
    stats = []
    ind = [i for i in range(len(lines)) if 'MAP' in lines[i]][0]
    params = [float(l.split()[1]) for l in lines[ind + 2:]]
    return params

class Features:
    """Base class to define basic functionality for extracting features from supernova datasets. Users are not
    restricted to inheriting from this class, but any Features class must contain the functions extract_features
    and fit_sn.
    """
    def __init__(self):
        self.p_limit=0.05 #At what point to we suggest a model has been a bad fit. 
        
    def extract_features(self):
        pass
        
    def fit_sn(self):
        pass
        
    def goodness_of_fit(self, d):
        """
        Test (for any feature set) how well the reconstruction from the features fits each of the objects in the dataset.

        Parameters
        ----------
        d : Dataset
            Dataset object.

        Returns
        -------
        astropy.table.Table
            Table with the reduced Chi2 for each object
        """
        if len(d.models)==0:
            print ('Call Dataset.set_models first.')
            return None
        filts=np.unique(d.data[d.object_names[0]]['filter'])
        filts=np.array(filts).tolist()
        rcs=Table(names=['Object']+filts, dtype=['U32']+['float64']*len(filts)) #Reduced chi2
        for obj in d.object_names:
            #Go through each filter
            chi2=[]
            lc=d.data[obj]
            mod=d.models[obj]
            for filt in filts:
                l=lc[lc['filter']==filt]
                m=mod[mod['filter']==filt]
                x=l['mjd']
                y=l['flux']
                e=l['flux_error']
                xmod=m['mjd']
                ymod=m['flux']
                #Interpolate
                fit=interp1d(xmod, ymod)
                yfit=fit(x)
                chi2.append(sum((yfit-y)**2/e**2)/(len(x)-1))
            rcs.add_row([obj]+chi2)
            
        return rcs

    def posterior_predictor(self, lc, nparams, chi2):
        """
        Computes posterior predictive p-value to see if the model fits sufficient well.
        ***UNTESTED***

        Parameters
        ----------
        lc : astropy.table.Table
            Light curve
        nparams : int
            The number of parameters in the model. For the wavelets, this will be the number of PCA coefficients. For the parametric
            models, this will be the number parameters per model multiplied by the number of filters. For the template models, this is simply the number
            of parameters in the model.
        chi2 : array-like
            An array of chi2 values for each set of parameter space in the parameter samples. This is easy to obtain as -2*loglikelihood output
            from a multinest or mcmc chain. For features such as the wavelets, this will have to be separately calculated by drawing thousands of curves
            consistent with the coefficients and their errors and then computing the chi2.

        Returns
        -------
        float
            The posterior predictive p-value. If this number is too close to 0 or 1 it implies the model is a poor fit.

        """

        #Count the number of data points over all filters.
        ndata=len(lc['mjd'])
        dof=ndata-nparams-1

        if dof<=0:
            dof=1
        chi2_limit=stats.chi2.ppf(1-self.p_limit, dof)
        print (chi2_limit)
        print (chi2.min(), chi2.max())
        chi2=np.sort(chi2)
        p_value=np.count_nonzero(chi2>chi2_limit)/len(chi2)
        
        if p_value>self.p_limit:
            print ('Model fit unsatisfactory, p value=', p_value)
        return p_value
    
    def convert_astropy_array(self, tab):
        """
        Convenience function to convert an astropy table (floats only) into a numpy array.
        """
        out_array= np.array([tab[c] for c in tab.columns]).T
        return out_array

        
class TemplateFeatures(Features):
    """
    Calls sncosmo to fit a variety of templates to the data. The number of features will depend on the templates 
    chosen (e.g. salt2, nugent2p etc.)
    """
    def __init__(self, model=['Ia'], sampler='leastsq',lsst_bands=False,lsst_dir='../lsst_bands/'):
        """
        To initialise, provide a list of models to fit for (defaults to salt2 Ia templates).

        Parameters
        ----------
        model : list-like, optional
            List of models. In theory you can fit Ia and non-Ia models and use all those as features. So far only tested with SALT2.
        sampler : str, optional
            A choice of 'mcmc', which uses the emcee sampler, or 'multinest' or 'leastsq' (default).
        lsst_bands : bool, optional
            Whether or not the LSST bands are required. Only need for LSST simulations to register bands with sncosmo.
        lsst_dir : str, optional
            Directory where LSST bands are stored.
        """

        Features.__init__(self)
        if lsst_bands:
            self.registerBands(lsst_dir,prefix='approxLSST_',suffix='_total.dat')
        self.model_names=model
        self.templates={'Ia':'salt2-extended','salt2':'salt2-extended', 'mlcs2k2':'mlcs2k2', 'II':'nugent-sn2n','IIn':'nugent-sn2n','IIp':'nugent-sn2p', 'IIl':'nugent-sn2l', 
        'Ibc':'nugent-sn1bc',  'Ib':'nugent-sn1bc',  'Ic':'nugent-sn1bc'}
        self.short_names={'Ia':'salt2', 'mlcs2k2':'mlcs'} #Short names because of limitations in Multinest
        if sampler=='nested':
            try:
                import pymultinest
            except ImportError:
                print ('Nested sampling selected but pymultinest is not installed. Defaulting to least squares.')
                sampler='leastsq'
        elif sampler=='mcmc':
            try:
                import emcee
            except ImportError:
                print ('MCMC sampling selected but emcee is not installed. Defaulting to least squares.')
                sampler='leastsq'
                
        self.sampler=sampler
        self.bounds={'salt2-extended':{'z':(0.01, 1.5), 't0':(-100,100),'x0':(-1e-3, 1e-3), 'x1':(-3, 3), 'c':(-0.5, 0.5)}, 
                    'mlcs2k2':{'z':(0.01, 1.5), 't0':(-100,100), 'amplitude':(0, 1e-17), 'delta':(-1.0,1.8),'hostebv':(0, 1),'hostr_v':(-7.0, 7.0)}, 
                    'nugent-sn2n':{'z':(0.01, 1.5)}, 
                    'nugent-sn2p':{'z':(0.01, 1.5)}, 
                    'nugent-sn2l':{'z':(0.01, 1.5)}, 
                    'nugent-sn1bc':{'z':(0.01, 1.5)}}
        
    def extract_features(self, d, save_chains=False, chain_directory='chains', use_redshift=False, nprocesses=1, restart=False, seed=-1):
        """
        Extract template features for a dataset.

        Parameters
        ----------
        d : Dataset object
            Dataset
        save_chains : bool
            Whether or not to save the intermediate output (if Bayesian inference is used instead of least squares)
        chain_directory : str
            Where to save the chains
        use_redshift : bool
            Whether or not to use provided redshift when fitting objects
        nprocesses : int, optional
            Number of processors to use for parallelisation (shared memory only)
        restart : bool
            Whether or not to restart from multinest chains

        Returns
        -------
        astropy.table.Table
            Table of fitted model parameters.

        """
        subprocess.call(['mkdir', chain_directory])
        print ('Fitting templates using', self.sampler, '...')
        all_output=[]
        t1=time.time()
        for mod_name in self.model_names:
            if mod_name=='mlcs2k2':
                dust=sncosmo.CCM89Dust()
                self.model=sncosmo.Model(self.templates[mod_name],effects=[dust],effect_names=['host'], effect_frames=['rest'])
            else:
                self.model=sncosmo.Model(self.templates[mod_name])
            params=['['+mod_name+']'+pname for pname in self.model.param_names]
            # err_plus=[pname+'_err+' for pname in params]
            # err_minus=[pname+'_err-' for pname in params]
            labels = ['Object'] + params
            # if self.sampler=='mcmc':
            #     labels=['Object']+params
            # elif self.sampler=='nested':
            #     labels=['Object']+params
            # else:
            #     labels=['Object']+params+['Chisq']
            
            #output=Table(names=labels, dtype=['S32']+['f']*(len(labels)-1))
            output = Table(names=labels, dtype=['U32'] + ['f'] * (len(labels) - 1))
            
            k=0
            if nprocesses<2:
                for obj in d.object_names:
                    if k%100==0:
                        print (k, 'objects fitted')
                    lc=d.data[obj]
                    
                    if self.sampler=='mcmc':
                        if seed!=-1:
                            np.random.seed(seed)
                        res, fitted_model = sncosmo.mcmc_lc(lc, self.model,  self.model.param_names, bounds=self.bounds[self.templates[mod_name]], nwalkers=20, nsamples=1500, nburn=300)
                        chain=res.samples
                        if save_chains:
                            tab=Table(chain, names=self.model.param_names)
                            tab.write(os.path.join(chain_directory, obj.split('.')[0]+'_emcee_'+mod_name), format='ascii')
                        best=res['parameters'].flatten('F').tolist()
                    elif self.sampler=='nested':
                        best=_run_multinest_templates(obj, d, self.templates[mod_name], self.bounds[self.templates[mod_name]], chain_directory=chain_directory,  
                        nlp=1000, convert_to_binary=False, use_redshift=use_redshift, short_name=self.short_names[mod_name], restart=restart, seed=seed)
                        best=best.tolist()
                    elif self.sampler=='leastsq':
                        if use_redshift:
                            self.model.set(z=lc.meta['z'])
                            prms=self.model.param_names
                            prms=prms[1:]
                            bnds=self.bounds[self.templates[mod_name]].copy()
                            bnds.pop('z', None)
                            res, fitted_model=sncosmo.fit_lc(lc, self.model, vparam_names=prms, 
                                bounds=bnds, minsnr=0)
                        else:
                            
                            res, fitted_model=sncosmo.fit_lc(lc, self.model, vparam_names=self.model.param_names, 
                                bounds=self.bounds[self.templates[mod_name]], minsnr=0)               
                        best=res['parameters'].flatten('F').tolist()#+[res['chisq']]
                    row=[obj]+best
                    output.add_row(row)
                    k+=1
                if len(all_output)==0:
                    all_output=output
                else:
                    all_output=join(all_output, output)
            
            else:
                if self.sampler=='leastsq':
                    p=Pool(nprocesses, maxtasksperchild=1)
                    partial_func=partial(_run_leastsq_templates, d=d, model_name=self.templates[mod_name], use_redshift=use_redshift, bounds=self.bounds[self.templates[mod_name]])
                    out=p.map(partial_func, d.object_names)
                    output=out[0]
                    for i in range(1, len(out)):
                        output=vstack((output, out[i]))
                    if len(all_output)==0:
                        all_output=output
                    else:
                        all_output=vstack((all_output, output))
                elif self.sampler=='nested':
                    p=Pool(nprocesses, maxtasksperchild=1)
                    partial_func=partial(_run_multinest_templates, d=d, model_name=self.templates[mod_name], bounds=self.bounds[self.templates[mod_name]], 
                    chain_directory=chain_directory, nlp=1000, convert_to_binary=True, use_redshift=use_redshift, short_name=self.short_names[mod_name], restart=restart, seed=seed)
                    out=p.map(partial_func, d.object_names)

                    for i in range(len(out)):
                        output.add_row([d.object_names[i]]+out[i].tolist())
                    if len(all_output)==0:
                        all_output=output
                    else:
                        all_output=vstack((all_output, output))
                        
        print (len(all_output), 'objects fitted')
        output_time(time.time()-t1)
        return all_output
        
        
    def fit_sn(self, lc, features):
        """
        Fits the chosen template model to a given light curve.

        Parameters
        ----------
        lc : astropy.table.Table
            Light curve
        features : astropy.table.Table
            Model parameters

        Returns
        -------
        astropy.table.Table
            Fitted light curve
        """
        obj=lc.meta['name']
        tab=features[features['Object']==obj]
        params=np.array([tab[c] for c in tab.columns[1:]]).flatten()
             
        if len(params)==0:
            print ('No feature set found for', obj)
            return None
        
        model_name=self.templates[self.model_names[0]]
        if model_name=='mlcs2k2':
            dust=sncosmo.CCM89Dust()
            model=sncosmo.Model(model_name,effects=[dust],effect_names=['host'], effect_frames=['rest'])
        else:
            model=sncosmo.Model(model_name)
        
        param_dict={}
        for i in range(len(model.param_names)):
            param_dict[model.param_names[i]]=params[i]
        model.set(**param_dict)

        filts=np.unique(lc['filter'])
        labels=['mjd', 'flux', 'filter']
        output=Table(names=labels, dtype=['f', 'f', 'U32'],  meta={'name':obj})
        for filt in filts:
            x=lc['mjd'][lc['filter']==filt]
            xnew=np.linspace(0, x.max()-x.min(), 100)
            P=params
            
            ynew=model.bandflux(filt, xnew, zp=27.5, zpsys='ab')

            newtable=Table([xnew+x.min(), ynew, [filt]*len(xnew)], names=labels)
            #newtable=Table([x, ynew, [filt]*len(x)], names=labels)
            output=vstack((output, newtable))
        return output

    def registerBands(self, dirname, prefix=None, suffix=None):
        """Register LSST bandpasses with sncosmo.
           Courtesy of Rahul Biswas"""
        bands = ['u', 'g', 'r', 'i', 'z', 'y']
        for band in bands:
            fname = os.path.join(dirname, prefix + band + suffix)
            data = np.loadtxt(fname)
            bp = sncosmo.Bandpass(wave=data[:, 0], trans=data[:, 1], name='lsst'+band)
            sncosmo.registry.register(bp, force=True)
            
class ParametricFeatures(Features):
    """
    Fits a few options of generalised, parametric models to the data.
    """
    
    def __init__(self, model_choice, sampler='leastsq', limits=None):
        """
        Initialisation

        Parameters
        ----------
        model_choice : str
            Which parametric model to use
        sampler : str, optional
            Choice of 'mcmc' (requires emcee), 'nested' (requires pymultinest) or 'leastsq'
        limits : dict, optional
            Parameter bounds if something other than the default is needed.
        """

        Features.__init__(self)

        self.model_choices={'newling':parametric_models.NewlingModel, 'karpenka':parametric_models.KarpenkaModel}

        try:
            self.model_name=model_choice #Used for labelling output files
            if limits is not None:
                self.model=self.model_choices[model_choice](limits=limits)
            else:
                self.model=self.model_choices[model_choice]()
        except KeyError:
            print ('Your selected model is not in the parametric_models package. Either choose an existing model,  or implement a new one in that package.')
            print ('Make sure any new models are included in the model_choices dictionary in the ParametricFeatures class.')
            sys.exit()
            
        if sampler=='nested' and not has_multinest:
            print ('Nested sampling selected but pymultinest is not installed. Defaulting to least squares.')
            sampler='leastsq'
                
        elif sampler=='mcmc' and not has_emcee:
            print ('MCMC sampling selected but emcee is not installed. Defaulting to least squares.')
            sampler='leastsq'
            
        self.sampler=sampler



    def extract_features(self, d, chain_directory='chains', save_output=True, n_attempts=20, nprocesses=1, n_walkers=100, 
    n_steps=500, walker_spread=0.1, burn=50, nlp=1000, starting_point=None, convert_to_binary=True, n_iter=0, restart=False, seed=-1):
        """
        Fit parametric models and return best-fitting parameters as features.

        Parameters
        ----------
        d : Dataset object
            Dataset
        chain_directory : str
            Where to save the chains
        save_output : bool
            Whether or not to save the intermediate output (if Bayesian inference is used instead of least squares)
        n_attempts : int
            Allow the minimiser to start in new random locations if the fit is bad. Put n_attempts=1 to fit only once
            with the default starting position.
        nprocesses : int, optional
            Number of processors to use for parallelisation (shared memory only)
        n_walkers : int
            emcee parameter - number of walkers to use
        n_steps : int
            emcee parameter - total number of steps
        walker_spread : float
            emcee parameter - standard deviation of distribution of starting points of walkers
        burn : int
            emcee parameter - length of burn-in
        nlp : int
            multinest parameter - number of live points
        starting_point : None or array-like
            Starting points of parameters for leastsq or emcee
        convert_to_binary : bool
            multinest parameter - whether or not to convert ascii output files to binary
        n_iter : int
            leastsq parameter - number of iterations to avoid local minima
        restart : bool
            Whether or not t restart from existing multinest chains

        Returns
        -------
        astropy.table.Table
            Best-fitting parameters

        """
        subprocess.call(['mkdir', chain_directory])
        self.chain_directory=chain_directory
        t1=time.time()
        output=[]
        
        #obj=d.object_names[0]
        if nprocesses<2:
            k=0
            for obj in d.object_names:
                if k%100==0:
                    print (k, 'objects fitted')
                if self.sampler=='leastsq':
                    newtable=_run_leastsq(obj, d, self.model, n_attempts, seed=seed)
                elif self.sampler=='mcmc':
                    if(seed!=-1):
                        np.random.seed(seed)
                    newtable=self.run_emcee(d, obj, save_output, chain_directory,   n_walkers, n_steps, walker_spread, burn, starting_point)
                else:
                    newtable=_run_multinest(obj, d, self.model, chain_directory, nlp, convert_to_binary, n_iter, restart, seed=seed)
                
                if len(output)==0:
                    output=newtable
                else:
                    output=vstack((output, newtable))
                k+=1
        else:
            if self.sampler=='leastsq':
                p=Pool(nprocesses, maxtasksperchild=1)
                partial_func=partial(_run_leastsq, d=d, model=self.model,  n_attempts=n_attempts, seed=seed)
                out=p.map(partial_func, d.object_names)
                output=out[0]
                for i in range(1, len(out)):
                    output=vstack((output, out[i]))
            elif self.sampler=='nested':
                p=Pool(nprocesses, maxtasksperchild=1)
                partial_func=partial(_run_multinest, d=d, model=self.model,chain_directory=chain_directory, 
                nlp=nlp, convert_to_binary=convert_to_binary, n_iter=n_iter, restart=restart, seed=seed)
                #Pool starts a number of threads, all of which may try to tackle all of the data. Better to take it in chunks
                output=[]
                k=0
                objs=d.object_names
                while k<len(objs):
                    objs_subset=objs[k:k+nprocesses]
                    out=p.map(partial_func, objs_subset)
                    for i in range(0, len(out)):
                        if out[i]==None:
                            print ('Fitting failed for', objs_subset[i])
                        else:
                            if len(output)==0:
                                output=out[i]
                            else:
                                output=vstack((output, out[i]))
                    k+=len(objs_subset)
        print (len(output), 'objects fitted')
        output_time(time.time()-t1)
        return output
        
        
    def fit_sn(self, lc, features):
        """
        Fits the chosen parametric model to a given light curve.

        Parameters
        ----------
        lc : astropy.table.Table
            Light curve
        features : astropy.table.Table
            Model parameters

        Returns
        -------
        astropy.table.Table
            Fitted light curve
        """
        obj=lc.meta['name']
        params=features[features['Object']==obj]
        
        if len(params)==0:
            print ('No feature set found for', obj)
            return None
        
        filts=np.unique(lc['filter'])
        labels=['mjd', 'flux', 'filter']
        output=Table(names=labels, dtype=['f', 'f', 'U32'],  meta={'name':obj})
        cols=params.columns[1:]
        prms=np.array([params[c] for c in cols])
        for filt in filts:
            x=lc['mjd'][lc['filter']==filt]
            xnew=np.linspace(0, x.max()-x.min(), 100)
            
            #inds=[s for s in cols if filt in s]
            inds=np.where([filt in s for s in cols])[0]
            P=np.array(prms[inds],dtype='float')

            ynew=self.model.evaluate(xnew, P)
            newtable=Table([xnew+x.min(), ynew, [filt]*len(xnew)], names=labels)
            output=vstack((output, newtable))
        return output
        

        
    def run_emcee(self, d, obj, save_output, chain_directory,  n_walkers, n_steps, walker_spread, burn, starting_point, seed=-1):
        """
        Runs emcee on all the filter bands of a given light curve, fitting the model to each one and extracting the best fitting parameters.

        Parameters
        ----------
        d : Dataset object
            Dataset
        obj : str
            Object name
        save_output : bool
            Whether or not to save the intermediate output
        chain_directory : str
            Where to save the chains
        n_walkers : int
            emcee parameter - number of walkers to use
        n_steps : int
            emcee parameter - total number of steps
        walker_spread : float
            emcee parameter - standard deviation of distribution of starting points of walkers
        burn : int
            emcee parameter - length of burn-in
        starting_point : None or array-like
            Starting points of parameters

        Returns
        -------
        astropy.table.Table
            Best fitting parameters (at the maximum posterior)
        """
        def get_params(starting_point, obj, filt):
            #Helper function to get parameters from the features astropy table
            X=starting_point[starting_point['Object']==obj]
            cols=X.columns
            inds=[s for s in cols if filt in s]
            P=X[inds]
            P=np.array([P[c] for c in P.columns]).flatten()
            return P
        
        lc=d.data[obj]
        filts=np.unique(lc['filter'])
        
        if(seed!=-1):
            np.random.seed(seed)

        n_params=len(self.model.param_names)
        
        #err_plus=[pname+'_err+' for pname in self.model.param_names]
        #err_minus=[pname+'_err-' for pname in self.model.param_names]
        labels=['Object']
        for f in d.filter_set:
            pams=self.model.param_names
            for p in pams:
                labels.append(f+'-'+p)

        output=Table(names=labels, dtype=['U32']+['f']*(len(labels)-1))

        t1=time.time()
        row=[obj]
        for f in d.filter_set:
            if f in filts:
                #This is pretty specific to current setup
                chain_name=os.path.join(self.chain_directory, '%s-%s-%s-' %(obj.split('.')[0], f, self.model_name))
                
                x, y, yerr=np.array(lc['mjd'][lc['filter']==f]), np.array(lc['flux'][lc['filter']==f]), np.array(lc['flux_error'][lc['filter']==f])
                
                
                if starting_point is None:
                    #Initialise randomly in parameter space
                    iniparams=np.random(n_params)*(self.model.upper_limit-self.model.lower_limit)+self.model.lower_limit
                else:
                    #A starting point from a least squares run can be given as an astropy table
                    iniparams=get_params(starting_point, obj, f)
                
                pos = [iniparams + walker_spread*np.randn(n_params) for i in range(n_walkers)]
                
                sampler = emcee.EnsembleSampler(n_walkers, n_params, self.lnprob_emcee, args=(x, y, yerr))
                pos, prob, state = sampler.run_mcmc(pos, burn) #Remove burn-in
                sampler.reset()
                pos, prob, state = sampler.run_mcmc(pos, n_steps)
                
                samples = sampler.flatchain
                lnpost=sampler.flatlnprobability
                
                if save_output:
                    np.savetxt(chain_directory+'emcee_chain_%s_%s_%s' %(self.model_name, f, (str)(obj)), np.column_stack((samples, lnpost)))
                #Maximum posterior params
                ind=lnpost.argmax()
                best_params=samples[ind, :]
                
                #Expects first the parameters, then -sigma then +sigma
                row+=best_params
            else:
                row+=[0]*len(self.model.param_names) #I'm not sure if it makes the most sense to fill in missing values with zeroes...
        output.add_row(row)

        print ('Time per filter', (time.time()-t1)/len(d.filter_set))
        return output
    

    def lnprob_emcee(self, params, x, y, yerr):
        """
        Likelihood function for emcee

        Parameters
        ----------
        params
        x
        y
        yerr

        Returns
        -------

        """
        #Uniform prior. Directly compares arrays
        if (np.any(params>self.model.upper_limit)) or (np.any(params<self.model.upper_limit)):
            return -np.inf
        else:
            ynew=self.model.evaluate(x, params)
            chi2=np.sum((y-ynew)*(y-ynew)/yerr/yerr)
            return -chi2/2.
        
        
class WaveletFeatures(Features):
    """
    Uses wavelets to decompose the data and then reduces dimensionality of the feature space using PCA.
    """

    def __init__(self, wavelet='sym2', ngp=100,**kwargs):
        """
        Initialises the pywt wavelet object and sets the maximum depth for deconstruction.

        Parameters
        ----------
        wavelet : str, optional
            String for which wavelet family to use.
        ngp : int, optional
            Number of points on the Gaussian process curve
        level : int, optional
            The maximum depth for wavelet deconstruction. If not provided, will use the maximum depth possible
            given the number of points in the Gaussian process curve.
        """
        Features.__init__(self)


        self.wav=pywt.Wavelet(wavelet)
        self.ngp=ngp #Number of points to use on the Gaussian process curve
        self.wavelet_list=pywt.wavelist() #All possible families

        if wavelet not in self.wavelet_list:
            print ('Wavelet not recognised in pywt')
            sys.exit()

        #If the user does not specify a level of depth for the wavelet, automatically calculate it
        if 'level' in kwargs:
            self.mlev=kwargs['level']
        else:
            self.mlev=pywt.swt_max_level(self.ngp)


    def extract_features(self, d, initheta=[500, 20], save_output='none',output_root='features', nprocesses=1, restart='none', gpalgo='george'):
        """
        Applies a wavelet transform followed by PCA dimensionality reduction to extract wavelet coefficients as features.

        Parameters
        ----------
        d : Dataset object
            Dataset
        initheta: list-like, optional
            Initial values for theta parameters. These should be roughly the scale length in the y & x directions.
        save_output : bool, optional
            Whether or not to save the output
        output_root : str, optional
         Output directory
        nprocesses : int, optional
            Number of processors to use for parallelisation (shared memory only)
        restart : str, optional
            Either 'none' to start from scratch, 'gp' to restart from saved Gaussian processes, or 'wavelet' to
            restart from saved wavelet decompositions (will look in output_root for the previously saved outputs).
        log : bool, optional
            Whether or not to take the logarithm of the final PCA components. Recommended setting is False (legacy code).

        Returns
        -------
        astropy.table.Table
            Table of features (first column object names, the rest are the PCA coefficient values)
        """

        if save_output is not 'none':
            subprocess.call(['mkdir', output_root])

        xmin=0
        xmax=d.get_max_length()

        if restart=='wavelet':
            wavout, waveout_err=self.restart_from_wavelets(d, output_root)
        else:
            if restart=='gp':
                self.restart_from_gp(d, output_root)
            else:
                self.extract_GP(d, self.ngp, xmin, xmax, initheta, save_output, output_root, nprocesses, gpalgo=gpalgo)

            wavout, waveout_err=self.extract_wavelets(d, self.wav, self.mlev,  nprocesses, save_output, output_root)
        self.features,vals,vec,mn=self.extract_pca(d.object_names.copy(), wavout)

        #Save the PCA information
        self.PCA_eigenvals = vals
        self.PCA_eigenvectors=vec
        self.PCA_mean=mn

        return self.features


    def fit_sn(self, lc, comps, vec,  mn, xmin, xmax, filter_set):
        """
        Fits a single object using previously run PCA components. Performs the full inverse wavelet transform.

        Parameters
        ----------
        lc : astropy.table.Table
            Light curve
        comps : astropy.table.Table
            The PCA coefficients for each object (i.e. the astropy table of wavelet features from by extract_features).
        vec : array-like
            PCA component vectors as array (each column is a vector, ordered from most to least significant)
        mn : array-like
            Mean vector
        xmin : float
            The minimum on the x axis (as defined for the original GP decomposition)
        xmax : float
            The maximum on the x axis (as defined for the original GP decomposition)
        filter_set : list-like
            The full set of filters of the original dataset

        Returns
        -------
        astropy.table.Table
            Fitted light curve
        """

        obj=lc.meta['name']
        filts=np.unique(lc['filter'])
        #The PCA will have been done over the full coefficient set, across all filters
        try:
            pca_comps=comps[comps['Object']==obj]
        except KeyError:
            print ('No feature set found for', obj)
            return None

        new_comps=np.array([pca_comps[c] for c in pca_comps.columns[1:]]).flatten()
        ncomp=len(new_comps)
        eigs=vec[:, :ncomp]

        coeffs=np.array(np.dot(new_comps, eigs.T)+mn).flatten()

        n=self.mlev*2*self.ngp
        xnew=np.linspace(xmin, xmax, self.ngp)
        output=[]
        for i in range(len(filter_set)):
            if filter_set[i] in filts:
                filt_coeffs=coeffs[i*n:(i+1)*n]
                filt_coeffs=filt_coeffs.reshape(self.mlev, 2, self.ngp, order='C')
                ynew=self.iswt(filt_coeffs, self.wav)

                newtable=Table([xnew, ynew, [filter_set[i]]*self.ngp], names=['mjd', 'flux', 'filter'], dtype=['f', 'f', 'U32'])
                if len(output)==0:
                    output=newtable
                else:
                    output=vstack((output, newtable))
        return output


    def restart_from_gp(self, d, output_root):
        """
        Allows the restarted of the feature extraction process from previously saved Gaussian Process curves.

        Parameters
        ----------
        d : Dataset object
            The same dataset (object) on which the previous GP analysis was performed.
        output_root : str
            Location of GP objects
        """

        print ('Restarting from stored Gaussian Processes...')
        for obj in d.object_names:
            fname=os.path.join(output_root, 'gp_'+obj)
            try:
                tab=Table.read(fname, format='ascii')
                d.models[obj]=tab
            except IOError:
                print ('IOError, file ',fname, 'does not exist.')

    def restart_from_wavelets(self, d, output_root):
        """
        Allows the restarted of the feature extraction process from previously saved wavelet decompositions. This
        allows you to quickly try different dimensionality reduction (e.g. PCA) algorithms on the wavelets.

        Parameters
        ----------
        d : Dataset object
            The same dataset (object) on which the previous wavelet analysis was performed.
        output_root : str
            Location of previously decomposed wavelet coefficients

        Returns
        -------
        wavout : array
            A numpy array of the wavelet coefficients where each row is an object and each column a different coefficient
        wavout_err :  array
            A similar numpy array storing the (assuming Gaussian) error on each coefficient.
        """

        print ('Restarting from stored wavelets...')
        nfilts=len(d.filter_set)
        wavout=np.zeros([len(d.object_names), self.ngp*2*self.mlev*nfilts]) #This is just a very big array holding coefficients in memory
        wavout_err=np.zeros([len(d.object_names), self.ngp*2*self.mlev*nfilts])

        for i in range(len(d.object_names)):
            obj=d.object_names[i]
            fname=os.path.join(output_root, 'wavelet_'+obj)
            try:
                out=Table.read(fname, format='ascii')
                cols=out.colnames[:-1]
                n=self.ngp*2*self.mlev
                for j in range(nfilts):
                    x=out[out['filter']==d.filter_set[j]]
                    coeffs=x[cols[:self.mlev*2]]
                    coeffs_err=x[cols[self.mlev*2:]]
                    newcoeffs=np.array([coeffs[c] for c in coeffs.columns]).T
                    newcoeffs_err=np.array([coeffs_err[c] for c in coeffs_err.columns]).T
                    wavout[i, j*n:(j+1)*n]=newcoeffs.flatten('F')
                    wavout_err[i, j*n:(j+1)*n]=newcoeffs_err.flatten('F')

            except IOError:
                print ('IOError, file ',fname, 'does not exist.')

        return wavout, wavout_err

    def extract_GP(self, d, ngp, xmin, xmax, initheta, save_output,  output_root, nprocesses, gpalgo='george'):
        """
        Runs Gaussian process code on entire dataset. The result is stored inside the models attribute of the dataset object.

        Parameters
        ----------
        d : Dataset object
            Dataset
        ngp : int
            Number of points to evaluate Gaussian Process at
        xmin : float
            Minimim time to evaluate at
        xmax : float
            Maximum time to evaluate at
        initheta : list-like
            Initial values for theta parameters. These should be roughly the scale length in the y & x directions.
        save_output : bool
            Whether or not to save the output
        output_root : str
            Output directory
        nprocesses : int, optional
            Number of processors to use for parallelisation (shared memory only)
        """
        print ('Performing Gaussian process regression')
        t1=time.time()
        #Check for parallelisation
        if nprocesses==1:
            for i in range(len(d.object_names)):
                obj=d.object_names[i]
                out=_GP(obj, d=d,ngp=ngp, xmin=xmin, xmax=xmax, initheta=initheta, save_output=save_output, output_root=output_root, gpalgo=gpalgo)
                d.models[obj]=out
                if save_output!='none':
                    out.write(os.path.join(output_root, 'gp_'+obj), format='ascii')
        else:
            p=Pool(nprocesses, maxtasksperchild=1)

            #Pool and map can only really work with single-valued functions
            partial_GP=partial(_GP, d=d, ngp=ngp, xmin=xmin, xmax=xmax, initheta=initheta, save_output=save_output, output_root=output_root, gpalgo=gpalgo)

            out=p.map(partial_GP, d.get_object_names())
            for i in range(len(out)):
                obj=d.get_object_names()[i]
                d.models[obj]=out[i]

        print ('Time taken for Gaussian process regression', time.time()-t1)

    def GP(self, obj, d, ngp=200, xmin=0, xmax=170, initheta=[500, 20], gpalgo='george'):
        """
        Fit a Gaussian process curve at specific evenly spaced points along a light curve.

        Parameters
        ----------
        obj : str
            Object name
        d : Dataset object
            Dataset
        ngp : int, optional
            Number of points to evaluate Gaussian Process at
        xmin : float, optional
            Minimim time to evaluate at
        xmax : float, optional
            Maximum time to evaluate at
        initheta : list-like, optional
            Initial values for theta parameters. These should be roughly the scale length in the y & x directions.

        Returns
        -------
        astropy.table.Table
            Table with evaluated Gaussian process curve and errors

        Notes
        -----
        Wraps internal module-level function in order to circumvent multiprocessing module limitations in dealing with
        objects when parallelising.
        """
        return _GP(obj, d, ngp, xmin, xmax, initheta, gpalgo)


    def wavelet_decomp(self, lc, wav, mlev):
        """
        Perform a wavelet decomposition on a single light curve.

        Parameters
        ----------
        lc : astropy.table.Table
            Light curve
        wav : str or swt.Wavelet object
            Which wavelet family to use
        mlev : int
            Max depth

        Returns
        -------
        astropy.table.Table
            Decomposed coefficients in each filter.
        """

        filters=np.unique(lc['filter'])
        ngp=len(lc['flux'][lc['filter']==filters[0]])
        #Store the output in another astropy table
        output=0
        for fil in filters:
            y=lc['flux'][lc['filter']==fil]
            err=lc['flux_error'][lc['filter']==fil]
            coeffs=np.array(pywt.swt(y, wav, level=mlev))
            coeffs_err=np.array(pywt.swt(err, wav, level=mlev)) #This actual gives a slight overestimate of the error

            #Create the column names (follows pywt convention)
            labels=[]
            for i in range(len(coeffs)):
                labels.append('cA%d' %(len(coeffs)-i))
                labels.append('cD%d' %(len(coeffs)-i))

            #For the erors
            err_labels=[]
            for i in range(len(labels)):
                err_labels.append(labels[i]+'_err')
            npoints=len(coeffs[0, 0, :])
            c=coeffs.reshape(mlev*2, npoints).T
            c_err=coeffs_err.reshape(mlev*2, npoints).T
            newtable1=Table(c, names=labels)
            newtable2=Table(c_err, names=err_labels)
            joined_table=hstack([newtable1, newtable2])
            #Add the filters
            joined_table['filter']=[fil]*npoints

            if output==0:
                output=joined_table
            else:
                output=vstack((output, joined_table))

        return output

    def extract_wavelets(self, d, wav, mlev, nprocesses, save_output, output_root):
        """
        Perform wavelet decomposition on all objects in dataset. Output is stored as astropy table for each object.

        Parameters
        ----------
        d : Dataset object
            Dataset
        wav : str or swt.Wavelet object
            Which wavelet family to use
        mlev : int
            Max depth
        nprocesses : int, optional
            Number of processors to use for parallelisation (shared memory only)
        save_output : bool, optional
            Whether or not to save the output
        output_root : str, optional
         Output directory

        Returns
        -------
        wavout : array
            A numpy array of the wavelet coefficients where each row is an object and each column a different coefficient
        wavout_err :  array
            A numpy array storing the (assuming Gaussian) error on each coefficient.
        """

        print ('Performing wavelet decomposition')

        nfilts=len(d.filter_set)
        wavout=np.zeros([len(d.object_names), self.ngp*2*mlev*nfilts]) #This is just a big array holding coefficients in memory
        wavout_err=np.zeros([len(d.object_names), self.ngp*2*mlev*nfilts])
        t1=time.time()
        for i in range(len(d.object_names)):
            obj=d.object_names[i]
            lc=d.models[obj]
            out= self.wavelet_decomp(lc, wav, mlev)
            if save_output=='wavelet' or save_output=='all':
                out.write(os.path.join(output_root, 'wavelet_'+obj), format='ascii')
            #We go by filter, then by set of coefficients
            cols=out.colnames[:-1]
            n=self.ngp*2*mlev
            filts=np.unique(lc['filter'])
            for j in range(nfilts):
                if d.filter_set[j] in filts:
                    x=out[out['filter']==d.filter_set[j]]
                    coeffs=x[cols[:mlev*2]]
                    coeffs_err=x[cols[mlev*2:]]
                    newcoeffs=np.array([coeffs[c] for c in coeffs.columns]).T
                    newcoeffs_err=np.array([coeffs_err[c] for c in coeffs_err.columns]).T
                    wavout[i, j*n:(j+1)*n]=newcoeffs.flatten('F')
                    wavout_err[i, j*n:(j+1)*n]=newcoeffs_err.flatten('F')
        print ('Time for wavelet decomposition', time.time()-t1)

        return wavout, wavout_err


    def pca(self, X):
        """
        Performs PCA decomposition of a feature array X.

        Parameters
        ----------
        X : array
            Array of features to perform PCA on.

        Returns
        -------
        vals : list-like
            Ordered array of eigenvalues
        vec : array
            Ordered array of eigenvectors, where each column is an eigenvector.
        mn : array
            The mean of the dataset, which is subtracted before PCA is performed.

        Notes
        -----
        Although SVD is considerably more efficient than eigh, it seems more numerically unstable and results in many more components
        being required to adequately describe the dataset, at least for the wavelet feature sets considered.
        """

        #Find the normalised spectra
        X=X.transpose()
        mn=np.mean(X, axis=1)
        mn.shape=(len(mn), 1)
        X=X-mn

        nor=np.sqrt(np.sum(X**2, axis=0))
        x_norm=X/nor
#
#        #Construct the covariance matrix
        C=np.dot(x_norm, x_norm.T)
        #C=np.cov(X.T)
        #print C.shape

        C=np.mat(C)
        vals, vec = np.linalg.eigh(C)

        inds=np.argsort(vals)[::-1]
        return vals[inds], vec[:, inds], mn

    def best_coeffs(self, vals, tol=0.98):
        """
        Determine the minimum number of PCA components required to adequately describe the dataset.

        Parameters
        ----------
        vals : list-like
            List of eigenvalues (ordered largest to smallest)
        tol : float, optional
            How much 'energy' or information must be retained in the dataset.

        Returns
        -------
        int
            The required number of coefficients to retain the requested amount of "information".

        """
        tot=np.sum(vals)
        tot2=0
        for i in range(len(vals)):
            tot2+=vals[i]
            if tot2>=tol*tot:
                return i
                break

        print ("No dimensionality reduction achieved. All components of the PCA are required.")
        return -1

    def project_pca(self, X, eig_vec):
        """
        Project a vector onto a PCA axis (i.e. transform data to PCA space).

        Parameters
        ----------
        X : array
            Vector of original data (for one object).
        eig_vec : array
            Array of eigenvectors, first column most significant.

        Returns
        -------
        array
            Coefficients of eigen vectors

        """

        A=np.linalg.lstsq(np.mat(eig_vec), np.mat(X).T)[0].flatten()
        return np.array(A)


    def extract_pca(self, object_names,  wavout):
        """
        Dimensionality reduction of wavelet coefficients using PCA.

        Parameters
        ----------
        object_names : list-like
            Object names corresponding to each row of the wavelet coefficient array.
        wavout : array
            Wavelet coefficient array, each row corresponds to an object, each column is a coefficient.
        log

        Returns
        -------
        astropy.table.Table
            Astropy table containing PCA features.

        """

        print ('Running PCA...')
        t1=time.time()
        #We now run PCA on this big matrix
        vals, vec, mn=self.pca(wavout)
        mn=mn.flatten()

        #
        np.savetxt('PCA_vals.txt', vals)
        np.savetxt('PCA_vec.txt', vec)
        np.savetxt('PCA_mean.txt', mn)

        #Actually fit the components
        ncomp=self.best_coeffs(vals)
        eigs=vec[:, :ncomp]
        comps=np.zeros([len(wavout), ncomp])

        for i in range(len(wavout)):
            coeffs=wavout[i]
            A=self.project_pca(coeffs-mn, eigs)
            comps[i]=A
        labels=['C%d' %i for i in range(ncomp)]
        wavs=Table(comps, names=labels)
        object_names.shape=(len(object_names), 1)
        objnames=Table(object_names, names=['Object'])
        wavs=hstack((objnames, wavs))
        #wavs.write('wavelet_features.dat', format='ascii')
        print ('Time for PCA', time.time()-t1)
        return wavs,vals,vec,mn

    def iswt(self, coefficients, wavelet):
        """
        Performs inverse wavelet transform.
        M. G. Marino to complement pyWavelets' swt.

        Parameters
        ----------
        coefficients : array
            approx and detail coefficients, arranged in level value
            exactly as output from swt:
            e.g. [(cA1, cD1), (cA2, cD2), ..., (cAn, cDn)]
        wavelet : str or swt.Wavelet
            Either the name of a wavelet or a Wavelet object

        Returns
        -------
        array
            The inverse transformed array
        """
        output = coefficients[0][0].copy() # Avoid modification of input data

        #num_levels, equivalent to the decomposition level, n
        num_levels = len(coefficients)
        for j in range(num_levels,0,-1):
            step_size = int(2**(j-1))
            last_index = step_size
            _, cD = coefficients[num_levels - j]
            for first in range(last_index): # 0 to last_index - 1

                # Getting the indices that we will transform
                indices = np.arange(first, len(cD), step_size)

                # select the even indices
                even_indices = indices[0::2]
                # select the odd indices
                odd_indices = indices[1::2]

                # perform the inverse dwt on the selected indices,
                # making sure to use periodic boundary conditions
                x1 = pywt.idwt(output[even_indices], cD[even_indices], wavelet, 'per')
                x2 = pywt.idwt(output[odd_indices], cD[odd_indices], wavelet, 'per')

                # perform a circular shift right
                x2 = np.roll(x2, 1)

                # average and insert into the correct indices
                output[indices] = (x1 + x2)/2.

        return output
