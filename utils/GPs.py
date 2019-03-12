"""
Module for extracting and saving GPs
"""

import numpy as np
import sys, time, subprocess, os, sncosmo
from scipy.interpolate import interp1d
import george
from astropy.table import Table, vstack, hstack, join
from multiprocessing import Pool
from functools import partial
import scipy.optimize as op
import pandas as pd
from scipy import interpolate
import scipy

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


def extract_GP(d, ngp, t_min, t_max, initheta, output_root, nprocesses, gp_algo='george', save_output=False):
    """
    Runs Gaussian process code on entire dataset. The result is stored inside the models attribute of the dataset object.

    Parameters
    ----------
    d : Dataset object
        Dataset
    ngp : int
        Number of points to evaluate Gaussian Process at
    t_min : float
        Minimim time to evaluate at
    t_max : float
        Maximum time to evaluate at
    initheta : list-like
        Initial values for theta parameters. These should be roughly the scale length in the y & x directions.
    output_root : str
        Output directory.
    nprocesses : int, optional
        Number of processors to use for parallelisation (shared memory only)
    gp_algo : str, optional
        which gp package is used for the Gaussian Process Regression, GaPP or george
    save_output : bool, optional
        whether or not to save the fitted GP means and errors
    """
    print ('Performing Gaussian process regression')
    initial_time = time.time()

    # Check for parallelisation
    if nprocesses == 1: # non parallelizing
        for i in range(len(d.object_names)):
            obj = d.object_names[i]
            try:
                output, gpdict, used_kernels_obj = _GP(obj, d=d,ngp=ngp, t_min=t_min, t_max=t_max, initheta=initheta, 
                                                        output_root=output_root, gp_algo=gp_algo)
                d.models[obj] = output
            except ValueError:
                print('Object {} has fallen over!'.format(obj))
    else: # parallelizing
        p = Pool(nprocesses, maxtasksperchild=10)

        #Pool and map can only really work with single-valued functions
        partial_GP = partial(_GP, d=d, ngp=ngp, t_min=t_min, t_max=t_max, initheta=initheta, output_root=output_root, gp_algo=gp_algo, save_output=save_output)

        out = p.map(partial_GP, d.object_names, chunksize=10)
        p.close()
        gp = {}
        used_kernels = {}

        out = np.reshape(out,(len(d.object_names),3))
        for i in range(len(out)):
            obj = d.object_names[i]
            d.models[obj] = out[i,0]
            gp[obj] = out[i,1]
            used_kernels[obj] = out[i,2]
            
        #with open(output_root+'/used_kernels.yaml', 'w') as kernels:
        #    yaml.dump(used_kernels, kernels, default_flow_style=False)

    print ('Time taken for Gaussian process regression', time.time()-initial_time)


def _GP(obj, d, ngp, t_min, t_max, initheta, output_root, gp_algo='george', save_output=False):
    """
    Fit a Gaussian process curve in every filter of an object.

    Parameters
    ----------
    obj : str
        Name of the object
    d : Dataset-like object
        Dataset
    ngp : int
        Number of points to evaluate Gaussian Process at
    t_min : float
        Minimim time to evaluate at
    t_max : float
        Maximum time to evaluate at
    initheta : list-like
        Initial values for theta parameters. These should be roughly the scale length in the y & x directions.
    output_root : str
        Output directory.
    gp_algo : str
        which gp package is used for the Gaussian Process Regression, GaPP or george
    save_output : bool, optional
        whether or not to save the fitted GP means and errors

    Returns
    -------
    output: astropy.table.Table
        Table with evaluated Gaussian process curve and errors
    """

    if gp_algo=='gapp' and not has_gapp:
        print('No GP module gapp. Defaulting to george instead.')
        gp_algo='george'
    lc      = d.data[obj]
    filters = np.unique(lc['filter'])
    gp_times = np.linspace(t_min, t_max, ngp)

    # Store the output in another astropy table
    output = []
    gpdict = {}
    filter_set = np.asarray(d.filter_set)
    used_kernels_obj = np.array([None]*len(filter_set)) # inilialize None kernel to each filter
    for fil in filter_set:
        if fil in filters:
            obj_times    = lc['mjd'][lc['filter']==fil]  # x
            obj_flux     = lc['flux'][lc['filter']==fil] # y
            obj_flux_err = lc['flux_error'][lc['filter']==fil] # y_err
            obj_obs = pd.DataFrame(columns=['mjd'], data=obj_times)
            obj_obs['flux']  = obj_flux
            obj_obs['flux_err'] = obj_flux_err

            if gp_algo=='gapp':
                gp         = dgp.DGaussianProcess(obj_times, obj_flux, obj_flux_err, cXstar=(t_min, t_max, ngp))
                rec, theta = gp.gp(theta=initheta)
            elif gp_algo=='george':
                metric  = initheta[1]**2
                gp_obs_0, redChi2_0, gp_0 = get_GP_redChi2(np.array([initheta[0]**2, metric, 2., 4., 4., 6., 6.]), 'ExpSquared', obj_obs, gp_times)
                if redChi2_0 < 2: # good gp
                    gp_obs, redChi2, gp, chosen_kernel = gp_obs_0, redChi2_0, gp_0, 'ExpSquared 0'
                else: # bad gp
                    gp_obs_1, redChi2_1, gp_1 = get_GP_redChi2(np.array([400., 200., 2., 4., 4., 6., 6.]), 'ExpSquared', obj_obs, gp_times)
                    if redChi2_1 < 2: # good gp
                        gp_obs, redChi2, gp, chosen_kernel = gp_obs_1, redChi2_1, gp_1, 'ExpSquared 1'
                    else:             # bad gp
                        gp_obs_all  = [gp_obs_0, gp_obs_1]
                        redChi2_all = [redChi2_0, redChi2_1]
                        gp_all      = [gp_0, gp_1]
                        gp_obs_2, redChi2_2, gp_2 = get_GP_redChi2(np.array([400., 20., 2., 4., 4., 6., 6.]), 'ExpSquared', obj_obs, gp_times)
                        if redChi2_2 < 2: # good gp
                            gp_obs, redChi2, gp, chosen_kernel =  gp_obs_2, redChi2_2, gp_2, 'ExpSquared 2'
                        else:             # bad gp
                            gp_obs_all.append(gp_obs_2)
                            redChi2_all.append(redChi2_2)
                            gp_all.append(gp_2)
                            gp_obs_3, redChi2_3, gp_3 = get_GP_redChi2(np.array([19., 9., 2., 4., 4., 6., 6.]), 'ExpSquared+ExpSine2',  obj_obs, gp_times)
                            if redChi2_3 < 2: # good gp
                                gp_obs, redChi2, gp, chosen_kernel =  gp_obs_3, redChi2_3, gp_3, 'ExpSquared+ExpSine2'
                            else:             # bad gp
                                gp_obs_all.append(gp_obs_3)
                                redChi2_all.append(redChi2_3)
                                gp_all.append(gp_3)
                                kernels = ['bad ExpSquared 0', 'bad ExpSquared 1', 'bad ExpSquared 2', 'bad ExpSquared+ExpSine2']
                                indMinRedChi2 = np.argmin(redChi2_all)
                                gp_obs, redChi2, gp = gp_obs_all[indMinRedChi2], redChi2_all[indMinRedChi2], gp_all[indMinRedChi2]
                                chosen_kernel = kernels[indMinRedChi2]

                used_kernels_obj[filter_set==fil] = chosen_kernel
                mu,cov = gp_obs.flux.values, gp_obs.flux_err.values
                std    = np.sqrt(np.diag(cov))
                rec    = np.column_stack((gp_times, mu, std))
            gpdict[fil] = gp
        else:
            rec=np.zeros([ngp, 3])
        newtable=Table([rec[:, 0], rec[:, 1], rec[:, 2], [fil]*ngp], names=['mjd', 'flux', 'flux_error', 'filter'])
        if len(output)==0:
            output=newtable
        else:
            output=vstack((output, newtable))

    if save_output:
        output.write(os.path.join(output_root, 'gp_'+obj), format='fits',overwrite=True)

    return output, gpdict, used_kernels_obj


def get_GP_redChi2(iniTheta, kernel_name, obj_obs, gp_times):
    """
    Fit a Gaussian process curve at specific evenly spaced points along a light curve.

    Parameters
    ----------
    initheta : list-like
        Initial values for theta parameters. These should be roughly the scale length in the y & x directions.
    kernel_name : str
        The kernel to fit the data. It can be ExpSquared or ExpSquared+ExpSine2
    obj_obs : pandas.core.frame.DataFrame
        Time, flux and flux error of the data (specific filter of an object)
    gp_times : 
        Times to evaluate the Gaussian Process at

    Returns
    -------
    gp_obs : pandas.core.frame.DataFrame
        Time, flux and flux error of the fitted Gaussian Process
    redChi2: float
        Reduced chi^2 of that particular object and filter
    gp : george.gp.GP
        The GP instance that was used to fit the object
    """
    obj_times = obj_obs.mjd
    obj_flux = obj_obs.flux
    obj_flux_err = obj_obs.flux_err

    def neg_log_like(p): # Objective function: negative log-likelihood
        gp.set_parameter_vector(p)
        loglike = gp.log_likelihood(obj_flux, quiet=True)
        return -loglike if np.isfinite(loglike) else 1e25

    def grad_neg_log_like(p): # Gradient of the objective function.
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(obj_flux, quiet=True)

    kernel = get_kernel(kernel_name, iniTheta)

    gp = george.GP(kernel)
    gp.compute(obj_times, obj_flux_err)
    results = op.minimize(neg_log_like, gp.get_parameter_vector(), jac=grad_neg_log_like,
                          method="L-BFGS-B", tol=1e-6)
    
    if np.sum(np.isnan(results.x)) != 0 : # the minimiser reaches a local minimum
        iniTheta[4] = iniTheta[4]+.1 # change a bit initial conditions so we don't go to that minima
        kernel = get_kernel(kernel_name, iniTheta)
        gp = george.GP(kernel)
        gp.compute(obj_times, obj_flux_err)
        results = op.minimize(neg_log_like, gp.get_parameter_vector(), jac=grad_neg_log_like,
                          method="L-BFGS-B", tol=1e-6)

    gp.set_parameter_vector(results.x)
    gp_mean, gp_cov = gp.predict(obj_flux, gp_times)
    gp_obs          = pd.DataFrame(columns=['mjd'], data=gp_times)
    gp_obs['flux']  = gp_mean
    if np.sum(np.diag(gp_cov)<0) == 0:
        gp_obs['flux_err'] = np.sqrt(np.diag(gp_cov))
        redChi2 = reducedChi2(obj_obs, gp_obs)
    else:
        gp_obs['flux_err'] = 66666
        redChi2            = 666666666 # do not choose this kernel
    return gp_obs, redChi2, gp


def get_kernel(kernel_name, iniTheta):
    """
    Fit the chosen kernel with the given initial conditions

    Parameters
    ----------
    initheta : list-like
        Initial values for theta parameters. These should be roughly the scale length in the y & x directions.
    kernel_name : str
        The kernel to fit the data. It can be ExpSquared or ExpSquared+ExpSine2

    Returns
    -------
    kernel : george.kernels
        The kernel instance to be used in the Gaussian Process
    """
    kExpSquared = iniTheta[0]*george.kernels.ExpSquaredKernel(metric=iniTheta[1])
    kExpSine2   = iniTheta[4]*george.kernels.ExpSine2Kernel(gamma=iniTheta[5],log_period=iniTheta[6])
    if kernel_name == 'ExpSquared':
        kernel = kExpSquared
    elif kernel_name == 'ExpSquared+ExpSine2':
        kernel = kExpSquared + kExpSine2
    return kernel


def reducedChi2(obj_obs, gp_obs):
    """
    Returns the reduced chi^2 calculated comparing the Gaussian Process and the object 

    Parameters
    ----------
    obj_obs : pandas.core.frame.DataFrame
        Time, flux and flux error of the data (specific filter of an object)
    gp_obs : pandas.core.frame.DataFrame
        Time, flux and flux error of the fitted Gaussian Process

    Returns
    -------
    redChi2 : float
        Reduced chi^2 of that particular object and filter
    """
    gp_times, gp_flux = gp_obs.mjd, gp_obs.flux
    obj_times, obj_flux = obj_obs.mjd, obj_obs.flux

    interpolate_flux = interpolate.interp1d(gp_times, gp_flux, kind='cubic')
    gp_flux_obj_times  = np.array(interpolate_flux(obj_times))
    chi2            = np.sum( ((obj_flux-gp_flux_obj_times)/obj_obs.flux_err)**2 )
    redChi2         = chi2 / len(obj_times)
    return redChi2
