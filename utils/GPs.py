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


def extract_GP(d, ngp, t_min, t_max, initheta, output_root, nprocesses, gp_algo='george'):
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
        Output directory. If None the GPs are not saved.
    nprocesses : int, optional
        Number of processors to use for parallelisation (shared memory only)
    gp_algo : str
        which gp package is used for the Gaussian Process Regression, GaPP or george
    """
    print ('Performing Gaussian process regression')
    initial_time = time.time()
    
    # Check for parallelisation
    if nprocesses == 1: # non parallelizing
        for i in range(len(d.object_names)):
            obj = d.object_names[i]
            try:
                out = _GP(obj, d=d,ngp=ngp, t_min=t_min, t_max=t_max, initheta=initheta, output_root=output_root, gp_algo=gp_algo)
                d.models[obj] = out
            except ValueError:
                print('Object {} has fallen over!'.format(obj))
    else: # parallelizing
        p = Pool(nprocesses, maxtasksperchild=10)

        #Pool and map can only really work with single-valued functions
        partial_GP = partial(_GP, d=d, ngp=ngp, t_min=t_min, t_max=t_max, initheta=initheta, output_root=output_root, gp_algo=gp_algo, return_gp=True)

        out = p.map(partial_GP, d.object_names, chunksize=10)
        p.close()
        gp = {}

        out = np.reshape(out,(len(d.object_names),2))
        for i in range(len(out)):
            obj = d.object_names[i]
            d.models[obj] = out[i,0]
            gp[obj] = out[i,1]

    print ('Time taken for Gaussian process regression', time.time()-initial_time)


def _GP(obj, d, ngp, t_min, t_max, initheta, output_root, gp_algo='george', return_gp=False):
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
    t_min : float
        Minimim time to evaluate at
    t_max : float
        Maximum time to evaluate at
    initheta : list-like
        Initial values for theta parameters. These should be roughly the scale length in the y & x directions.
    output_root : str
        Output directory. If None the GPs are not saved.
    gp_algo : str
        which gp package is used for the Gaussian Process Regression, GaPP or george
    return_gp : bool, optional
        do we return the mean, or the mean and a dict of the fitted GPs

    Returns
    -------
    astropy.table.Table
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
    for fil in d.filter_set:
        if fil in filters:
            obs_times    = lc['mjd'][lc['filter']==fil]  # x
            obs_flux     = lc['flux'][lc['filter']==fil] # y
            obs_flux_err = lc['flux_error'][lc['filter']==fil] # y_err

            if gp_algo=='gapp':
                g          = dgp.DGaussianProcess(obs_times, obs_flux, obs_flux_err, cXstar=(t_min, t_max, ngp))
                rec, theta = g.gp(theta=initheta)
            elif gp_algo=='george':
                metric  = initheta[1]**2
                gp_obs, redChi2, g = get_GP_redChi2(iniTheta=np.array([initheta[0]**2, metric, 2., 4., 4., 6., 6.]), kernel_name='ExpSquared',
                                                    obs_times=obs_times,obs_flux=obs_flux,obs_flux_err=obs_flux_err, gp_times=gp_times)
                if redChi2 > 2: # bad gp
                    gp_obs, g, redChi2, chosen_kernel = get_hier_GP(gp_obs, redChi2, g, obs_times,obs_flux,obs_flux_err, gp_times)
                else:
                    chosen_kernel = 'ExpSquared 0'

                print(obj, fil, chosen_kernel+' \t\t redX2 = {:09.2f}'.format(redChi2))
                mu,cov = gp_obs.flux.values, gp_obs.flux_err.values
                std    = np.sqrt(np.diag(cov))
                rec    = np.column_stack((gp_times, mu, std))
            gpdict[fil] = g
        else:
            rec=np.zeros([ngp, 3])
        newtable=Table([rec[:, 0], rec[:, 1], rec[:, 2], [fil]*ngp], names=['mjd', 'flux', 'flux_error', 'filter'])
        if len(output)==0:
            output=newtable
        else:
            output=vstack((output, newtable))
    if output_root != None:
        output.write(os.path.join(output_root, 'gp_'+obj), format='fits',overwrite=True)
    if return_gp:
        return output,gpdict
    else:
        return output

def get_GP_redChi2(iniTheta, kernel_name, obs_times,obs_flux,obs_flux_err, gp_times):
    def neg_log_like(p): # Objective function: negative log-likelihood
        gp.set_parameter_vector(p)
        loglike = gp.log_likelihood(obs_flux, quiet=True)
        return -loglike if np.isfinite(loglike) else 1e25

    def grad_neg_log_like(p): # Gradient of the objective function.
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(obs_flux, quiet=True)

    kernel = get_kernel(kernel_name, iniTheta)

    gp = george.GP(kernel)
    gp.compute(obs_times, obs_flux_err)
    results = op.minimize(neg_log_like, gp.get_parameter_vector(), jac=grad_neg_log_like,
                          method="L-BFGS-B", tol=1e-6)
    print(results.x)
    if np.sum(np.isnan(results.x)) != 0 :
        print('iniTheta before = '+str(iniTheta))
        print(iniTheta[4])
        iniTheta[4] = iniTheta[4]+.1 # change a bit initial conditions so we don't go to a minima
        print('iniTheta = '+str(iniTheta))
        kernel = get_kernel(kernel_name, iniTheta)
        results = op.minimize(neg_log_like, gp.get_parameter_vector(), jac=grad_neg_log_like,
                          method="L-BFGS-B", tol=1e-6)
        print('Changing')
        print(results.x)

    gp.set_parameter_vector(results.x)
    gp_mean, gp_cov = gp.predict(obs_flux, gp_times)
    gp_obs          = pd.DataFrame(columns=['mjd'], data=gp_times)
    gp_obs['flux']  = gp_mean
    if np.sum(np.diag(gp_cov)<0) == 0:
        gp_obs['flux_err'] = np.sqrt(np.diag(gp_cov))
        redChi2 = reducedChi2(obs_times,obs_flux,obs_flux_err, gp_obs)
    else:
        gp_obs['flux_err'] = 66666
        redChi2            = 666666666 # do not choose this kernel
    return gp_obs, redChi2, gp

def get_kernel(kernel_name, iniTheta):
    kExpSquared = iniTheta[0]*george.kernels.ExpSquaredKernel(metric=iniTheta[1])
    kExpSine2   = iniTheta[4]*george.kernels.ExpSine2Kernel(gamma=iniTheta[5],log_period=iniTheta[6])
    if kernel_name == 'ExpSquared':
        kernel = kExpSquared
    elif kernel_name == 'ExpSquared+ExpSine2':
        kernel = kExpSquared + kExpSine2
    return kernel


def reducedChi2(obs_times,obs_flux,obs_flux_err, gp_obs):
    gp_times, gp_flux = gp_obs.mjd, gp_obs.flux

    interpolate_flux = interpolate.interp1d(gp_times, gp_flux, kind='cubic')
    gp_flux_obs_times  = np.array(interpolate_flux(obs_times))
    chi2            = np.sum( ((obs_flux-gp_flux_obs_times)/obs_flux_err)**2 )
    redChi2         = chi2 / len(obs_times)
    return redChi2


def get_hier_GP(gp_obs, redChi2, gp, obs_times,obs_flux,obs_flux_err, gp_times):
    gp_obs_1, redChi2_1, gp_1 = get_GP_redChi2(iniTheta=np.array([400., 200., 2., 4., 4., 6., 6.]), kernel_name='ExpSquared',
                                         obs_times=obs_times,obs_flux=obs_flux,obs_flux_err=obs_flux_err, gp_times=gp_times)
    if redChi2_1 < 2: # good gp
        return gp_obs_1, gp_1, redChi2_1, 'ExpSquared 1'
    else:             # bad gp
        gp_obs_all   = [gp_obs, gp_obs_1]
        redChi2_all = [redChi2, redChi2_1]
        gp_all      = [gp, gp_1]
        gp_obs_2, redChi2_2, gp_2 = get_GP_redChi2(iniTheta=np.array([400., 20., 2., 4., 4., 6., 6.]), kernel_name='ExpSquared',
                                         obs_times=obs_times,obs_flux=obs_flux,obs_flux_err=obs_flux_err, gp_times=gp_times)
        if redChi2_2 < 2: # good gp
            gp_obs, redChi2, gp, chosen_kernel =  gp_obs_2, redChi2_2, gp_2, 'ExpSquared 2'
        else:             # bad gp
            gp_obs_all.append(gp_obs_2)
            redChi2_all.append(redChi2_2)
            gp_all.append(gp_2)
            gp_obs_3, redChi2_3, gp_3 = get_GP_redChi2(iniTheta=np.array([19., 9., 2., 4., 4., 6., 6.]),
                                kernel_name='ExpSquared+ExpSine2', obs_times=obs_times,obs_flux=obs_flux,obs_flux_err=obs_flux_err, gp_times=gp_times)
            if redChi2_3 < 2: # good gp
                gp_obs, redChi2, gp, chosen_kernel =  gp_obs_3, redChi2_3, gp_3, 'ExpSquared+ExpSine2'
            else:             # bad gp
                gp_obs_all.append(gp_obs_3)
                redChi2_all.append(redChi2_3)
                gp_all.append(gp_3)
                kernels = ['bad ExpSquared 0', 'bad ExpSquared 1', 'bad ExpSquared 2', 'bad ExpSquared+ExpSine2']
                indMinRedChi2 = np.argmin(redChi2_all)
                gp_obs, redChi2, gp = gp_obs_all[indMinRedChi2], redChi2_all[indMinRedChi2], gp_all[indMinRedChi2]
                try:
                    chosen_kernel = kernels[indMinRedChi2]
                except:
                    print('(-_-) ... '+str(indMinRedChi2)+' '+str(kernels))
    return gp_obs, gp, redChi2, chosen_kernel