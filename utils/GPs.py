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


def extract_GP(d, ngp, xmin, xmax, initheta, output_root, nprocesses, gp_algo='george'):
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
                out = _GP(obj, d=d,ngp=ngp, xmin=xmin, xmax=xmax, initheta=initheta, output_root=output_root, gp_algo=gp_algo)
                d.models[obj] = out
            except ValueError:
                print('Object {} has fallen over!'.format(obj))
    else: # parallelizing
        p = Pool(nprocesses, maxtasksperchild=10)

        #Pool and map can only really work with single-valued functions
        partial_GP = partial(_GP, d=d, ngp=ngp, xmin=xmin, xmax=xmax, initheta=initheta, output_root=output_root, gp_algo=gp_algo, return_gp=True)

        out = p.map(partial_GP, d.object_names, chunksize=10)
        p.close()
        gp = {}

        out = np.reshape(out,(len(d.object_names),2))
        for i in range(len(out)):
            obj = d.object_names[i]
            d.models[obj] = out[i,0]
            gp[obj] = out[i,1]

    print ('Time taken for Gaussian process regression', time.time()-initial_time)


def _GP(obj, d, ngp, xmin, xmax, initheta, output_root, gp_algo='george', return_gp=False):
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
    gpTimes = np.linspace(xmin, xmax, ngp) # times to plot the GP

    # Store the output in another astropy table
    output = []
    gpdict = {}
    for fil in d.filter_set:
        if fil in filters:
            x   = lc['mjd'][lc['filter']==fil]
            y   = lc['flux'][lc['filter']==fil]
            err = lc['flux_error'][lc['filter']==fil]
            obs_times, obsFlux, obsFluxErr = x, y, err # more descriptive names

            if gp_algo=='gapp':
                g          = dgp.DGaussianProcess(x, y, err, cXstar=(xmin, xmax, ngp))
                rec, theta = g.gp(theta=initheta)
            elif gp_algo=='george':
                metric  = initheta[1]**2
                gpObs, redChi2, g = get_GP_redChi2(iniTheta=np.array([initheta[0]**2, metric, 2, 4, 4, 6, 6]), kernel='ExpSquared',
                                                    obs_times=x,obs_flux=y,obs_flux_err=err, gp_times=gpTimes)
                if redChi2 > 2: # bad gp
                    gpObs, g, redChi2, chosenKernel = get_hier_GP(gpObs, redChi2, g, x,y,err, gpTimes)
                else:
                    chosenKernel = 'ExpSquared 0'

                print(obj, fil, chosenKernel+' \t\t redX2 = {:09.2f}'.format(redChi2))
                mu,cov = gpObs.flux.values, gpObs.flux_err.values
                std    = np.sqrt(np.diag(cov))
                rec    = np.column_stack((gpTimes, mu, std))
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


def get_GP_redChi2(iniTheta, kernel, obs_times,obs_flux,obs_flux_err, gp_times):
    def neg_log_like(p): # Objective function: negative log-likelihood
        gp.set_parameter_vector(p)
        loglike = gp.log_likelihood(obs_flux, quiet=True)
        return -loglike if np.isfinite(loglike) else 1e25

    def grad_neg_log_like(p): # Gradient of the objective function.
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(obs_flux, quiet=True)

    kExpSquared = iniTheta[0]*george.kernels.ExpSquaredKernel(metric=iniTheta[1])
    kExpSine2   = iniTheta[4]*george.kernels.ExpSine2Kernel(gamma=iniTheta[5],log_period=iniTheta[6])
    if kernel == 'ExpSquared':
        kernel = kExpSquared
    elif kernel == 'ExpSquared+ExpSine2':
        kernel = kExpSquared + kExpSine2

    gp = george.GP(kernel)
    gp.compute(obs_times, obs_flux_err)
    results = op.minimize(neg_log_like, gp.get_parameter_vector(), jac=grad_neg_log_like,
                          method="L-BFGS-B", tol=1e-6)
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


def reducedChi2(obs_times,obs_flux,obs_flux_err, gp_obs):
    gp_times, gp_flux = gp_obs.mjd, gp_obs.flux

    interpolate_flux = interpolate.interp1d(gp_times, gp_flux, kind='cubic')
    gp_flux_obs_times  = np.array(interpolate_flux(obs_times))
    chi2            = np.sum( ((obs_flux-gp_flux_obs_times)/obs_flux_err)**2 )
    redChi2         = chi2 / len(obs_times)
    return redChi2


def get_hier_GP(gpObs, redChi2, gp, x,y,err, gpTimes):
    gpObs_1, redChi2_1, gp_1 = get_GP_redChi2(iniTheta=np.array([400, 200, 2, 4, 4, 6, 6]), kernel='ExpSquared',
                                         obs_times=x,obs_flux=y,obs_flux_err=err, gp_times=gpTimes)
    if redChi2_1 < 2: # good gp
        return gpObs_1, gp_1, redChi2_1, 'ExpSquared 1'
    else:             # bad gp
        gpObsAll   = [gpObs, gpObs_1]
        redChi2All = [redChi2, redChi2_1]
        gpAll      = [gp, gp_1]
        gpObs_2, redChi2_2, gp_2 = get_GP_redChi2(iniTheta=np.array([400, 20, 2, 4, 4, 6, 6]), kernel='ExpSquared',
                                         obs_times=x,obs_flux=y,obs_flux_err=err, gp_times=gpTimes)
        if redChi2_2 < 2: # good gp
            gpObs, redChi2, gp, chosenKernel =  gpObs_2, redChi2_2, gp_2, 'ExpSquared 2'
        else:             # bad gp
            gpObsAll.append(gpObs_2)
            redChi2All.append(redChi2_2)
            gpAll.append(gp_2)
            gpObs_3, redChi2_3, gp_3 = get_GP_redChi2(iniTheta=np.array([19, 9, 2, 4, 4, 6, 6]),
                                                 kernel='ExpSquared+ExpSine2', obs_times=x,obs_flux=y,obs_flux_err=err, gp_times=gpTimes)
            if redChi2_3 < 2: # good gp
                gpObs, redChi2, gp, chosenKernel =  gpObs_3, redChi2_3, gp_3, 'ExpSquared+ExpSine2'
            else:             # bad gp
                gpObsAll.append(gpObs_3)
                redChi2All.append(redChi2_3)
                gpAll.append(gp_3)
                kernels = ['bad ExpSquared 0', 'bad ExpSquared 1', 'bad ExpSquared 2', 'bad ExpSquared+ExpSine2']
                indMinRedChi2 = np.argmin(redChi2All)
                gpObs, redChi2, gp = gpObsAll[indMinRedChi2], redChi2All[indMinRedChi2], gpAll[indMinRedChi2]
                try:
                    chosenKernel = kernels[indMinRedChi2]
                except:
                    print('(-_-) ... '+str(indMinRedChi2)+' '+str(kernels))
    return gpObs, gp, redChi2, chosenKernel