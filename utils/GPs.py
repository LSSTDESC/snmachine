"""
Module for extracting and saving GPs
"""

from __future__ import division, print_function
import numpy as np
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
import pickle
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
        print('non parallel')
        for i in range(len(d.object_names)):
            obj = d.object_names[i]
            try:
                out = _GP(obj, d=d,ngp=ngp, xmin=xmin, xmax=xmax, initheta=initheta, output_root=output_root, gp_algo=gp_algo)
            except ValueError:
                print('Object %s has fallen over!'%obj)
            d.models[obj] = out
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


def _GP(obj, d, ngp, xmin, xmax, initheta, output_root, gp_algo='george',return_gp=False):
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
    xstar   = np.linspace(xmin, xmax, ngp) # times to plot the GP
    gpTimes = xstar # more descriptive name

    #Store the output in another astropy table
    output = []
    gpdict = {}
    for fil in d.filter_set:
        if fil in filters:
            x   = lc['mjd'][lc['filter']==fil]
            y   = lc['flux'][lc['filter']==fil]
            err = lc['flux_error'][lc['filter']==fil]
            obsTimes, obsFlux, obsFluxErr = x, y, err # more descriptive names
            if gp_algo=='gapp':
                g          = dgp.DGaussianProcess(x, y, err, cXstar=(xmin, xmax, ngp))
                rec, theta = g.gp(theta=initheta)
            elif gp_algo=='george':
                metric  = initheta[1]**2
                gpObs, redChi2, g = getGPChi2(iniTheta=np.array([initheta[0]**2, metric, 2, 4, 4, 6, 6]), kernel='ExpSquared',
                                               x=x,y=y,err=err, gpTimes=gpTimes)
                if redChi2 > 2: # bad gp
                    gpObs, g, redChi2, chosenKernel = getHierGP(gpObs, redChi2, g, x,y,err, gpTimes)
                else:
                    chosenKernel = 'ExpSquared 0'

                print(obj, fil, chosenKernel+' \t\t redX2 = {:09.2f}'.format(redChi2))
                mu,cov = gpObs.flux.values, gpObs.flux_err.values
                std    = np.sqrt(np.diag(cov))
                rec    = np.column_stack((xstar,mu,std))
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


def getGPChi2(iniTheta, kernel, x,y,err, gpTimes):
    def negLoglike(p): # Objective function: negative log-likelihood
        gp.set_parameter_vector(p)
        loglike = gp.log_likelihood(y, quiet=True)
        return -loglike if np.isfinite(loglike) else 1e25

    def gradNegLoglike(p): # Gradient of the objective function.
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y, quiet=True)

    kExpSquared = iniTheta[0]*george.kernels.ExpSquaredKernel(metric=iniTheta[1])
    kExpSine2   = iniTheta[4]*george.kernels.ExpSine2Kernel(gamma=iniTheta[5],log_period=iniTheta[6])
    if kernel == 'ExpSquared':
        kernel = kExpSquared
    elif kernel == 'ExpSquared+ExpSine2':
        kernel = kExpSquared + kExpSine2

    gp      = george.GP(kernel)
    gp.compute(x, err)
    results = op.minimize(negLoglike, gp.get_parameter_vector(), jac=gradNegLoglike,
                          method="L-BFGS-B", tol=1e-6)
    gp.set_parameter_vector(results.x)
    gpMean, gpCov     = gp.predict(y, gpTimes)
    gpObs             = pd.DataFrame(columns=['mjd'], data=gpTimes)
    gpObs['flux']     = gpMean
    if np.sum(np.diag(gpCov)<0) == 0:
        gpObs['flux_err'] = np.sqrt(np.diag(gpCov))
        redChi2 = reducedChi2(x,y,err, gpObs)
    else:
        gpObs['flux_err'] = 66666
        redChi2           = 666666666 # do not choose this kernel
    return gpObs, redChi2, gp


def reducedChi2(x,y,err, gpObs):
    obsTimes            = x
    obsFlux, obsFluxErr = y, err
    gpTimes, gpFlux     = gpObs.mjd, gpObs.flux

    interpolateFlux = interpolate.interp1d(gpTimes, gpFlux, kind='cubic')
    gpFluxObsTimes  = np.array(interpolateFlux(obsTimes))
    chi2            = np.sum( ((obsFlux-gpFluxObsTimes)/obsFluxErr)**2 )
    redChi2         = chi2 / len(x) # reduced X^2
    return redChi2


def getHierGP(gpObs, redChi2, gp, x,y,err, gpTimes):
    gpObs_1, redChi2_1, gp_1 = getGPChi2(iniTheta=np.array([400, 200, 2, 4, 4, 6, 6]), kernel='ExpSquared',
                                         x=x,y=y,err=err, gpTimes=gpTimes)
    if redChi2_1 < 2: # good gp
        return gpObs_1, gp_1, redChi2_1, 'ExpSquared 1'
    else:             # bad gp
        gpObsAll   = [gpObs, gpObs_1]
        redChi2All = [redChi2, redChi2_1]
        gpAll      = [gp, gp_1]
        gpObs_2, redChi2_2, gp_2 = getGPChi2(iniTheta=np.array([400, 20, 2, 4, 4, 6, 6]), kernel='ExpSquared',
                                         x=x,y=y,err=err, gpTimes=gpTimes)
        if redChi2_2 < 2: # good gp
            gpObs, redChi2, gp, chosenKernel =  gpObs_2, redChi2_2, gp_2, 'ExpSquared 2'
        else:             # bad gp
            gpObsAll.append(gpObs_2)
            redChi2All.append(redChi2_2)
            gpAll.append(gp_2)
            gpObs_3, redChi2_3, gp_3 = getGPChi2(iniTheta=np.array([19, 9, 2, 4, 4, 6, 6]),
                                                 kernel='ExpSquared+ExpSine2', x=x,y=y,err=err, gpTimes=gpTimes)
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