"""
Module for feature extraction on supernova light curves.
"""

__all__ = []
# 'get_MAP', 'Features', 'TemplateFeatures', 'ParametricFeatures',
#           'WaveletFeatures']

import os
import pickle
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pywt
import sncosmo

from . import parametric_models
from astropy.table import Table, vstack, join
from functools import partial
from iminuit import Minuit
from multiprocessing import Pool
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from snmachine import gps, chisq

try:
    import pymultinest
    has_multinest = True
    print('Module pymultinest found')
except (ImportError, SystemExit) as exception:
    print(exception)
    if str(exception) == "No module named 'pymultinest'":
        errmsg = """
                PyMultinest not found. If you would like to use, please install
                Mulitnest with 'sh install/multinest_install.sh; source
                install/setup.sh'
                """
        print(errmsg)
        has_multinest = False
    else:
        errmsg = """
                Multinest installed but not linked.
                Please ensure $LD_LIBRARY_PATH set correctly with:

                    source install/setup.sh
                """
        raise OSError(errmsg) from exception

try:
    import emcee
    has_emcee = True
except ImportError:
    has_emcee = False

# Central passbands wavelengths
pb_wavelengths = {"lsstu": 3685., "lsstg": 4802., "lsstr": 6231.,
                  "lssti": 7542., "lsstz": 8690., "lssty": 9736.}
wavelengths_pb = {v: k for k, v in pb_wavelengths.items()}  # inverted map


def _run_leastsq(obj, d, model, n_attempts, seed=-1):
    """Minimises the chi2 on all the filter bands of a given light curve,
    fitting the model to each one and extracting the best fitting parameters

    Parameters
    ----------
    obj : str
        Object name
    d : Dataset
        Dataset object
    model : parametric model object
        Parametric model
    n_attempts : int
        We run this multiple times to try to avoid local minima and take the
        best fitting values.
    seed : int, optional
        Here you can set the random seed for the minimisation to a specific
        value. If -1, the default value will be used instead.

    Returns
    -------
    astropy.table.Table
        Table of best fitting parameters for all filters

    """
    lc = d.data[obj]
    filts = np.unique(lc['filter'])

    # How many times to keep trying to fit each object to obtain a good fit
    # (defined as reduced chi2 of less than 2)
    if n_attempts < 1:
        n_attempts = 1

    labels = ['Object']
    for f in d.filter_set:
        pams = model.param_names
        for p in pams:
            labels.append(f+'-'+p)
    output = Table(names=labels, dtype=['U32']+['f']*(len(labels)-1))

    if seed != -1:
        np.random.seed(seed)

    row = [obj]
    for f in d.filter_set:
        if f in filts:
            x = np.array(lc['mjd'][lc['filter'] == f])
            y = np.array(lc['flux'][lc['filter'] == f])
            err = np.array(lc['flux_error'][lc['filter'] == f])

            def mini_func(*params):
                # This function changes with each object so is necessary to
                # redefine (since you can't pass arguments through Minuit)
                for i in range(len(params)):
                    p = params[i]
                    is_outside_limits = (
                        p < model.limits[model.param_names[i]][0] or
                        p > model.limits[model.param_names[i]][1])
                    if is_outside_limits:
                        return np.inf
                ynew = model.evaluate(x, params)
                chi2 = np.sum((y-ynew)*(y-ynew)/err/err)
                return chi2

            # For the sake of speed, we stop as soon as we get to reduced
            # chi2<2, failing that we use the best fit found so far
            fmin = np.inf
            min_params = []

            for i in range(n_attempts):
                if i == 0:
                    # Try the default starting point
                    input_args = model.initial.copy()
                else:
                    # Pick a new, random starting point
                    input_args = {}
                    for p in model.param_names:
                        val = np.random.uniform(model.limits[p][0],
                                                model.limits[p][1])
                        input_args[p] = val

                m = Minuit(mini_func,
                           name=model.param_names, **input_args)
                for p in model.param_names:
                    m.limits[p] = model.limits[p]

                m.migrad()
                parm = []
                for p in model.param_names:
                    parm.append(m.values[p])

                rchi2 = m.fval/len(x)
                if rchi2 < 2:
                    fmin = m.fval
                    min_params = parm
                    break
                elif m.fval < fmin:
                    fmin = m.fval
                    min_params = parm

            outfl = open('out', 'a')
            outfl.write('%s\t%s\t%f\t%d\n' % (obj, f, fmin/len(x), i))
            outfl.close()
            row += min_params
        else:
            row += [0]*len(model.param_names)  # Fill missing values with zeros
    output.add_row(row)
    return output


def _run_multinest(obj, d, model, chain_directory,  nlp, convert_to_binary,
                   n_iter, restart=False, seed=-1):
    """Runs multinest on all the filter bands of a given light curve, fitting
    the model to each one and extracting the best fitting parameters.

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
    restart : bool, optional
        Whether to restart from existing chain files.
    seed : int, optional
        Here you can set the random seed for the minimisation to a specific
        value. If -1, the default value will be used instead.


    Returns
    -------
    astropy.table.Table
        Table of best fitting parameters and their errors for all filters
    """

    try:
        def prior_multinest(cube, ndim, nparams):
            """Prior function specifically for multinest.

            This would be called for one filter, for one object.

            Parameters
            ----------
            cube
                A ctypes pointer to the parameter cube (actually just the
                current parameter values).
            ndim
                Number of dimensions
            nparams
                Number of parameters. Usually the same as ndim unless you have
                unsampled (e.g. calculated) parameters. These are assumed to
                be the first (ndim-nparams) parameters.
            """
            up = model.upper_limit
            low = model.lower_limit

            for i in range(nparams):
                cube[i] = cube[i]*(up[i]-low[i])+low[i]
            return cube

        lc = d.data[obj]
        filts = np.unique(lc['filter'])

        n_params = len(model.param_names)

        labels = ['Object']
        for f in d.filter_set:
            pams = model.param_names
            for p in pams:
                labels.append(f+'-'+p)

        output = Table(names=labels, dtype=['U32']+['f']*(len(labels)-1))

        row = [obj]
        for f in d.filter_set:
            t1 = time.time()
            if f in filts:
                # Current values for x,y and err are set each time multinest
                # is called
                x = lc['mjd'][lc['filter'] == f]
                y = lc['flux'][lc['filter'] == f]
                err = lc['flux_error'][lc['filter'] == f]

                def loglike_multinest(cube, ndim, nparams):
                    """Loglikelihood function specifically for multinest. This
                    would be called for one filter, for one object.

                    Parameters
                    ----------
                    cube
                        A ctypes pointer to the parameter cube (actually just
                        the current parameter values).
                    ndim
                        Number of dimensions
                    nparams
                        Number of parameters. Usually the same as ndim unless
                        you have unsampled (e.g. calculated) parameters. These
                        are assumed to be the first (ndim-nparams) parameters.
                    """
                    params = np.zeros(nparams)
                    # This is the only obvious way to convert a ctypes pointer
                    # to a numpy array
                    for i in range(nparams):
                        params[i] = cube[i]
                    # params=[ 26.97634888, 45.13123322, 2.59183478,
                    #          0.12057552, 7.65392637]
                    ynew = model.evaluate(x, params)

                    chi2 = np.sum(((y-ynew)*(y-ynew))/err/err)
                    return -chi2/2.

                chain_name = os.path.join(chain_directory,
                                          '%s-%s-%s-' % (obj.split('.')[0], f,
                                                         model.model_name))

                if not restart or not os.path.exists(chain_name+'stats.dat'):
                    # Gives the ability to restart from existing chains if
                    # they exist
                    pymultinest.run(loglike_multinest, prior_multinest,
                                    n_params, importance_nested_sampling=False,
                                    init_MPI=False, resume=False, seed=seed,
                                    verbose=False, n_live_points=nlp,
                                    sampling_efficiency='parameter',
                                    outputfiles_basename=chain_name,
                                    multimodal=False, max_iter=n_iter)

                best_params = get_MAP(chain_name)

                if convert_to_binary and not restart:
                    # These are the files we can convert
                    ext = ['ev.dat', 'phys_live.points', 'live.points', '.txt',
                           'post_equal_weights.dat']
                    for e in ext:

                        infile = os.path.join(chain_directory,
                                              '{}-{}-{}-{}'.format(
                                                  obj.split('.')[0], f,
                                                  model.model_name, e))
                        outfile = infile+'.npy'
                        try:
                            x = np.loadtxt(infile)
                            np.save(outfile, x)
                            os.system('rm %s' % infile)
                        except:
                            print('ERROR reading file', infile)
                            print('File unconverted')

                row += best_params
            else:
                # I'm not sure if it makes the most sense to fill in missing
                # values with zeroes...
                row += [0]*len(model.param_names)
            # print 'Time for object', obj, 'filter', f,':', (time.time()-t1)
            np.savetxt(os.path.join(chain_directory,
                       '%s-%s-%s-.time' % (obj.split('.')[0], f,
                                           model.model_name)),
                       [time.time()-t1])

        output.add_row(row)
        return output

    except:
        # Sometimes things just break
        print('ERROR in', obj)
        print(sys.exc_info()[0])
        print(sys.exc_info()[1])
        traceback.print_tb(sys.exc_info()[2])
        return None


def _run_leastsq_templates(obj, d, model_name, use_redshift, bounds, seed=-1):
    """Fit template-based supernova models using least squares.

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
    seed : int, optional
        Here you can set the random seed for the minimisation to a specific
        value. If -1, the default value will be used instead.

    Returns
    -------
    astropy.table.Table
        Table of best fitting parameters
    """

    lc = d.data[obj]

    if model_name == 'mlcs2k2':
        dust = sncosmo.CCM89Dust()
        model = sncosmo.Model(model_name, effects=[dust],
                              effect_names=['host'], effect_frames=['rest'])
    else:
        model = sncosmo.Model(model_name)

    labels = ['Object'] + model.param_names
    output = Table(names=labels, dtype=['U32']+['f']*(len(model.param_names)))

    row = [obj]
    try:
        if use_redshift:
            model.set(z=lc.meta['z'])
            prms = model.param_names
            prms = prms[1:]
            bnds = bounds.copy()
            bnds.pop('z', None)
            res, fitted_model = sncosmo.fit_lc(lc, model, vparam_names=prms,
                                               bounds=bnds, minsnr=0)
        else:
            res, fitted_model = sncosmo.fit_lc(lc, model,
                                               vparam_names=model.param_names,
                                               bounds=bounds, minsnr=0)
        best = res['parameters']
        best = best.tolist()
        row += best
    except RuntimeError:  # the event failed
        output = Table(names=labels,
                       dtype=['U32']+['f']*(len(model.param_names)))
        row += [lc.meta['z']]
        row += (len(model.param_names) - 1) * [None]
    output.add_row(row)
    return output


def _run_multinest_templates(obj, d, model_name, bounds, chain_directory='./',
                             nlp=1000, convert_to_binary=True, seed=-1,
                             use_redshift=False, short_name='', restart=False):
    """Fit template-based supernova models using multinest.

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
        A shorter name for the chains (to overcome Multinest's character
        limitation)
    restart : bool
        Whether or not to restart from existing chains

    Returns
    -------
    array-like
        List of best-fitting parameters
    """
    try:
        def prior_multinest(cube, ndim, nparams):
            """Prior function specifically for multinest. This would be called
            for one filter, for one object.

            Parameters
            ----------
            cube
                A ctypes pointer to the parameter cube (actually just the
                current parameter values).
            ndim
                Number of dimensions
            nparams
                Number of parameters. Usually the same as ndim unless you have
                unsampled (e.g. calculated) parameters. These are assumed to
                be the first (ndim-nparams) parameters.
            """
            params = model.param_names
            if use_redshift:
                params = params[1:]
            for i in range(ndim):
                p = params[i]
                cube[i] = cube[i]*(bounds[p][1]-bounds[p][0])+bounds[p][0]
            return cube

        def loglike_multinest(cube, ndim, nparams):
            """Loglikelihood function specifically for multinest. This would
            be called for one filter, for one object.

            Parameters
            ----------
            cube
                A ctypes pointer to the parameter cube (actually just the
                current parameter values).
            ndim
                Number of dimensions
            nparams
                Number of parameters. Usually the same as ndim unless you have
                unsampled (e.g. calculated) parameters. These are assumed to
                be the first (ndim-nparams) parameters.
            """
            dic = {}
            # This is the only obvious way to convert a ctypes pointer to a
            # numpy array
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

        lc = d.data[obj]
        filts = np.unique(lc['filter'])
        if model_name == 'mlcs2k2':
            dust = sncosmo.CCM89Dust()
            model = sncosmo.Model(model_name, effects=[dust],
                                  effect_names=['host'],
                                  effect_frames=['rest'])
        else:
            model = sncosmo.Model(model_name)

        t1 = time.time()

        # Convert the astropy table to numpy array outside the likelihood
        # function to avoid repeated calls
        X, Y, E = {}, {}, {}
        for filt in filts:
            x = lc['mjd'][lc['filter'] == filt]
            y = lc['flux'][lc['filter'] == filt]
            err = lc['flux_error'][lc['filter'] == filt]
            X[filt] = x
            Y[filt] = y
            E[filt] = err

        chain_name = os.path.join(chain_directory,
                                  '%s-%s-' % (obj.split('.')[0], short_name))

        if use_redshift:
            ndim = len(model.param_names)-1
        else:
            ndim = len(model.param_names)

        if not restart or not os.path.exists(chain_name+'stats.dat'):
            pymultinest.run(loglike_multinest, prior_multinest, ndim,
                            importance_nested_sampling=False, init_MPI=False,
                            resume=False, verbose=False, seed=seed,
                            sampling_efficiency='parameter', n_live_points=nlp,
                            outputfiles_basename=chain_name, multimodal=False)

        best_params = get_MAP(chain_name)

        if use_redshift:
            best_params = [lc.meta['z']]+best_params

        if convert_to_binary and not restart:
            # These are the files we can convert
            ext = ['ev.dat', 'phys_live.points', 'live.points', '.txt',
                   'post_equal_weights.dat']
            for e in ext:
                infile = os.path.join(
                    chain_directory, '%s-%s-%s' % (obj.split('.')[0],
                                                   short_name, e))
                outfile = infile+'.npy'
                x = np.loadtxt(infile)
                np.save(outfile, x)
                os.system('rm %s' % infile)
        np.savetxt(os.path.join(chain_directory,
                                '%s-%s-.time' % (obj.split('.')[0],
                                                 short_name)),
                   [time.time()-t1])
        return np.array(best_params)

    except:
        # Sometimes things just break
        print('ERROR in', obj)
        print(sys.exc_info()[0])
        print(sys.exc_info()[1])
        traceback.print_tb(sys.exc_info()[2])
        return None


def output_time(tm):
    """Simple function to output the time nicely formatted.

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
    print('Time taken is', out)


def get_MAP(chain_name):
    """Read maximum posterior parameters from a stats file of multinest.

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
    ind = [i for i in range(len(lines)) if 'MAP' in lines[i]][0]
    params = [float(l.split()[1]) for l in lines[ind + 2:]]
    return params


class Features:
    """Base class to define basic functionality for extracting features from
    supernova datasets. Users are not restricted to inheriting from this class,
    but any Features class must contain the functions `extract_features` and
    `fit_sn`.
    """
    def __init__(self):
        # At what point to we suggest a model has been a bad fit.
        self.p_limit = 0.05

    def extract_features(self):
        raise NotImplementedError('This method has not been implemented yet.')

    def fit_sn(self):
        raise NotImplementedError('This method has not been implemented yet.')

    @staticmethod
    def _exists_path(path_to_test):
        """Check if the inputed path exists.

        Parameters
        ----------
        path_to_test: str
            Path to test the existence.

        Raises
        ------
        ValueError
            If the provided path does not exist.
        """
        exists_path = os.path.exists(path_to_test)
        if not exists_path:
            raise ValueError('The path {} does not exist. Provide a valid path'
                             '.'.format(path_to_test))


class TemplateFeatures(Features):
    """Calls sncosmo to fit a variety of templates to the data. The number of
    features will depend on the templates chosen (e.g. salt2, nugent2p etc.)
    """
    def __init__(self, model=['Ia'], sampler='leastsq', lsst_bands=False,
                 lsst_dir='../lsst_bands/'):
        """ To initialise, provide a list of models to fit for (defaults to
        salt2 Ia templates).

        Parameters
        ----------
        model : list-like, optional
            List of models. In theory you can fit Ia and non-Ia models and use
            all those as features. So far only tested with SALT2.
        sampler : str, optional
            A choice of 'mcmc', which uses the emcee sampler, or 'nested' or
            'leastsq' (default).
        lsst_bands : bool, optional
            Whether or not the LSST bands are required. Only need for LSST
            simulations to register bands with sncosmo.
        lsst_dir : str, optional
            Directory where LSST bands are stored.
        """
        Features.__init__(self)
        if lsst_bands:
            self.registerBands(lsst_dir, prefix='approxLSST_',
                               suffix='_total.dat')
        self.model_names = model
        self.templates = {'Ia': 'salt2-extended', 'salt2': 'salt2-extended',
                          'mlcs2k2': 'mlcs2k2', 'II': 'nugent-sn2n',
                          'IIn': 'nugent-sn2n', 'IIp': 'nugent-sn2p',
                          'IIl': 'nugent-sn2l', 'Ibc': 'nugent-sn1bc',
                          'Ib': 'nugent-sn1bc', 'Ic': 'nugent-sn1bc'}
        # Short names because of limitations in Multinest
        self.short_names = {'Ia': 'salt2', 'mlcs2k2': 'mlcs'}
        if sampler == 'nested':
            try:
                import pymultinest
            except ImportError:
                print('Nested sampling selected but pymultinest is not '
                      'installed. Defaulting to least squares.')
                sampler = 'leastsq'
        elif sampler == 'mcmc':
            try:
                import emcee
            except ImportError:
                print('MCMC sampling selected but emcee is not installed. '
                      'Defaulting to least squares.')
                sampler = 'leastsq'

        self.sampler = sampler
        self.bounds = {'salt2-extended': {'z': (0.01, 1.5), 't0': (-100, 100),
                                          'x0': (-1e-3, 1e-3), 'x1': (-3, 3),
                                          'c': (-0.5, 0.5)},
                       'mlcs2k2': {'z': (0.01, 1.5), 't0': (-100, 100),
                                   'amplitude': (0, 1e-17),
                                   'delta': (-1.0, 1.8), 'hostebv': (0, 1),
                                   'hostr_v': (-7.0, 7.0)},
                       'nugent-sn2n': {'z': (0.01, 1.5)},
                       'nugent-sn2p': {'z': (0.01, 1.5)},
                       'nugent-sn2l': {'z': (0.01, 1.5)},
                       'nugent-sn1bc': {'z': (0.01, 1.5)}}

    def extract_features(self, d, save_output=False, chain_directory='chains',
                         use_redshift=False, number_processes=1, restart=False,
                         seed=-1):
        """Extract template features for a dataset.

        Parameters
        ----------
        d : Dataset object
            Dataset
        save_output : bool
            Whether or not to save the intermediate output (if Bayesian
            inference is used instead of least squares)
        chain_directory : str
            Where to save the chains
        use_redshift : bool
            Whether or not to use provided redshift when fitting objects
        number_processes : int, optional
            Number of processors to use for parallelisation (shared memory
            only).
        restart : bool
            Whether or not to restart from multinest chains

        Returns
        -------
        astropy.table.Table
            Table of fitted model parameters.
        """
        subprocess.call(['mkdir', chain_directory])
        print('Fitting templates using', self.sampler, '...')
        all_output = []
        t1 = time.time()
        for mod_name in self.model_names:
            if mod_name == 'mlcs2k2':
                dust = sncosmo.CCM89Dust()
                self.model = sncosmo.Model(self.templates[mod_name],
                                           effects=[dust],
                                           effect_names=['host'],
                                           effect_frames=['rest'])
            else:
                self.model = sncosmo.Model(self.templates[mod_name])
                print(F'MODEL-NAME: {mod_name}')
            params = ['['+mod_name+']'+pname
                      for pname in self.model.param_names]
            labels = ['Object'] + params
            output = Table(names=labels,
                           dtype=['U32'] + ['f'] * (len(labels) - 1))
            k = 0
            if number_processes < 2:
                for obj in d.object_names:
                    if k % 100 == 0:
                        print(k, 'objects fitted')
                    lc = d.data[obj]

                    if self.sampler == 'mcmc':
                        if seed != -1:
                            np.random.seed(seed)
                        res, fitted_model = sncosmo.mcmc_lc(
                            lc, self.model, self.model.param_names,
                            bounds=self.bounds[self.templates[mod_name]],
                            nwalkers=20, nsamples=1500, nburn=300)
                        chain = res.samples
                        if save_output:
                            tab = Table(chain, names=self.model.param_names)
                            path_to_save = os.path.join(
                                chain_directory,
                                obj.split('.')[0]+'_emcee_'+mod_name)
                            tab.write(path_to_save, format='ascii')
                        best = res['parameters'].flatten('F').tolist()
                    elif self.sampler == 'nested':
                        best = _run_multinest_templates(
                            obj, d, self.templates[mod_name],
                            self.bounds[self.templates[mod_name]],
                            chain_directory=chain_directory, nlp=1000,
                            convert_to_binary=False, use_redshift=use_redshift,
                            short_name=self.short_names[mod_name],
                            restart=restart, seed=seed)
                        best = best.tolist()
                    elif self.sampler == 'leastsq':
                        if use_redshift:
                            self.model.set(z=lc.meta['z'])
                            prms = self.model.param_names
                            prms = prms[1:]
                            bnds = self.bounds[self.templates[mod_name]].copy()
                            bnds.pop('z', None)
                            try:
                                res, fitted_model = sncosmo.fit_lc(
                                    lc, self.model, vparam_names=prms,
                                    bounds=bnds, minsnr=0)
                            except RuntimeError:  # the event failed to fit
                                print(f'Obj. {obj} failed.')
                        else:
                            try:
                                res, fitted_model = sncosmo.fit_lc(
                                    lc, self.model, minsnr=0,
                                    vparam_names=self.model.param_names,
                                    bounds=self.bounds[
                                        self.templates[mod_name]])
                                print(res)
                            except RuntimeError:  # the event failed to fit
                                print(f'Obj. {obj} failed.')
                                res = {}
                        best = res['parameters'].flatten('F').tolist()
                    row = [obj]+best
                    output.add_row(row)
                    k += 1
                if len(all_output) == 0:
                    all_output = output
                else:
                    all_output = join(all_output, output)
            else:
                if self.sampler == 'leastsq':
                    p = Pool(number_processes, maxtasksperchild=1)
                    partial_func = partial(_run_leastsq_templates, d=d,
                                           model_name=self.templates[mod_name],
                                           use_redshift=use_redshift,
                                           bounds=self.bounds[
                                               self.templates[mod_name]])
                    out = p.map(partial_func, d.object_names)
                    output = out[0]
                    for i in range(1, len(out)):
                        output = vstack((output, out[i]))
                    if len(all_output) == 0:
                        all_output = output
                    else:
                        all_output = vstack((all_output, output))
                elif self.sampler == 'nested':
                    p = Pool(number_processes, maxtasksperchild=1)
                    partial_func = partial(_run_multinest_templates, d=d,
                                           model_name=self.templates[mod_name],
                                           bounds=self.bounds[
                                               self.templates[mod_name]],
                                           chain_directory=chain_directory,
                                           nlp=1000, convert_to_binary=True,
                                           use_redshift=use_redshift,
                                           short_name=self.short_names[
                                               mod_name],
                                           restart=restart, seed=seed)
                    out = p.map(partial_func, d.object_names)

                    for i in range(len(out)):
                        output.add_row([d.object_names[i]]+out[i].tolist())
                    if len(all_output) == 0:
                        all_output = output
                    else:
                        all_output = vstack((all_output, output))
        print(len(all_output), 'objects fitted')
        output_time(time.time()-t1)
        return all_output

    def fit_sn(self, lc, features):
        """Fits the chosen template model to a given light curve.

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
        obj = lc.meta['name']
        tab = features[features['Object'] == obj]
        params = np.array([tab[c] for c in tab.columns[1:]]).flatten()

        if len(params) == 0:
            print('No feature set found for', obj)
            return None

        model_name = self.templates[self.model_names[0]]
        if model_name == 'mlcs2k2':
            dust = sncosmo.CCM89Dust()
            model = sncosmo.Model(model_name, effects=[dust],
                                  effect_names=['host'],
                                  effect_frames=['rest'])
        else:
            model = sncosmo.Model(model_name)

        param_dict = {}
        for i in range(len(model.param_names)):
            param_dict[model.param_names[i]] = params[i]
        model.set(**param_dict)

        filts = np.unique(lc['filter'])
        labels = ['mjd', 'flux', 'filter']
        output = Table(names=labels, dtype=['f', 'f', 'U32'],
                       meta={'name': obj})
        for filt in filts:
            x = lc['mjd'][lc['filter'] == filt]
            xnew = np.linspace(0, x.max()-x.min(), 100)
            ynew = model.bandflux(filt, xnew, zp=27.5, zpsys='ab')
            newtable = Table([xnew+x.min(), ynew, [filt]*len(xnew)],
                             names=labels)
            output = vstack((output, newtable))
        return output

    def registerBands(self, dirname, prefix=None, suffix=None):
        """Register LSST bandpasses with sncosmo.
           Courtesy of Rahul Biswas"""
        bands = ['u', 'g', 'r', 'i', 'z', 'y']
        for band in bands:
            fname = os.path.join(dirname, prefix + band + suffix)
            data = np.loadtxt(fname)
            bp = sncosmo.Bandpass(wave=data[:, 0], trans=data[:, 1],
                                  name='lsst'+band)
            sncosmo.registry.register(bp, force=True)

    def goodness_of_fit(self, d):
        """Legacy code - to be deprecated. Use `compute_overall_chisq_over_pts`
        from `chisq.py`.

        Test (for any feature set) how well the reconstruction from the
        features fits each of the objects in the dataset.

        Parameters
        ----------
        d : Dataset
            Dataset object.

        Returns
        -------
        astropy.table.Table
            Table with the reduced Chi2 for each object
        """
        if len(d.models) == 0:
            print('Call Dataset.set_models first.')
            return None
        filts = np.unique(d.data[d.object_names[0]]['filter'])
        filts = np.array(filts).tolist()
        rcs = Table(names=['Object']+filts,
                    dtype=['U32']+['float64']*len(filts))  # Reduced chi2
        for obj in d.object_names:
            # Go through each filter
            chi2 = []
            lc = d.data[obj]
            mod = d.models[obj]
            for filt in filts:
                lc_filt = lc[lc['filter'] == filt]
                m = mod[mod['filter'] == filt]
                x = lc_filt['mjd']
                y = lc_filt['flux']
                e = lc_filt['flux_error']
                xmod = m['mjd']
                ymod = m['flux']
                # Interpolate
                fit = interp1d(xmod, ymod)
                yfit = fit(x)
                chi2.append(sum((yfit-y)**2/e**2)/(len(x)-1))
            rcs.add_row([obj]+chi2)
        return rcs


class ParametricFeatures(Features):
    """Fits a few options of generalised, parametric models to the data.
    """

    def __init__(self, model_choice, sampler='leastsq', limits=None):
        """
        Initialisation

        Parameters
        ----------
        model_choice : str
            Which parametric model to use
        sampler : str, optional
            Choice of 'mcmc' (requires emcee), 'nested' (requires pymultinest)
            or 'leastsq'
        limits : dict, optional
            Parameter bounds if something other than the default is needed.
        """

        Features.__init__(self)

        self.model_choices = {'newling': parametric_models.NewlingModel,
                              'karpenka': parametric_models.KarpenkaModel}

        try:
            self.model_name = model_choice  # Used for labelling output files
            if limits is not None:
                self.model = self.model_choices[model_choice](limits=limits)
            else:
                self.model = self.model_choices[model_choice]()
        except KeyError:
            print('Your selected model is not in the parametric_models package'
                  '. Either choose an existing model,  or implement a new one '
                  'in that package.')
            print('Make sure any new models are included in the model_choices '
                  'dictionary in the ParametricFeatures class.')
            sys.exit()

        if sampler == 'nested' and not has_multinest:
            print('Nested sampling selected but pymultinest is not installed. '
                  'Defaulting to least squares.')
            sampler = 'leastsq'

        elif sampler == 'mcmc' and not has_emcee:
            print('MCMC sampling selected but emcee is not installed. '
                  'Defaulting to least squares.')
            sampler = 'leastsq'

        self.sampler = sampler

    def extract_features(self, d, chain_directory='chains', save_output=True,
                         n_attempts=20, number_processes=1, n_walkers=100,
                         n_steps=500, walker_spread=0.1, burn=50, nlp=1000,
                         starting_point=None, convert_to_binary=True, n_iter=0,
                         restart=False, seed=-1):
        """Fit parametric models and return best-fitting parameters as features.

        Parameters
        ----------
        d : Dataset object
            Dataset
        chain_directory : str
            Where to save the chains
        save_output : bool
            Whether or not to save the intermediate output (if Bayesian
            inference is used instead of least squares)
        n_attempts : int
            Allow the minimiser to start in new random locations if the fit is
            bad. Put n_attempts=1 to fit only once with the default starting
            position.
        number_processes : int, optional
            Number of processors to use for parallelisation (shared memory
            only).
        n_walkers : int
            emcee parameter - number of walkers to use
        n_steps : int
            emcee parameter - total number of steps
        walker_spread : float
            emcee parameter - standard deviation of distribution of starting
            points of walkers.
        burn : int
            emcee parameter - length of burn-in
        nlp : int
            multinest parameter - number of live points
        starting_point : None or array-like
            Starting points of parameters for leastsq or emcee
        convert_to_binary : bool
            multinest parameter - whether or not to convert ascii output files
            to binary
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
        self.chain_directory = chain_directory
        t1 = time.time()
        output = []

        if number_processes < 2:
            k = 0
            for obj in d.object_names:
                if k % 100 == 0:
                    print(k, 'objects fitted')
                if self.sampler == 'leastsq':
                    newtable = _run_leastsq(obj, d, self.model, n_attempts,
                                            seed=seed)
                elif self.sampler == 'mcmc':
                    if (seed != -1):
                        np.random.seed(seed)
                    newtable = self.run_emcee(d, obj, save_output,
                                              chain_directory, n_walkers,
                                              n_steps, walker_spread, burn,
                                              starting_point)
                else:
                    newtable = _run_multinest(obj, d, self.model,
                                              chain_directory, nlp,
                                              convert_to_binary, n_iter,
                                              restart, seed=seed)

                if len(output) == 0:
                    output = newtable
                else:
                    output = vstack((output, newtable))
                k += 1
        else:
            if self.sampler == 'leastsq':
                p = Pool(number_processes, maxtasksperchild=1)
                partial_func = partial(_run_leastsq, d=d, model=self.model,
                                       n_attempts=n_attempts, seed=seed)
                out = p.map(partial_func, d.object_names)
                output = out[0]
                for i in range(1, len(out)):
                    output = vstack((output, out[i]))
            elif self.sampler == 'nested':
                p = Pool(number_processes, maxtasksperchild=1)
                partial_func = partial(_run_multinest, d=d,
                                       model=self.model,
                                       chain_directory=chain_directory,
                                       nlp=nlp, n_iter=n_iter,
                                       convert_to_binary=convert_to_binary,
                                       restart=restart, seed=seed)
                # Pool starts a number of threads, all of which may try to
                # tackle all of the data. Better to take it in chunks
                output = []
                k = 0
                objs = d.object_names
                while k < len(objs):
                    objs_subset = objs[k:k+number_processes]
                    out = p.map(partial_func, objs_subset)
                    for i in range(0, len(out)):
                        if out[i] is None:
                            print('Fitting failed for', objs_subset[i])
                        else:
                            if len(output) == 0:
                                output = out[i]
                            else:
                                output = vstack((output, out[i]))
                    k += len(objs_subset)
        print(len(output), 'objects fitted')
        output_time(time.time()-t1)
        return output

    def fit_sn(self, lc, features):
        """Fits the chosen parametric model to a given light curve.

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
        obj = lc.meta['name']
        params = features[features['Object'] == obj]

        if len(params) == 0:
            print('No feature set found for', obj)
            return None

        filts = np.unique(lc['filter'])
        labels = ['mjd', 'flux', 'filter']
        output = Table(names=labels, dtype=['f', 'f', 'U32'],
                       meta={'name': obj})
        cols = params.columns[1:]
        prms = np.array([params[c] for c in cols])
        for filt in filts:
            x = lc['mjd'][lc['filter'] == filt]
            xnew = np.linspace(0, x.max()-x.min(), 100)

            inds = np.where([filt in s for s in cols])[0]
            P = np.array(prms[inds], dtype='float')

            ynew = self.model.evaluate(xnew, P)
            newtable = Table([xnew+x.min(), ynew, [filt]*len(xnew)],
                             names=labels)
            output = vstack((output, newtable))
        return output

    def run_emcee(self, d, obj, save_output, chain_directory,  n_walkers,
                  n_steps, walker_spread, burn, starting_point, seed=-1):
        """Runs emcee on all the filter bands of a given light curve, fitting
        the model to each one and extracting the best fitting parameters.

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
            emcee parameter - standard deviation of distribution of starting
            points of walkers
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
            # Helper function to get parameters from the features astropy table
            X = starting_point[starting_point['Object'] == obj]
            cols = X.columns
            inds = [s for s in cols if filt in s]
            P = X[inds]
            P = np.array([P[c] for c in P.columns]).flatten()
            return P

        lc = d.data[obj]
        filts = np.unique(lc['filter'])

        if (seed != -1):
            np.random.seed(seed)

        n_params = len(self.model.param_names)
        labels = ['Object']
        for f in d.filter_set:
            pams = self.model.param_names
            for p in pams:
                labels.append(f+'-'+p)

        output = Table(names=labels, dtype=['U32']+['f']*(len(labels)-1))

        t1 = time.time()
        row = [obj]
        for f in d.filter_set:
            if f in filts:
                # This is pretty specific to current setup
                x = np.array(lc['mjd'][lc['filter'] == f])
                y = np.array(lc['flux'][lc['filter'] == f])
                yerr = np.array(lc['flux_error'][lc['filter'] == f])

                if starting_point is None:
                    # Initialise randomly in parameter space
                    iniparams = (np.random(n_params)*(self.model.upper_limit
                                                      - self.model.lower_limit)
                                 + self.model.lower_limit)
                else:
                    # A starting point from a least squares run can be given
                    # as an astropy table
                    iniparams = get_params(starting_point, obj, f)

                pos = [iniparams + walker_spread*np.randn(n_params)
                       for i in range(n_walkers)]

                sampler = emcee.EnsembleSampler(n_walkers, n_params,
                                                self.lnprob_emcee,
                                                args=(x, y, yerr))
                # Remove burn-in
                pos, prob, state = sampler.run_mcmc(pos, burn)
                sampler.reset()
                pos, prob, state = sampler.run_mcmc(pos, n_steps)

                samples = sampler.flatchain
                lnpost = sampler.flatlnprobability

                if save_output:
                    np.savetxt(chain_directory+'emcee_chain_%s_%s_%s' % (
                        self.model_name, f, (str)(obj)), np.column_stack(
                            (samples, lnpost)))
                # Maximum posterior params
                ind = lnpost.argmax()
                best_params = samples[ind, :]

                # Expects first the parameters, then -sigma then +sigma
                row += best_params
            else:
                # I'm not sure if it makes the most sense to fill in missing
                # values with zeroes...
                row += [0]*len(self.model.param_names)
        output.add_row(row)
        print('Time per filter', (time.time()-t1)/len(d.filter_set))
        return output

    def lnprob_emcee(self, params, x, y, yerr):
        """Likelihood function for emcee

        Parameters
        ----------
        params
        x
        y
        yerr

        Returns
        -------
        """
        # Uniform prior. Directly compares arrays
        is_not_within_limits = ((np.any(params > self.model.upper_limit))
                                or (np.any(params < self.model.upper_limit)))
        if is_not_within_limits:
            return -np.inf
        else:
            ynew = self.model.evaluate(x, params)
            chi2 = np.sum((y-ynew)*(y-ynew)/yerr/yerr)
            return -chi2/2.

    def goodness_of_fit(self, d):
        """Legacy code - to be deprecated. Use `compute_overall_chisq_over_pts`
        from `chisq.py`.

        Test (for any feature set) how well the reconstruction from the
        features fits each of the objects in the dataset.

        Parameters
        ----------
        d : Dataset
            Dataset object.

        Returns
        -------
        astropy.table.Table
            Table with the reduced Chi2 for each object
        """
        if len(d.models) == 0:
            print('Call Dataset.set_models first.')
            return None
        filts = np.unique(d.data[d.object_names[0]]['filter'])
        filts = np.array(filts).tolist()
        rcs = Table(names=['Object']+filts,
                    dtype=['U32']+['float64']*len(filts))  # Reduced chi2
        for obj in d.object_names:
            # Go through each filter
            chi2 = []
            lc = d.data[obj]
            mod = d.models[obj]
            for filt in filts:
                lc_filt = lc[lc['filter'] == filt]
                m = mod[mod['filter'] == filt]
                x = lc_filt['mjd']
                y = lc_filt['flux']
                e = lc_filt['flux_error']
                xmod = m['mjd']
                ymod = m['flux']
                # Interpolate
                fit = interp1d(xmod, ymod)
                yfit = fit(x)
                chi2.append(sum((yfit-y)**2/e**2)/(len(x)-1))
            rcs.add_row([obj]+chi2)
        return rcs


class WaveletFeatures(Features):
    """Uses wavelets to decompose the data and then reduces dimensionality of
    the feature space using PCA.
    """

    def __init__(self, output_root=None, **kwargs):
        """
        Parameters
        ----------
        output_root : str, optional
            Path where the wavelet features are saved. The eigendecomposition
            is also saved there by default.
        """
        Features.__init__(self)
        self.output_root = output_root

    def extract_features(self, dataset, number_gp, t_min, t_max, output_root,
                         number_processes, gp_dim, number_comps,
                         path_saved_eigendecomp=None, seed=1,
                         **kwargs):
        """Fit Gaussian Processes and compute the dimensionality reduced features.

        Parameters
        ----------
        dataset : Dataset object (sndata class)
            Dataset to work with.
        number_gp : int
            Number of points to evaluate the Gaussian Process Regression at.
        t_min : float
            Minimim time to evaluate the Gaussian Process Regression at.
        t_max : float
            Maximum time to evaluate the Gaussian Process Regression at.
        output_root : {None, str}, optional
            If None, don't save anything. If str, it is the output directory,
            so save the flux and error estimates and used kernels there.
        number_processes : int, optional
            Number of processors to use for parallelisation (shared memory
            only).
        gp_dim : int, optional
            The dimension of the Gaussian Process. If  `gp_dim` is 1, the
            filters are fitted independently. If `gp_dim` is 2, the Matern
            kernel is used with cross-information between the passbands.
        number_comps : int
            Dimension of the reduced wavelet space; Number of components to
            keep from the eigendecomposition.
        path_saved_eigendecomp : {None, str}, optional
            Path where the eigendecomposition is saved. If None, the
            eigendecomposition is calculated and saved in
            `path_save_eigendecomp` (see kwargs bellow).
        seed : int, optional
            Seed to have reproducible results. By default, `seed=1`.
        **kwargs : dict, optional
            kernel_param : list-like, default = [500., 20.]
                Initial values for kernel parameters. These should be roughly
                the scale length in the y & x directions.
            gp_algo: str, default = 'george'
                Which gp package is used for the Gaussian Process Regression,
                GaPP or george.
            do_subtract_background : bool, default = False
                Whether to estimate a new background subtracting the current.
            wavelet_name : {'sym2', str}, optional
                Name of the wavelets used.
            number_decomp_levels : {'max', int}, optional
                The number of decomposition steps to perform.
            path_saved_gp_files : {None, str}, optional
                Path for the Gaussian Process curve files.
            normalise_var : bool, optional
                If True, the feature space is scaled so that each feature has
                unit variance. By default it is False.
            path_save_eigendecomp : {'output_root', str}, optional
                Path where the eigendecomposition is saved. By default, it is
                saved in `self.output_root`, the same place as the wavelet
                features were saved.

        Returns
        -------
        reduced_features : pandas.DataFrame
            Projection of the events onto a lower dimensional space of size
            `number_comps`. It is then the reduced feature space.
            Shape (# events, `number_comps`).
        """
        np.random.seed(seed)
        print('Extracting features of the dataset.')
        initial_time = time.time()

        kwarg_gps = kwargs.copy()
        kwarg_gps.pop('wavelet_name', None)
        kwarg_gps.pop('number_decomp_levels', None)
        kwarg_gps.pop('path_saved_gp_files', None)
        kwarg_gps.pop('normalise_var', None)
        kwarg_gps.pop('path_save_eigendecomp', None)
        gps.compute_gps(dataset=dataset, number_gp=number_gp, t_min=t_min,
                        t_max=t_max, output_root=output_root,
                        number_processes=number_processes, gp_dim=gp_dim,
                        **kwarg_gps)

        kwargs_features = kwargs.copy()
        kwargs_features.pop('kernel_param', None)
        kwargs_features.pop('gp_algo', None)
        kwargs_features.pop('do_subtract_background', None)
        reduced_features = self.compute_reduced_features(
            dataset=dataset, number_comps=number_comps,
            path_saved_eigendecomp=path_saved_eigendecomp,
            **kwargs_features)

        print('Time taken to extract features: {:.2f}s.'
              ''.format(time.time()-initial_time))
        return reduced_features

    def fit_sn(self, lc, features, dataset, wavelet_name,
               path_saved_eigendecomp, filter_set):
        """Reconstruct the observations in real space from reduced features.

        Parameters
        ----------
        lc : astropy.table.Table
            Light curve: Time, flux and flux error observations in each
            passband of the event.
        features : pandas.DataFrame
            Projection of events onto a lower dimensional feature space.
            Shape (# events, `number_comps`), where `number_comps` is the
            # dimensions of lower dimensional feature space.
        dataset : Dataset object (sndata class)
            Dataset with the Gaussian process (GP) loaded in the models. These
            must contain the GP estimated light curve: Time, flux and flux
            error predictions in each passband of the event.
        wavelet_name : str
            Name of the wavelets used.
        path_saved_eigendecomp : str
            Path where the eigendecomposition is saved.
        filter_set : list
            List of the filters in the order they were put in the features.

        Returns
        -------
        astropy.table.Table
            Reconstructed light curve: Time, flux and flux error in each
            passband of the event. The times correspond to the uniform
            time-grid set to fit the Gaussian process curve. The flux and
            respective error are reconstructed from the low-dimension wavelet
            features.
        """
        # Make the names consisten with the rest of the WaveletFeatures class
        obj_data = lc
        obj = obj_data.meta['name']
        reduced_features = features
        # Obtain the Gaussian process estimated light curve: Time, flux and
        # flux error predictions in each passband of the event.
        lc_gps = dataset.models[obj]  # obtain the
        obj_gps = lc_gps.to_pandas()

        self._filter_set = filter_set
        self._number_gp = int(sum(obj_gps['filter'] == obj_gps['filter'][0]))
        self._is_wavelet_valid(wavelet_name)

        rec_space = self.reconstruct_feature_space(reduced_features,
                                                   path_saved_eigendecomp)
        # Feature space has dimensions:
        #  # passbands * # levels * 2 * # gp evaluations
        denominator = 2 * self.number_gp * len(self.filter_set)
        self.number_decomp_levels = np.shape(rec_space)[1] / denominator

        obj_coeffs_list = rec_space.loc[obj].values
        obj_gps_reconstruct = self._reconstruct_obj_real_space(
            obj_gps, obj_coeffs_list)
        obj_gps_reconstruct['flux'] = obj_gps_reconstruct['flux_reconstruct']
        obj_gps_reconstruct.drop(columns=['flux_reconstruct'], inplace=True)

        return Table().from_pandas(obj_gps_reconstruct)

    def compute_reduced_features(self, dataset, number_comps,
                                 path_saved_eigendecomp=None, **kwargs):
        """Compute the dimensionality reduced wavelet features.

        Parameters
        ----------
        dataset : Dataset object (sndata class)
            Dataset to work with.
        number_comps : int
            Dimension of the reduced wavelet space; Number of components to
            keep from the eigendecomposition.
        path_saved_eigendecomp : {None, str}, optional
            Path where the eigendecomposition is saved. If None, the
            eigendecomposition is calculated and saved in
            `path_save_eigendecomp` (see kwargs bellow).
        **kwargs : dict, optional
            wavelet_name : {'sym2', str}, optional
                Name of the wavelets used.
            number_decomp_levels : {'max', int}, optional
                The number of decomposition steps to perform.
            path_saved_gp_files : {None, str}, optional
                Path for the Gaussian Process curve files.
            normalise_var : bool, optional
                If True, the feature space is scaled so that each feature has
                unit variance. By default it is False.
            path_save_eigendecomp : {'output_root', str}, optional
                Path where the eigendecomposition is saved. By default, it is
                saved in `self.output_root`, the same place as the wavelet
                features were saved.

        Returns
        -------
        reduced_features : pandas.DataFrame
            Projection of the events onto a lower dimensional space of size
            `number_comps`. It is then the reduced feature space.
            Shape (# events, `number_comps`).
        """
        kwargs_wavelet_decomp = kwargs.copy()
        kwargs_wavelet_decomp.pop('normalise_var', None)
        kwargs_wavelet_decomp.pop('path_save_eigendecomp', None)
        self.compute_wavelet_decomp(dataset, **kwargs_wavelet_decomp)
        feature_space = self.load_feature_space(dataset)

        if path_saved_eigendecomp is None:
            kwargs_eigendecomp = kwargs.copy()
            kwargs_eigendecomp.pop('wavelet_name', None)
            kwargs_eigendecomp.pop('number_decomp_levels', None)
            kwargs_eigendecomp.pop('path_saved_gp_files', None)
            self.compute_eigendecomp(dataset, **kwargs_eigendecomp)

            path_saved_eigendecomp = kwargs.get('path_save_eigendecomp',
                                                self.output_root)

        reduced_features = self.project_to_space(feature_space,
                                                 path_saved_eigendecomp,
                                                 number_comps)
        return reduced_features

    def compute_wavelet_decomp(self, dataset, wavelet_name='sym2',
                               number_decomp_levels='max',
                               path_saved_gp_files=None):
        """Computes the wavelet decomposition of the dataset events.

        The function assumes the Gaussian Processes fit of the events has been
        done and that the estimated flux at regular intervals is saved in
        `dataset.models` or in the path provided on `path_saved_gp_files`.

        Parameters
        ----------
        dataset : Dataset object (sndata class)
            Dataset to work with.
        wavelet_name : {'sym2', str}, optional
            Name of the wavelets used.
        number_decomp_levels : {`max`, int}, optional
            The number of decomposition steps to perform.
        path_saved_gp_files : {None, str}, optional
            Path for the Gaussian Process curve files.
        """
        self._filter_set = dataset.filter_set
        self._read_gps_into_models(dataset, path_saved_gp_files)
        self._number_gp = self._extract_number_gp(dataset)
        self._is_wavelet_valid(wavelet_name)
        self.number_decomp_levels = number_decomp_levels

        print('Performing wavelet decomposition.')
        initial_time = time.time()

        objs = dataset.object_names
        for i in range(len(objs)):
            obj = objs[i]
            obj_gps = dataset.models[obj].to_pandas()
            coeffs = self._compute_obj_wavelet_decomp(obj_gps)

            if self.output_root is not None:
                self._save_obj_wavelet_decomp(obj, coeffs)
        print('Time taken for wavelet decomposition: {:.2f}s.'
              ''.format(time.time()-initial_time))

    def load_feature_space(self, dataset):
        """Load the wavelet feature space.

        The wavelet coefficients of the events in the dataset were previously
        calculated. This function loads then to memory and format them into a
        large table where each column corresponds to a wavelet coefficient in a
        specific passband.

        Parameters
        ----------
        dataset : Dataset object (sndata class)
            Dataset to work with.

        Returns
        -------
        feature_space : pandas.DataFrame
            Table of shape (# events, # wavelet features).
            Row `i` has the wavelet features of the event
            `dataset.object_names[i]`.
            The wavelet features are ordered as a flat version of `pywt.swt`
            function:
                [cAn, cDn, ..., cA2, cD2, cA1, cD1]
            where n equals the number of decomposition levels.
        """
        self._filter_set = dataset.filter_set
        objs = dataset.object_names
        filter_set = self.filter_set
        feature_space = []
        for i in range(len(objs)):
            obj = objs[i]
            path_saved_wavelet_decomp = os.path.join(self.output_root,
                                                     'wavelet_dict_{}.pckl'
                                                     ''.format(obj))
            with open(path_saved_wavelet_decomp, 'rb') as input:
                obj_coeffs = pickle.load(input)

            if i == 0:  # all events/passbands have the same number of levels
                number_levels = np.shape(obj_coeffs[filter_set[0]])[0]

            coeffs_list = []
            for pb in filter_set:
                pb_coeffs = obj_coeffs[pb]
                for level in np.arange(number_levels):
                    level_coeffs = pb_coeffs[level]
                    coeffs_list.append(level_coeffs[0])  # cA
                    coeffs_list.append(level_coeffs[1])  # cD
            coeffs_list = np.array(coeffs_list).flatten()
            feature_space.append(coeffs_list)
        feature_space = np.array(feature_space)  # change list -> array
        feature_space = pd.DataFrame(feature_space, index=objs)
        feature_space.index.rename('object_id', inplace=True)
        return feature_space

    def compute_eigendecomp(self, dataset, normalise_var=False,
                            path_save_eigendecomp='output_root'):
        """Compute eigendecomposition of the feature space.

        The eigendecomposition is performed using Singular Value Decomposition
        (SVD). The input data is always centered before applying the SVD. The
        features' values can also be scaled if `normalise_var` is set to
        `True`.

        Parameters
        ----------
        dataset : Dataset object (sndata class)
            Dataset to work with.
        normalise_var : bool, optional
            If True, the feature space is scaled so that each feature has unit
            variance. By default it is False.
        path_save_eigendecomp : {'output_root', str}, optional
            Path where the eigendecomposition is saved. By default, it is
            saved in `self.output_root`, the same place as the wavelet
            features were saved.

        Raises
        ------
        ValueError
            The eigendecomposition must be saved, hence a valid path must be
            provided.
        """
        if path_save_eigendecomp == 'output_root':
            path_save_eigendecomp = self.output_root
        elif path_save_eigendecomp is None:
            raise ValueError('A valid path to save the eigendecomposotion must'
                             ' be provided. At the moment, `None` was provided'
                             ' instead.')
        self._exists_path(path_save_eigendecomp)

        print('Performing eigendecomposition.')
        initial_time = time.time()

        self._filter_set = dataset.filter_set
        feature_space = self.load_feature_space(dataset)
        # Center the feature_space to perform eigendecomposition
        feature_space_new, means, scales = self._center_matrix(
            feature_space, normalise_var=normalise_var)

        # Row ith has the eigenvector corresponding to the ith eigenvalue
        # (eigenvectors in descending order of eigenvalue)
        u, singular_vals, eigenvecs = np.linalg.svd(feature_space_new,
                                                    full_matrices=False)

        number_objs = np.shape(feature_space)[0]
        eigenvals = singular_vals**2 / (number_objs-1)  # by definition

        path_save = path_save_eigendecomp
        np.save(os.path.join(path_save, 'means.npy'), means)
        np.save(os.path.join(path_save, 'scales.npy'), scales)
        np.save(os.path.join(path_save, 'eigenvalues.npy'), eigenvals)
        np.save(os.path.join(path_save, 'eigenvectors.npy'), eigenvecs)

        print('Time taken for eigendecomposition: {:.2f}s.'
              ''.format(time.time()-initial_time))

    @staticmethod
    def load_pca(path_saved_eigendecomp, number_comps=None):
        """Load the principal component analysis of the feature space.

        Load the means and scales needed to transform a matrix onto the same
        space as the matrix used to calculate the eigendecomposition.
        Load the first `number_comps` eigenvalues to transform any new data
        onto a lower-dimensional space.

        Parameters
        ----------
        path_saved_eigendecomp : str
            Path where the eigendecomposition is saved.
        number_comps : {None, 'all', int}, optional
            If None, returns the enough eingenvectors for the explained
            variance be >= 95%. If 'all, all eigenvectors are returned.
            Otherwise, only the first `number_comps` are returned.

        Returns
        -------
        means : array
            Mean of each feature across all the samples. Shape (# features, ).
        scales : {None, array}
            If `normalise_var` is false, `scales` is not used. Otherwise, it
            is the value used to rescale `matrix` so that the variance of each
            feature is unitaty. Shape (# features, ).
        eigenvecs : array
            First `number_comps` eigenvectors of the original feature space.
            Each row corresponds to a eigenvector and they are in descending
            order of eigenvalue.
        """
        means = np.load(os.path.join(path_saved_eigendecomp, 'means.npy'))
        scales = np.load(os.path.join(path_saved_eigendecomp, 'scales.npy'),
                         allow_pickle=True)
        eigenvecs = np.load(os.path.join(path_saved_eigendecomp,
                            'eigenvectors.npy'))

        if np.shape(scales) == ():  # the saved output has a weird format
            scales = None

        if number_comps is None:
            tol = .95  # Explain at least 95% of the variance
            eigenvals = np.load(os.path.join(path_saved_eigendecomp,
                                'eigenvalues.npy'))
            eigenvals_cumsum = np.cumsum(eigenvals/sum(eigenvals))
            index_greater_tol = np.where(eigenvals_cumsum >= tol)
            first_index = index_greater_tol[0][0]
            # +1 because it is inclusive that index
            number_comps = first_index + 1
            eigenvecs = eigenvecs[:number_comps, :]
        elif number_comps != 'all':  # a number was given
            eigenvecs = eigenvecs[:number_comps, :]
        else:
            number_comps = np.shape(eigenvecs)[0]
        print('Dimensionality reduced feature space with {} components.'
              ''.format(number_comps))
        return means, scales, eigenvecs

    def project_to_space(self, feature_space, path_saved_eigendecomp,
                         number_comps=None):
        """Project dataset onto a previously calculated feature space.

        The feature space correspond to the wavelet decomposition of the
        events to use. It has not been centered/ scaled yet.

        Parameters
        ----------
        feature_space : pandas.DataFrame
            Table of shape (# events, # features).
            Row `i` has the wavelet features of the event
            `dataset.object_names[i]`.
            The wavelet features are ordered as a flat version of `pywt.swt`
            function:
                [cAn, cDn, ..., cA2, cD2, cA1, cD1]
            where n equals the number of decomposition levels.
        path_saved_eigendecomp : str
            Path where the eigendecomposition is saved.
        number_comps : {None, 'all', int}, optional
            If None, returns the enough eingenvectors for the explained
            variance be >= 95%. If 'all, all eigenvectors are returned.
            Otherwise, only the first `number_comps` are returned.

        Returns
        -------
        reduced_space : pandas.DataFrame
            Projection of the events onto a lower dimensional space of size
            `number_comps`. It is then the reduced feature space.
            Shape (# events, `number_comps`).
        """
        means, scales, eigenvecs = self.load_pca(path_saved_eigendecomp,
                                                 number_comps)

        feature_space_new = self._preprocess_matrix(feature_space, means,
                                                    scales)

        reduced_space = feature_space_new @ eigenvecs.T
        return reduced_space

    def _compute_obj_wavelet_decomp(self, obj_gps):
        """Stationary wavelet transform of each passband of the event.

        Parameters
        ----------
        obj_gps : pandas.DataFrame
            Table with evaluated Gaussian process curve and errors at each
            passband.

        Returns
        -------
        coeffs : dict
            Coefficients of the wavelet decompostion per passband of `obj`.
        """
        coeffs = {}
        for pb in self.filter_set:
            obj_pb = obj_gps[obj_gps['filter'] == pb]
            coeffs[pb] = pywt.swt(obj_pb['flux'], wavelet=self.wavelet,
                                  level=self.number_decomp_levels)
        return coeffs

    def _save_obj_wavelet_decomp(self, obj, coeffs):
        """Save the wavelet decomposition of the event.

        Parameters
        ----------
        obj : str
            Name of the event whose wavelet decomposition belong.
        coeffs : dict
            Coefficients of the wavelet decompostion per passband of `obj`.
        """
        path_save_wavelet_decomp = os.path.join(self.output_root,
                                                'wavelet_dict_{}.pckl'
                                                ''.format(obj))
        with open(path_save_wavelet_decomp, 'wb') as f:
            pickle.dump(coeffs, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _center_matrix(matrix, normalise_var=False):
        """Centers the matrix and normalises its variance if chosen.

        This step is needed to compute the Singular Value Decomposition,
        Eigendecomposition or computing covariance.

        Parameters
        -----------
        matrix : array
            Matrix of shape (# samples, # features).
        normalise_var: bool, optional
            If True, `matrix` is scaled so that each feature has unit variance.

        Returns
        -------
        matrix_new : pandas.DataFrame
            Centered matrix of shape (# samples, # features). If
            `normalise_var` is true, `matrix_new` has unit variance across the
            features.
        means : array
            Mean of each feature across all the samples. Shape (# features, ).
        scales : {None, array}
            If `normalise_var` is false, `scales` is not used. Otherwise, it
            is the value used to rescale `matrix` so that the variance of each
            feature is unitaty. Shape (# features, ).

        Notes
        -----
        Let :math:`X` be the initial matrix with shape (# events, # features)
        and :math:`X_{new}` the output matrix.

        .. math::  X_{\mathrm{new}} = (X - mean(X))

        If `normalise_var` is true, :math:`X` is further rescaled to have
        unit variance.
        The option of scaling the variance has been retained for consistency
        with previous methods and various ML resources which suggest this may
        help in balancing cases where the variances of different features vary
        a lot. In a few tests we have done with specific datasets, we have not
        seen the suggested benefits.
        """
        means = np.mean(matrix, axis=0)  # mean of each feature
        matrix_new = matrix - means

        scales = None
        if normalise_var:  # L2 normalization
            scales = np.sqrt(np.sum(matrix_new**2, axis=0)).values
            matrix_new /= scales

        return matrix_new, means.values, scales

    @staticmethod
    def _preprocess_matrix(matrix, means, scales):
        """Subtract the features' mean and scales their variance.

        Change the matrix to be in the same feature space the
        eigendecomposition was calculated.

        Parameters
        -----------
        matrix : pandas.DataFrame
            Matrix of shape (# samples, # features).
        means : array
            Mean of each feature across all the samples. Shape (# features, ).
        scales : {None, array}
            If `normalise_var` is false, `scales` is not used. Otherwise, it
            is the value used to rescale `matrix` so that the variance of each
            feature is unitaty. Shape (# features, ).

        Returns
        -------
        matrix_new : array
            Preprocessed matrix of shape (# samples, # features).
        """
        matrix_new = matrix - means
        if scales is not None:
            matrix_new /= scales

        return matrix_new

    def save_reduced_features(self, reduced_features,
                              path_save_reduced_features='output_root',
                              file_name=None):
        """Save dimensionality reduced wavelet features.

        Parameters
        ----------
        reduced_features : pandas.DataFrame
            Projection of the events onto a lower dimensional space of size
            `number_comps`. It is then the reduced feature space.
            Shape (# events, `number_comps`).
        path_save_reduced_features : {'output_root', str}, optional
            Path where the dimensionality reduced wavelet features is saved.
            By default, it is saved in `self.output_root`, the same place as
            the wavelet features were saved.
        file_name : {None, str}, optional
            Name of the dimensionality reduced wavelet features file. By
            default (None), it is `reduced_features_{# components}.pckl`.
        """
        if file_name is None:
            number_comps = np.shape(reduced_features)[1]
            file_name = 'reduced_features_{}.pckl'.format(number_comps)
        if path_save_reduced_features == 'output_root':
            path_save_reduced_features = self.output_root

        path_save_file = os.path.join(path_save_reduced_features, file_name)
        with open(path_save_file, 'wb') as f:
            pickle.dump(reduced_features, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def create_readable_table(coeffs):
        """Create readable table from `pywt` coefficients.

        Parameters
        ----------
        coeffs : list
            List of approximation and details coefficients pairs in the same
            order as `pywt.swt` function:
                [(cAn, cDn), ..., (cA2, cD2), (cA1, cD1)]
            where n equals the number of decomposition levels.

        Returns
        -------
        table : pandas.DataFrame
            Table with the name of each column denoting the decomposition
            level and type.
        """
        table = pd.DataFrame()
        number_levels = np.shape(coeffs)[0]
        for level in np.arange(number_levels):
            table['cA{}'.format(number_levels-level)] = coeffs[level][0]
            table['cD{}'.format(number_levels-level)] = coeffs[level][1]
        return table

    def compute_reconstruct_error(self, dataset, true_obs=False, **kwargs):
        """X^2/datapoints between original and reconstructed observations.

        The original observations are the points at which the Gaussian Process
        (GP) was evaluated. The uncertainty is then the GP uncertainty at
        those points.

        Parameters
        ----------
        dataset : Dataset object (sndata class)
            Dataset to work with.
        true_obs : bool, optional
            If the reconstructed error is in relation with the original
            observations (True) or with the GP estimates (False, default).

        Returns
        -------
        chisq_over_pts : pandas.DataFrame
            Table with the X^2/datapoints per object.

        Raises
        ------
        KeyError
            If the reconstructed flux was not saved into the dataset models,
            the reconstructed feature space and wavelet name must be given as
            `**kwargs` with the names `feature_space` and `wavelet_name`,
            respectively.
        """
        objs = dataset.object_names
        try:
            dataset.models[objs[0]]['flux_reconstruct']
        except KeyError:
            try:
                kwargs['feature_space']
                kwargs['wavelet_name']
            except KeyError:
                raise KeyError('If the reconstructed flux was not saved into '
                               'the dataset models, the reconstructed feature '
                               'space and wavelet name must be given as '
                               '`**kwargs` with the names `feature_space` and '
                               '`wavelet_name`, respectively.')
            self.reconstruct_real_space(dataset, **kwargs)

        chisq_over_pts = np.zeros(len(objs))
        for i in range(len(objs)):
            obj = objs[i]
            obj_gps = dataset.models[obj].to_pandas()
            obj_reconstruct = obj_gps.copy()
            obj_reconstruct['flux'] = obj_reconstruct['flux_reconstruct']

            if true_obs:  # compare with true observations
                obj_data = dataset.data[obj].to_pandas()
                obj_true = obj_data
            else:  # compare with estimated observations
                obj_true = obj_gps
            chisq_over_pts[i] = (
                chisq.compute_overall_chisq_over_pts(obj_true,
                                                     obj_reconstruct))
        chisq_over_pts = pd.DataFrame(data=chisq_over_pts, index=objs,
                                      columns=['chisq_over_pts'])
        chisq_over_pts.index.rename('object_id', inplace=True)
        return chisq_over_pts

    def reconstruct_real_space(self, dataset, feature_space, wavelet_name,
                               path_saved_gp_files=None):
        """Reconstruct the observations in real space from the feature space.

        Parameters
        ----------
        dataset : Dataset object (sndata class)
            Dataset to work with.
        feature_space : pandas.DataFrame
            Table of shape (# events, # features).
            Row `i` has the wavelet features of the event
            `dataset.object_names[i]`.
            The wavelet features are ordered as a flat version of `pywt.swt`
            function:
                [cAn, cDn, ..., cA2, cD2, cA1, cD1]
            where n equals the number of decomposition levels.
        wavelet_name : str
            Name of the wavelets used.
        path_saved_gp_files : {None, str}, optional
            Path for the Gaussian Process curve files.
        """
        objs = dataset.object_names

        try:
            dataset.models[objs[0]]
        except KeyError:
            self._read_gps_into_models(dataset, path_saved_gp_files)

        self._filter_set = dataset.filter_set
        self._number_gp = self._extract_number_gp(dataset)
        self._is_wavelet_valid(wavelet_name)

        # Feature space has dimensions:
        #  # passbands * # levels * 2 * # gp evaluations
        denominator = 2 * self.number_gp * len(self.filter_set)
        self.number_decomp_levels = np.shape(feature_space)[1] / denominator

        for i in range(len(objs)):
            obj = objs[i]
            obj_gps = dataset.models[obj].to_pandas()
            obj_coeffs_list = feature_space.loc[obj].values
            obj_gps_reconstruct = self._reconstruct_obj_real_space(
                obj_gps, obj_coeffs_list)
            # Write back to Astropy Table for consistency
            dataset.models[obj] = Table().from_pandas(obj_gps_reconstruct)

    def _reconstruct_obj_real_space(self, obj_gps, coeffs_list):
        """Reconstructs the flux using the wavelet decomposition.

        Parameters
        ----------
        obj_gps : pandas.DataFrame
            Table with evaluated Gaussian process curve and errors at each
            passband.
        coeffs_list : array
            Wavelet features of the event ordered as a flat version of
            `pywt.swt` function:
                [cAn, cDn, ..., cA2, cD2, cA1, cD1]
            where n equals the number of decomposition levels.
            Shape (1, # features).

        Returns
        -------
        obj_gps : pandas.DataFrame
            Table with evaluated Gaussian process curve, errors and the
            reconstructed flux (`flux_reconstruct`) at each passband.
        """
        obj_coeffs = self._reshape_coeffs(coeffs_list)
        obj_gps['flux_reconstruct'] = -666.  # initialize the column
        for pb in self.filter_set:
            is_pb_obs = obj_gps['filter'] == pb
            pb_coeffs = obj_coeffs[pb]
            pb_flux_reconstruct = pywt.iswt(pb_coeffs, self.wavelet)
            obj_gps.loc[is_pb_obs, 'flux_reconstruct'] = pb_flux_reconstruct
        return obj_gps

    def _reshape_coeffs(self, coeffs_list):
        """Reshape the coefficients list into a differentiated table.

        Parameters
        ----------
        coeffs_list : array
            Wavelet features of the event ordered as a flat version of
            `pywt.swt` function:
                [cAn, cDn, ..., cA2, cD2, cA1, cD1]
            where n equals the number of decomposition levels.
            Shape (1, # features).

        Returns
        -------
        coeffs : dict
            Coefficients of the wavelet decompostion per passband of the event.
            The coefficients of each passband are the list of approximation
            and details coefficients pairs in the same order as `pywt.swt`
            function:
                [(cAn, cDn), ..., (cA2, cD2), (cA1, cD1)]
            where n equals the number of decomposition levels.
        """
        coeffs = {}
        number_pbs = len(self.filter_set)
        coeffs_list = np.split(np.array(coeffs_list), number_pbs)
        for i in np.arange(number_pbs):
            pb = self.filter_set[i]
            new_coeff_format = []
            pb_coeffs_list = np.split(coeffs_list[i],
                                      self.number_decomp_levels)
            for level in np.arange(self.number_decomp_levels):
                level_coeffs_list = np.split(pb_coeffs_list[level], 2)
                new_coeff_format.append(level_coeffs_list)
            coeffs[pb] = new_coeff_format
        return coeffs

    def reconstruct_feature_space(self, reduced_space, path_saved_eigendecomp):
        """Reconstruct the original feature space from the reduced one.

        Parameters
        ----------
        reduced_space : pandas.DataFrame
            Projection of the events onto a lower dimensional space of size
            `number_comps`. It is then the reduced feature space.
            Shape (# events, `number_comps`).
        path_saved_eigendecomp : str
            Path where the eigendecomposition is saved.

        Returns
        -------
        reconstruct_space : pandas.DataFrame
            Reconstructed features in the original feature space. This matrix
            has a lower rank than the original features matrix because there
            is some loss in the dimensionality reduction.
            Shape (# events, # features).
            Row `i` has the wavelet features of the event
            `dataset.object_names[i]`.
            The wavelet features are ordered as a flat version of `pywt.swt`
            function:
                [cAn, cDn, ..., cA2, cD2, cA1, cD1]
            where n equals the number of decomposition levels.
        """
        number_comps = np.shape(reduced_space)[1]
        means, scales, eigenvecs = self.load_pca(path_saved_eigendecomp,
                                                 number_comps)
        reconstruct_space = reduced_space @ eigenvecs
        reconstruct_space = self._postprocess_matrix(reconstruct_space, means,
                                                     scales)
        return reconstruct_space

    @staticmethod
    def _postprocess_matrix(matrix, means, scales):
        """Add the features' mean and scales back their variance.

        Change the matrix from the feature space where the
        eigendecomposition was calculated to the original feature space.

        Parameters
        -----------
        matrix : pandas.DataFrame
            Matrix of shape (# samples, # features).
        means : array
            Mean of each feature across all the samples. Shape (# features, ).
        scales : {None, array}
            If `normalise_var` is false, `scales` is not used. Otherwise, it
            is the value used to rescale `matrix` so that the variance of each
            feature is unitaty. Shape (# features, ).

        Returns
        -------
        matrix_new : pandas.DataFrame
            Postprocessed matrix of shape (# samples, # features).
        """
        matrix_new = matrix + means
        if scales is not None:
            matrix_new *= scales

        return matrix_new

    @staticmethod
    def _read_gps_into_models(dataset, path_saved_gp_files):
        """Read the saved files into the dataset models.

        If the models already exist, nothing is done.

        Parameters
        ----------
        dataset : Dataset object (sndata class)
            Dataset.
        path_saved_gp_files : {None, str}
            Path for the Gaussian Process curve files.

        Raises
        ------
        AttributeError
            The Gaussian Processes fit of the events must have been done and
            it must be either saved in `dataset.models` or in the path
            provided on `path_saved_gp_files`.
        """
        if len(dataset.models) != 0:
            pass
        elif path_saved_gp_files is not None:
            gps.read_gp_files_into_models(dataset, path_saved_gp_files)
        else:
            raise AttributeError('The Gaussian Processes fit of the events '
                                 'must have been done and it must be either '
                                 'saved in `dataset.models` or in the path '
                                 'provided on `path_saved_gp_files`.')

    @property
    def filter_set(self):
        """Return passbands the events could be observed on.

        Returns
        -------
        list-like
            Passbands in which the events could be observed on.
        """
        return self._filter_set

    def _is_wavelet_valid(self, wavelet_name):
        """Check if the wavelet is valid.

        The available families can be checked using `pywt.wavelist()` or going
        to https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html .

        Parameters
        ----------
        wavelet_name : str
            Name of the wavelets used.
        """
        try:
            self._wavelet = pywt.Wavelet(wavelet_name)
            print('The wavelet used is {}.'.format(wavelet_name))
        except ValueError:
            raise ValueError('Unknown wavelet name {}. Check `pywt.wavelist()`'
                             ' for the list of available builtin wavelets.'
                             ''.format(wavelet_name))

    @property
    def wavelet(self):
        """Return the wavelets used.

        Returns
        -------
        str
            Name of the wavelets used.
        """
        return self._wavelet

    @property
    def number_gp(self):
        """Return the number of points the Gaussian Process was evaluated at.

        Returns
        -------
        int
            Number of points the Gaussian Process was evaluated at.
        """
        return self._number_gp

    def _extract_number_gp(self, dataset):
        """Extract the number of points the Gaussian Process was evaluated at.

        It assumes all the events and passbands were evaluated in the same
        number of points, as supposed. It extracts the number of the points
        from the first passband of the first event.

        Parameters
        ----------
        dataset : Dataset object (sndata class)
            Dataset to work with.

        Returns
        -------
        int
            Number of points the Gaussian Process was evaluated at.

        Raises
        ------
        AttributeError
            The Gaussian processes have not been saved into `dataset.models`.
        """
        obj = dataset.object_names[0]
        try:
            obj_gps = dataset.models[obj].to_pandas()
        except KeyError:
            raise AttributeError('The Gaussian processes have not been saved '
                                 'into `dataset.models`.')
        return np.sum(obj_gps['filter'] == self.filter_set[0])

    @property
    def number_decomp_levels(self):
        """Return the number of decomposition steps to perform.

        Returns
        -------
        dict
            The number of decomposition steps to perform.
        """
        return self._number_decomp_levels

    @number_decomp_levels.setter
    def number_decomp_levels(self, value):
        """Set the number of decomposition steps to perform.

        Parameters
        ----------
        value : {`max`, int}, optional
            The number of decomposition steps to perform.

        Raises
        ------
        ValueError
            The number of decomposition levels must be a positive number
            smaller or equal to the maximum number of decomposition levels
            allowed by the data.
        """
        max_number_levels = pywt.swt_max_level(self.number_gp)
        if value == 'max':
            number_levels = int(max_number_levels)
        elif 1 <= value <= max_number_levels:
            number_levels = int(value)
        else:
            raise ValueError('This dataset can only be decomposed into a '
                             'positive number of levels smaller or equal to {}'
                             '.'.format(max_number_levels))
        print('Each passband is decomposed in {} levels.'
              ''.format(number_levels))
        self._number_decomp_levels = number_levels

    @property
    def output_root(self):
        """Return the path to the output files.

        Returns
        -------
        {None, str}
            Path to the output files.
        """
        return self._output_root

    @output_root.setter
    def output_root(self, value):
        """Set the path to the output files.

        Parameters
        ----------
        value : {None, str}, optional
            Path to the output files.
        """
        if value is not None:
            self._exists_path(value)
        self._output_root = value


class AvocadoFeatures(Features):
    """Class adapted from `avocado` augmentation [1]_.

    References
    ----------
    .. [1] Boone, Kyle. "Avocado: Photometric classification of astronomical
    transients with gaussian process augmentation." The Astronomical Journal
    158.6 (2019): 257."""

    def __init__(self, output_root=None, **kwargs):
        """
        Parameters
        ----------
        dataset : Dataset object (sndata class)
            Dataset to augment.
        output_root : str, optional
            Path where the wavelet features are saved. The eigendecomposition
            is also saved there by default.
        """
        Features.__init__(self)
        self.output_root = output_root

    def extract_features(self, dataset, number_gp, t_min, t_max, output_root,
                         number_processes, gp_dim, seed=1,
                         **kwargs):
        """Fit Gaussian Processes and compute the avocado features.

        The features extracted can be directly used for classification.

        Parameters
        ----------
        dataset : Dataset object (sndata class)
            Dataset to work with.
        number_gp : int
            Number of points to evaluate the Gaussian Process Regression at.
        t_min : float
            Minimim time to evaluate the Gaussian Process Regression at.
        t_max : float
            Maximum time to evaluate the Gaussian Process Regression at.
        output_root : {None, str}, optional
            If None, don't save anything. If str, it is the output directory,
            so save the flux and error estimates and used kernels there.
        number_processes : int, optional
            Number of processors to use for parallelisation (shared memory
            only).
        gp_dim : int, optional
            The dimension of the Gaussian Process. If  `gp_dim` is 1, the
            filters are fitted independently. If `gp_dim` is 2, the Matern
            kernel is used with cross-information between the passbands.
        seed : int, optional
            Seed to have reproducible results. By default, `seed=1`.
        **kwargs : dict, optional
            kernel_param : list-like, default = [500., 20.]
                Initial values for kernel parameters. These should be roughly
                the scale length in the y & x directions.
            gp_algo: str, default = 'george'
                Which gp package is used for the Gaussian Process Regression,
                GaPP or george.
            do_subtract_background : bool, default = False
                Whether to estimate a new background subtracting the current.

        Returns
        -------
        classification_features : pandas.DataFrame
            The processed features that can be fed to a classifier.
        """
        np.random.seed(seed)
        print('Extracting features of the dataset.')
        initial_time = time.time()

        kwarg_gps = kwargs.copy()
        gps.compute_gps(dataset=dataset, number_gp=number_gp, t_min=t_min,
                        t_max=t_max, output_root=output_root,
                        number_processes=number_processes, gp_dim=gp_dim,
                        **kwarg_gps)

        raw_features = self.compute_raw_features(dataset)
        classification_features = self.compute_classification_features(
            raw_features)

        self.save_avo_features(classification_features)

        print('Time taken to extract features: {:.2f}s.'
              ''.format(time.time()-initial_time))
        return classification_features

    def fit_sn(self, lc_gps):
        """Plot the Gaussian Process estimated light curve.

        avocado features do not change the models used to fit the objects, so
        this function returns the Gaussian Process estimated light curve.

        Parameters
        ----------
        lc_gps : astropy.table.Table
            Gaussian Process estimated light curve: Time, flux and flux error
            predictions in each passband of the event.

        Returns
        -------
        lc_gps : astropy.table.Table
            Gaussian Process estimated light curve: Time, flux and flux error
            predictions in each passband of the event.
        """
        return lc_gps

    def compute_raw_features(self, dataset):
        """Compute raw features from the dataset.

        The raw features are later used to compute the features that will be
        used for classification. The raw features are saved as
        `self.raw_features`.
        From [1]_:
            "Featurizing is slow, so the idea here is to extract a lot of
            different things, and then in `select_features` these features are
            postprocessed to select the ones that are actually fed into the
            classifier. This allows for rapid iteration of training on
            different feature sets. Note that the features produced by this
            method are often unsuitable for classification, and may include
            data leaks. Make sure that you understand what features can be
            used for real classification before making any changes.
            This class implements a featurizer that is tuned to the PLAsTiCC
            dataset."

        Parameters
        ----------
        dataset : Dataset object (sndata class)
            Dataset to compute the raw features.

        Returns
        -------
        raw_features : pandas.DataFrame
            The computed raw features of the entire dataset.

        Notes
        -----
        Function adapted from `avocado` augmentation [2]_ ([1]_).

        References
        ----------
        .. [1] Boone, Kyle.
        https://avocado-classifier.readthedocs.io/en/latest/
        .. [2] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        self.filter_set = dataset.filter_set
        list_raw_features = []
        objs = dataset.object_names
        metadata = dataset.metadata
        for obj in objs:
            obj_data = dataset.data[obj].to_pandas()
            obj_gps = dataset.models[obj].to_pandas()
            obj_metadata = metadata.loc[obj]
            obj_features = self._compute_obj_raw_features(obj_data, obj_gps,
                                                          obj_metadata)
            list_raw_features.append(obj_features.values())

        # The features' keys are the same for every obj.
        keys = obj_features.keys()

        raw_features = pd.DataFrame(list_raw_features, index=objs,
                                    columns=keys)
        raw_features.index.name = 'object_id'
        self.raw_features = raw_features
        return raw_features

    def _compute_obj_raw_features(self, obj_data, obj_gps, obj_metadata):
        """Compute the raw features for an object.

        Parameters
        ----------
        obj_data : pandas.DataFrame
            Time, flux, flux error and passbands of the object.
        obj_gps : pandas.DataFrame
            Table with evaluated Gaussian process curve and errors at each
            passband.
        obj_metadata : pandas.DataFrame
            Metadata of the event.

        Returns
        -------
        features : pandas.DataFrame
            The computed raw features of the event provided.

        Notes
        -----
        Function adapted from `avocado` augmentation [1]_ ([2]_).

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        .. [2] Boone, Kyle.
        https://avocado-classifier.readthedocs.io/en/latest/
        """
        features = dict()
        pad = 250
        plasticc_start_time = np.min(obj_gps['mjd'])
        plasticc_end_time = np.max(obj_gps['mjd'])

        obj = obj_metadata['object_id']

        # Load the GP to make more predictions
        gp_params, gp_predict = self._load_gp(obj, return_gp_params=True)

        # Fit the GP and produce an output model
        gp_start_time = plasticc_start_time - pad
        gp_end_time = plasticc_end_time + pad
        gp_times = np.arange(gp_start_time, gp_end_time + 1)

        filter_set = np.asarray(self.filter_set)
        gp_wavelengths = np.vectorize(pb_wavelengths.get)(filter_set)
        obj_gps_new = gps.predict_2d_gp(gp_predict, gp_times,
                                        gp_wavelengths).to_pandas()

        times = obj_data['mjd']
        fluxes = obj_data['flux']
        flux_errors = obj_data['flux_error']
        s2ns = fluxes / flux_errors

        # Features from the metadata
        features["host_specz"] = obj_metadata["hostgal_specz"]
        features["host_photoz"] = obj_metadata["hostgal_photoz"]
        features["host_photoz_error"] = obj_metadata["hostgal_photoz_err"]
        features["ra"] = obj_metadata["ra"]
        features["decl"] = obj_metadata["decl"]
        features["mwebv"] = obj_metadata["mwebv"]
        features["ddf"] = obj_metadata["ddf"]

        # Count how many observations there are
        features["count"] = len(fluxes)

        # Features from GP fit parameters
        for i, fit_param in enumerate(gp_params):
            features[f"gp_fit_{i}"] = fit_param

        # Maximum and minimum fluxes and times of maximum.
        pbs = self.filter_set
        max_fluxes = pd.DataFrame(columns=pbs, dtype=float)
        min_fluxes = pd.DataFrame(columns=pbs, dtype=float)
        time_max_flux = pd.DataFrame(columns=pbs, dtype=float)
        for pb in pbs:
            is_pb = obj_gps_new['filter'] == pb
            obj_pb = obj_gps_new[is_pb].reset_index()
            max_fluxes.loc[obj, pb] = np.max(obj_pb['flux'])
            min_fluxes.loc[obj, pb] = np.min(obj_pb['flux'])

            index_max = np.argmax(obj_pb['flux'])
            time_max_flux.loc[obj, pb] = obj_pb.loc[index_max, 'mjd']
        med_time_max = np.median(time_max_flux)
        max_dts = time_max_flux - med_time_max

        features["max_time"] = med_time_max
        for pb in pbs:
            features[f"max_flux_{pb}"] = max_fluxes.loc[:, pb][0]
            features[f"max_dt_{pb}"] = max_dts.loc[:, pb][0]
            features[f"min_flux_{pb}"] = min_fluxes.loc[:, pb][0]

        for pb in pbs:
            is_pb = obj_gps_new['filter'] == pb
            obj_pb = obj_gps_new[is_pb].reset_index()
            obj_flux = obj_pb['flux'].values

            # Calculate the positive and negative integrals of the lightcurve,
            # normalized to the respective peak fluxes. This gives a measure
            # of the "width" of the lightcurve, even for non-bursty objects.
            positive_width = (np.sum(np.clip(obj_flux, 0, None))
                              / max_fluxes.loc[:, pb][0])
            negative_width = (np.sum(np.clip(obj_flux, None, 0))
                              / min_fluxes.loc[:, pb][0])
            features[f"positive_width_{pb}"] = positive_width
            features[f"negative_width_{pb}"] = negative_width

            # Calculate the total absolute differences of the lightcurve. For
            # supernovae, they typically go up and down a single time. Periodic
            # objects will have many more ups and downs.
            abs_diff = np.sum(np.abs(obj_flux[1:] - obj_flux[:-1]))
            features[f"abs_diff_{pb}"] = abs_diff

            # Find times to fractions of the peak amplitude
            fractions = [0.8, 0.5, 0.2]
            forward_times = self._compute_time_to_fractions(obj_flux,
                                                            fractions)
            backward_times = self._compute_time_to_fractions(obj_flux,
                                                             fractions,
                                                             forward=False)
            for fraction, forward_time, backward_time in zip(fractions,
                                                             forward_times,
                                                             backward_times):
                features["time_fwd_max_{:.1f}_{}".format(
                    fraction, pb)] = forward_time
                features["time_bwd_max_{:.1f}_{}".format(
                    fraction, pb)] = backward_time

        # Count the number of data points with significant positive/negative
        # fluxes
        thresholds = [-20, -10, -5, -3, 3, 5, 10, 20]
        for threshold in thresholds:
            if threshold < 0:
                count = np.sum(s2ns < threshold)
            else:
                count = np.sum(s2ns > threshold)
            features[f"count_s2n_{threshold}"] = count

        # Count the fraction of data points that are "background", i.e. less
        # than a 3 sigma detection of something.
        features["frac_background"] = np.sum(np.abs(s2ns) < 3) / len(s2ns)

        for pb in pbs:
            is_pb = obj_gps_new['filter'] == pb
            obj_pb = obj_gps_new[is_pb].reset_index()
            obj_flux = obj_pb['flux'].values
            obj_flux_error = obj_pb['flux_error'].values

            # Sum up the total signal-to-noise in each band
            total_band_s2n = np.sqrt(np.sum((obj_flux / obj_flux_error) ** 2))
            features[f"total_s2n_{pb}"] = total_band_s2n

            # Calculate percentiles of the data in each band.
            for percentile in (10, 30, 50, 70, 90):
                try:
                    val = np.percentile(obj_flux, percentile)
                except IndexError:
                    val = np.nan
                features[f"percentile_{pb}_{percentile}"] = val

        # Count the time delay between the first and last significant fluxes
        thresholds = [5, 10, 20]
        for threshold in thresholds:
            significant_times = times[np.abs(s2ns) > threshold]
            if len(significant_times) < 2:
                dt = -1
            else:
                dt = np.max(significant_times) - np.min(significant_times)
            features[f"time_width_s2n_{threshold}"] = dt

        # Count how many data points are within a certain number of days of
        # maximum light. This provides some estimate of the robustness of the
        # determination of maximum light and rise/fall times.
        time_bins = [
            (-5, 5, "center"),
            (-20, -5, "rise_20"),
            (-50, -20, "rise_50"),
            (-100, -50, "rise_100"),
            (-200, -100, "rise_200"),
            (-300, -200, "rise_300"),
            (-400, -300, "rise_400"),
            (-500, -400, "rise_500"),
            (-600, -500, "rise_600"),
            (-700, -600, "rise_700"),
            (-800, -700, "rise_800"),
            (5, 20, "fall_20"),
            (20, 50, "fall_50"),
            (50, 100, "fall_100"),
            (100, 200, "fall_200"),
            (200, 300, "fall_300"),
            (300, 400, "fall_400"),
            (400, 500, "fall_500"),
            (500, 600, "fall_600"),
            (600, 700, "fall_700"),
            (700, 800, "fall_800"),
        ]

        diff_times = times - med_time_max
        for start, end, label in time_bins:
            is_in_bin = (diff_times > start) & (diff_times < end)

            # Count how many observations there are in the time bin
            count = np.sum(is_in_bin)
            features[f"count_max_{label}"] = count

            if count == 0:
                bin_mean_fluxes = np.nan
                bin_std_fluxes = np.nan
            else:
                # Measure the GP flux level relative to the peak flux. We do
                # this by taking the median flux in each band and comparing it
                # to the peak flux.
                time_start = med_time_max + start
                time_end = med_time_max + end
                is_in_bin_gp = ((obj_gps_new['mjd'] >= time_start)
                                & (obj_gps_new['mjd'] < time_end))
                scale_gp_fluxes = []
                for pb in pbs:
                    is_pb = obj_gps_new['filter'] == pb
                    obj_pb = obj_gps_new[is_pb & is_in_bin_gp].reset_index()
                    obj_flux = obj_pb['flux'].values
                    scale_gp_fluxes.append(obj_flux / max_fluxes[pb].values[0])
                bin_mean_fluxes = np.mean(scale_gp_fluxes)
                bin_std_fluxes = np.std(scale_gp_fluxes)

            features[f"mean_max_{label}"] = bin_mean_fluxes
            features[f"std_max_{label}"] = bin_std_fluxes

        # Do peak detection on the GP output
        for positive in (True, False):
            for band_idx, pb in enumerate(pbs):
                is_pb = obj_gps_new['filter'] == pb
                obj_pb = obj_gps_new[is_pb].reset_index()
                obj_flux = obj_pb['flux'].values
                if positive:
                    band_flux = obj_flux
                    base_name = f"peaks_pos_{pb}"
                else:
                    band_flux = - obj_flux
                    base_name = f"peaks_neg_{pb}"
                peaks, properties = find_peaks(
                    band_flux, height=np.max(np.abs(band_flux) / 5.0))
                num_peaks = len(peaks)

                features[f"{base_name}_count"] = num_peaks

                sort_heights = np.sort(properties["peak_heights"])[::-1]
                # Measure the fractional height of the other peaks.
                for i in range(1, 3):
                    if num_peaks > i:
                        rel_height = sort_heights[i] / sort_heights[0]
                    else:
                        rel_height = np.nan
                    features[f"{base_name}_frac_{i + 1}"] = rel_height

        return features

    def compute_classification_features(self, raw_features=None):
        """Compute the features used for classification from raw features.

        This method should take a DataFrame or dictionary of raw features,
        produced by `compute_raw_features`, and output a list of processed
        features that can be fed to a classifier.
        The classification features are saved as
        `self.classification_features`.

        Parameters
        ----------
        raw_features : pandas.DataFrame or dict, optional
            The raw features extracted using `compute_raw_features`.

        Returns
        -------
        features : pandas.DataFrame or dict
            The processed features that can be fed to a classifier.

        Raises
        ------
        AttributeError
            The raw features must be provided as input or have been saved in
            `self.raw_features`.

        Notes
        -----
        Function adapted from `avocado` augmentation [1]_ ([2]_).

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        .. [2] Boone, Kyle.
        https://avocado-classifier.readthedocs.io/en/latest/
        """
        if raw_features is None:
            try:
                raw_features = self.raw_features
            except AttributeError:
                raise AttributeError('The raw features must be provided as '
                                     'input or have been saved in '
                                     '`self.raw_features`.')
        rf = raw_features

        # Make a new dict or pandas DataFrame for the features. Everything is
        # agnostic about whether raw_features is a dict or a pandas DataFrame
        # and the output will be the same as the input.
        features = type(rf)()

        # Keys that we want to use directly for classification.
        copy_keys = ["host_photoz", "host_photoz_error"]

        for copy_key in copy_keys:
            features[copy_key] = rf[copy_key]

        features["length_scale"] = rf["gp_fit_1"]

        max_flux = rf["max_flux_lssti"]
        max_mag = -2.5 * np.log10(np.abs(max_flux))

        features["max_mag"] = max_mag

        features["pos_flux_ratio"] = rf["max_flux_lssti"] / (
            rf["max_flux_lssti"] - rf["min_flux_lssti"]
        )
        features["max_flux_ratio_red"] = np.abs(rf["max_flux_lssty"]) / (
            np.abs(rf["max_flux_lssty"]) + np.abs(rf["max_flux_lssti"])
        )
        features["max_flux_ratio_blue"] = np.abs(rf["max_flux_lsstg"]) / (
            np.abs(rf["max_flux_lssti"]) + np.abs(rf["max_flux_lsstg"])
        )

        features["min_flux_ratio_red"] = np.abs(rf["min_flux_lssty"]) / (
            np.abs(rf["min_flux_lssty"]) + np.abs(rf["min_flux_lssti"])
        )
        features["min_flux_ratio_blue"] = np.abs(rf["min_flux_lsstg"]) / (
            np.abs(rf["min_flux_lssti"]) + np.abs(rf["min_flux_lsstg"])
        )

        features["max_dt"] = rf["max_dt_lssty"] - rf["max_dt_lsstg"]

        features["positive_width"] = rf["positive_width_lssti"]
        features["negative_width"] = rf["negative_width_lssti"]

        features["time_fwd_max_0.5"] = rf["time_fwd_max_0.5_lssti"]
        features["time_fwd_max_0.2"] = rf["time_fwd_max_0.2_lssti"]

        features["time_fwd_max_0.5_ratio_red"] = (
            rf["time_fwd_max_0.5_lssty"] / (rf["time_fwd_max_0.5_lssty"]
                                            + rf["time_fwd_max_0.5_lssti"]))
        features["time_fwd_max_0.5_ratio_blue"] = (
            rf["time_fwd_max_0.5_lsstg"] / (rf["time_fwd_max_0.5_lsstg"]
                                            + rf["time_fwd_max_0.5_lssti"]))
        features["time_fwd_max_0.2_ratio_red"] = (
            rf["time_fwd_max_0.2_lssty"] / (rf["time_fwd_max_0.2_lssty"]
                                            + rf["time_fwd_max_0.2_lssti"]))
        features["time_fwd_max_0.2_ratio_blue"] = (
            rf["time_fwd_max_0.2_lsstg"] / (rf["time_fwd_max_0.2_lsstg"]
                                            + rf["time_fwd_max_0.2_lssti"]))

        features["time_bwd_max_0.5"] = rf["time_bwd_max_0.5_lssti"]
        features["time_bwd_max_0.2"] = rf["time_bwd_max_0.2_lssti"]

        features["time_bwd_max_0.5_ratio_red"] = (
            rf["time_bwd_max_0.5_lssty"] / (rf["time_bwd_max_0.5_lssty"]
                                            + rf["time_bwd_max_0.5_lssti"]))
        features["time_bwd_max_0.5_ratio_blue"] = (
            rf["time_bwd_max_0.5_lsstg"] / (rf["time_bwd_max_0.5_lsstg"]
                                            + rf["time_bwd_max_0.5_lssti"]))
        features["time_bwd_max_0.2_ratio_red"] = (
            rf["time_bwd_max_0.2_lssty"] / (rf["time_bwd_max_0.2_lssty"]
                                            + rf["time_bwd_max_0.2_lssti"]))
        features["time_bwd_max_0.2_ratio_blue"] = (
            rf["time_bwd_max_0.2_lsstg"] / (rf["time_bwd_max_0.2_lsstg"]
                                            + rf["time_bwd_max_0.2_lssti"]))

        features["frac_s2n_5"] = rf["count_s2n_5"] / rf["count"]
        features["frac_s2n_-5"] = rf["count_s2n_-5"] / rf["count"]
        features["frac_background"] = rf["frac_background"]

        features["time_width_s2n_5"] = rf["time_width_s2n_5"]

        features["count_max_center"] = rf["count_max_center"]
        features["count_max_rise_20"] = (
            rf["count_max_rise_20"] + features["count_max_center"]
        )
        features["count_max_rise_50"] = (
            rf["count_max_rise_50"] + features["count_max_rise_20"]
        )
        features["count_max_rise_100"] = (
            rf["count_max_rise_100"] + features["count_max_rise_50"]
        )
        features["count_max_fall_20"] = (
            rf["count_max_fall_20"] + features["count_max_center"]
        )
        features["count_max_fall_50"] = (
            rf["count_max_fall_50"] + features["count_max_fall_20"]
        )
        features["count_max_fall_100"] = (
            rf["count_max_fall_100"] + features["count_max_fall_50"]
        )

        all_peak_pos_frac_2 = [rf["peaks_pos_lsstu_frac_2"],
                               rf["peaks_pos_lsstg_frac_2"],
                               rf["peaks_pos_lsstr_frac_2"],
                               rf["peaks_pos_lssti_frac_2"],
                               rf["peaks_pos_lsstz_frac_2"],
                               rf["peaks_pos_lssty_frac_2"]]

        with np.warnings.catch_warnings():
            np.warnings.filterwarnings("ignore", r"All-NaN slice encountered")
            features["peak_frac_2"] = np.nanmedian(all_peak_pos_frac_2, axis=0)

        features["total_s2n"] = np.sqrt(rf["total_s2n_lsstu"] ** 2
                                        + rf["total_s2n_lsstg"] ** 2
                                        + rf["total_s2n_lsstr"] ** 2
                                        + rf["total_s2n_lssti"] ** 2
                                        + rf["total_s2n_lsstz"] ** 2
                                        + rf["total_s2n_lssty"] ** 2)

        all_frac_percentiles = []
        for percentile in (10, 30, 50, 70, 90):
            frac_percentiles = []
            for pb in self.filter_set:
                percentile_flux = rf[f"percentile_{pb}_{percentile}"]
                max_flux = rf[f"max_flux_{pb}"]
                min_flux = rf[f"min_flux_{pb}"]
                frac_percentiles.append(
                    (percentile_flux - min_flux) / (max_flux - min_flux)
                )
            all_frac_percentiles.append(np.nanmedian(frac_percentiles, axis=0))

        features["percentile_diff_10_50"] = (all_frac_percentiles[0]
                                             - all_frac_percentiles[2])
        features["percentile_diff_30_50"] = (all_frac_percentiles[1]
                                             - all_frac_percentiles[2])
        features["percentile_diff_70_50"] = (all_frac_percentiles[3]
                                             - all_frac_percentiles[2])
        features["percentile_diff_90_50"] = (all_frac_percentiles[4]
                                             - all_frac_percentiles[2])

        self.classification_features = features
        return features

    def _load_gp(self, obj, return_gp_params=False):
        """Load the Gaussian Process predict object.

        Parameters
        ----------
        obj : str
            Name of the original event.
        return_gp_params : bool, optional (False)
            Whether to return the used GP fit parameters.

        Returns
        -------
        gp_params : numpy.ndarray, optional
            The resulting GP fit parameters.
        gp_predict: functools.partial with bound method GP.predict
            Function to predict the Gaussian Process flux and uncertainty at
            any time and wavelength.
        """
        # The name format of the saved Gaussian Processes is hard coded
        path_saved_obj_gp = os.path.join(self.path_saved_gps,
                                         f'used_gp_{obj}.pckl')
        with open(path_saved_obj_gp, 'rb') as input:
            gp_predict = pickle.load(input)
        if return_gp_params:
            path_saved_obj_gp_params = os.path.join(self.path_saved_gps,
                                                    f'used_params_{obj}.pckl')
            with open(path_saved_obj_gp_params, 'rb') as input:
                gp_params = pickle.load(input)
            return gp_params, gp_predict
        else:
            return gp_predict

    @staticmethod
    def _exists_path(path_to_test):
        """Check if the inputed path exists.

        Parameters
        ----------
        path_to_test: str
            Path to test the existence.

        Raises
        ------
        ValueError
            If the provided path does not exist.
        """
        exists_path = os.path.exists(path_to_test)
        if not exists_path:
            raise ValueError('The path {} does not exist. Provide a valid path'
                             '.'.format(path_to_test))

    @staticmethod
    def _compute_time_to_fractions(fluxes, fractions, forward=True):
        """Compute the time to decline to a fraction of maximum flux.

        Find the time for a lightcurve to decline to specific fractions of
        maximum flux.

        Parameters
        ----------
        fluxes : numpy.array
            A list of the fluxes predicted by the Gaussian Processes.
        fractions : list
            A decreasing list of the fractions of maximum light that will be
            computed (eg: [0.8, 0.5, 0.2]).
        forward : bool, optional (True)
            If True (default), look forward in time. Otherwise, look backward.

        Returns
        -------
        times : numpy.array
            A list of times for the lightcurve to decline to each of the given
            fractions of maximum flux.

        Notes
        -----
        Function adapted from `avocado` augmentation [1]_ ([2]_).

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        .. [1] Boone, Kyle.
        https://avocado-classifier.readthedocs.io/en/latest/
        """
        max_time = np.argmax(fluxes)
        max_flux = fluxes[max_time]

        times = np.zeros(len(fractions))
        times[:] = np.nan

        frac_idx = 0

        # Start at maximum light, and move along the spectrum. Add every
        # threshold crossed to the list. If a given threshold is not crossed,
        # return a large number for that time.
        offset = 0
        while True:
            offset += 1
            if forward:
                new_time = max_time + offset
                if new_time >= fluxes.shape:
                    break
            else:
                new_time = max_time - offset
                if new_time < 0:
                    break

            test_flux = fluxes[new_time]
            while test_flux < max_flux * fractions[frac_idx]:
                times[frac_idx] = offset
                frac_idx += 1
                if frac_idx == len(fractions):
                    break

            if frac_idx == len(fractions):
                break
        return times

    def save_avo_features(self, features, path_save_features='output_root',
                          file_name=None):
        """Save the processed avocado features that can be fed to a classifier.

        Parameters
        ----------
        features : pandas.DataFrame
            The processed features that can be fed to a classifier.
        path_save_reduced_features : {'output_root', str}, optional
            Path where the avocado features is saved. By default, it is saved
            in `self.output_root`.
        file_name : {None, str}, optional
            Name of the dimensionality reduced wavelet features file. By
            default (None), it is `avocado_features.pckl`.
        """
        if file_name is None:
            file_name = 'avocado_features.pckl'
        if path_save_features == 'output_root':
            path_save_features = self.output_root

        path_save_file = os.path.join(path_save_features, file_name)
        features.to_pickle(path_save_file)
