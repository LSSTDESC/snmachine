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
from astropy.table import Table, vstack, hstack, join
from functools import partial
from iminuit import Minuit
from multiprocessing import Pool
from scipy.interpolate import interp1d
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
                    if (p < model.limits[model.param_names[i]][0] or
                        p > model.limits[model.param_names[i]][1]):
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
                for p in model.param_names:
                    input_args['limit_'+p] = model.limits[p]

                m = Minuit(mini_func, pedantic=False, print_level=0,
                           forced_parameters=model.param_names, **input_args)

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
        # TODO: CAT: Why do we have this empty method? Either we write that is
        # not implemented and then it is overwritten or erase it
        pass

    def fit_sn(self):
        # TODO: CAT: Why do we have this empty method? Either we write that is
        # not implemented and then it is overwritten or erase it
        pass

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
                          'Ib':'nugent-sn1bc', 'Ic': 'nugent-sn1bc'}
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
            Number of processors to use for parallelisation (shared memory only)
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
            params = ['['+mod_name+']'+pname for pname in self.model.param_names]
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
                            res, fitted_model = sncosmo.fit_lc(
                                lc, self.model, vparam_names=prms, bounds=bnds,
                                minsnr=0)
                        else:
                            res, fitted_model = sncosmo.fit_lc(
                                lc, self.model, minsnr=0,
                                vparam_names=self.model.param_names,
                                bounds=self.bounds[self.templates[mod_name]])
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
        tab = features[features['Object']==obj]
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
        """
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

                pos = [iniparams + walker_spread*np.randn(n_params) for i in range(n_walkers)]

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
        if ((np.any(params > self.model.upper_limit)) or
            (np.any(params < self.model.upper_limit))):
            return -np.inf
        else:
            ynew = self.model.evaluate(x, params)
            chi2 = np.sum((y-ynew)*(y-ynew)/yerr/yerr)
            return -chi2/2.

    def goodness_of_fit(self, d):
        """
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

    def __init__(self, wavelet_name='sym2', number_gp=100, **kwargs):
        """Initialises the pywt wavelet object and sets the maximum depth for
        deconstruction.

        Parameters
        ----------
        wavelet : str, optional
            String for which wavelet family to use.
        number_gp : int, optional
            Number of points on the Gaussian process curve
        level : int, optional
            The maximum depth for wavelet deconstruction. If not provided,
            will use the maximum depth possible given the number of points in
            the Gaussian process curve.
        """
        Features.__init__(self)

    def compute_wavelet_decomp(self, dataset, wavelet_name='sym2',
                               number_decomp_levels='max', output_root=None,
                               path_saved_gp_files=None):
        """Computes the wavelet decomposition of the dataset events.

        The function assumes the Gaussian Processes fit of the events has been
        done and that the estimated flux at regular intervals is saved in
        `dataset.models` or in the path provided on `path_saved_gp_files`.
        """
        self._filter_set = dataset.filter_set
        self._number_gp = self._extract_number_gp(dataset)
        self._is_wavelet_valid(wavelet_name)
        self.number_decomp_levels = number_decomp_levels
        self.output_root = output_root

        self._read_gps_into_models(dataset, path_saved_gp_files)

        objs = dataset.object_names
        for i in range(len(objs)):
            obj = objs[i]
            obj_gps = dataset.models[obj].to_pandas()
            coeffs = self._compute_obj_wavelet_decomp(obj_gps)

            if self.output_root is not None:
                self._save_obj_wavelet_decomp(obj, coeffs)

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
            coeffs[pb] = pywt.swt(obj_pb['flux'], wavelet=self.wavelet_name,
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

    def compute_eigendecomp(self, dataset, normalise_var=False,
                            path_save_eigendecomp=None):
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
            variance.
        """
        feature_space = self._load_feature_space(dataset)
        # Center the feature_space to perform eigendecomposition
        feature_space_new, means, scales = self._center_matrix(
            feature_space, normalise_var=normalise_var)

        # Row ith has the eigenvector corresponding to the ith eigenvalue
        # (eigenvectors in descending order of eigenvalue)
        u, singular_vals, eigenvecs = np.linalg.svd(feature_space_new,
                                                    full_matrices=False)

        number_objs = np.shape(feature_space)[0]
        eigenvals = singular_vals**2 / (number_objs-1)

        if path_save_eigendecomp is not None:
            path_save = path_save_eigendecomp
            np.save(os.path.join(path_save, 'means.npy'), means)
            np.save(os.path.join(path_save, 'scales.npy'), scales)
            np.save(os.path.join(path_save, 'eigenvalues.npy'), eigenvals)
            np.save(os.path.join(path_save, 'eigenvectors.npy'), eigenvecs)

    @staticmethod
    def load_pca(path_save_eigendecomp, number_comps=None):
        """Load the principal component analysis of the feature space.

        Load the means and scales needed to transform a matrix onto the same
        space as the matrix used to calculate the eigendecomposition.
        Load the first `number_comps` eigenvalues to transform any new data
        onto a lower-dimensional space.

        Parameters
        ----------
        path_save_eigendecomp : str
            Path where the eigendecomposition is saved.
        number_comps : {None, int}, optional
            If `None`, all eigenvectors are returned. Otherwise, only the
            first `number_comps` are returned.

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
        means = np.load(os.path.join(path_save_eigendecomp, 'means.npy'))
        scales = np.load(os.path.join(path_save_eigendecomp, 'scales.npy'))
        eigenvecs = np.load(os.path.join(path_save_eigendecomp,
                            'eigenvectors.npy'))

        if number_comps is not None:
            eigenvecs = eigenvecs[:number_comps, :]
        return means, scales, eigenvecs

    def _load_feature_space(self, dataset):
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
        feature_space : array
            Table of shape (# events, # wavelet features).
            Row `i` has the wavelet features of the event
            `dataset.object_names[i]`.
            The wavelet features are ordered as a flat version of `pywt.swt`
            function:
                [cAn, cDn, ..., cA2, cD2, cA1, cD1]
            where n equals the number of decomposition levels.
        """
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
        return feature_space

    def project_to_space(self, feature_space, path_save_eigendecomp,
                         number_comps):
        """Project dataset onto a previously calculated feature space.

        The feature space correspond to the wavelet decomposition of the
        events to use. It has not been centered/ scaled yet.

        Parameters
        ----------
        feature_space : array
            Table of shape (# events, # features).
            Row `i` has the wavelet features of the event
            `dataset.object_names[i]`.
            The wavelet features are ordered as a flat version of `pywt.swt`
            function:
                [cAn, cDn, ..., cA2, cD2, cA1, cD1]
            where n equals the number of decomposition levels.
        path_save_eigendecomp : str
            Path where the eigendecomposition is saved.
        number_comps : {None, int}
            If `None`, the same feature space is returned. Otherwise, a
            reduced feature space with `number_comps` features is returned.

        Returns
        -------
        red_space : array
            Projection of the events onto a lower dimensional space of size
            `number_comps`. It is then the reduced feature space.
            Shape (# events, `number_comps`).
        """
        means, scales, eigenvecs = self.load_pca(path_save_eigendecomp,
                                                 number_comps)

        feature_space_new = self._preprocess_matrix(feature_space, means,
                                                    scales)

        red_space = feature_space_new @ eigenvecs.T
        return red_space

    def reconstruct_feature_space(self, red_space, path_save_eigendecomp,
                                  number_comps):
        """Reconstruct the original feature space from the reduced one.

        Parameters
        ----------
        red_space : array
            Projection of the events onto a lower dimensional space of size
            `number_comps`. It is then the reduced feature space.
            Shape (# events, `number_comps`).
        path_save_eigendecomp : str
            Path where the eigendecomposition is saved.
        number_comps : {None, int}
            If `None`, the same feature space is returned. Otherwise, a
            reduced feature space with `number_comps` features is returned.

        Returns
        -------
        reconstruct_space : array
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
        means, scales, eigenvecs = self.load_pca(path_save_eigendecomp,
                                                 number_comps)
        reconstruct_space = red_space @ eigenvecs
        reconstruct_space = self._postprocess_matrix(reconstruct_space, means,
                                                     scales)
        return reconstruct_space

    def reconstruct_real_space(self, dataset, feature_space,
                               wavelet_name='sym2'):
        """Reconstruct the observations in real space from the feature space.

        Parameters
        ----------
        dataset : Dataset object (sndata class)
            Dataset to work with.
        feature_space : array
            Table of shape (# events, # features).
            Row `i` has the wavelet features of the event
            `dataset.object_names[i]`.
            The wavelet features are ordered as a flat version of `pywt.swt`
            function:
                [cAn, cDn, ..., cA2, cD2, cA1, cD1]
            where n equals the number of decomposition levels.
        wavelet_name : str
            Name of the wavelets used.
        """
        self._filter_set = dataset.filter_set
        self._number_gp = self._extract_number_gp(dataset)
        self._is_wavelet_valid(wavelet_name)

        # Feature space has dimensions:
        #  # passbands * # levels * 2 * # gp evaluations
        denominator = 2 * self.number_gp * len(self.filter_set)
        self.number_decomp_levels = np.shape(feature_space)[1] / denominator

        objs = dataset.object_names
        for i in range(len(objs)):
            obj = objs[i]
            obj_gps = dataset.models[obj].to_pandas()
            obj_coeffs_list = feature_space[i]
            obj_gps_reconstruct = self.reconstruct_obj_real_space(
                obj_gps, obj_coeffs_list)
            dataset.models[obj] = obj_gps_reconstruct

    def reconstruct_obj_real_space(self, obj_gps, coeffs_list):
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
        obj_gps['flux_reconstruct'] = None  # initialize the column
        for pb in self.filter_set:
            is_pb_obs = obj_gps['filter'] == pb
            pb_coeffs = obj_coeffs[pb]
            pb_flux_reconstruct = pywt.iswt(pb_coeffs, self.wavelet_name)
            obj_gps['flux_reconstruct'][is_pb_obs] = pb_flux_reconstruct
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
        for i in number_pbs:
            pb = self.filter_set[i]
            new_coeff_format = []
            pb_coeffs_list = np.split(coeffs_list[i],
                                      self.number_decomp_levels)
            for level in np.arange(self.number_decomp_levels):
                level_coeffs_list = np.split(pb_coeffs_list[level], 2)
                new_coeff_format.append(level_coeffs_list)
            coeffs[pb] = new_coeff_format
        return coeffs

    def compute_reconstruct_error(self, dataset, **kwargs):
        """X^2/datapoints between original and reconstructed observations.

        The original observations are the points at which the Gaussian Process
        (GP) was evaluated. The uncertainty is then the GP uncertainty at
        those points.

        Parameters
        ----------
        dataset : Dataset object (sndata class)
            Dataset to work with.

        Returns
        -------
        chisq_over_datapoints : pandas.DataFrame
            Table with the X^2/datapoints per object.
        """
        objs = dataset.object_names
        chisq_over_datapoints = np.zeros(len(objs))
        for i in range(len(objs)):
            obj = objs[i]
            obj_gps = dataset.models[obj]
            try:
                obj_reconstruct = np.copy(obj_gps)
            except KeyError:
                obj_coeffs_list = kwargs['feature_space'][i]
                obj_reconstruct = self.reconstruct_obj_real_space(
                    obj_gps, obj_coeffs_list)
            except KeyError:
                raise KeyError('If the reconstructed flux was not saved into '
                               'the dataset models, the reconstructed feature '
                               'space must be given as `**kwargs` with the '
                               'name `feature_space`.')
            obj_reconstruct['flux'] = obj_gps['flux_reconstruct']
            chisq_over_datapoints[i] = (
                chisq.compute_overall_chisq_over_datapoints(obj_gps,
                                                            obj_reconstruct))
        chisq_over_datapoints = pd.DataFrame(data=chisq_over_datapoints,
                                             index=objs,
                                             columns=['chisq_over_datapoints'])
        chisq_over_datapoints.index.rename('object_id', inplace=True)
        return chisq_over_datapoints

    @staticmethod
    def _postprocess_matrix(matrix, means, scales):
        """Add the features' mean and scales back their variance.

        Change the matrix from the feature space where the
        eigendecomposition was calculated to the original feature space.

        Parameters
        -----------
        matrix : array
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
            Postprocessed matrix of shape (# samples, # features).
        """
        matrix_new = matrix + means
        if scales is not None:
            matrix_new *= scales

        return matrix_new, means, scales

    @staticmethod
    def _preprocess_matrix(matrix, means, scales):
        """Subtract the features' mean and scales their variance.

        Change the matrix to be in the same feature space the
        eigendecomposition was calculated.

        Parameters
        -----------
        matrix : array
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

        return matrix_new, means, scales

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
        matrix_new : array
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
            scales = np.sqrt(np.sum(matrix_new**2, axis=0))
            matrix_new /= scales

        return matrix_new, means, scales

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
        """
        if len(dataset.models) != 0:
            pass
        elif path_saved_gp_files is not None:
            gps.read_gp_files_into_models(dataset, path_saved_gp_files)
        else:
            raise AttributeError('The Gaussian Processes fit of the events '
                                 'must have been done and that the estimated '
                                 'flux at regular intervals is saved in '
                                 '`dataset.models` or in the path provided on '
                                 '`path_saved_gp_files`.')

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
            self._wavelet_name = pywt.Wavelet(wavelet_name)
        except ValueError:
            print('Unknown wavelet name {}. Check `pywt.wavelist()` for the '
                  'list of available builtin wavelets.'.format(wavelet_name))
            sys.exit()

    @property
    def wavelet_name(self):
        """Return name of the wavelets used.

        Returns
        -------
        str
            Name of the wavelets used.
        """
        return self._wavelet_name

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
        """
        obj = dataset.object_names[0]
        obj_gps = dataset.models[obj].to_pandas()
        return np.sum(obj_gps == self.filter_set[0])

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
        """
        max_number_levels = pywt.swt_max_level(self.number_gp)
        if value == 'max':
            number_levels = max_number_levels
        elif 1 <= value <= max_number_levels:
            number_levels = value
        else:
            raise ValueError('This dataset can only be decomposed into a '
                             'positive number of levels smaller or equal to {}'
                             '.'.format(max_number_levels))
        print('Each passband will be decomposed in {} levels.'
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
