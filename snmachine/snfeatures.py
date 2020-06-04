"""
Module for feature extraction on supernova light curves.
"""

__all__ = []
# 'get_MAP', 'Features', 'TemplateFeatures', 'ParametricFeatures',
#           'WaveletFeatures']

import os
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pywt
import sncosmo

from . import parametric_models
from .gps import compute_gps
from astropy.table import Table, vstack, hstack, join
from functools import partial
from iminuit import Minuit
from multiprocessing import Pool
from scipy.interpolate import interp1d

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
                               number_decomp_levels='max', output_root=None):
        """Assumes the GPs have already been computed and are stored in
        `dataset.models` or in a specific path.
        """
        self._filter_set = dataset.filter_set
        self._extract_number_gp(dataset)
        self._is_wavelet_name_valid(wavelet_name)
        self.number_decomp_levels = number_decomp_levels

        for i in range(len(dataset.object_names)):
            obj = dataset.object_names[i]
            obj_gps = dataset.models[obj].to_pandas()
            self._compute_obj_wavelet_decomp(obj_gps)

    def _compute_obj_wavelet_decomp(self, obj_gps):
        """stationary wavelet transform"""
        for pb in self.filter_set:
            obj_pb = obj_gps[obj_gps == pb]
            self._compute_pb_wavelet_decomp(obj_pb)

    def _compute_pb_wavelet_decomp(self, obj_pb):
        pb_flux = obj_pb['flux']
        pywt.swt(pb_flux, wavelet=self.wavelet_name, level=self.number_decomp_levels)

    def _extract_number_gp(self, dataset):
        """Extract the number of points the Gaussian Process was evaluated at.

        It assumes all the events and passbands were evaluated in the same
        number of points as supposed. It extracts the number of the points
        from the first passband of the first event.

        Parameters
        ----------
        dataset : Dataset object (sndata class)
            Dataset to augment.
        """
        obj = dataset.object_names[0]
        obj_gps = dataset.models[obj].to_pandas()
        self.number_gp = np.sum(obj_gps == self.filter_set[0])

    def compute_eigendecomp():
        3

    def project_to_space():
        3

    def reconstruct_lc():
        3

    def compute_reconstruct_error():
        3

    @property
    def filter_set(self):
        """Return passbands the events could be observed on.

        Returns
        -------
        list-like
            Passbands in which the events could be observed on.
        """
        return self._filter_set

    def _is_wavelet_name_valid(self, wavelet_name):
        """Check the wavelet name is valid.

        The available families can be checked using `pywt.wavelist()` or going
        to https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html .
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

    def extract_features(self, d, initheta=[500, 20], save_output=False,
                         output_root='features', number_processes=24,
                         restart='none', gp_algo='george', xmin=None,
                         xmax=None, recompute_pca=True, pca_path=None):
        """Applies a wavelet transform followed by PCA dimensionality
        reduction to extract wavelet coefficients as features.

        Parameters
        ----------
        d : Dataset object
            Dataset
        initheta: list-like, optional
            Initial values for theta parameters. These should be roughly the
            scale length in the y & x directions.
        save_output : bool, optional
            Whether or not to save the output
        output_root : str, optional
            Output directory
        number_processes : int, optional
            Number of processors to use for parallelisation (shared memory
            only)
        restart : str, optional
            Either 'none' to start from scratch, 'gp' to restart from saved
            Gaussian processes, or 'wavelet' to restart from saved wavelet
            decompositions (will look in output_root for the previously saved
            outputs).
        log : bool, optional
            Whether or not to take the logarithm of the final PCA components.
            Recommended setting is False (legacy code).

        Returns
        -------
        astropy.table.Table
            Table of features (first column object names, the rest are the PCA
            coefficient values)
        """

        if save_output and not os.path.exists(output_root):
            print('No output directory found; creating output root directory :'
                  '\n{}'.format(output_root))
            subprocess.call(['mkdir', output_root])
        if xmin is None:
            xmin = 0
        if xmax is None:
            xmax = d.get_max_length()

        if restart == 'wavelet':
            wavout, waveout_err = self.restart_from_wavelets(d, output_root)
        else:
            if restart == 'gp':
                self.restart_from_gp(d, output_root)
            else:
                compute_gps(d, self.number_gp, xmin, xmax, initheta,
                            output_root, number_processes, gp_algo=gp_algo,
                            save_output=save_output)

            wavout, waveout_err = self.extract_wavelets(d, self.wav, self.mlev,
                                                        number_processes,
                                                        save_output,
                                                        output_root)
        self.features, vals, vec, mn, s = self.extract_pca(
            d.object_names.copy(), wavout, recompute_pca=recompute_pca,
            pca_path=pca_path, output_root=output_root)

        # Save the PCA information
        self.PCA_eigenvals = vals
        self.PCA_eigenvectors = vec
        self.PCA_mean = mn
        return self.features

    def fit_sn(self, lc, comps, vec,  mn, xmin, xmax, filter_set, waveUni=0):
        """Fits a single object using previously run PCA components. Performs
        the full inverse wavelet transform.

        Parameters
        ----------
        lc : astropy.table.Table
            Light curve
        comps : astropy.table.Table
            The PCA coefficients for each object (i.e. the astropy table of
            wavelet features from by extract_features).
        vec : array-like
            PCA component vectors as array (each column is a vector, ordered
            from most to least significant)
        mn : array-like
            Mean vector
        xmin : float
            The minimum on the x axis (as defined for the original GP
            decomposition)
        xmax : float
            The maximum on the x axis (as defined for the original GP
            decomposition)
        filter_set : list-like
            The full set of filters of the original dataset

        Returns
        -------
        astropy.table.Table
            Fitted light curve
        """

        obj = lc.meta['name']
        filts = np.unique(lc['filter'])
        # The PCA will have been done over the full coefficient set, across
        # all filters
        if waveUni == 0:  # tweak
            try:
                pca_comps = comps[comps['Object'] == obj]
            except KeyError:
                print('No feature set found for', obj)
                return None

            new_comps = np.array([pca_comps[c] for c in pca_comps.columns[1:]]).flatten()
            number_comp = len(new_comps)
            eigs = vec[:, :number_comp]

            coeffs = np.array(np.dot(new_comps, eigs.T)+mn).flatten()

        n = self.mlev*2*self.number_gp
        xnew = np.linspace(xmin, xmax, self.number_gp)
        output = []
        for i in range(len(filter_set)):
            if filter_set[i] in filts:
                if waveUni == 0:
                    filt_coeffs = coeffs[i*n:(i+1)*n]
                    filt_coeffs = filt_coeffs.reshape(self.mlev, 2,
                                                      self.number_gp,
                                                      order='C')
                else:  # tweak things
                    ifFil = comps['filter'] == filter_set[i]
                    filt_coeffs = ((np.array(comps[ifFil]['cA2']),
                                    np.array(comps[ifFil]['cD2'])),
                                   (np.array(comps[ifFil]['cA1']),
                                    np.array(comps[ifFil]['cD1'])))

                ynew = self.iswt(filt_coeffs, self.wav)

                newtable = Table([xnew, ynew, [filter_set[i]]*self.number_gp],
                                 names=['mjd', 'flux', 'filter'],
                                 dtype=['f', 'f', 'U32'])
                if len(output) == 0:
                    output = newtable
                else:
                    output = vstack((output, newtable))
        return output

    def restart_from_gp(self, d, output_root):
        """Allows the restarted of the feature extraction process from
        previously saved Gaussian Process curves.

        Parameters
        ----------
        d : Dataset object
            The same dataset (object) on which the previous GP analysis was
            performed.
        output_root : str
            Location of GP objects
        """
        print('Restarting from stored Gaussian Processes...')
        for obj in d.object_names:
            # Currently it is only working for the train GPs
            fname = os.path.join(output_root, 'gp_train_'+obj)
            try:
                tab = Table.read(fname, format='ascii')
                d.models[obj] = tab
            except:
                try:
                    tab = Table.read(fname, format='fits')
                    d.models[obj] = tab
                except IOError:
                    print('IOError, file ', fname, 'does not exist.')

    def restart_from_wavelets(self, d, output_root):
        """Allows the restarted of the feature extraction process from
        previously saved wavelet decompositions. This allows you to quickly
        try different dimensionality reduction (e.g. PCA) algorithms on the
        wavelets.

        Parameters
        ----------
        d : Dataset object
            The same dataset (object) on which the previous wavelet analysis
            was performed.
        output_root : str
            Location of previously decomposed wavelet coefficients

        Returns
        -------
        wavout : array
            A numpy array of the wavelet coefficients where each row is an
            object and each column a different coefficient
        wavout_err :  array
            A similar numpy array storing the (assuming Gaussian) error on
            each coefficient.
        """
        print('Restarting from stored wavelets...')
        initial_time = time.time()
        nfilts = len(d.filter_set)
        # `wavout` is just a very big array holding coefficients in memory
        wavout = np.zeros([len(d.object_names),
                           self.number_gp*2*self.mlev*nfilts])
        wavout_err = np.zeros([len(d.object_names),
                               self.number_gp*2*self.mlev*nfilts])

        for i in range(len(d.object_names)):
            obj = d.object_names[i]
            fname = os.path.join(output_root, 'wavelet_'+obj)
            try:
                # out=Table.read(fname, format='ascii')
                out = Table.read(fname, format='fits')
                cols = out.colnames[:-1]
                n = self.number_gp*2*self.mlev
                for j in range(nfilts):
                    # I think I can do this in a more clear/ easy to
                    # understand way
                    x = out[out['filter'] == d.filter_set[j]]  # select the filter
                    coeffs = x[cols[:self.mlev*2]]  # select the coeeficients ['cA2', 'cD2', 'cA1', 'cD1'] of that filter
                    coeffs_err = x[cols[self.mlev*2:]]
                    newcoeffs = np.array([coeffs[c] for c in coeffs.columns]).T # (np.shape(newcoeffs) = 100, 4)
                    newcoeffs_err = np.array([coeffs_err[c] for c in coeffs_err.columns]).T
                    wavout[i, j*n:(j+1)*n] = newcoeffs.flatten('F') # [cA2 cD2 cA1 cD1]
                    wavout_err[i, j*n:(j+1)*n] = newcoeffs_err.flatten('F')

            except IOError:
                print('IOError, file ', fname, 'does not exist.')
        final_time = time.time()
        time_spent = pd.to_timedelta(int(final_time-initial_time), unit='s')
        print('Time spent: {}.'.format(time_spent))
        return wavout, wavout_err

    def wavelet_decomp(self, lc, wav, mlev):
        """Perform a wavelet decomposition on a single light curve.

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
        filters = np.unique(lc['filter'])
        # Store the output in another astropy table
        output = 0
        for fil in filters:
            y = lc['flux'][lc['filter'] == fil]
            err = lc['flux_error'][lc['filter'] == fil]
            coeffs = np.array(pywt.swt(y, wav, level=mlev))
            coeffs_err = np.array(pywt.swt(err, wav, level=mlev))  # This actual gives a slight overestimate of the error

            # Create the column names (follows pywt convention)
            labels = []
            for i in range(len(coeffs)):
                labels.append('cA%d' % (len(coeffs)-i))
                labels.append('cD%d' % (len(coeffs)-i))

            # For the erors
            err_labels = []
            for i in range(len(labels)):
                err_labels.append(labels[i]+'_err')
            npoints = len(coeffs[0, 0, :])
            c = coeffs.reshape(mlev*2, npoints).T
            c_err = coeffs_err.reshape(mlev*2, npoints).T
            newtable1 = Table(c, names=labels)
            newtable2 = Table(c_err, names=err_labels)
            joined_table = hstack([newtable1, newtable2])
            # Add the filters
            joined_table['filter'] = [fil]*npoints

            if output == 0:
                output = joined_table
            else:
                output = vstack((output, joined_table))
        return output

    def extract_wavelets(self, d, wav, mlev, number_processes, save_output,
                         output_root):
        """Perform wavelet decomposition on all objects in dataset. Output is
        stored as astropy table for each object.

        Parameters
        ----------
        d : Dataset object
            Dataset
        wav : str or swt.Wavelet object
            Which wavelet family to use
        mlev : int
            Max depth
        number_processes : int, optional
            Number of processors to use for parallelisation (shared memory
            sonly)
        save_output : bool, optional
            Whether or not to save the output
        output_root : str, optional
         Output directory

        Returns
        -------
        wavout : array
            A numpy array of the wavelet coefficients where each row is an
            object and each column a different coefficient
        wavout_err :  array
            A numpy array storing the (assuming Gaussian) error on each
            coefficient.
        """
        print('Performing wavelet decomposition')
        nfilts = len(d.filter_set)
        wavout = np.zeros([len(d.object_names), self.number_gp*2*mlev*nfilts]) # This is just a big array holding coefficients in memory
        wavout_err = np.zeros([len(d.object_names),
                               self.number_gp*2*mlev*nfilts])
        t1 = time.time()
        for i in range(len(d.object_names)):
            obj = d.object_names[i]
            lc = d.models[obj]
            out = self.wavelet_decomp(lc, wav, mlev)
            if save_output:
                # out.write(os.path.join(output_root, 'wavelet_'+obj), format='ascii')
                out.write(os.path.join(output_root, 'wavelet_'+obj),
                          format='fits', overwrite=True)
            # We go by filter, then by set of coefficients
            cols = out.colnames[:-1]
            n = self.number_gp*2*mlev
            filts = np.unique(lc['filter'])
            for j in range(nfilts):
                if d.filter_set[j] in filts:
                    x = out[out['filter'] == d.filter_set[j]]
                    coeffs = x[cols[:mlev*2]]
                    coeffs_err = x[cols[mlev*2:]]
                    newcoeffs = np.array([coeffs[c] for c in coeffs.columns]).T
                    newcoeffs_err = np.array([coeffs_err[c] for c in coeffs_err.columns]).T
                    wavout[i, j*n:(j+1)*n] = newcoeffs.flatten('F')
                    wavout_err[i, j*n:(j+1)*n] = newcoeffs_err.flatten('F')
        print('Time for wavelet decomposition', time.time()-t1)
        return wavout, wavout_err

    @staticmethod
    def get_svd(X):
        """Obtain Singular Value Decomposition of X, such that X =  U SDiag VT

        Parameters
        ----------
        X : `np.ndarray`
            Reduced Data Matrix that is centered and normalized of
            shape (Nsamps, Nfeats)

        Returns
        -------
        U : `np.ndarray`
            Left Singular Matrix of shape (Nsamps, min(Nsamps, Nfeats))
        SDiag : `np.ndarray`
            Singular values in an array of shape (,min(Nsamps, Nfeats))
        VT : `np.ndarray`
            Transpose of Right Singular Matrix of shape
            (min(Nfeats, Nsamps), Nfeats).

        """
        return np.linalg.svd(X, full_matrices=False)

    def get_pca_eigendecomposition(self, data_matrix, number_comp=None,
                                   tol=0.999, normalize_variance=False):
        """Perform Principal Component Analysis using an eigendecomposition

        Parameters
        ----------
        data_matrix : `np.ndarray` of shape (Nsamps, Nfeats)
            Data Matrix
        number_comp : int, defaults to `None`
            Number of components of PCA to keep. If `None`
            gets determined from a tolerance level.
        tol: `np.float`
            tolerance level above which the explained variance must be
            to determine the number of principal components number_comp to
            keep.
            Only used if `number_comp` is `None`
        normalize_variance : Bool, defaults to `False`
            If `True` pass to `normalize_variance` method so that the features
            are scaled to have unit variance.

        Returns
        -------
        vecs : `np.ndarray` of shape (Nsamps, number_comp)
            `number_comp` PCA basis vectors
        Z : `np.ndarray`
            Components of the vectors forming the data matrix in the PCA bases
            of shape (Nsamps, number_comps)
        M : `np.ndarray`
            Means of the features of the data matrix over the samples, should
            have shape (Nfeats,)
        s : `np.ndarray`
            scalings used to rescale X so that the variance of each feature in
            X is 1. Should have shape (Nfeats, ) or be `None`
        vals : `np.ndarray` of size number_comp
            The highest `number_comp` eigenvalues of the covariance matrix in
            descending order

        Notes
        -----
        normalize_variance defaults to False. Please read notes in
        `normalize_datamatrix` on `normalize_variance`.
        """
        # We are dealing with data matrix of shape (Nsamps, Nfeats)
        err_msg = 'data_matrix input to svd function expected to be 2D'
        assert len(data_matrix.shape) == 2, err_msg  # Sanity check

        # perform eigendecomposition on covariance

        X, M, s = self.normalize_datamatrix(
            data_matrix, normalize_variance=normalize_variance)

        # This is the same as np.dot(D.T, D) / (N-1) if D is centered
        cov = np.dot(X.T, X)

        # Symmetric, numpy method gives these in ascending order of eigenvalues
        # Change to descending order
        vals, vecs = np.linalg.eigh(cov)
        vals = vals[::-1]
        vecs = vecs[:, ::-1]

        # Components in the principal component basis.
        Z = np.dot(X, vecs[:, :number_comp])

        return vecs[:, :number_comp], Z, M, s, vals[:number_comp]

    def get_pca_svd(self, data_matrix, number_comp=None, tol=0.999,
                    normalize_variance=False):
        """Perform Principal Component Analysis of `data_matrix` using
        Singular Value Decomposition.

        Parameters
        ----------
        data_matrix : `np.ndarray`
            Data Matrix
        number_comp : int, defaults to `None`
            Number of components of PCA to keep. If `None`
            gets determined from a tolerance level.
        tol: `np.float`
            tolerance level above which the explained variance must be
            to determine the number of principal components number_comp to
            keep. Only used if `number_comp` is `None`
        normalize_variance : Bool, defaults to `False`
            If `True` pass to `normalize_variance` method so that the features
            are scaled to have unit variance.

        Returns
        -------
        V : `np.ndarray`
            Right Singular Matrix, with shape
            (Nsamps, min(`number_comp`, Nfeats))
        Z : `np.ndarray`
            Components of the vectors forming the data matrix in the PCA bases
            of shape (Nsamps, `number_comp`)
        M : `np.ndarray`
            Means of the features of the data matrix over the samples, should
            have shape (Nfeats,)
        s : `np.ndarray`
            scalings used to rescale X so that the variance of each feature in
            X is 1. Should have shape (Nfeats, ) or be `None`
        eigenvalues : `np.ndarray`
            eigenvalues corresponding to the retained components in descending
            order. Only as many as the number of components kept. Of size
            `number_comp`

        Notes
        -----
        `normalize_variance defaults` to False. Please read notes in
        `normalize_datamatrix` on `normalize_variance`.
        """
        # We are dealing with data matrix of shape (Nsamps, Nfeats)
        err_msg = 'data_matrix input to svd function has wrong shape'
        assert len(data_matrix.shape) == 2, err_msg  # Sanity check

        # perform SVD on normalized Data Matrix
        X, M, s = self.normalize_datamatrix(
            data_matrix, normalize_variance=normalize_variance)
        U, sDiag, VT = self.get_svd(X)
        assert len(U.shape) == 2

        # eigenvals in descending order
        vals = sDiag * sDiag # shape = (nsamples, nsamples)

        # eigenvalues in descending order
        eigenvalues = sDiag * sDiag

        # Find number of components to keep
        if number_comp is None:
            assert isinstance(tol, np.float)  # sanity check (eg. not arrays)
            number_comp = self.number_comps_for_tolerance(eigenvalues, tol=tol)
        else:
            assert isinstance(number_comp, np.int)  # sanity check

        # Coefficients of Data in basis of Principal Components
        Z = np.dot(U[:, :number_comp], np.diag(sDiag[:number_comp]))

        return VT.T[:, :number_comp], Z, M, s, eigenvalues[:number_comp]

    @staticmethod
    def normalize_datamatrix(D, normalize_variance=True):
        """Normalize data matrix for Singular Value Decomposition (SVD) or
        computing covariance.

        This does X = (D - mean(D))/sqrt(N - 1), where N is len(D). D is
        assumed to have shape (Nsamps, Nfeats) while M has shape (1, Nfeats)
        and X has shape (Nsamps, Nfeats). If `normalize_variance` is `True`, X
        is further rescaled to have unit variance.

        Parameters
        -----------
        D : `np.ndarray`
            Data Matrix of shape (Nsamps, Nfeats)
        normalize_variance: Bool, defaults to True
            If True, transform features so that each feature has variance
            of 1.

        Returns
        -------
        X : `np.ndarray`
            normalized and centered data matrix X. Should have shape
            (Nsamps, Nfeats)
        M : `np.ndarray`
            Means of the features of the data matrix over the samples, should
            have shape (Nfeats,)
        s : `np.ndarray`
            scalings used to rescale X so that the variance of each feature in
            X is 1. Should have shape (Nfeats, ) or be `None`

        Notes
        ----
        The option of normalizing variances has been retained for consisitency
        with previous methods and various ML resources which suggest this may
        help in balancing cases where the variances of different features vary
        a lot. In a few tests we have done with specific datasets, we have not
        seen benefits in doing this.
        """
        # We are dealing with data matrix of shape (Nsamps, Nfeats)
        err_msg = 'data_matrix expected to be 2D'
        assert len(D.shape) == 2, err_msg  # Sanity check

        M = D.mean(axis=0)
        N = len(D)
        X = (D - M) / np.sqrt(N - 1)

        s = None
        if normalize_variance:
            s = np.sqrt(np.sum(X**2, axis=0))
            X /= s

        return X, M, s

    @staticmethod
    def number_comps_for_tolerance(vals, tol=.99):
        """
        Determine the minimum number of Principal Components required to
        adequately describe the dataset.

        Parameters
        ----------
        vals : `np.ndarray`
            eigenvalues (ordered largest to smallest)
        tol : np.float, defaults to 0.99
            The fraction of total eigenvalues or variance that must be
            explained

        Returns
        -------
        int
            The required number of coefficients to retain the requested amount
            of "information".

        """
        total = np.sum(vals)
        cum_totals = np.cumsum(vals) / total

        # (max components that would still capture < tol fraction of variance)
        # + 1
        return cum_totals[cum_totals < tol].size + 1

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
        A = np.linalg.lstsq(np.mat(eig_vec), np.mat(X).T)[0].flatten()
        return np.array(A)

    def read_pca(self, pca_path):
        if pca_path is None:
            print('If you want me to use a precomputed PCA, you need to '
                  'provide a path from where I can read it in.')
            # TODO: throw an error
        if not os.path.exists(pca_path):
            print('Invalid path!')
            # TODO: throw an error
        print('Reading in PCA frame from %s ...' % pca_path)
        vals = np.load(os.path.join(pca_path, 'PCA_vals.npy'))
        vec = np.load(os.path.join(pca_path, 'PCA_vec.npy'))
        mn = np.load(os.path.join(pca_path, 'PCA_mean.npy'))
        return vals, vec, mn

    def _pca(self, data_matrix, number_comp, tol, normalize_variance, method):
        """
        Parameters
        ----------
        data_matrix : `np.ndarray`
            Data Matrix

        number_comp : int, defaults to `None`
            Number of components of PCA to keep. If `None`
            gets determined from a tolerance level.
        tol: `np.float`
            tolerance level above which the explained variance must be
            to determine the number of principal components number_comp to
            keep. Only used if `number_comp` is `None`
        normalize_variance : Bool
            If `True` pass to `normalize_variance` method so that the features
            are scaled to have unit variance.
        method : {'svd'| 'eigendecomposition'}

        Notes
        -----
        normalize_variance defaults to False. Please read notes in
        `normalize_datamatrix` on `normalize_variance`.
        """
        if method == 'svd':
            return self.get_pca_svd(data_matrix, number_comp, tol,
                                    normalize_variance)
        elif method == 'eigendecomposition':
            return self.get_pca_eigendecomposition(data_matrix, number_comp,
                                                   tol, normalize_variance)

    @staticmethod
    def reconstruct_datamatrix_lossy(Z, vec, M=None, s=None):
        """Reconstruct (lossily) the original Data Matrix from the data compressed
        by Principal Component Analysis. ie. the coefficients of the
        eigenvectors to represent the data.

        Parameters
        ----------
        Z : `np.ndarray`
            Array of coefficients of the Principal Component Vectors of the
            Normalized Data Matrix. Must have shape (Nsamps, number_comp)
        vec : `np.ndarray`
            Array with normalized retained eigenvectors as columns. Has shape
            (Nfeats, number_comp)
        M : `np.ndarray`, defaults to `None`
            Matrix subtracted from original Data Matrix to center it.
            Must have shape (Nfeats, ). If `None`, M is assumed to be 0
        s : `np.ndarry`
            scale factor applied to normalize data matrix so that each feature
            vector has variance 1. Must have shape (Nfeats, ) or be `None`
        Returns
        -------
        D : `np.ndarray`
            Reconstructed un-normalized data matrix of shape (Nsamps, Nfeats)
            that was compressed via PCA
        """
        # Go to the space of normalized data
        Nsamps, number_comps_ = Z.shape
        Nfeats, number_comps = vec.shape

        # While we have enough information to create a zero matrix of the right
        # shape, we will avoid using the memory.
        if M is not None:
            Nfeats_ = M.size
            assert Nfeats == Nfeats_  # Sanity check

        if s is not None:
            assert s.shape == (Nfeats,)  # Sanity check

        # Sometimes Z may be made an array from an `astropy.table.Table`,
        # with different data types. In this case, the array will be an
        # `np.recarray` which will show number_comps_ = 1, even though the
        # shape is different. This could also happen if the `object_names` are
        # not removed. However, to do the matrix multiplication below, this
        # has to be fixed.

        assert number_comps_ == number_comps  # Sanity check

        X = np.dot(Z, vec.T)

        if s is not None:
            X *= s

        # De-normalize the datamatrix
        D = X * np.sqrt(len(X) - 1)

        if M is not None:
            D += M

        return D

    def extract_pca(self, object_names, wavout, recompute_pca=True,
                    method='svd', number_comp=None, tol=0.999, pca_path=None,
                    save_output=False, output_root=None,
                    normalize_variance=False):
        """Obtain Principal Components from wavelets using Principal Component
        Analysis.

        Parameters
        ----------
        object_names : list-like
            Object names corresponding to each row of the wavelet coefficient
            array.
        wavout : array
            Wavelet coefficient array, each row corresponds to an object, each
            column is a coefficient.
        recompute_pca : Bool, default to `True`
            If `True`, calculate the PCA, `False` should require a valid
            `pca_path` to read
            pca information from
        method: {'svd'|'eigendecomposition'|None} , defaults to `svd`
            strings to pick the SVD or eigenDecompostition method. Ignored if
            `recompute_PCA` is `True`, and may be `None` in that case. `svd`
            invokes the `get_pca_svd` method, while `eigenDecomposition`
            invokes the `get_pca_eigendecomposition` method.
        number_comp: int, defaults to `None`
            Number of components of PCA kept for analysis. If `None`,
            determined internally from `tol` instead.
        tol: float, defaults to 0.99
            fraction of variance that must be explained by retained PCA
            components. To override this and use `number_comp` directly, tol
            should be set to `None`.
        normalize_variance : Bool, defaults to `False`
            If `True` pass to `normalize_variance` method so that the features
            are scaled to have unit variance.

        Returns
        -------
        wavs :`astropy.table.Table`
            table containing PCA features.
        vals : `np.ndarray`
            array of eigenvalues in the descending orders, keeping only
            retained components
        vec : `np.ndarray`
            array of shape (Nfeat, Ncomp) whose columns are the
            eigenvectors of the covariance matrix.
        M : `np.ndarray`
            Means of the features of the data matrix over the samples, should
            have shape (Nfeats,)
        s : `np.ndarray`
            scalings used to rescale X so that the variance of each feature in
            X is 1. Should have shape (Nfeats, ) or be `None`

        Notes
        -----
        normalize_variance defaults to False. Please read notes in
        `normalize_datamatrix` on `normalize_variance`.
        """
        object_names = np.asarray(object_names)
        assert object_names.shape == (wavout.shape[0],)  # Sanity check
        # This is necessary for the creation of `Table` objnames, as shapes
        # must match

        t1 = time.time()

        if recompute_pca:
            method = method.lower()
            if method not in ('svd', 'eigendecomposition'):
                raise NotImplementedError('PCA method not implemented')

            print("OUTPUT ROOT: {}\n".format(output_root))
            print('Running PCA...')

            # PCA on the data matrix wavout after centering
            vec, comps, M, s, vals = self._pca(
                wavout, number_comp=number_comp, tol=tol, method=method,
                normalize_variance=normalize_variance)

            # Get number_comp if run determined by tol, ie. number_comp is
            # `None`
            if number_comp is None:
                number_comp = vals.size
        else:
            # We need to add some reading to make it consistent with new code
            vals, vec, mn = self.read_pca(pca_path)
            M = mn

            # Actually fit the components
            tolerance = tol
            number_comp = self.best_coeffs(vals, tol=tolerance)
            eigs = vec[:, :number_comp]
            print('Number of components used is '+str(number_comp))
            comps = np.zeros([len(wavout), number_comp])

            for i in range(len(wavout)):
                if i % 100 == 0:
                    print('I am still here!! i ='+str(i))
                coeffs = wavout[i]
                a = self.project_pca(coeffs-mn, eigs)
                comps[i] = a
            print('finish projecting PCA')

        # Now reformat the components as a table
        labels = ['C%d' % i for i in range(number_comp)]
        reduced_wavelet_components = Table(comps, names=labels)
        objnames = Table(object_names.reshape(len(object_names), 1),
                         names=['Object'])
        reduced_wavelet_components = hstack((objnames,
                                             reduced_wavelet_components))
        print('Time for PCA', time.time() - t1)

        if save_output:
            # We need to change the output to make it consistent with new code
            np.save(os.path.join(output_root,
                                 'eigenvalues_{}.npy'.format(number_comp)),
                    vals)
            np.save(os.path.join(output_root,
                                 'eigenvectors_{}.npy'.format(number_comp)),
                    vec)
            np.save(os.path.join(output_root,
                                 'comps_{}.npy'.format(number_comp)), comps)
            np.save(os.path.join(output_root,
                                 'means_{}.npy'.format(number_comp)), M)
            # Write the astropy table containing the wavelet features to disk
            # after converting to pandas dataframe
            reduced_wavelet_components = reduced_wavelet_components.to_pandas()
            reduced_wavelet_components.to_pickle(
                os.path.join(output_root,
                'reduced_wavelet_components_{}.pickle'.format(number_comp)))
        return reduced_wavelet_components, vals, vec, M, s

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
        output = coefficients[0][0].copy()  # Avoid modification of input data

        # num_levels, equivalent to the decomposition level, n
        num_levels = len(coefficients)
        for j in range(num_levels, 0, -1):
            step_size = int(2**(j-1))
            last_index = step_size
            _, cD = coefficients[num_levels - j]
            for first in range(last_index):  # 0 to last_index - 1

                # Getting the indices that we will transform
                indices = np.arange(first, len(cD), step_size)

                # select the even indices
                even_indices = indices[0::2]
                # select the odd indices
                odd_indices = indices[1::2]

                # perform the inverse dwt on the selected indices,
                # making sure to use periodic boundary conditions
                x1 = pywt.idwt(output[even_indices], cD[even_indices], wavelet,
                               'per')
                x2 = pywt.idwt(output[odd_indices], cD[odd_indices], wavelet,
                               'per')

                # perform a circular shift right
                x2 = np.roll(x2, 1)

                # average and insert into the correct indices
                output[indices] = (x1 + x2)/2.

        return output
