"""
Module for extracting and saving GPs.
"""

__all__ = []

import os
import pickle
import time

import george
import numpy as np
import pandas as pd
import scipy.optimize as op

from astropy.stats import biweight_location
from astropy.table import Table, vstack
from functools import partial
from multiprocessing import Pool
from snmachine import chisq as cs


# Central passbands wavelengths
pb_wavelengths = {"lsstu": 3685., "lsstg": 4802., "lsstr": 6231.,
                  "lssti": 7542., "lsstz": 8690., "lssty": 9736.}
wavelengths_pb = {v: k for k, v in pb_wavelengths.items()}  # inverted map


def compute_gps(dataset, number_gp, t_min, t_max, output_root=None,
                number_processes=1, gp_dim=1, **kwargs):
    """Runs Gaussian process code on entire dataset.

    The result is stored inside the models attribute of the dataset object.
    The result is also saved on external files, if an output root is given.

    Parameters
    ----------
    dataset : Dataset object (sndata class)
        Dataset.
    number_gp : int
        Number of points to evaluate the Gaussian Process Regression at.
    t_min : float
        Minimim time to evaluate the Gaussian Process Regression at.
    t_max : float
        Maximum time to evaluate the Gaussian Process Regression at.
    output_root : {None, str}, optional
        If None, don't save anything. If str, it is the output directory, so
        save the flux and error estimates and used kernels there.
    number_processes : int, optional
        Number of processors to use for parallelisation (shared memory only).
        By default `number_processes` = 1.
    gp_dim : int, optional
        The dimension of the Gaussian Process. If  `gp_dim` is 1, the filters
        are fitted independently. If `gp_dim` is 2, the Matern kernel is used
        with cross-information between the passbands.
    **kwargs : dict
        Additional keyword arguments that can replace default paremeters in
        other funtions. At the moment, we have:
          - kernel_param : list-like, default = [500., 20.]
                Initial values for kernel parameters. These should be roughly
                the scale length in the y & x directions.
          - do_subtract_background : Bool, default = False
                Whether to estimate a new background subtracting the current.

    Raises
    ------
    ValueError
        The Gaussian Processes Regression must be evaluated beyond the minimum
        and maximum time in the dataset. This requires `t_min` and `t_max` to
        be smaller and greater than the minimum and maximum observations of
        `dataset`, respectively.
    """
    if is_dataset_in_t_range(dataset, t_min=t_min, t_max=t_max) is False:
        raise ValueError('There are events in the dataset with observations '
                         'beyond `t_min` and `t_max`. Increase the time range '
                         'so they can be fully captured by the Gaussian '
                         'Process Regression.')
    print('Performing Gaussian process regression.')
    initial_time = time.time()

    # Check for parallelisation
    if number_processes == 1:  # non parallelizing
        _compute_gps_single_core(dataset, number_gp, t_min, t_max, output_root,
                                 gp_dim, **kwargs)

    else:  # parallelizing
        _compute_gps_parallel(dataset, number_gp, t_min, t_max, output_root,
                              number_processes, gp_dim, **kwargs)

    print('Time taken for Gaussian process regression: {:.2f}s.'
          ''.format(time.time()-initial_time))


def is_dataset_in_t_range(dataset, t_min, t_max):
    """Check if all the events in the dataset are inside a specific time range.

    Parameters
    ----------
    dataset : Dataset object (sndata class)
        Dataset.
    t_min : float
        Minimum of the considerend time range.
    t_max : float
        Maximum of the considerend time range.

    Returns
    -------
    bool
        True if all the events in the dataset are inside the specific time
        range. False otherwise.
    """
    for obj in dataset.object_names:
        obj_data = dataset.data[obj]
        t_min_obj = np.min(obj_data['mjd'])
        t_max_obj = np.max(obj_data['mjd'])
        if (t_min_obj < t_min) or (t_max_obj > t_max):
            return False
    return True


def read_gp_files_into_models(dataset, path_saved_gp_files):
    """Reads the saved files into the dataset models.

    By reading the saved files into the models, we can start from a previously
    saved Gaussian Processes. The files can have been stored as `.ascii` or
    `.fits`.

    Parameters
    ----------
    dataset : Dataset object (sndata class)
        Dataset.
    path_saved_gp_files : str
        Path for the Gaussian Process curve files.
    """
    print('Restarting from stored Gaussian Processes...')
    time_start_reading = time.time()
    for obj in dataset.object_names:
        obj_saved_gps_file = os.path.join(path_saved_gp_files, 'gp_'+obj)
        try:
            obj_saved_gps = Table.read(obj_saved_gps_file, format='ascii')
        except UnicodeDecodeError:
            obj_saved_gps = Table.read(obj_saved_gps_file, format='fits',
                                       character_as_bytes=False)
        except FileNotFoundError:
            print('The file {} does not exist.'.format(obj_saved_gps_file))
        dataset.models[obj] = obj_saved_gps
    print('Models fitted with the Gaussian Processes values.')
    print_time_difference(time_start_reading, time.time())


def _compute_gps_single_core(dataset, number_gp, t_min, t_max, output_root,
                             gp_dim, **kwargs):
    """Computes the Gaussian process code on entire dataset in a single core.

    Parameters
    ----------
    dataset : Dataset object (sndata class)
        Dataset.
    number_gp : int
        Number of points to evaluate the Gaussian Process Regression at.
    t_min : float
        Minimim time to evaluate the Gaussian Process Regression at.
    t_max : float
        Maximum time to evaluate the Gaussian Process Regression at.
    output_root : {None, str}, optional
        If None, don't save anything. If str, it is the output directory, so
        save the flux and error estimates and used kernels there.
    gp_dim : int, optional
        The dimension of the Gaussian Process. If  `gp_dim` is 1, the filters
        are fitted independently. If `gp_dim` is 2, the Matern kernel is used
        with cross-information between the passbands.
    **kwargs : dict
        Additional keyword arguments that can replace default paremeters in
        other funtions. At the moment, we have:
          - kernel_param : list-like, default = [500., 20.]
                Initial values for kernel parameters. These should be roughly
                the scale length in the y & x directions.
          - do_subtract_background : Bool, default = False
                Whether to estimate a new background subtracting the current.
    """
    for i in range(len(dataset.object_names)):
        obj = dataset.object_names[i]
        try:
            obj_gps = _compute_gp_all_passbands(obj, dataset, number_gp, t_min,
                                                t_max, output_root=output_root,
                                                gp_dim=gp_dim, **kwargs)
            dataset.models[obj] = obj_gps
        except ValueError:
            print('Object {} has fallen over!'.format(obj))
    print('Models fitted with the Gaussian Processes values.')


def _compute_gps_parallel(dataset, number_gp, t_min, t_max, output_root,
                          number_processes, gp_dim, **kwargs):
    """Computes the Gaussian process code on entire dataset in a parallel way.

    Parameters
    ----------
    dataset : Dataset object (sndata class)
        Dataset.
    number_gp : int
        Number of points to evaluate the Gaussian Process Regression at.
    t_min : float
        Minimim time to evaluate the Gaussian Process Regression at.
    t_max : float
        Maximum time to evaluate the Gaussian Process Regression at.
    output_root : {None, str}, optional
        If None, don't save anything. If str, it is the output directory, so
        save the flux and error estimates and used kernels there.
    number_processes : int, optional
        Number of processors to use for parallelisation (shared memory only).
        By default `number_processes` = 1.
    gp_dim : int, optional
        The dimension of the Gaussian Process. If  `gp_dim` is 1, the filters
        are fitted independently. If `gp_dim` is 2, the Matern kernel is used
        with cross-information between the passbands.
    **kwargs : dict
        Additional keyword arguments that can replace default paremeters in
        other funtions. At the moment, we have:
          - kernel_param : list-like, default = [500., 20.]
                Initial values for kernel parameters. These should be roughly
                the scale length in the y & x directions.
          - do_subtract_background : Bool, default = False
                Whether to estimate a new background subtracting the current.
    """
    p = Pool(number_processes, maxtasksperchild=10)

    # Pool and map can only really work with single-valued functions
    partial_gp = partial(_compute_gp_all_passbands, dataset=dataset,
                         number_gp=number_gp, t_min=t_min, t_max=t_max,
                         output_root=output_root, gp_dim=gp_dim, **kwargs)

    dataset_gps = p.map(partial_gp, dataset.object_names, chunksize=10)
    p.close()

    for i in range(len(dataset.object_names)):
        obj = dataset.object_names[i]
        obj_gps = dataset_gps[i]
        dataset.models[obj] = obj_gps
    print('Models fitted with the Gaussian Processes values.')


def _compute_gp_all_passbands(obj, dataset, number_gp, t_min, t_max,
                              output_root=None, gp_dim=1, **kwargs):
    """Compute/ Fit a Gaussian process curve in every passband of an object.

    If asked to save the output, it saves the Gaussian process curve in every
    passband of an object and the GP instances and kernel used.
    This function can be used for the 1D fit to each passband independently or
    the 2D cross-band fit.

    Parameters
    ----------
    obj : str
        Name of the object.
    dataset : Dataset object (sndata class)
        Dataset.
    number_gp : int
        Number of points to evaluate the Gaussian Process Regression at.
    t_min : float
        Minimim time to evaluate the Gaussian Process Regression at.
    t_max : float
        Maximum time to evaluate the Gaussian Process Regression at.
    output_root : {None, str}, optional
        If None, don't save anything. If str, it is the output directory, so
        save the flux and error estimates and used kernels there.
    gp_dim : int, optional
        The dimension of the Gaussian Process. If  `gp_dim` is 1, the filters
        are fitted independently. If `gp_dim` is 2, the Matern kernel is used
        with cross-information between the passbands.
    **kwargs : dict
        Additional keyword arguments that can replace default paremeters in
        other funtions. At the moment, we have:
          - kernel_param : list-like, default = [500., 20.]
                Initial values for kernel parameters. These should be roughly
                the scale length in the y & x directions.
          - do_subtract_background : Bool, default = False
                Whether to estimate a new background subtracting the current.

    Returns
    -------
    astropy.table.Table
        Table with evaluated Gaussian process curve and errors at each
        passband.
    """
    # Check for number of Gaussian Processes dimension
    if gp_dim == 1:  # independent passbands
        return _compute_gp_all_passbands_1D(obj, dataset, number_gp, t_min,
                                            t_max, output_root=output_root,
                                            **kwargs)

    elif gp_dim == 2:  # cross-band information
        return _compute_gp_all_passbands_2D(obj, dataset, number_gp, t_min,
                                            t_max, output_root=output_root,
                                            **kwargs)


def _compute_gp_all_passbands_1D(obj, dataset, number_gp, t_min, t_max,
                                 output_root=None, **kwargs):
    """Compute/ Fit a Gaussian process curve in every passband independently.

    If asked to save the output, it saves the Gaussian process curve in every
    passband of an object and the GP instances and kernel used.

    Parameters
    ----------
    obj : str
        Name of the object.
    dataset : Dataset object (sndata class)
        Dataset.
    number_gp : int
        Number of points to evaluate the Gaussian Process Regression at.
    t_min : float
        Minimim time to evaluate the Gaussian Process Regression at.
    t_max : float
        Maximum time to evaluate the Gaussian Process Regression at.
    output_root : {None, str}, optional
        If None, don't save anything. If str, it is the output directory, so
        save the flux and error estimates and used kernels there.
    **kwargs : dict
        Additional keyword arguments that can replace default paremeters in
        other funtions. At the moment, we have:
          - kernel_param : list-like, default = [500., 20.]
                Initial values for kernel parameters. These should be roughly
                the scale length in the y & x directions.

    Returns
    -------
    obj_gps : astropy.table.Table
        Table with evaluated Gaussian process curve and errors at each
        passband.
    """
    try:
        kernel_param = kwargs["kernel_param"]
    except KeyError:
        kernel_param = [500., 20.]

    obj_data = dataset.data[obj]  # object's light curve
    obj_data = cs.rename_passband_column(obj_data.to_pandas())
    unique_pbs = np.unique(obj_data.passband)
    gp_times = np.linspace(t_min, t_max, number_gp)

    # Store the output in another astropy table
    obj_gps = []
    used_gp_dict = {}
    used_kernels_dict = {}
    filter_set = np.asarray(dataset.filter_set)
    for pb in filter_set:
        used_kernels_dict[pb] = None  # inilialize None kernel to each passband
        if pb in unique_pbs:
            is_pb = obj_data.passband == pb  # observations in this passband
            obj_data_pb = obj_data.loc[is_pb]

            gp_obs, gp, chosen_kernel = fit_best_gp(kernel_param,
                                                    obj_data_pb, gp_times)

            mu, std = gp_obs.flux.values, gp_obs.flux_error.values
            # stack the GP results in a array momentarily
            obj_gp_pb_array = np.column_stack((gp_times, mu, std))
            used_kernels_dict[pb] = chosen_kernel
            # Save the GP already conditioned on a specific set of observations
            gp_predict = partial(gp.predict, obj_data_pb.flux)
            used_gp_dict[pb] = gp_predict
        else:
            obj_gp_pb_array = np.zeros([number_gp, 3])
        obj_gp_pb = Table([obj_gp_pb_array[:, 0], obj_gp_pb_array[:, 1],
                           obj_gp_pb_array[:, 2], [pb]*number_gp],
                          names=['mjd', 'flux', 'flux_error', 'filter'])
        if len(obj_gps) == 0:  # initialize the table for 1st passband
            obj_gps = obj_gp_pb
        else:
            obj_gps = vstack((obj_gps, obj_gp_pb))

    if output_root is not None:
        obj_gps.write(os.path.join(output_root, 'gp_'+obj), format='fits',
                      overwrite=True)
        path_save_gps = os.path.join(output_root, 'used_gp_dict_{}.pckl'
                                     ''.format(obj))
        path_save_kernels = os.path.join(output_root, 'used_kernels_dict_{}.'
                                         'pckl'.format(obj))
        with open(path_save_gps, 'wb') as f:
            pickle.dump(used_gp_dict, f, pickle.HIGHEST_PROTOCOL)
        with open(path_save_kernels, 'wb') as f:
            pickle.dump(used_kernels_dict, f, pickle.HIGHEST_PROTOCOL)

    return obj_gps


def fit_best_gp(kernel_param, obj_data, gp_times):
    """Fits Gaussian Processes in a hierarchical way.

    It keeps fitting until it finds one Gaussian Process whose reduced X^2 < 1
    or tried all.

    Parameters
    ----------
    kernel_param : list-like
        Initial values for kernel parameters. These should be roughly the
        scale length in the y & x directions.
    obj_data : pandas.core.frame.DataFrame
        Time, flux and flux error of the data (specific filter of an object).
    gp_times : numpy.ndarray
        Times to evaluate the Gaussian Process at.

    Returns
    -------
    gp_obs : pandas.core.frame.DataFrame
        Time, flux and flux error of the fitted Gaussian Process.
    gp_instance : george.gp.GP
        The GP instance that was used to fit the object.
    kernel_id : str
        The id of the kernel chosen. This can then be used as a feature.
    """
    possible_kernel_ids = ['kernel 0', 'kernel 1', 'kernel 2', 'kernel 3',
                           'kernel 4']
    possible_kernel_names = ['ExpSquared', 'ExpSquared', 'ExpSquared',
                             'ExpSquared+ExpSine2', 'ExpSquared+ExpSine2']
    possible_kernel_params = [[kernel_param[0]**2, kernel_param[1]**2, None,
                               None, None, None, None],
                              [400., 200., None, None, None, None, None],
                              [400., 20., None, None, None, None, None],
                              [400., 20., 2., 4., 4., 6., 6.],
                              [19., 9., 2., 4., 4., 6., 6.]]
    number_diff_kernels = len(possible_kernel_names)
    i = 0  # initializing the while loop
    all_obj_gp = number_diff_kernels*['']  # initializing
    all_gp_instances = number_diff_kernels*['']  # initializing
    # just a random number > 1 to initialize
    all_chisq_over_pts = np.zeros(number_diff_kernels) + 666
    threshold_chisq_over_pts = 2  # because that is a good number
    while ((i < number_diff_kernels) and
           (all_chisq_over_pts[i-1] > threshold_chisq_over_pts)):
        obj_gp, gp_instance = fit_gp(kernel_name=possible_kernel_names[i],
                                     kernel_param=possible_kernel_params[i],
                                     obj_data=obj_data, gp_times=gp_times)
        chisq_over_pts = cs.compute_chisq_over_pts(obj_data, obj_gp)
        kernel_id = possible_kernel_ids[i]
        all_obj_gp[i] = obj_gp
        all_gp_instances[i] = gp_instance
        all_chisq_over_pts[i] = chisq_over_pts
        i += 1
        is_above_threshold = (
            chisq_over_pts > threshold_chisq_over_pts)
        if ((i == number_diff_kernels) and is_above_threshold):
            # All kernels/parameters are bad for this object
            output = _choose_less_bad_kernel(all_obj_gp, all_gp_instances,
                                             all_chisq_over_pts,
                                             possible_kernel_ids)
            obj_gp, gp_instance, chisq_over_pts, kernel_id = output
    return obj_gp, gp_instance, kernel_id


def _choose_less_bad_kernel(all_obj_gp, all_gp_instances,
                            all_chisq_over_pts, possible_kernel_ids):
    """If all kernels give a bad reduced X^2, choose the less bad of them.

    Parameters
    ----------
    all_obj_gp : list-like
        List of the DataFrames containing time, flux and flux error for each
        GP instance in `all_gp_instances`.
    all_gp_instances : list-like
        List of all GP instances used to fit the object.
    all_chisq_over_pts : list-like
        List of the X^2/number of datapoints values for each GP instance in
        `all_gp_instances`.
    possible_kernel_ids : list-like
        List of all the possible kernel ids.

    Returns
    -------
    less_bad_obj_gp : pandas.core.frame.DataFrame
        Time, flux and flux error of the fitted Gaussian Process `gp`.
    less_bad_gp_instance : george.gp.GP
        The GP instance that was used to fit the object. It is the one that
        gives the lower reduced X^2.
    less_bad_chisq_over_pts : float
        The X^2/number of datapoints given by the Gaussian Process `gp`.
    """
    index_min = np.argmin(all_chisq_over_pts)
    less_bad_obj_gp = all_obj_gp[index_min]
    less_bad_gp_instance = all_gp_instances[index_min]
    less_bad_chisq_over_pts = all_chisq_over_pts[index_min]
    less_bad_kernel = possible_kernel_ids[index_min]
    return (less_bad_obj_gp, less_bad_gp_instance,
            less_bad_chisq_over_pts, less_bad_kernel)


def fit_gp(kernel_name, kernel_param, obj_data, gp_times):
    """Fit a Gaussian process curve at evenly spaced points along a light curve.

    Parameters
    ----------
    kernel_name : str
        The kernel to fit the data. It can be ExpSquared or
        ExpSquared+ExpSine2.
    kernel_param : list-like
        Initial values for kernel parameters. These should be roughly the
        scale length in the y & x directions.
    obj_data : pandas.core.frame.DataFrame
        Time, flux and flux error of the data (specific filter of an object).
    gp_times : numpy.ndarray
        Times to evaluate the Gaussian Process at.

    Returns
    -------
    obj_gp : pandas.core.frame.DataFrame
        Time, flux and flux error of the fitted Gaussian Process.
    gp : george.gp.GP
        The GP instance that was used to fit the object.
    """
    obj_times = obj_data.mjd
    obj_flux = obj_data.flux
    obj_flux_error = obj_data.flux_error

    def neg_log_like(p):  # Objective function: negative log-likelihood
        gp.set_parameter_vector(p)
        loglike = gp.log_likelihood(obj_flux, quiet=True)
        return -loglike if np.isfinite(loglike) else 1e25

    def grad_neg_log_like(p):  # Gradient of the objective function.
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(obj_flux, quiet=True)

    kernel = get_kernel(kernel_name, kernel_param)

    gp = george.GP(kernel)
    gp.compute(obj_times, obj_flux_error)
    results = op.minimize(neg_log_like, gp.get_parameter_vector(),
                          jac=grad_neg_log_like, method="L-BFGS-B", tol=1e-6)

    if np.sum(np.isnan(results.x)) != 0:
        # The minimiser reaches a local minimum.
        # Change a bit initial conditions so we don't go to that minima
        kernel_param[4] = kernel_param[4]+.1
        kernel = get_kernel(kernel_name, kernel_param)
        gp = george.GP(kernel)
        gp.compute(obj_times, obj_flux_error)
        results = op.minimize(neg_log_like, gp.get_parameter_vector(),
                              jac=grad_neg_log_like, method="L-BFGS-B",
                              tol=1e-6)

    gp.set_parameter_vector(results.x)
    gp_mean, gp_cov = gp.predict(obj_flux, gp_times)
    obj_gp = pd.DataFrame(columns=['mjd'], data=gp_times)
    obj_gp['flux'] = gp_mean
    if np.sum(np.diag(gp_cov) < 0) == 0:
        obj_gp['flux_error'] = np.sqrt(np.diag(gp_cov))
    else:  # do not choose this kernel
        obj_gp['flux_error'] = 666666
    return obj_gp, gp


def get_kernel(kernel_name, kernel_param):
    """Get the chosen kernel with the given initial conditions.

    Parameters
    ----------
    kernel_name : str
        The kernel to fit the data. It can be ExpSquared or
        ExpSquared+ExpSine2.
    kernel_param : list-like
        Initial values for kernel parameters.

    Returns
    -------
    kernel : george.kernels
        The kernel instance to be used in the Gaussian Process.

    Raises
    ------
    AttributeError
        The only available kernels are 'ExpSquared' and 'ExpSquared+ExpSine2'.
    """
    if kernel_name not in ['ExpSquared', 'ExpSquared+ExpSine2']:
        raise AttributeError("The only available kernels are 'ExpSquared' and "
                             "'ExpSquared+ExpSine2'.")
    kExpSquared = kernel_param[0] * george.kernels.ExpSquaredKernel(
        metric=kernel_param[1])
    if kernel_name == 'ExpSquared':
        kernel = kExpSquared
    elif kernel_name == 'ExpSquared+ExpSine2':
        kExpSine2 = kernel_param[4] * george.kernels.ExpSine2Kernel(
            gamma=kernel_param[5], log_period=kernel_param[6])
        kernel = kExpSquared + kExpSine2
    return kernel


def _compute_gp_all_passbands_2D(obj, dataset, number_gp, t_min, t_max,
                                 output_root=None, **kwargs):
    """Compute/ Fit a Gaussian process curve in every passband jointly.

    If asked to save the output, it saves the Gaussian process curve in every
    passband of an object and the GP instances and kernel used.

    Parameters
    ----------
    obj : str
        Name of the object.
    dataset : Dataset object (sndata class)
        Dataset.
    number_gp : int
        Number of points to evaluate the Gaussian Process Regression at.
    t_min : float
        Minimim time to evaluate the Gaussian Process Regression at.
    t_max : float
        Maximum time to evaluate the Gaussian Process Regression at.
    output_root : {None, str}, optional
        If None, don't save anything. If str, it is the output directory, so
        save the flux and error estimates and used kernels there.
    **kwargs : dict
        Additional keyword arguments that can replace default parameters in
        other funtions. At the moment, we have:
          - do_subtract_background : Bool, default = False
                Whether to estimate a new background subtracting the current.

    Returns
    -------
    obj_gps : astropy.table.Table
        Table with evaluated Gaussian process curve and errors at each
        passband.
    """
    obj_data = dataset.data[obj]  # object's lightcurve
    gp_times = np.linspace(t_min, t_max, number_gp)
    filter_set = np.asarray(dataset.filter_set)

    kernel, gp_params, gp_predict = fit_2d_gp(obj_data, return_kernel=True,
                                              return_gp_params=True)
    gp_wavelengths = np.vectorize(pb_wavelengths.get)(filter_set)
    obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)

    if output_root is not None:
        obj_gps.write(os.path.join(output_root, f'gp_{obj}'), format='fits',
                      overwrite=True)
        path_save_gps = os.path.join(output_root, f'used_gp_{obj}.pckl')
        path_save_kernels = os.path.join(output_root, f'used_kernels_{obj}'
                                                      '.pckl')
        path_save_params = os.path.join(output_root, f'used_params_{obj}.pckl')
        # Save the GP already conditioned on a specific set of observations
        with open(path_save_gps, 'wb') as f:
            pickle.dump(gp_predict, f, pickle.HIGHEST_PROTOCOL)
        with open(path_save_kernels, 'wb') as f:
            pickle.dump(kernel, f, pickle.HIGHEST_PROTOCOL)
        with open(path_save_params, 'wb') as f:
            pickle.dump(gp_params, f, pickle.HIGHEST_PROTOCOL)

    return obj_gps


def preprocess_obs(obj_data, **kwargs):
    """Apply preprocessing to the observations.

    This function is intended to be used to transform the raw observations
    table into one that can actually be used for classification. For now,
    all that this step does is apply background subtraction.

    Parameters
    ----------
    obj_data : pandas.core.frame.DataFrame or astropy.table.Table
        Time, flux, flux error and passbands of the object.
    kwargs : dict
        Additional keyword arguments that are ignored at the moment. We allow
        additional keyword arguments so that the various functions that
        call this one can be called with the same arguments.

    Returns
    -------
    preprocessed_obj_data : pandas.core.frame.DataFrame
        Modified version of the observations on `obj_data` after preprocessing
        that can be used for further analyses.
    """
    try:  # transform into a pandas DataFrame
        obj_data = obj_data.to_pandas()
    except AttributeError:  # is it already a pandas DataFrame
        pass
    obj_data = cs.rename_passband_column(obj_data)

    try:
        do_subtract_background = kwargs["do_subtract_background"]
    except KeyError:
        do_subtract_background = False

    if do_subtract_background:  # estimate a background flux and remove it
        #                         from the light curve.
        preprocessed_obj_data = subtract_background(obj_data)
    else:  # do nothing
        preprocessed_obj_data = obj_data

    return preprocessed_obj_data


def subtract_background(obj_data):
    """Subtract the background flux levels from each band independently.

    The background levels are estimated using a biweight location estimator.
    This estimator will calculate a robust estimate of the background level
    for objects that have short-lived light curves, and it will return
    something like the median flux level for periodic or continuous light
    curves.

    Parameters
    ----------
    obj_data : pandas.core.frame.DataFrame
        Time, flux, flux error and passbands of the object.

    Returns
    -------
    obj_subtracted_obs_data : pandas.core.frame.DataFrame
        Modified version of the observations on `obj_data` with the background
        level flux removed.
    """
    obj_subtracted_obs_data = obj_data.copy()
    unique_pbs = np.unique(obj_data.passband)

    for pb in unique_pbs:
        is_pb = obj_data.passband == pb
        obj_data_pb = obj_data.loc[is_pb]  # the observations in this passband

        # Use a biweight location to estimate the background
        ref_flux = biweight_location(obj_data_pb["flux"])

        obj_subtracted_obs_data.loc[is_pb, "flux"] -= ref_flux
    return obj_subtracted_obs_data


def fit_2d_gp(obj_data, return_kernel=False, return_gp_params=False, **kwargs):
    """Fit a 2D Gaussian process.

    If required, predict the Gaussian process (GP) at evenly spaced points
    along a light curve.

    Parameters
    ----------
    obj_data : pandas.core.frame.DataFrame or astropy.table.Table
        Time, flux and flux error of the data (specific filter of an object).
    return_kernel : bool, optional (False)
        Whether to return the used kernel.
    return_gp_params : bool, optional (False)
        Whether to return the used GP fit parameters.
    kwargs : dict
        Additional keyword arguments that are ignored at the moment. We allow
        additional keyword arguments so that the various functions that
        call this one can be called with the same arguments.

    Returns
    -------
    kernel: george.gp.GP.kernel, optional
        The kernel used to fit the GP.
    gp_params : numpy.ndarray, optional
        The resulting GP fit parameters.
    gp_predict : functools.partial of george.gp.GP
        The GP instance that was used to fit the object.
    """
    guess_length_scale = 20.0  # a parameter of the Matern32Kernel

    obj_data = preprocess_obs(obj_data, **kwargs)  # preprocess obs

    obj_times = obj_data.mjd
    obj_flux = obj_data.flux
    obj_flux_error = obj_data.flux_error
    obj_wavelengths = obj_data['passband'].map(pb_wavelengths)

    def neg_log_like(p):  # Objective function: negative log-likelihood
        gp.set_parameter_vector(p)
        loglike = gp.log_likelihood(obj_flux, quiet=True)
        return -loglike if np.isfinite(loglike) else 1e25

    def grad_neg_log_like(p):  # Gradient of the objective function.
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(obj_flux, quiet=True)

    # Use the highest signal-to-noise observation to estimate the scale. We
    # include an error floor so that in the case of very high
    # signal-to-noise observations we pick the maximum flux value.
    signal_to_noises = np.abs(obj_flux) / np.sqrt(
        obj_flux_error ** 2 + (1e-2 * np.max(obj_flux)) ** 2
    )
    scale = np.abs(obj_flux[signal_to_noises.idxmax()])

    kernel = (0.5 * scale) ** 2 * george.kernels.Matern32Kernel([
        guess_length_scale ** 2, 6000 ** 2], ndim=2)
    kernel.freeze_parameter("k2:metric:log_M_1_1")

    gp = george.GP(kernel)
    default_gp_param = gp.get_parameter_vector()
    x_data = np.vstack([obj_times, obj_wavelengths]).T
    gp.compute(x_data, obj_flux_error)

    bounds = [(0, np.log(1000 ** 2))]
    bounds = [(default_gp_param[0] - 10, default_gp_param[0] + 10)] + bounds
    results = op.minimize(neg_log_like, gp.get_parameter_vector(),
                          jac=grad_neg_log_like, method="L-BFGS-B",
                          bounds=bounds, tol=1e-6)

    if results.success:
        gp.set_parameter_vector(results.x)
    else:
        # Fit failed. Print out a warning, and use the initial guesses for fit
        # parameters.
        obj = obj_data['object_id'][0]
        print("GP fit failed for {}! Using guessed GP parameters.".format(obj))
        gp.set_parameter_vector(default_gp_param)

    gp_predict = partial(gp.predict, obj_flux)

    return_results = []
    if return_kernel:
        return_results.append(kernel)
    if return_gp_params:
        return_results.append(results.x)
    return_results.append(gp_predict)
    return return_results


def predict_2d_gp(gp_predict, gp_times, gp_wavelengths):
    """Outputs the predictions of a Gaussian Process.

    Parameters
    ----------
    gp_predict : functools.partial of george.gp.GP
        The GP instance that was used to fit the object.
    gp_times : numpy.ndarray
        Times to evaluate the Gaussian Process at.
    gp_wavelengths : numpy.ndarray
        Wavelengths to evaluate the Gaussian Process at.

    Returns
    -------
    obj_gps : astropy.table.Table
        Table with evaluated Gaussian process curve and errors at each
        passband.
    """
    unique_wavelengths = np.unique(gp_wavelengths)
    number_gp = len(gp_times)
    obj_gps = []
    for wavelength in unique_wavelengths:
        gp_wavelengths = np.ones(number_gp) * wavelength
        pred_x_data = np.vstack([gp_times, gp_wavelengths]).T
        pb_pred, pb_pred_var = gp_predict(pred_x_data, return_var=True)
        # stack the GP results in a array momentarily
        obj_gp_pb_array = np.column_stack((gp_times, pb_pred,
                                           np.sqrt(pb_pred_var)))
        obj_gp_pb = Table([obj_gp_pb_array[:, 0], obj_gp_pb_array[:, 1],
                           obj_gp_pb_array[:, 2], [wavelength]*number_gp],
                          names=['mjd', 'flux', 'flux_error', 'filter'])

        if len(obj_gps) == 0:  # initialize the table for 1st passband
            obj_gps = obj_gp_pb
        else:  # add more entries to the table
            obj_gps = vstack((obj_gps, obj_gp_pb))

    # Map the wavelenghts to the original passband denominations
    obj_gps['filter'] = np.vectorize(wavelengths_pb.get)(obj_gps['filter'])

    return obj_gps


def print_time_difference(initial_time, final_time):
    """Print the time interval.

    Parameters
    ----------
    initial_time : float
        Time at which the time interval starts.
    final_time : float
        Time at which the time interval ends.
    """
    time_spent = pd.to_timedelta(int(final_time-initial_time), unit='s')
    print('Time spent: {}.'.format(time_spent))
