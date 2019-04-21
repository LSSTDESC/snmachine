"""
Module containing some basic functions that do not depend on any other file and that can be imported to all the files needing them.

The X^2 related functions work only for 1 object. If you want to calculate them for several, you need to contruct your own function
to loop through.
"""

import numpy as np
import pandas as pd

from scipy import interpolate

def compute_overall_reduced_chi_squared(obj_data_with_passband, obj_model_with_passband):
    """Calculates the reduced X^2 statistic between an objects's observations with different passbands and the interpolated computed data.

    `obj_data_with_passband` is considered the real data and `obj_model_with_passband` the data we get from a model we
    wish to compare the goodness of fit using the reduced X^2 statictic. The reduced X^2 of the observation is the sum
    of the reduced X^2 of each `passband` (filter/passband).
    `obj_model_with_passband` does not need to contain the fluxes at the same times as `obj_data_with_passband` as
    this function performs an interpolation.

    Parameters
    ----------
    obj_data : pandas.core.frame.DataFrame with `mjd`, `flux`, `flux_error` and 'passband' columns
        DataFrame containing all the times, flux, flux errors and passband of the real data of the object.
    obj_model : pandas.core.frame.DataFrame with `mjd`, `flux` and 'passband' columns
        DataFrame containing all the times and flux of the computed data/ model of the object. The
        fluxes don't need to be calculated for the same times `obj_data` have.

    Returns
    -------
    reduced_chi_squared : float
        Reduced_chi_squared statistic between observations and the interpolated computed data over all passbands
    """
    obj_data_with_passband = rename_passband_column(obj_data_with_passband) # rename the passband column if needed
    obj_model_with_passband = rename_passband_column(obj_model_with_passband)
    unique_passbands = np.unique(obj_data_with_passband.passband)
    chi_squared = 0
    for pb in unique_passbands:
        obj_data_pb = obj_data_with_passband.loc[obj_data_with_passband.passband == pb]
        obj_model_pb = obj_model_with_passband.loc[obj_model_with_passband.passband == pb]
        chi_squared += compute_chi_squared(obj_data_pb, obj_model_pb)
    number_data_points = np.shape(obj_data_with_passband)[0]
    reduced_chi_squared = chi_squared/number_data_points
    return reduced_chi_squared


def compute_reduced_chi_squared(obj_data, obj_model):
    """Calculates the reduced X^2 statistic between an object's observations and the interpolated computed data.

    `obj_data` is considered the real data and `obj_model` the data we get from a model
    we wish to compare the goodness of fit using the reduced X^2 statictic. The X^2
    statistic is calculated as the sum of squared errors between the fluxes
    of `obj_data` and `obj_model` divided by the flux errors of `obj_data`. The reduced
    X^2 statictic is the X^2 statistic divided by the number of points in `obj_data`.
    `obj_model` does not need to contain the fluxes at the same times as `obj_data` as
    this function performs an interpolation.

    Parameters
    ----------
    obj_data : pandas.core.frame.DataFrame with `mjd`, `flux` and `flux_error` columns
        DataFrame containing all the times, flux and flux errors of the real data of the object.
    obj_model : pandas.core.frame.DataFrame with `mjd` and `flux` columns
        DataFrame containing all the times and flux of the computed data/ model of the object. The
        fluxes don't need to be calculated for the same times `obj_data` have.

    Returns
    -------
    reduced_chi_squared : float
        Reduced_chi_squared statistic between observations and the interpolated computed data

    Raises
    ------
    AttributeError
        Both `obj_data` and `obj_model` need to contain the columns `mjd` and `flux`. `obj_data` also needs `flux_error`.
    """
    number_freedom_degrees = np.shape(obj_data)[0] # number of objects in the data
    chi_squared = compute_chi_squared(obj_data, obj_model)
    reduced_chi_squared = chi_squared/number_freedom_degrees
    return reduced_chi_squared


def compute_chi_squared(obj_data, obj_model):
    """Calculates the X^2 statistic between an object's observations and the interpolated computed data.

    `obj_data` is considered the real data and `obj_model` the data we get from a model
    we wish to compare the goodness of fit using the X^2 statictic. The X^2 statistic is
    calculated as the sum of squared errors between the fluxes of `obj_data` and `obj_model`
    divided by the flux errors of `obj_data`.
    `obj_model` does not need to contain the fluxes at the same times as `obj_data` as
    this function performs an interpolation.

    Parameters
    ----------
    obj_data : pandas.core.frame.DataFrame with `mjd`, `flux` and `flux_error` columns
        DataFrame containing all the times, flux and flux errors of the real data of the object.
    obj_model : pandas.core.frame.DataFrame with `mjd` and `flux` columns
        DataFrame containing all the times and flux of the computed data/ model of the object. The
        fluxes don't need to be calculated for the same times `obj_data` have.

    Returns
    -------
    chi_squared : float
        chi_squared statistic between observations and the interpolated computed data

    Raises
    ------
    AttributeError
        Both `obj_data` and `obj_model` need to contain the columns `mjd` and `flux`. `obj_data` also needs `flux_error`.
    """
    try:
        assert({'mjd', 'flux', 'flux_error'}.issubset(set(obj_data)))
        assert({'mjd', 'flux'}.issubset(set(obj_model)))
    except:
        raise AttributeError("Both `obj_data` and `obj_model` need to contain the columns `mjd` and `flux`. `obj_data` also needs `flux_error`.")

    interpolate_model_flux_at_times = interpolate.interp1d(obj_model.mjd, obj_model.flux, kind='cubic')
    interpolated_model_flux = interpolate_model_flux_at_times(obj_data.mjd)

    chi_squared = np.sum(((obj_data.flux-interpolated_model_flux) / obj_data.flux_error)**2)
    return chi_squared


def rename_passband_column(obj_obs, original_passband_column_name=None):
    """Rename the passband column of a DataFrame to `passband`.

    If the column that contains the passbands is not named `passband`, rename it in that way. It will only work if
    the column is named `pb` or `filter`.

    Parameters
    ----------
    obj_obs : pandas.core.frame.DataFrame with 'passband'/`pb`/`filter` column
        DataFrame containing a column corrsponding to the passbands named 'passband'/`pb`/`filter`/`original_passband_column_name`.
    original_passband_column_name : str, optional
        The name of the passband column, if different of 'passband'/`pb`/`filter`.

    Returns
    -------
    obj_obs_with_passband : pandas.core.frame.DataFrame with 'passband' column
        DataFrame containing a column corrsponding to the passbands named 'passband'

    Raises
    ------
    AttributeError
        `obj_obs` need to contain the column `passband` or a similar one (`pb` or `filter`). If it is none of these,
        input the name of the column in `original_passband_column_name`.
    """
    obj_obs_with_passband = obj_obs.rename(index=str, columns={"pb": "passband", "": "passband", "filter": "passband",
                                                                "original_passband_column_name": "passband"})
    try:
        obj_obs_with_passband.passband
    except:
        raise AttributeError("`obj_obs` need to contain the column `passband` as: `pb`, `filter` or inputed in `original_passband_column_name`.")
    return obj_obs_with_passband