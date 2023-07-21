"""
Module handling the data augmentation of a snmachine dataset.
"""

import copy
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd

from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from functools import partial  # TODO: erase when old sndata is deprecated
from scipy.special import erf
from snmachine import gps


# Functions to choose spectroscopic redshift for `GPAugment`.
# They are inputed as the parameter `choose_z` and their arguments as `*kwargs`
def choose_z_wfd_plasticc(z_ori, pb_wavelengths, random_state):
    """Choose the new spectroscopic redshift for an WFD augmented event.

    The new spectroscopic redshift is based on the redhsift of the original
    event.
    This target distribution of the redshift is class-agnostic and modeled
    after the PLAsTiCC supernovae simulated in the Wide-Fast-Deep Survey.

    Parameters
    ----------
    z_ori : float
        Redshift of the original event.
    pb_wavelengths : dict
        Mapping between the passbands name and central wavelength.
    random_state : numpy.random.mtrand.RandomState
        Container for the slow Mersenne Twister pseudo-random number generator.
        It allows reproducible results.

    Returns
    -------
    z_new : float
        Redshift of the new event.
    """
    z_min = max(10**(-4), (1 + z_ori) * (2 - pb_wavelengths['lsstz'] /
                pb_wavelengths['lssty'])**(-1) - 1)
    z_max = ((1 + z_ori) * (2 - pb_wavelengths['lsstg'] / 
             pb_wavelengths['lsstu'])**(-1) - 1)

    log_z_star = random_state.triangular(left=np.log(z_min),
                                         mode=np.log(z_min),
                                         right=np.log(z_max))
    z_new = - np.exp(log_z_star) + z_min + z_max

    return z_new


def choose_z_ddf_plasticc(z_ori, pb_wavelengths, random_state):
    """Choose the new spectroscopic redshift for an DDF augmented event.

    The new spectroscopic redshift is based on the redhsift of the original
    event.
    This target distribution of the redshift is class-agnostic and modeled
    after the PLAsTiCC supernovae simulated in the Deep Drilling Field Survey.

    Parameters
    ----------
    z_ori : float
        Redshift of the original event.
    pb_wavelengths : dict
        Mapping between the passbands name and central wavelength.
    random_state : numpy.random.mtrand.RandomState
        Container for the slow Mersenne Twister pseudo-random number generator.
        It allows reproducible results.

    Returns
    -------
    z_new: float
        Redshift of the new event.
    """
    z_min = max(10**(-4), (1 + z_ori) * (2 - pb_wavelengths['lsstz'] /
                                         pb_wavelengths['lssty'])**(-1) - 1)
    z_max = 1.4 * ((1 + z_ori) * (2 - pb_wavelengths['lsstg'] / 
                                  pb_wavelengths['lsstu'])**(-1) - 1)

    log_z_star = random_state.triangular(left=np.log(z_min),
                                         mode=np.log(z_min),
                                         right=np.log(z_max))
    z_new = - np.exp(log_z_star) + z_min + z_max

    return z_new


def choose_z_wfd_basev2(z_ori, pb_wavelengths, random_state):
    """Choose the new spectroscopic redshift for an WFD augmented event.

    The new spectroscopic redshift is based on the redhsift of the original
    event.
    This target distribution of the redshift is class-agnostic and modeled
    after the supernovae simulated in the Wide-Fast-Deep Survey with the
    Baseline v2.0 cadence.

    Parameters
    ----------
    z_ori : float
        Redshift of the original event.
    pb_wavelengths : dict
        Mapping between the passbands name and central wavelength.
    random_state : numpy.random.mtrand.RandomState
        Container for the slow Mersenne Twister pseudo-random number generator.
        It allows reproducible results.

    Returns
    -------
    z_new : float
        Redshift of the new event.
    """
    z_min = max(10**(-4), (1 + z_ori) * (2 - pb_wavelengths['lsstz'] /
                                         pb_wavelengths['lssty'])**(-1) - 1)
    z_max = ((1 + z_ori) * (2 - pb_wavelengths['lsstg'] / 
                            pb_wavelengths['lsstu'])**(-1) - 1)

    # Draw a value from a custom trapezoid distribution
    right_val = .1 * 2 / (np.log(z_max) - np.log(z_min))  # 10% of triang. peak
    log_z_star = trapezoid(left=np.log(z_min), right=np.log(z_max),
                           right_val=right_val, random_state=random_state)
    z_new = - np.exp(log_z_star) + z_min + z_max

    return z_new


def trapezoid(left, right, right_val, random_state, size=None):
    """Draw samples from a trapezoid distribution over an interval.

    TODO: add unit tests for this function

    This trapezoid distribution is a continuous probability distribution with
    lower limit `left` and upper limit `right`, and whose paralel sides are in
    the lower and upper limits (not a generalized trapezoidal distribution). To
    define the shape of the probability density function, we add the value of
    the distribution at the upper limit, `right_val`. This trapezoid
    distribution simplifies to a triangular distribution when `right_val == 0`
    or `right_val == 2/(right-left)`, and to a uniform distribution when
    `right_val == 1/(right-left)`.

    Parameters
    ----------
    left : float
        Lower limit.
    right : float
        Upper limit.
    right_val : float
        Value of the trapezoid distribution at the upper limit `right`.
    size : float, optional
        Default is None, in which case a single value is returned.

    Returns
    -------
    samples : ndarray or float
        The returned samples all lie in the interval [`left`, `right`].

    Raises
    ------
    ValueError
        The lower limit `left` must be smaller than the upper limit `right`.
    ValueError
        The value of the distribution at the upper limit `right_val` must be '
        'non-negative and smaller or equal than 2/(`right`-`left`).'

    Notes
    -----
    This function contains the analytical formula for the inverse of the
    cumulative distribution function of the trapezoid distribution.

    With `left`, `right`, and `right_val` parameterised as
    :math:`a, b, \\text{ and } c`, respectively, the probability density
    function for the trapezoid distribution is

    .. math:: P(x;a, b, c) = \\begin{cases}
                  \\frac{2x}{b-a} \\left(c-\\frac{1}{b-a}\\right)
                    + \\frac{2b}{\\left(b-a\\right)^{2}}-c
                    & \\text{for $a \\leq x \\leq b$},\\\\
                  0& \\text{otherwise}.
                  \\end{cases}

    For a generalised trapezoid distribution see the PR at
        https://github.com/numpy/numpy/pull/3770
    """
    dx = right - left
    if dx <= 0:
        raise ValueError('The lower limit `left` must be smaller than the '
                         'upper limit `right`.')
    if (right_val < 0) or (right_val > 2 / dx):
        raise ValueError('The value of the distribution at the upper limit '
                         '`right_val` must be non-negative and smaller or '
                         'equal than 2/(`right`-`left`).')

    # Numbers from unif. dist. to transform into trapez. distr.
    number_unif = random_state.uniform(size=size)
    # The numbers correespond to independent coordinate y and we want to know
    # the dependent x
    y = number_unif

    left_val = 2 / dx - right_val  # value of trapez. distr. at the lower limit

    # Components of the quadratic formula: ax^2 + bx + c = 0
    a = right_val - left_val
    b = 2 * (left_val * right - right_val * left)
    c = - (2 * dx * y + left * (2 * left_val * right - 
                                left * (left_val + right_val)))

    # Formula for the inverse of the cdf of the triangular distribution
    x = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)  # quadratic formula
    return x


class SNAugment:
    """Base class outlining the structure for the augmentation of a sndata
    instance. Classes that encapsulate a specific data augmentation procedure
    are derived from this class.
    """

    def __init__(self, dataset):
        """Class constructor.

        Parameters:
        ----------
        dataset : Dataset object (sndata class)
            Dataset.
        """
        self.dataset = dataset
        # This can contain any metadata that the augmentation
        # process produces, and we want to keep track of.
        self.meta = {}
        self.augmentation_method = None
        # Name of the events that were in the data set prior to augmenting.
        self.original_object_names = dataset.object_names.copy()

    def augment(self):
        """Augment the dataset.
        """
        return NotImplementedError('This method should be defined on child '
                                   'classes.')

    @property
    def dataset(self):
        """Return the original dataset.

        Returns
        -------
        Dataset object (sndata class)
            Dataset to augment.
        """
        return self._dataset

    @property
    def random_seed(self):
        """Return the random state used to augment.

        Returns
        -------
        int
            Random seed used. Saving this seed allows reproducible results.
            If given, it must be between 0 and 2**32 - 1.
        """
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value):
        """Set the seed to the random state used to augment.

        It also initilizes the random state generator used to augment.

        Parameters
        ----------
        value: int, optional
            Random seed used. Saving this seed allows reproducible results.
            If given, it must be between 0 and 2**32 - 1.
        """
        # Initialise the random state
        self._rs = np.random.RandomState(value)
        self._random_seed = value


class GPAugment(SNAugment):
    """Augment the dataset using the Gaussian Process extrapolation of the
    known events.
    """

    def __init__(self, dataset, path_saved_gps, objs_number_to_aug=None,
                 choose_z=None, z_table=None, max_duration=None,
                 cosmology=FlatLambdaCDM(**{"H0": 70, "Om0": 0.3,
                                            "Tcmb0": 2.725}),
                 random_seed=None, **kwargs):
        """Class enclosing the Gaussian Process augmentation.

        Parameters
        ----------
        dataset : Dataset object (`sndata` class)
            Dataset to augment.
        path_saved_gps: str
            Path to the Gaussian Process files.
        objs_number_to_aug : {`None`, 'all', dict}, optional
            Specify which events to augment and by how much. If `None`, the
            dataset it not augmented. If `all`, all the events are augmented
            10 times. If a dictionary is provided, it should be in the form of:
                event: number of times to augment that event.
        choose_z : {None, function}, optional
            Function used to choose the new true redshift of the augmented
            events. If `None`, the new events have the same redshift as the
            original event. If a function is provided, arguments can be
            included as `**kwargs`.
        z_table : {None, pandas.DataFrame}, optional
            Dataset of the spectroscopic and photometric redshift, and
            photometric redshift error of events. This table is used to
            generate the photometric redshift and respective error for the
            augmented events. If `None`, this table is generated from the
            events in the original dataset.
        max_duration : {None, float}, optional
            Maximum duration of the augmented light curves. If `None`, it is
            set to the length of the longest event in `dataset`.
        cosmology : astropy.cosmology.core.COSMOLOGY, optional
            Cosmology from `astropy` with the cosmologic parameters already
            defined. By default it assumes Flat LambdaCDM with parameters
            `H0 = 70`, `Om0 = 0.3` and `T_cmb0 = 2.725`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
            If given, it must be between 0 and 2**32 - 1.
        **kwargs : dict, optional
            Optional keywords to pass arguments into `choose_z` and into
            `snamchine.gps.compute_gps`.

        Notes
        -----
        This augmentation is based on [1]_.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        self._dataset = dataset
        self._aug_method = 'GP augmentation'
        self.random_seed = random_seed
        self._original_object_names = self.dataset.object_names.copy()
        self.objs_number_to_aug = objs_number_to_aug
        self.z_table = z_table
        self._path_saved_gps = path_saved_gps
        self._cosmology = cosmology
        self.max_duration = max_duration
        self._kwargs = dict(kwargs, pb_wavelengths=self.dataset.pb_wavelengths,
                            random_state=self._rs)
        self.choose_z = choose_z

    def augment(self):
        """Augment the dataset.

        Returns
        -------
        str, optional
            If there are no events choosen to augment, a message is returned
            with that information.
        """
        print('Augmenting the dataset...')
        initial_time = time.time()

        aug_objs_data = []
        aug_objs_metadata = []
        if np.size(self.objects_to_aug) == 0:
            return ('No events were choosen to augment, so no augmentation '
                    'was performed.')
        for obj in self.objects_to_aug:
            new_aug_objs_data, new_aug_objs_metadata = self.augment_obj(obj)
            aug_objs_data += new_aug_objs_data
            aug_objs_metadata += new_aug_objs_metadata

        self._add_augment_objs_to_dataset(aug_objs_data, aug_objs_metadata)

        time_spent = pd.to_timedelta(int(time.time() - initial_time), unit='s')
        print('Time spent augmenting: {}.'.format(time_spent))

    def _add_augment_objs_to_dataset(self, aug_objs_data, aug_objs_metadata):
        """Add the new events to the dataset.

        There are two new datasets created:
            - aug_dataset: original + augmented events
            - only_new_dataset: augmented events

        Parameters
        ----------
        aug_objs_data : list of pandas.DataFrame
            List containing the observations of each augmentation of each
            event.
        aug_objs_metadata : list of pandas.DataFrame
            Ordered list containing the metadata of each augmentation of each
            event.
        """
        aug_dataset = copy.deepcopy(self.dataset)

        metadata = self.dataset.metadata
        new_entries = pd.concat(aug_objs_metadata, axis=1).T
        aug_metadata = pd.concat([metadata, new_entries])
        # Flag the original events as non augmented
        augmented_col = aug_metadata.augmented
        aug_metadata.augmented[augmented_col.isna()] = False
        aug_dataset.metadata = aug_metadata

        aug_dataset.object_names = aug_metadata.object_id

        for obj_data in aug_objs_data:
            try:
                obj = obj_data['object_id'].iloc[0]
                aug_dataset.data[obj] = Table.from_pandas(obj_data)
            except (IndexError, TypeError):
                print(obj_data)
                print('Failed attempt.')
        self.aug_dataset = aug_dataset

        only_new_dataset = copy.deepcopy(aug_dataset)
        only_new_dataset.update_dataset(new_entries.object_id)
        self.only_new_dataset = only_new_dataset

    def augment_obj(self, obj):
        """Create the specified number of augmentations of the event.

        The number of augmented events to create is specified in
        `self.objs_number_to_aug[obj]`.

        Parameters
        ----------
        obj : str
            Name of the original event.

        Returns
        -------
        aug_objs_data : list of pandas.DataFrame
            Ordered list containing the observations of each augmentation of
            `obj`.
        aug_objs_metadata : list of pandas.DataFrame
            Ordered list containing the metadata of each augmentation of `obj`.
        """
        obj_data = self.dataset.data[obj].to_pandas()
        obj_metadata = self.dataset.metadata.loc[obj]
        z_obj = obj_metadata['hostgal_specz']

        number_aug = self.objs_number_to_aug[obj]  # # of new events
        number_tries = 10  # # tries to generate an augmented event
        aug_objs_data = []
        aug_objs_metadata = []
        for i in np.arange(number_aug):
            aug_obj = '{}_aug{}'.format(obj, i)
            j = 0
            while j < number_tries:
                aug_obj_metadata = self.create_aug_obj_metadata(aug_obj,
                                                                obj_metadata)
                if aug_obj_metadata is None:
                    # If the metadata creation failed, do not try to generate
                    # observations
                    aug_obj_data = []
                else:
                    aug_obj_data = self.create_aug_obj_obs(aug_obj_metadata,
                                                           obj_data, z_obj)
                # Failed attempt at creating observations
                if len(aug_obj_data) == 0:
                    j += 1
                # Successful attempt at creating observations
                else:
                    j = number_tries  # finish the loop
                    aug_objs_data.append(aug_obj_data)
                    aug_objs_metadata.append(aug_obj_metadata)
        return aug_objs_data, aug_objs_metadata

    def _choose_obs_times(self, aug_obj_metadata, obj_data, z_ori):
        """Choose the times at which mock observations will be made.

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        obj_data : pandas.DataFrame
            Observations of the original event.
        z_ori : float
            Redshift of the original event.

        Returns
        -------
        aug_obj_data : pandas.DataFrame
            Table containing the times and passbands of the augmented event
            observations. The other columns contain the information relative
            to the original event.

        Notes
        -----
        This function is adapted from the code developed in [1]_. In
        particular, the funtion `Augmentor._choose_sampling_times` of
        `avocado/augment.py`.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        z_aug = aug_obj_metadata['hostgal_specz']

        # Generate a copy of the original event
        aug_obj_data = obj_data.copy()
        aug_obj_data['object_id'] = aug_obj_metadata['object_id']
        aug_obj_data['ref_mjd'] = aug_obj_data['mjd'].copy()

        # Stretch the observed epochs of the original event to account for the
        # time dilation due to the difference between the original and
        # augmented redshifts
        z_scale = (1 + z_ori) / (1 + z_aug)

        # Keep the time of the maximum flux invariant so that the interesting
        # part of the light curve remains inside the observing window.
        time_peak = obj_data['mjd'].iloc[np.argmax(obj_data['flux'].values)]
        aug_obj_data['mjd'] = time_peak + z_scale**-1 * (
            obj_data['mjd'] - time_peak)
        # Removed any observations outside the observing window
        is_not_seen = aug_obj_data['mjd'] < 0
        aug_obj_data = aug_obj_data[~is_not_seen]  # before 0
        aug_obj_data = self.trim_obj(aug_obj_data, self.max_duration)  # after

        # Ensure the augmented event still has observations. If not, stop this
        # augmentation
        if len(aug_obj_data) == 0:
            print('obj {} failed.'.format(aug_obj_metadata['object_id']))
            return None

        # Randomly choose a target number of observations for the new event.
        target_number_obs = self._choose_target_number_obs(aug_obj_metadata)

        # Events shifted to higher redshifts have a lower density of
        # observations than the events observed at those redshifts. In order
        # to account for this, we add more observations to these higher
        # redshift events.
        num_fill = int(target_number_obs * (z_scale**-1 - 1))
        if num_fill > 0:
            # At the most, create 50% more data; It prevents augmented events
            # with many observations that provide no extra information
            if num_fill > len(obj_data) / 2:
                num_fill = int(len(obj_data) / 2)
            new_indices = self._rs.choice(aug_obj_data.index, num_fill,
                                          replace=True)
            new_rows = aug_obj_data.loc[new_indices]

            # Choose new passbands randomly
            obj_pbs = np.unique(aug_obj_data['filter'])
            new_rows['filter'] = self._rs.choice(obj_pbs, num_fill,
                                                 replace=True)
            aug_obj_data = pd.concat([aug_obj_data, new_rows])

            # Reorder observations in chronological order
            aug_obj_data.sort_values(by=['mjd'], ignore_index=True,
                                     inplace=True)

        # If the augmented event has more observations than the target number
        # of observations, randomly drop the difference. In any case, to
        # introduce additional variability, randomly drop at least 10% of the
        # synthetic observations.
        drop_fraction = 0.1
        number_drop = int(max(len(aug_obj_data) - target_number_obs,
                              drop_fraction * len(aug_obj_data)))
        drop_indices = self._rs.choice(aug_obj_data.index, number_drop,
                                       replace=False)
        aug_obj_data = aug_obj_data.drop(drop_indices).copy()

        # For consistency between all datasets, the first observation is at t=0
        aug_obj_data['mjd'] -= np.min(aug_obj_data['mjd'])

        aug_obj_data.reset_index(inplace=True, drop=True)
        return aug_obj_data

    def create_aug_obj_obs(self, aug_obj_metadata, obj_data, z_ori):
        """Create observations for the augmented event.

        The new observations are based on the observations of the original
        event.

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        obj_data : pandas.DataFrame
            Observations of the original event.
        z_ori : float
            Redshift of the original event.

        Returns
        -------
        aug_obj_data : pandas.DataFrame
            Observations of the augmented event.
        """
        z_aug = aug_obj_metadata['hostgal_specz']
        ori_obj = str(obj_data['object_id'][0])

        aug_obj_data = self._choose_obs_times(aug_obj_metadata, obj_data,
                                              z_ori)
        if aug_obj_data is None:
            return []  # failed attempt
        aug_obj_data['wavelength_z_ori'] = self.compute_new_wavelength(
            z_ori, z_aug, aug_obj_data)
        gp_predict = self.load_gp(ori_obj)

        # Predict the augmented observations seen at `z_ori`
        pred_x_data = np.vstack([aug_obj_data['ref_mjd'],
                                 aug_obj_data['wavelength_z_ori']]).T
        flux_pred, flux_pred_var = gp_predict(pred_x_data, return_var=True)
        flux_pred_error = np.sqrt(flux_pred_var)

        # Redshift flux values
        z_scale = (1 + z_ori) / (1 + z_aug)
        dist_scale = (self.cosmology.luminosity_distance(z_ori) / 
                      self.cosmology.luminosity_distance(z_aug))**2
        aug_obj_data['flux'] = flux_pred * z_scale * dist_scale
        aug_obj_data['flux_error'] = flux_pred_error * z_scale * dist_scale

        # Add flux uncertainty to the light curve observations
        aug_obj_data = self._compute_obs_uncertainty(aug_obj_data,
                                                     aug_obj_metadata)

        # Apply quality cuts
        # The event has at least two detections
        aug_obj_data, pass_detection = self._simulate_detection(
            aug_obj_data, aug_obj_metadata)
        # Since two observations are insufficient for constraining a GP, we
        # require an additional observation, regardless of its S/N.
        if pass_detection and (len(aug_obj_data) >= 3):
            return aug_obj_data

        # Failed attempt
        return []

    def load_gp(self, obj):
        """Load the Gaussian Process predict object.

        Parameters
        ----------
        obj : str
            Name of the original event.

        Returns
        -------
        gp_predict : functools.partial with bound method GP.predict
            Function to predict the Gaussian Process flux and uncertainty at
            any time and wavelength.
        """
        # The name format of the saved Gaussian Processes is hard coded
        path_saved_obj_gp = os.path.join(self.path_saved_gps,
                                         'used_gp_' + obj + '.pckl')
        with open(path_saved_obj_gp, 'rb') as input:
            gp_predict = pickle.load(input)
        try:  # old format - TODO: deprecate the old sndata format
            obj_flux = self.dataset.data[obj]['flux']
            gp_predict = partial(gp_predict.predict, obj_flux)
            warnings.warn('This is an old format and it will be removed soon.',
                          DeprecationWarning)
        except AttributeError:
            pass
        return gp_predict

    def create_aug_obj_metadata(self, aug_obj, obj_metadata):
        """Create metadata for the augmented event.

        The new metadata is based based on the metadata of the original event.

        Parameters
        ----------
        aug_obj : str
            Name of the augmented event in the form
                `[original event name]_[number of the augmentation]`.
        obj_metadata: pandas.DataFrame
            Metadata of the original event.

        Returns
        -------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        """
        raise NotImplementedError('This method should be defined on child '
                                  'classes')

    def compute_new_z_photo(self, z_spec):
        """Compute a new photometric redshift and error.

        The new values are randomly withdrawn from a redshift table containing
        differences between spectroscopic and photometric redshifts and the
        respective photometric redshift error.

        Parameters
        ----------
        z_spec : float
            Spectroscopic redshift of the augmented event.
        i: int, optional
            Number of the current iteration. This acts as a stopping criteria.

        Returns
        -------
        z_photo : float
            Photometric redshift of the augmented event.
        z_photo_error : float
            Photometric redshift error of the augmented event.

        Raises
        ------
        ValueError
            If none of the generated `z_photo` or `z_photo_error` are positive.
        """
        z_table = self.z_table
        number_tries = 100  # number of tries to get positive z values
        rd_zs_triple = z_table.sample(random_state=self._rs, n=number_tries,
                                      replace=True)
        zs_diff = rd_zs_triple['z_diff']
        zs_photo = z_spec + (self._rs.choice([-1, 1], size=number_tries) * 
                             zs_diff)
        zs_photo_error = (rd_zs_triple['hostgal_photoz_err'] * 
                          self._rs.normal(1, .05, size=number_tries))

        are_zs_pos = (zs_photo > 0) & (zs_photo_error > 0)
        try:  # choose the first appropriate value
            z_photo = np.array(zs_photo[are_zs_pos])[0]
            z_photo_error = np.array(zs_photo_error[are_zs_pos])[0]
        except (KeyError, IndexError):
            raise ValueError('The new redshift and respective error must be '
                             'positive and they were not after 100 tries so '
                             'something is wrong.')
        return z_photo, z_photo_error

    def fit_gps(self, path_to_save_gps, **kwargs):
        """Fit Gaussian Processes to the augmented events.

        Parameters
        ----------
        path_to_save_gps : str
            Path where to save the new Gaussian Processes outputs.
        """
        # Confirm the data was augmented and the path to save the GPs exist
        self._is_dataset_augmented()
        self._exists_path(path_to_save_gps)

        self._kwargs.update(kwargs)  # add more arguments

        gps.compute_gps(self.only_new_dataset, output_root=path_to_save_gps,
                        **self._kwargs)

    def _is_dataset_augmented(self):
        """Check if the dataset was already augmented.

        Raises
        ------
        AttributeError
            If the dataset has not been augmented yet and therefore the
            attribute `self.only_new_dataset` does not exist.
        """
        try:
            self.only_new_dataset
        except AttributeError:
            raise AttributeError('The original dataset must be augmented '
                                 'before Gaussian Processes can be fitted to '
                                 'the augmented events.')

    @property
    def max_duration(self):
        """Return the maximum duration any light curve can have.

        All the events in the original dataset must be shorter than this value.

        Returns
        -------
        float
            Maximum duration any light curve can have.
        """
        return self._max_duration

    @max_duration.setter
    def max_duration(self, value):
        """Set the maximum duration any light curve can have.

        Parameters
        ----------
        value : {None, float}, optional
            Maximum duration of the light curve. If `None`, it is set to the
            maximum lenght of an event in `dataset`.

        Raises
        ------
        ValueError
            If any event in the original dataset is longer than the maximum
            duration any light curve can have.
        """
        max_duration_ori = self.dataset.get_max_length()
        if value is None:
            duration = max_duration_ori
        elif max_duration_ori > value:
            raise ValueError('All the events in the original dataset must be '
                             'shorter than the required maximum duration any '
                             'light curve. At the moment the maximum duration '
                             'of an event is {:.0f} days.'
                             ''.format(max_duration_ori))
        else:
            duration = value
        self._max_duration = duration

    @staticmethod
    def trim_obj(obj_data, max_duration):
        """Remove the event edges so it is shorter than a threshold.

        Remove at least 5 days each time. The observations removed are the
        first or the last, depending on which ones have the mean flux closer
        to zero.

        Parameters
        ----------
        obj_data : pandas.DataFrame
            Observations of an event.
        max_duration : float
            Maximum duration of the light curve.

        Returns
        -------
        obj_data : pandas.DataFrame
            Trimmed observations of an event.
        """
        obj_duration = np.max(obj_data['mjd'])
        while obj_duration > max_duration:
            diff_duration = max_duration - obj_duration
            number_days_remove = np.max([5., diff_duration / 3])  # >= 5 days

            index_initial_obs = obj_data['mjd'] < number_days_remove
            mean_initial_flux = np.mean(obj_data['flux'][index_initial_obs])

            index_final_obs = obj_data['mjd'] < (obj_duration - 
                                                 number_days_remove)
            mean_final_flux = np.mean(obj_data['flux'][index_final_obs])

            if np.abs(mean_initial_flux) > np.abs(mean_final_flux):
                obj_data = obj_data[~index_final_obs]  # remove end
            else:  # remove beginning
                obj_data = obj_data[~index_initial_obs]
                obj_data['mjd'] -= np.min(obj_data['mjd'])
            obj_duration = np.max(obj_data['mjd'])
        return obj_data

    @property
    def cosmology(self):
        """Return the cosmology.

        Returns
        -------
        astropy.cosmology.core.COSMOLOGY
            Cosmology from `astropy` with the cosmologic parameters already
            defined.

        Raises
        ------
        TypeError
            If the form of the cosmology inputed does not allow the
            computation of the distance modulus.
        """
        try:
            self._cosmology.distmod(z=1)
            return self._cosmology
        except TypeError:
            raise TypeError('The cosmology must be given as '
                            '`astropy.cosmology.core.COSMOLOGY`. It must be '
                            'possible to compute the distance modulus at '
                            'redshift 1 by calling `cosmology.distmod(z=1)`.')

    def compute_new_wavelength(self, z_ori, z_new, obj_data):
        """Compute the new observations wavelength at the original redshift.

        The observation flux is measured at specific wavelengths. This
        function calculates the wavelength of the new event as seen at
        redshift `z_ori`.

        Parameters
        ----------
        z_ori : float
            Redshift of the original event.
        z_new : float
            Redshift of the new event.
        obj_data : pandas.DataFrame
            Observations of the original event.

        Returns
        -------
        wavelength_new : list-like
            Wavelength of the new observations at redshift `z_ori`.
        """
        z_scale = (1 + z_ori) / (1 + z_new)
        wavelength_ori = obj_data['filter'].map(self.dataset.pb_wavelengths)
        wavelength_new = wavelength_ori * z_scale
        return wavelength_new

    @property
    def aug_method(self):
        """Return the augmentation method.

        Returns
        -------
        str
            Name of the augmentation method.
        """
        return self._aug_method

    @property
    def original_object_names(self):
        """Return the names of the events in the original dataset.

        Returns
        -------
        list-like
            Name of the events in the original dataset.
        """
        return self._original_object_names

    @property
    def z_table(self):
        """Return a table with all the redshift differences and uncertainties.

        This redshift table is used to map the difference between
        spectroscopic and photometric redshift.

        Returns
        -------
        pandas.DataFrame
            Table with `z_diff` and `hostgal_photoz_err`
        """
        return self._z_table

    @z_table.setter
    def z_table(self, value):
        """Set the redshift table to use in the augmentation.

        Parameters
        ----------
        value : {None, pandas.DataFrame}, optional
            Dataset of the spectroscopic and photometric redshift and
            photometric redshift error of events. This table is used to
            generate the photometric redshift and respective error for the
            augmented events. If `None`, this table is generated from the
            events in the original dataset.
        """
        if value is None:
            metadata = self.dataset.metadata
            z_table = metadata[['hostgal_specz', 'hostgal_photoz',
                                'hostgal_photoz_err']]
            z_table['z_diff'] = (z_table['hostgal_photoz'] - 
                                 z_table['hostgal_specz'])
        else:
            z_table = self._standardise_z_table(value)
        self._z_table = z_table

    @staticmethod
    def _standardise_z_table(z_table):
        """Standardise redshift table to have redshift difference and error.

        Returns
        -------
        z_table : pandas.DataFrame
            Table with `z_diff` and `hostgal_photoz_err`

        Raises
        ------
        KeyError
            If the table provided does not contain enough information to
            compute the redshift difference and error. It also raises an error
            if table type/ format is wrong.
        """
        try:  # does it have z_spec, z_photo and z_photo_err?
            z_table[['hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err']]
        except KeyError:
            try:  # does it have z_diff, and z_photo_err?
                z_table[['z_diff', 'hostgal_photoz_err']]
            except KeyError:
                raise KeyError('The redshift table must be a '
                               '`pandas.DataFrame` with `hostgal_photoz_err` '
                               'and either `z_diff` or `hostgal_specz` and '
                               '`hostgal_photoz`.')
            return z_table
        z_table['z_diff'] = (z_table['hostgal_photoz'] - 
                             z_table['hostgal_specz'])
        return z_table

    @staticmethod
    def _exists_path(path_to_test):
        """Check if the inputed path exists.

        Parameters
        ----------
        path_to_test : str
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

    @property
    def path_saved_gps(self):
        """Return the path to the Gaussian Process files.

        Returns
        -------
        str
            Path to the Gaussian Process files.
        """
        self._exists_path(self._path_saved_gps)
        return self._path_saved_gps

    @property
    def objs_number_to_aug(self):
        """Return which events to augment and by how much.

        Returns
        -------
        dict
            The events to augment and how many of each in the form of:
                event: number of times to augment that event.
        """
        return self._objs_number_to_aug

    @objs_number_to_aug.setter
    def objs_number_to_aug(self, value):
        """Set which events to augment and by how much.

        Parameters
        ----------
        value : {`None`, 'all', dict}, optional
            Specify which events to augment and by how much. If `None`, the
            dataset it not augmented. If `all`, all the events are augmented
            10 times. If a dictionary is provided, it should be in the form of:
                event: number of times to augment that event.
        """
        self._objs_number_to_aug = self._standardise_objs_number_to_aug(value)

    def _standardise_objs_number_to_aug(self, objs_number_to_aug):
        """Standardise the number and which events to augment.

        The events to augment can be specified in several different ways. This
        function standardises them all into a dictionary.

        Parameters
        ----------
        objs_number_to_aug : {`None`, 'all', dict}, optional
            Specify which events to augment and by how much. If `None`, the
            dataset it not augmented. If `all`, all the events are augmented
            10 times. If a dictionary is provided, it should be in the form of:
                event: number of times to augment that event.

        Returns
        -------
        objs_number_to_aug : dict
            The events used to augment and how many of each in the form of:
                event: number of times to augment that event.

        Raises
        ------
        ValueError
            If the events to augment do not exist in the original dataset.
        TypeError
            If `objs_number_to_aug` is not `None`, `all` or a dictionary
            because the function is only built to deal with these input types.
        """
        if objs_number_to_aug is None:
            objs_number_to_aug = {}
        elif objs_number_to_aug == 'all':  # use all events
            ori_obj_names = self.original_object_names
            objs_number_to_aug = {obj: 10 for obj in ori_obj_names}
        elif type(objs_number_to_aug) == dict:
            objs_to_aug = np.array(list(objs_number_to_aug.keys()))
            ori_obj_names = self.original_object_names
            # Verify all the events to augment exist in the original dataset
            is_in_ori_obj_names = np.in1d(objs_to_aug, ori_obj_names)
            if np.sum(~is_in_ori_obj_names) != 0:
                raise ValueError('The events specified in `objs_number_to_aug`'
                                 ' must exist in the original dataset.')
        else:
            raise TypeError('`objs_number_to_aug` specifies which events to '
                            'augment and by how much. If `None`, the dataset '
                            'it not augmented. If `all`, all the events are '
                            'augmented 10 times. If a dictionary is provided, '
                            'it should be in the form of: \n event: number of '
                            'times to augment that event.')
        return objs_number_to_aug

    @property
    def objects_to_aug(self):
        """Return the name of the events to augment.

        Returns
        -------
        list-like
            The id of the events used to augment.
        """
        objs_number_to_aug = self.objs_number_to_aug
        return np.array(list(objs_number_to_aug.keys()))

    def _choose_target_number_obs(self, aug_obj_metadata):
        """Randomly choose the target number of light curve observations.

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.

        Returns
        -------
        target_number_obs : int
            The target number of observations in the new light curve.
        """
        raise NotImplementedError('This method should be defined on child '
                                  'classes')

    def _compute_obs_uncertainty(self, aug_obj_data, aug_obj_metadata):
        """Compute and add uncertainty to the light curve observations.

        Following [1]_, we estimate the flux uncertainties for each
        passband with a lognormal distribution for the Wide-Fast-Deep (WFD)
        and Deep Drilling Field (DDF) surveys. Each passband in each survey
        was modeled individually with test set events.
        The flux uncertanty of the augmented events is the combination of the
        flux uncertainty of the augmented events predicted by the GP in
        quadrature with a value drawn from the flux uncertainty distribution
        described above.

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        obj_data : pandas.DataFrame
            Observations of the original event.

        Returns
        -------
        aug_obj_data : pandas.DataFrame
            Observations of the augmented event.

        Notes
        -----
        This function is adapted from the code developed in [1]_. In
        particular, the funtion
        `PlasticcAugmentor._simulate_light_curve_uncertainties` of
        `avocado/plasticc.py`.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        raise NotImplementedError('This method should be defined on child '
                                  'classes')

    def _simulate_detection(self, aug_obj_data, aug_obj_metadata):
        """Simulate the detection process for a light curve.

        We impose quality cuts on the augmented events. Following [1]_, we
        require at least two detections: at least two observations above the
        signal-to-noise (S/N) threshold. [1]_ calculated this threshold by
        fitting an error function to the observations from the PLAsTiCC
        dataset to predict the probability of detection as a function of S/N.

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        obj_data : pandas.DataFrame
            Observations of the original event.

        Returns
        -------
        aug_obj_data : pandas.DataFrame
            Observations of the augmented event.
        pass_detection : bool
            Whether or not the event passes the detection threshold.

        Notes
        -----
        This method is adapted from the code developed in [1]_. In particular,
        the funtion `PlasticcAugmentor._simulate_detection` of
        `avocado/plasticc.py`.
        Note that this method should be overridden in child classes if the
        quality cuts desired are different.


        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        # Calculate the S/N of the observations
        s2n = np.abs(aug_obj_data["flux"]) / aug_obj_data["flux_error"]

        # Apply the S/N threshold
        prob_detected = (erf((s2n - 5.5) / 2) + 1) / 2.0
        aug_obj_data["detected"] = self._rs.rand(len(s2n)) < prob_detected
        pass_detection = np.sum(aug_obj_data["detected"]) >= 2

        return aug_obj_data, pass_detection


class PlasticcWFDAugment(GPAugment):
    """Augment the Wide-Fast-Deep (WFD) events in the PLAsTiCC dataset using
    Gaussian Process extrapolation method (see `snaugment.GPAugment`).
    """

    def __init__(self, dataset, path_saved_gps, objs_number_to_aug=None,
                 z_table=None, max_duration=None,
                 cosmology=FlatLambdaCDM(**{"H0": 70, "Om0": 0.3,
                                            "Tcmb0": 2.725}),
                 random_seed=None, **kwargs):
        """Class enclosing the Gaussian Process augmentation of WFD events.

        This class augments the Wide-Fast-Deep (WFD) events in the PLAsTiCC
        dataset.

        Parameters
        ----------
        dataset : Dataset object (`sndata` class)
            Dataset to augment.
        path_saved_gps: str
            Path to the Gaussian Process files.
        objs_number_to_aug : {`None`, 'all', dict}, optional
            Specify which events to augment and by how much. If `None`, the
            dataset it not augmented. If `all`, all the events are augmented
            10 times. If a dictionary is provided, it should be in the form of:
                event: number of times to augment that event.
        z_table : {None, pandas.DataFrame}, optional
            Dataset of the spectroscopic and photometric redshift, and
            photometric redshift error of events. This table is used to
            generate the photometric redshift and respective error for the
            augmented events. If `None`, this table is generated from the
            events in the original dataset.
        max_duration : {None, float}, optional
            Maximum duration of the augmented light curves. If `None`, it is
            set to the length of the longest event in `dataset`.
        cosmology : astropy.cosmology.core.COSMOLOGY, optional
            Cosmology from `astropy` with the cosmologic parameters already
            defined. By default it assumes Flat LambdaCDM with parameters
            `H0 = 70`, `Om0 = 0.3` and `T_cmb0 = 2.725`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
            If given, it must be between 0 and 2**32 - 1.
        **kwargs : dict, optional
            Optional keywords to pass arguments into
            `snamchine.gps.compute_gps`.

        Notes
        -----
        This augmentation is based on [1]_.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        super().__init__(dataset=dataset, path_saved_gps=path_saved_gps,
                         objs_number_to_aug=objs_number_to_aug,
                         choose_z=choose_z_wfd_plasticc, z_table=z_table,
                         max_duration=max_duration, cosmology=cosmology,
                         random_seed=random_seed, **kwargs)

        self._aug_method = 'GP augmentation; PLAsTiCC WFD'

    def create_aug_obj_metadata(self, aug_obj, obj_metadata):
        """Create metadata for the WFD augmented event.

        The new metadata is based based on the metadata of the original event.

        Parameters
        ----------
        aug_obj : str
            Name of the augmented event in the form
                `[original event name]_[number of the augmentation]`.
        obj_metadata: pandas.DataFrame
            Metadata of the original event.

        Returns
        -------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        """
        aug_obj_metadata = obj_metadata.copy()
        aug_obj_metadata.name = aug_obj
        aug_obj_metadata.object_id = aug_obj
        aug_obj_metadata['augmented'] = True
        aug_obj_metadata['original_event'] = aug_obj.split('_')[0]

        z_spec = self.choose_z(obj_metadata['hostgal_specz'], **self._kwargs)
        z_photo, z_photo_error = self.compute_new_z_photo(z_spec)
        aug_obj_metadata['hostgal_specz'] = z_spec
        aug_obj_metadata['hostgal_photoz'] = z_photo
        aug_obj_metadata['hostgal_photoz_err'] = z_photo_error

        # The new event will be in WFD regardless of the original event; we
        # can always degrade DDF events to simulate a WFD event
        aug_obj_metadata['ddf'] = False

        return aug_obj_metadata

    def _choose_target_number_obs(self, aug_obj_metadata):
        """Randomly choose the target number of light curve observations.

        Using Gaussian mixture models, we model the number of observations in
        the test set events simulated on the Wide-Fast-Deep (WFD).

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.

        Returns
        -------
        target_number_obs : int
            The target number of observations in the new light curve.

        Notes
        -----
        This function is adapted from the code developed in [1]_. In
        particular, the funtion
        `PlasticcAugmentor._choose_target_observation_count` of
        `avocado/plasticc.py`.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        target_number_obs = (self._rs.normal(24.5006006, np.sqrt(72.5106613)))
        target_number_obs = int(np.clip(target_number_obs, 3, None))

        return target_number_obs

    def _compute_obs_uncertainty(self, aug_obj_data, aug_obj_metadata):
        """Compute and add uncertainty to the light curve observations.

        Following [1]_, we estimate the flux uncertainties for each
        passband with a lognormal distribution for the Wide-Fast-Deep (WFD)
        survey. Each passband was modeled individually with test set events.
        The flux uncertanty of the augmented events is the combination of the
        flux uncertainty of the augmented events predicted by the GP in
        quadrature with a value drawn from the flux uncertainty distribution
        described above.

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        obj_data : pandas.DataFrame
            Observations of the original event.

        Returns
        -------
        aug_obj_data : pandas.DataFrame
            Observations of the augmented event.

        Notes
        -----
        This function is adapted from the code developed in [1]_. In
        particular, the funtion
        `PlasticcAugmentor._simulate_light_curve_uncertainties` of
        `avocado/plasticc.py`.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        # Make a copy of the original data to avoid modifying it
        aug_obj_data = aug_obj_data.copy()

        # Skip this function if there are no observations
        if len(aug_obj_data) == 0:
            return aug_obj_data

        # The uncertainty levels of the observations in each passband can be
        # modeled with a lognormal distribution. See [1].
        # Lognormal parameters
        pb_noises = {'lsstu': (2.34, 0.43), 'lsstg': (0.94, 0.41),
                     'lsstr': (1.30, 0.41), 'lssti': (1.82, 0.42),
                     'lsstz': (2.56, 0.36), 'lssty': (3.33, 0.37)}

        # Calculate the new uncertainty levels for each passband
        lognormal_parameters = []
        for pb in aug_obj_data['filter']:
            try:
                lognormal_parameters.append(pb_noises[pb])
            except KeyError:
                raise ValueError(f'The noise properties of the passband {pb} '
                                 f'are not known. Add them to '
                                 f'`GPAugment._compute_obs_uncertainty`.')
        lognormal_parameters = np.array(lognormal_parameters)

        # Combine the flux uncertainty of the augmented events predicted by
        # the GP in quadrature with a value drawn from the flux uncertainty
        # distribution of the test set
        add_stds = self._rs.lognormal(
            lognormal_parameters[:, 0], lognormal_parameters[:, 1])
        noise_add = self._rs.normal(loc=0.0, scale=add_stds)
        aug_obj_data['flux'] += noise_add  # add noise to increase variability
        aug_obj_data['flux_error'] = np.sqrt(aug_obj_data['flux_error'] ** 2 + 
                                             add_stds ** 2)
        return aug_obj_data


class PlasticcDDFAugment(GPAugment):
    """Augment the Deep Drilling Field (DDF) events in the PLAsTiCC dataset 
    using Gaussian Process extrapolation method (see `snaugment.GPAugment`).
    """

    def __init__(self, dataset, path_saved_gps, objs_number_to_aug=None,
                 z_table=None, max_duration=None,
                 cosmology=FlatLambdaCDM(**{"H0": 70, "Om0": 0.3,
                                            "Tcmb0": 2.725}),
                 random_seed=None, **kwargs):
        """Class enclosing the Gaussian Process augmentation of DDF events.

        This class augments the Deep Drilling Field (DDF) events in the
        PLAsTiCC dataset.

        Parameters
        ----------
        dataset : Dataset object (`sndata` class)
            Dataset to augment.
        path_saved_gps: str
            Path to the Gaussian Process files.
        objs_number_to_aug : {`None`, 'all', dict}, optional
            Specify which events to augment and by how much. If `None`, the
            dataset it not augmented. If `all`, all the events are augmented
            10 times. If a dictionary is provided, it should be in the form of:
                event: number of times to augment that event.
        z_table : {None, pandas.DataFrame}, optional
            Dataset of the spectroscopic and photometric redshift, and
            photometric redshift error of events. This table is used to
            generate the photometric redshift and respective error for the
            augmented events. If `None`, this table is generated from the
            events in the original dataset.
        max_duration : {None, float}, optional
            Maximum duration of the augmented light curves. If `None`, it is
            set to the length of the longest event in `dataset`.
        cosmology : astropy.cosmology.core.COSMOLOGY, optional
            Cosmology from `astropy` with the cosmologic parameters already
            defined. By default it assumes Flat LambdaCDM with parameters
            `H0 = 70`, `Om0 = 0.3` and `T_cmb0 = 2.725`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
            If given, it must be between 0 and 2**32 - 1.
        **kwargs : dict, optional
            Optional keywords to pass arguments into
            `snamchine.gps.compute_gps`.

        Notes
        -----
        This augmentation is based on [1]_.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        super().__init__(dataset=dataset, path_saved_gps=path_saved_gps,
                         objs_number_to_aug=objs_number_to_aug,
                         choose_z=choose_z_ddf_plasticc, z_table=z_table,
                         max_duration=max_duration, cosmology=cosmology,
                         random_seed=random_seed, **kwargs)

        self._aug_method = 'GP augmentation; PLAsTiCC DDF'

    def create_aug_obj_metadata(self, aug_obj, obj_metadata):
        """Create metadata for the DDF augmented event.

        The new metadata is based based on the metadata of the original event.

        Parameters
        ----------
        aug_obj : str
            Name of the augmented event in the form
                `[original event name]_[number of the augmentation]`.
        obj_metadata: pandas.DataFrame
            Metadata of the original event.

        Returns
        -------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        """
        aug_obj_metadata = obj_metadata.copy()
        aug_obj_metadata.name = aug_obj
        aug_obj_metadata.object_id = aug_obj
        aug_obj_metadata['augmented'] = True
        aug_obj_metadata['original_event'] = aug_obj.split('_')[0]

        z_spec = self.choose_z(obj_metadata['hostgal_specz'], **self._kwargs)
        z_photo, z_photo_error = self.compute_new_z_photo(z_spec)
        aug_obj_metadata['hostgal_specz'] = z_spec
        aug_obj_metadata['hostgal_photoz'] = z_photo
        aug_obj_metadata['hostgal_photoz_err'] = z_photo_error

        if obj_metadata['ddf']:
            aug_obj_metadata['ddf'] = True
        else:
            # If the original event was not a DDF observation, cannot simulate
            # a DDF event.
            aug_obj_metadata = None

        return aug_obj_metadata

    def _choose_target_number_obs(self, aug_obj_metadata):
        """Randomly choose the target number of light curve observations.

        Using Gaussian mixture models, we model the number of observations in
        the test set events simulated on the Deep Drilling Field (DDF).

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.

        Returns
        -------
        target_number_obs : int
            The target number of observations in the new light curve.

        Notes
        -----
        This function is adapted from the code developed in [1]_. In
        particular, the funtion
        `PlasticcAugmentor._choose_target_observation_count` of
        `avocado/plasticc.py`.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        # Estimate the distribution of number of observations in the
        # DDF regions with a mixture of 2 gaussian distributions.
        gauss_choice = self._rs.choice(2, p=[0.34393457, 0.65606543])
        if gauss_choice == 0:
            mean = 57.36015146
            var = np.sqrt(271.58889272)
        elif gauss_choice == 1:
            mean = 92.7741619
            var = np.sqrt(338.53085446)
        target_number_obs = int(
            np.clip(self._rs.normal(mean, var), 20, None))

        return target_number_obs

    def _compute_obs_uncertainty(self, aug_obj_data, aug_obj_metadata):
        """Compute and add uncertainty to the light curve observations.

        Following [1]_, we estimate the flux uncertainties for each
        passband with a lognormal distribution for the Deep Drilling Field
        (DDF) survey. Each passband was modeled individually with test set
        events.
        The flux uncertanty of the augmented events is the combination of the
        flux uncertainty of the augmented events predicted by the GP in
        quadrature with a value drawn from the flux uncertainty distribution
        described above.

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        obj_data : pandas.DataFrame
            Observations of the original event.

        Returns
        -------
        aug_obj_data : pandas.DataFrame
            Observations of the augmented event.

        Notes
        -----
        This function is adapted from the code developed in [1]_. In
        particular, the funtion
        `PlasticcAugmentor._simulate_light_curve_uncertainties` of
        `avocado/plasticc.py`.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        # Make a copy of the original data to avoid modifying it
        aug_obj_data = aug_obj_data.copy()

        # Skip this function if there are no observations
        if len(aug_obj_data) == 0:
            return aug_obj_data

        # The uncertainty levels of the observations in each passband can be
        # modeled with a lognormal distribution. See [1].
        # Lognormal parameters
        pb_noises = {'lsstu': (0.68, 0.26), 'lsstg': (0.25, 0.50),
                     'lsstr': (0.16, 0.36), 'lssti': (0.53, 0.27),
                     'lsstz': (0.88, 0.22), 'lssty': (1.76, 0.23)}

        # Calculate the new uncertainty levels for each passband
        lognormal_parameters = []
        for pb in aug_obj_data['filter']:
            try:
                lognormal_parameters.append(pb_noises[pb])
            except KeyError:
                raise ValueError(f'The noise properties of the passband {pb} '
                                 f'are not known. Add them to '
                                 f'`GPAugment._compute_obs_uncertainty`.')
        lognormal_parameters = np.array(lognormal_parameters)

        # Combine the flux uncertainty of the augmented events predicted by
        # the GP in quadrature with a value drawn from the flux uncertainty
        # distribution of the test set
        add_stds = self._rs.lognormal(
            lognormal_parameters[:, 0], lognormal_parameters[:, 1])
        noise_add = self._rs.normal(loc=0.0, scale=add_stds)
        aug_obj_data['flux'] += noise_add  # add noise to increase variability
        aug_obj_data['flux_error'] = np.sqrt(aug_obj_data['flux_error'] ** 2 + 
                                             add_stds ** 2)
        return aug_obj_data

    def _add_augment_objs_to_dataset(self, aug_objs_data, aug_objs_metadata):
        """Add the new events to the dataset.

        There are two new datasets created:
            - aug_dataset: original + augmented events
            - only_new_dataset: augmented events

        Parameters
        ----------
        aug_objs_data : list of pandas.DataFrame
            List containing the observations of each augmentation of each
            event.
        aug_objs_metadata : list of pandas.DataFrame
            Ordered list containing the metadata of each augmentation of each
            event.

        Notes
        -----
        This method further adds an inner metadata to the events. This was used
        in snmachine < v2.0.
        """
        super(PlasticcWFDAugment, self)._add_augment_objs_to_dataset()
        for obj_data in aug_objs_data:
            try:
                obj = obj_data['object_id'].iloc[0]
                self.dataset.set_inner_metadata(obj)
            except (IndexError, TypeError):
                print(obj_data)
                print('Failed attempt.')


class BaselineV20PlasticcWFDAugment(GPAugment):
    """Augment the Wide-Fast-Deep (WFD) events in the LSST-like dataset
    corresponding to the PLAsTiCC-like `baseline_v2.0_10yrs.db` cadence. The
    augmentation uses Gaussian Process extrapolation method (see
    `snaugment.GPAugment`).
    """

    def __init__(self, dataset, path_saved_gps, objs_number_to_aug=None,
                 z_table=None, max_duration=None,
                 cosmology=FlatLambdaCDM(**{"H0": 70, "Om0": 0.3,
                                            "Tcmb0": 2.725}),
                 random_seed=None, **kwargs):
        """Class enclosing the Gaussian Process augmentation of DDF events.

        This class augments the Deep Drilling Field (DDF) events in the
        LSST-like dataset corresponding to the `baseline_v2.0_10yrs.db`
        cadence.

        Parameters
        ----------
        dataset : Dataset object (`sndata` class)
            Dataset to augment.
        path_saved_gps: str
            Path to the Gaussian Process files.
        objs_number_to_aug : {`None`, 'all', dict}, optional
            Specify which events to augment and by how much. If `None`, the
            dataset it not augmented. If `all`, all the events are augmented
            10 times. If a dictionary is provided, it should be in the form of:
                event: number of times to augment that event.
        z_table : {None, pandas.DataFrame}, optional
            Dataset of the spectroscopic and photometric redshift, and
            photometric redshift error of events. This table is used to
            generate the photometric redshift and respective error for the
            augmented events. If `None`, this table is generated from the
            events in the original dataset.
        max_duration : {None, float}, optional
            Maximum duration of the augmented light curves. If `None`, it is
            set to the length of the longest event in `dataset`.
        cosmology : astropy.cosmology.core.COSMOLOGY, optional
            Cosmology from `astropy` with the cosmologic parameters already
            defined. By default it assumes Flat LambdaCDM with parameters
            `H0 = 70`, `Om0 = 0.3` and `T_cmb0 = 2.725`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
            If given, it must be between 0 and 2**32 - 1.
        **kwargs : dict, optional
            Optional keywords to pass arguments into
            `snamchine.gps.compute_gps`.

        Notes
        -----
        This augmentation is based on [1]_.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        super().__init__(dataset=dataset, path_saved_gps=path_saved_gps,
                         objs_number_to_aug=objs_number_to_aug,
                         choose_z=choose_z_wfd_plasticc, z_table=z_table,
                         max_duration=max_duration, cosmology=cosmology,
                         random_seed=random_seed, **kwargs)

        self._aug_method = 'GP augmentation; baseline_v2.0_10yrs DDF'

    def create_aug_obj_metadata(self, aug_obj, obj_metadata):
        """Create metadata for the DDF augmented event.

        The new metadata is based based on the metadata of the original event.

        Parameters
        ----------
        aug_obj : str
            Name of the augmented event in the form
                `[original event name]_[number of the augmentation]`.
        obj_metadata: pandas.DataFrame
            Metadata of the original event.

        Returns
        -------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        """
        aug_obj_metadata = obj_metadata.copy()
        aug_obj_metadata.name = aug_obj
        aug_obj_metadata.object_id = aug_obj
        aug_obj_metadata['augmented'] = True
        aug_obj_metadata['original_event'] = aug_obj.split('_')[0]

        z_spec = self.choose_z(obj_metadata['hostgal_specz'], **self._kwargs)
        z_photo, z_photo_error = self.compute_new_z_photo(z_spec)
        aug_obj_metadata['hostgal_specz'] = z_spec
        aug_obj_metadata['hostgal_photoz'] = z_photo
        aug_obj_metadata['hostgal_photoz_err'] = z_photo_error

        if obj_metadata['ddf']:
            aug_obj_metadata['ddf'] = True
        else:
            # If the original event was not a DDF observation, cannot simulate
            # a DDF event.
            aug_obj_metadata = None

        return aug_obj_metadata

    def _choose_target_number_obs(self, aug_obj_metadata):
        """Randomly choose the target number of light curve observations.

        Using Gaussian mixture models, we model the number of observations in
        the test set events simulated on the Deep Drilling Field (DDF).

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.

        Returns
        -------
        target_number_obs : int
            The target number of observations in the new light curve.

        Notes
        -----
        This function is adapted from the code developed in [1]_. In
        particular, the funtion
        `PlasticcAugmentor._choose_target_observation_count` of
        `avocado/plasticc.py`.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        # Estimate the distribution of number of observations in the
        # DDF regions with a mixture of 3 gaussian distributions.
        gauss_choice = self._rs.choice(3,
                                       p=[0.33829852, 0.36584055, 0.29586093])
        if gauss_choice == 0:
            mean = 17.57122788
            var = np.sqrt(88.81445042)
        elif gauss_choice == 1:
            mean = 98.76335824
            var = np.sqrt(175.56964147)
        elif gauss_choice == 2:
            mean = 55.84976549
            var = np.sqrt(90.92855255)
        target_number_obs = int(
            np.clip(self._rs.normal(mean, var), 4, None))

        return target_number_obs

    def _compute_obs_uncertainty(self, aug_obj_data, aug_obj_metadata):
        """Compute and add uncertainty to the light curve observations.

        We estimate the flux uncertainties for each passband of the Drilling
        Field (DDF) survey with the logarithm of a Gaussian mixture model.
        Each passband was modeled individually with test set events.
        The flux uncertanty of the augmented events is the combination of the
        flux uncertainty of the augmented events predicted by the GP in
        quadrature with a value drawn from the flux uncertainty distribution
        described above.

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        obj_data : pandas.DataFrame
            Observations of the original event.

        Returns
        -------
        aug_obj_data : pandas.DataFrame
            Observations of the augmented event.

        Notes
        -----
        This function is adapted from the code developed in [1]_. In
        particular, the funtion
        `PlasticcAugmentor._simulate_light_curve_uncertainties` of
        `avocado/plasticc.py`.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        # Make a copy of the original data to avoid modifying it
        aug_obj_data = aug_obj_data.copy()

        # Skip this function if there are no observations
        if len(aug_obj_data) == 0:
            return aug_obj_data

        # The uncertainty levels of the observations in each passband can be
        # modeled with the logaritm of a Gaussian mixture model.
        # Passband parameters
        pb_noises = {'lsstu': {'weights': [0.50890351, 0.1674677, 0.3236288],
                               'means': [1.15910188, 1.57513178, 0.75304931],
                               'covars': [0.02570773, 0.12252229, 0.02907247]},
                     'lsstg': {'weights': [0.31367609, 0.14129098, 0.54503293],
                               'means': [-0.25116208, 0.6020499, 0.1060065],
                               'covars': [0.03202531, 0.06563274, 0.02546574]},
                     'lsstr': {'weights': [0.31646206, 0.17148511, 0.51205283],
                               'means': [-0.20202721, 0.46961199, 0.15050021],
                               'covars': [0.02614601, 0.1284051, 0.02230432]},
                     'lssti': {'weights': [0.3383791, 0.50591912, 0.15570179],
                               'means': [0.25100765, 0.53670117, 0.95752681],
                               'covars': [0.02520334, 0.01984151, 0.12698874]},
                     'lsstz': {'weights': [0.44948967, 0.33357324, 0.21693709],
                               'means': [0.92402643, 0.71355807, 1.25239874],
                               'covars': [0.01066763, 0.01201745, 0.03381948]},
                     'lssty': {'weights': [0.16064206, 0.34760465, 0.49175329],
                               'means': [2.23060362, 1.57224717, 1.81965783],
                               'covars': [0.06961287, 0.02034714, 0.01822751]}}

        filter_set = self.dataset.filter_set
        add_stds = np.zeros(len(aug_obj_data))
        for pb in filter_set:
            is_pb = aug_obj_data['filter'] == pb
            aug_obj_data_pb = aug_obj_data[is_pb]

            # Generate new uncertainty levels for each observation
            try:
                pb_unc_vals = pb_noises[pb]
            except KeyError:
                raise ValueError(f'The noise properties of the passband {pb} '
                                 f'are not known. Add them to '
                                 f'`GPAugment._compute_obs_uncertainty`.')
            number_comps = len(pb_unc_vals['weights'])
            number_obs_pb = len(aug_obj_data_pb)
            gauss_choice = self._rs.choice(number_comps, size=number_obs_pb,
                                           p=pb_unc_vals['weights'])
            gmm_sample = np.zeros(number_obs_pb)
            for i in np.arange(number_comps):
                is_i = gauss_choice == i
                gmm_sample[is_i] = self._rs.normal(
                    pb_unc_vals['means'][i], np.sqrt(pb_unc_vals['covars'][i]),
                    size=np.sum(is_i))

            # Store the new uncertainty levels for each observation
            add_stds[is_pb] = gmm_sample
        add_stds = np.exp(add_stds)

        # Combine the flux uncertainty of the augmented events predicted by
        # the GP in quadrature with a value drawn from the flux uncertainty
        # distribution of the test set
        noise_add = self._rs.normal(loc=0.0, scale=add_stds)
        aug_obj_data['flux'] += noise_add  # add noise to increase variability
        aug_obj_data['flux_error'] = np.sqrt(aug_obj_data['flux_error'] ** 2 + 
                                             add_stds ** 2)
        return aug_obj_data


class BaselineV20WFDAugment(GPAugment):
    """Augment the Wide-Fast-Deep (WFD) events in the LSST-like dataset
    corresponding to the `baseline_v2.0_10yrs.db` cadence. The
    augmentation uses Gaussian Process extrapolation method (see
    `snaugment.GPAugment`).
    """

    def __init__(self, dataset, path_saved_gps, objs_number_to_aug=None,
                 z_table=None, max_duration=None,
                 cosmology=FlatLambdaCDM(**{"H0": 70, "Om0": 0.3,
                                            "Tcmb0": 2.725}),
                 random_seed=None, **kwargs):
        """Class enclosing the Gaussian Process augmentation of WFD events.

        This class augments the Wide-Fast-Deep (WFD) events in the
        LSST-like dataset corresponding to the `baseline_v2.0_10yrs.db`
        cadence.

        Parameters
        ----------
        dataset : Dataset object (`sndata` class)
            Dataset to augment.
        path_saved_gps: str
            Path to the Gaussian Process files.
        objs_number_to_aug : {`None`, 'all', dict}, optional
            Specify which events to augment and by how much. If `None`, the
            dataset it not augmented. If `all`, all the events are augmented
            10 times. If a dictionary is provided, it should be in the form of:
                event: number of times to augment that event.
        z_table : {None, pandas.DataFrame}, optional
            Dataset of the spectroscopic and photometric redshift, and
            photometric redshift error of events. This table is used to
            generate the photometric redshift and respective error for the
            augmented events. If `None`, this table is generated from the
            events in the original dataset.
        max_duration : {None, float}, optional
            Maximum duration of the augmented light curves. If `None`, it is
            set to the length of the longest event in `dataset`.
        cosmology : astropy.cosmology.core.COSMOLOGY, optional
            Cosmology from `astropy` with the cosmologic parameters already
            defined. By default it assumes Flat LambdaCDM with parameters
            `H0 = 70`, `Om0 = 0.3` and `T_cmb0 = 2.725`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
            If given, it must be between 0 and 2**32 - 1.
        **kwargs : dict, optional
            Optional keywords to pass arguments into
            `snamchine.gps.compute_gps`.

        Notes
        -----
        This augmentation is based on [1]_.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        super().__init__(dataset=dataset, path_saved_gps=path_saved_gps,
                         objs_number_to_aug=objs_number_to_aug,
                         choose_z=choose_z_wfd_basev2, z_table=z_table,
                         max_duration=max_duration, cosmology=cosmology,
                         random_seed=random_seed, **kwargs)

        self._aug_method = 'GP augmentation; baseline_v2.0_10yrs WFD'

    def create_aug_obj_metadata(self, aug_obj, obj_metadata):
        """Create metadata for the WFD augmented event.

        The new metadata is based based on the metadata of the original event.

        Parameters
        ----------
        aug_obj : str
            Name of the augmented event in the form
                `[original event name]_[number of the augmentation]`.
        obj_metadata: pandas.DataFrame
            Metadata of the original event.

        Returns
        -------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        """
        aug_obj_metadata = obj_metadata.copy()
        aug_obj_metadata.name = aug_obj
        aug_obj_metadata.object_id = aug_obj
        aug_obj_metadata['augmented'] = True
        aug_obj_metadata['original_event'] = aug_obj.split('_')[0]

        z_spec = self.choose_z(obj_metadata['hostgal_specz'], **self._kwargs)
        z_photo, z_photo_error = self.compute_new_z_photo(z_spec)
        aug_obj_metadata['hostgal_specz'] = z_spec
        aug_obj_metadata['hostgal_photoz'] = z_photo
        aug_obj_metadata['hostgal_photoz_err'] = z_photo_error

        # The new event will be in WFD regardless of the original event; we
        # can always degrade DDF events to simulate a WFD event
        aug_obj_metadata['ddf'] = False

        return aug_obj_metadata

    def _choose_target_number_obs(self, aug_obj_metadata):
        """Randomly choose the target number of light curve observations.

        Using Gaussian mixture models, we model the number of observations in
        the test set events simulated on the Wide-Fast-Deep (WFD).

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.

        Returns
        -------
        target_number_obs : int
            The target number of observations in the new light curve.

        Notes
        -----
        This function is adapted from the code developed in [1]_. In
        particular, the funtion
        `PlasticcAugmentor._choose_target_observation_count` of
        `avocado/plasticc.py`.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        # Estimate the distribution of number of observations in WFD
        # with a mixture of 2 gaussian distributions.
        gauss_choice = self._rs.choice(2, p=[0.679, 0.321])
        if gauss_choice == 0:
            mean = 33.6
            var = np.sqrt(199.09)
        elif gauss_choice == 1:
            mean = 65.1
            var = np.sqrt(453.67)
        target_number_obs = int(
            np.clip(self._rs.normal(mean, var), 3, None))

        return target_number_obs

    def _compute_obs_uncertainty(self, aug_obj_data, aug_obj_metadata):
        """Compute and add uncertainty to the light curve observations.

        Following [1]_, we estimate the flux uncertainties for each
        passband with a lognormal distribution for the Wide-Fast-Deep (WFD)
        survey. Each passband was modeled individually with test set events.
        The flux uncertanty of the augmented events is the combination of the
        flux uncertainty of the augmented events predicted by the GP in
        quadrature with a value drawn from the flux uncertainty distribution
        described above.

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        obj_data : pandas.DataFrame
            Observations of the original event.

        Returns
        -------
        aug_obj_data : pandas.DataFrame
            Observations of the augmented event.

        Notes
        -----
        This function is adapted from the code developed in [1]_. In
        particular, the funtion
        `PlasticcAugmentor._simulate_light_curve_uncertainties` of
        `avocado/plasticc.py`.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        # Make a copy of the original data to avoid modifying it
        aug_obj_data = aug_obj_data.copy()

        # Skip this function if there are no observations
        if len(aug_obj_data) == 0:
            return aug_obj_data

        # The uncertainty levels of the observations in each passband can be
        # modeled with the logaritm of a Gaussian mixture model.
        # Lognormal parameters from GMM:
        pb_noises = {'lsstu': {'weights': np.array([0.29, 0.71]),
                               'means': np.array([2.24, 1.93]),
                               'covars': np.array([0.38, 0.09])},
                     'lsstg': {'weights': np.array([0.88, 0.12]),
                               'means': np.array([1.26, 1.98]),
                               'covars': np.array([0.11, 0.85])},
                     'lsstr': {'weights': np.array([0.29, 0.71]),
                               'means': np.array([1.89, 1.58]),
                               'covars': np.array([0.34, 0.08])},
                     'lssti': {'weights': np.array([0.63, 0.37]),
                               'means': np.array([1.99, 2.45]),
                               'covars': np.array([0.08, 0.20])},
                     'lsstz': {'weights': np.array([1.]),
                               'means': np.array([2.67]),
                               'covars': np.array([0.14])},
                     'lssty': {'weights': np.array([1.]),
                               'means': np.array([3.23]),
                               'covars': np.array([0.14])}}

        # Calculate the new uncertainty levels for each passband
        # Initialize the variable to store the mean and standard deviation
        lognormal_parameters = np.zeros((len(aug_obj_data), 2))  # mean, std

        filter_set = self.dataset.filter_set
        for pb in filter_set:  # go through each passband
            is_pb = aug_obj_data['filter'] == pb
            aug_obj_data_pb = aug_obj_data[is_pb]

            # Generate new uncertainty levels for each observation
            try:
                pb_unc_vals = pb_noises[pb]
            except KeyError:
                raise ValueError(f'The noise properties of the passband {pb} '
                                 f'are not known. Add them to '
                                 f'`GPAugment._compute_obs_uncertainty`.')
            # For each event, choose from which gaussian the unc. will come
            number_comps = len(pb_unc_vals['weights'])
            number_obs_pb = len(aug_obj_data_pb)
            gauss_choice = self._rs.choice(number_comps, size=number_obs_pb,
                                           p=pb_unc_vals['weights'])

            # Store the lognormal parameters to later use
            means = pb_unc_vals['means'][gauss_choice]
            covars = pb_unc_vals['covars'][gauss_choice]
            lognormal_parameters[is_pb, 0] = means
            lognormal_parameters[is_pb, 1] = np.sqrt(covars)

        # Combine the flux uncertainty of the augmented events predicted by
        # the GP in quadrature with a value drawn from the flux uncertainty
        # distribution of the test set
        add_stds = self._rs.lognormal(lognormal_parameters[:, 0],
                                      lognormal_parameters[:, 1])
        noise_add = self._rs.normal(loc=0.0, scale=add_stds)
        aug_obj_data['flux'] += noise_add  # add noise to increase variability
        aug_obj_data['flux_error'] = np.sqrt(aug_obj_data['flux_error'] ** 2 + 
                                             add_stds ** 2)
        return aug_obj_data


class PrestoV20WFDAugment(GPAugment):
    """Augment the Wide-Fast-Deep (WFD) events in the LSST-like dataset
    corresponding to the `presto_gap2.5_mix_v2.0_10yrs.db` cadence. The
    augmentation uses Gaussian Process extrapolation method (see
    `snaugment.GPAugment`).
    """

    def __init__(self, dataset, path_saved_gps, objs_number_to_aug=None,
                 z_table=None, max_duration=None,
                 cosmology=FlatLambdaCDM(**{"H0": 70, "Om0": 0.3,
                                            "Tcmb0": 2.725}),
                 random_seed=None, **kwargs):
        """Class enclosing the Gaussian Process augmentation of WFD events.

        This class augments the Wide-Fast-Deep (WFD) events in the
        LSST-like dataset corresponding to the
        `presto_gap2.5_mix_v2.0_10yrs.db` cadence.

        Parameters
        ----------
        dataset : Dataset object (`sndata` class)
            Dataset to augment.
        path_saved_gps: str
            Path to the Gaussian Process files.
        objs_number_to_aug : {`None`, 'all', dict}, optional
            Specify which events to augment and by how much. If `None`, the
            dataset it not augmented. If `all`, all the events are augmented
            10 times. If a dictionary is provided, it should be in the form of:
                event: number of times to augment that event.
        z_table : {None, pandas.DataFrame}, optional
            Dataset of the spectroscopic and photometric redshift, and
            photometric redshift error of events. This table is used to
            generate the photometric redshift and respective error for the
            augmented events. If `None`, this table is generated from the
            events in the original dataset.
        max_duration : {None, float}, optional
            Maximum duration of the augmented light curves. If `None`, it is
            set to the length of the longest event in `dataset`.
        cosmology : astropy.cosmology.core.COSMOLOGY, optional
            Cosmology from `astropy` with the cosmologic parameters already
            defined. By default it assumes Flat LambdaCDM with parameters
            `H0 = 70`, `Om0 = 0.3` and `T_cmb0 = 2.725`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
            If given, it must be between 0 and 2**32 - 1.
        **kwargs : dict, optional
            Optional keywords to pass arguments into
            `snamchine.gps.compute_gps`.

        Notes
        -----
        This augmentation is based on [1]_.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        super().__init__(dataset=dataset, path_saved_gps=path_saved_gps,
                         objs_number_to_aug=objs_number_to_aug,
                         choose_z=choose_z_wfd_basev2, z_table=z_table,
                         max_duration=max_duration, cosmology=cosmology,
                         random_seed=random_seed, **kwargs)

        self._aug_method = 'GP augmentation; presto_gap2.5_mix_v2.0_10yrs WFD'

    def create_aug_obj_metadata(self, aug_obj, obj_metadata):
        """Create metadata for the WFD augmented event.

        The new metadata is based based on the metadata of the original event.

        Parameters
        ----------
        aug_obj : str
            Name of the augmented event in the form
                `[original event name]_[number of the augmentation]`.
        obj_metadata: pandas.DataFrame
            Metadata of the original event.

        Returns
        -------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        """
        aug_obj_metadata = obj_metadata.copy()
        aug_obj_metadata.name = aug_obj
        aug_obj_metadata.object_id = aug_obj
        aug_obj_metadata['augmented'] = True
        aug_obj_metadata['original_event'] = aug_obj.split('_')[0]

        z_spec = self.choose_z(obj_metadata['hostgal_specz'], **self._kwargs)
        z_photo, z_photo_error = self.compute_new_z_photo(z_spec)
        aug_obj_metadata['hostgal_specz'] = z_spec
        aug_obj_metadata['hostgal_photoz'] = z_photo
        aug_obj_metadata['hostgal_photoz_err'] = z_photo_error

        # The new event will be in WFD regardless of the original event; we
        # can always degrade DDF events to simulate a WFD event
        aug_obj_metadata['ddf'] = False

        return aug_obj_metadata

    def _choose_target_number_obs(self, aug_obj_metadata):
        """Randomly choose the target number of light curve observations.

        Using Gaussian mixture models, we model the number of observations in
        the test set events simulated on the Wide-Fast-Deep (WFD).

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.

        Returns
        -------
        target_number_obs : int
            The target number of observations in the new light curve.

        Notes
        -----
        This function is adapted from the code developed in [1]_. In
        particular, the funtion
        `PlasticcAugmentor._choose_target_observation_count` of
        `avocado/plasticc.py`.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        # Estimate the distribution of number of observations in WFD
        # with a mixture of 2 gaussian distributions.
        gauss_choice = self._rs.choice(2, p=[0.626, 0.374])
        if gauss_choice == 0:
            mean = 27.11
            var = np.sqrt(125.04)
        elif gauss_choice == 1:
            mean = 53.72
            var = np.sqrt(252.05)
        target_number_obs = int(
            np.clip(self._rs.normal(mean, var), 3, None))

        return target_number_obs

    def _compute_obs_uncertainty(self, aug_obj_data, aug_obj_metadata):
        """Compute and add uncertainty to the light curve observations.

        Following [1]_, we estimate the flux uncertainties for each
        passband with a lognormal distribution for the Wide-Fast-Deep (WFD)
        survey. Each passband was modeled individually with test set events.
        The flux uncertanty of the augmented events is the combination of the
        flux uncertainty of the augmented events predicted by the GP in
        quadrature with a value drawn from the flux uncertainty distribution
        described above.

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        obj_data : pandas.DataFrame
            Observations of the original event.

        Returns
        -------
        aug_obj_data : pandas.DataFrame
            Observations of the augmented event.

        Notes
        -----
        This function is adapted from the code developed in [1]_. In
        particular, the funtion
        `PlasticcAugmentor._simulate_light_curve_uncertainties` of
        `avocado/plasticc.py`.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        # Make a copy of the original data to avoid modifying it
        aug_obj_data = aug_obj_data.copy()

        # Skip this function if there are no observations
        if len(aug_obj_data) == 0:
            return aug_obj_data

        # The uncertainty levels of the observations in each passband can be
        # modeled with the logaritm of a Gaussian mixture model.
        # Lognormal parameters from GMM:
        pb_noises = {'lsstu': {'weights': np.array([0.64, 0.36]),
                               'means': np.array([1.96, 2.35]),
                               'covars': np.array([0.09, 0.28])},
                     'lsstg': {'weights': np.array([0.10, 0.90]),
                               'means': np.array([2.12, 1.41]),
                               'covars': np.array([0.98, 0.14])},
                     'lsstr': {'weights': np.array([0.23, 0.45, 0.32]),
                               'means': np.array([2.82, 1.60, 2.20]),
                               'covars': np.array([0.30, 0.08, 0.12])},
                     'lssti': {'weights': np.array([0.66, 0.34]),
                               'means': np.array([1.96, 2.48]),
                               'covars': np.array([0.10, 0.26])},
                     'lsstz': {'weights': np.array([1.]),
                               'means': np.array([2.68]),
                               'covars': np.array([0.17])},
                     'lssty': {'weights': np.array([0.62, 0.38]),
                               'means': np.array([3.07, 3.52]),
                               'covars': np.array([0.07, 0.13])}}

        # Calculate the new uncertainty levels for each passband
        # Initialize the variable to store the mean and standard deviation
        lognormal_parameters = np.zeros((len(aug_obj_data), 2))  # mean, std

        filter_set = self.dataset.filter_set
        for pb in filter_set:  # go through each passband
            is_pb = aug_obj_data['filter'] == pb
            aug_obj_data_pb = aug_obj_data[is_pb]

            # Generate new uncertainty levels for each observation
            try:
                pb_unc_vals = pb_noises[pb]
            except KeyError:
                raise ValueError(f'The noise properties of the passband {pb} '
                                 f'are not known. Add them to '
                                 f'`GPAugment._compute_obs_uncertainty`.')
            # For each event, choose from which gaussian the unc. will come
            number_comps = len(pb_unc_vals['weights'])
            number_obs_pb = len(aug_obj_data_pb)
            gauss_choice = self._rs.choice(number_comps, size=number_obs_pb,
                                           p=pb_unc_vals['weights'])

            # Store the lognormal parameters to later use
            means = pb_unc_vals['means'][gauss_choice]
            covars = pb_unc_vals['covars'][gauss_choice]
            lognormal_parameters[is_pb, 0] = means
            lognormal_parameters[is_pb, 1] = np.sqrt(covars)

        # Combine the flux uncertainty of the augmented events predicted by
        # the GP in quadrature with a value drawn from the flux uncertainty
        # distribution of the test set
        add_stds = self._rs.lognormal(lognormal_parameters[:, 0],
                                      lognormal_parameters[:, 1])
        noise_add = self._rs.normal(loc=0.0, scale=add_stds)
        aug_obj_data['flux'] += noise_add  # add noise to increase variability
        aug_obj_data['flux_error'] = np.sqrt(aug_obj_data['flux_error'] ** 2 + 
                                             add_stds ** 2)
        return aug_obj_data


class BaselineV20SubsetWFDAugment(GPAugment):
    """Augment the Wide-Fast-Deep (WFD) events in the LSST-like dataset
    corresponding to the years 1.5 to 3 in the  `baseline_v2.0_10yrs.db`
    cadence. The augmentation uses Gaussian Process extrapolation method (see
    `snaugment.GPAugment`).
    """

    def __init__(self, dataset, path_saved_gps, objs_number_to_aug=None,
                 z_table=None, max_duration=None,
                 cosmology=FlatLambdaCDM(**{"H0": 70, "Om0": 0.3,
                                            "Tcmb0": 2.725}),
                 random_seed=None, **kwargs):
        """Class enclosing the Gaussian Process augmentation of WFD events.

        This class augments the Wide-Fast-Deep (WFD) events in the
        LSST-like dataset corresponding to the `baseline_v2.0_10yrs.db`
        cadence.

        Parameters
        ----------
        dataset : Dataset object (`sndata` class)
            Dataset to augment.
        path_saved_gps: str
            Path to the Gaussian Process files.
        objs_number_to_aug : {`None`, 'all', dict}, optional
            Specify which events to augment and by how much. If `None`, the
            dataset it not augmented. If `all`, all the events are augmented
            10 times. If a dictionary is provided, it should be in the form of:
                event: number of times to augment that event.
        z_table : {None, pandas.DataFrame}, optional
            Dataset of the spectroscopic and photometric redshift, and
            photometric redshift error of events. This table is used to
            generate the photometric redshift and respective error for the
            augmented events. If `None`, this table is generated from the
            events in the original dataset.
        max_duration : {None, float}, optional
            Maximum duration of the augmented light curves. If `None`, it is
            set to the length of the longest event in `dataset`.
        cosmology : astropy.cosmology.core.COSMOLOGY, optional
            Cosmology from `astropy` with the cosmologic parameters already
            defined. By default it assumes Flat LambdaCDM with parameters
            `H0 = 70`, `Om0 = 0.3` and `T_cmb0 = 2.725`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
            If given, it must be between 0 and 2**32 - 1.
        **kwargs : dict, optional
            Optional keywords to pass arguments into
            `snamchine.gps.compute_gps`.

        Notes
        -----
        This augmentation is based on [1]_.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        super().__init__(dataset=dataset, path_saved_gps=path_saved_gps,
                         objs_number_to_aug=objs_number_to_aug,
                         choose_z=choose_z_wfd_basev2, z_table=z_table,
                         max_duration=max_duration, cosmology=cosmology,
                         random_seed=random_seed, **kwargs)

        self._aug_method = 'GP augmentation; subset baseline_v2.0_10yrs WFD'

    def create_aug_obj_metadata(self, aug_obj, obj_metadata):
        """Create metadata for the WFD augmented event.

        The new metadata is based based on the metadata of the original event.

        Parameters
        ----------
        aug_obj : str
            Name of the augmented event in the form
                `[original event name]_[number of the augmentation]`.
        obj_metadata: pandas.DataFrame
            Metadata of the original event.

        Returns
        -------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        """
        aug_obj_metadata = obj_metadata.copy()
        aug_obj_metadata.name = aug_obj
        aug_obj_metadata.object_id = aug_obj
        aug_obj_metadata['augmented'] = True
        aug_obj_metadata['original_event'] = aug_obj.split('_')[0]

        z_spec = self.choose_z(obj_metadata['hostgal_specz'], **self._kwargs)
        z_photo, z_photo_error = self.compute_new_z_photo(z_spec)
        aug_obj_metadata['hostgal_specz'] = z_spec
        aug_obj_metadata['hostgal_photoz'] = z_photo
        aug_obj_metadata['hostgal_photoz_err'] = z_photo_error

        # The new event will be in WFD regardless of the original event; we
        # can always degrade DDF events to simulate a WFD event
        aug_obj_metadata['ddf'] = False

        return aug_obj_metadata

    def _choose_target_number_obs(self, aug_obj_metadata):
        """Randomly choose the target number of light curve observations.

        Using Gaussian mixture models, we model the number of observations in
        the test set events simulated on the Wide-Fast-Deep (WFD).

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.

        Returns
        -------
        target_number_obs : int
            The target number of observations in the new light curve.

        Notes
        -----
        This function is adapted from the code developed in [1]_. In
        particular, the funtion
        `PlasticcAugmentor._choose_target_observation_count` of
        `avocado/plasticc.py`.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        # Estimate the distribution of number of observations in WFD
        # with a mixture of 2 gaussian distributions.
        gauss_choice = self._rs.choice(2, p=[0.322, 0.678])
        if gauss_choice == 0:
            mean = 16.86
            var = np.sqrt(46.62)
        elif gauss_choice == 1:
            mean = 61.15
            var = np.sqrt(561.06)
        target_number_obs = int(
            np.clip(self._rs.normal(mean, var), 3, None))

        return target_number_obs

    def _compute_obs_uncertainty(self, aug_obj_data, aug_obj_metadata):
        """Compute and add uncertainty to the light curve observations.

        Following [1]_, we estimate the flux uncertainties for each
        passband with a lognormal distribution for the Wide-Fast-Deep (WFD)
        survey. Each passband was modeled individually with test set events.
        The flux uncertanty of the augmented events is the combination of the
        flux uncertainty of the augmented events predicted by the GP in
        quadrature with a value drawn from the flux uncertainty distribution
        described above.

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        obj_data : pandas.DataFrame
            Observations of the original event.

        Returns
        -------
        aug_obj_data : pandas.DataFrame
            Observations of the augmented event.

        Notes
        -----
        This function is adapted from the code developed in [1]_. In
        particular, the funtion
        `PlasticcAugmentor._simulate_light_curve_uncertainties` of
        `avocado/plasticc.py`.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        # Make a copy of the original data to avoid modifying it
        aug_obj_data = aug_obj_data.copy()

        # Skip this function if there are no observations
        if len(aug_obj_data) == 0:
            return aug_obj_data

        # The uncertainty levels of the observations in each passband can be
        # modeled with the logaritm of a Gaussian mixture model.
        # Lognormal parameters from GMM:
        pb_noises = {'lsstu': {'weights': np.array([0.24, 0.76]),
                               'means': np.array([2.34, 1.92]),
                               'covars': np.array([0.46, 0.10])},
                     'lsstg': {'weights': np.array([0.88, 0.12]),
                               'means': np.array([1.26, 2.06]),
                               'covars': np.array([0.11, 0.93])},
                     'lsstr': {'weights': np.array([0.27, 0.73]),
                               'means': np.array([1.92, 1.57]),
                               'covars': np.array([0.38, 0.09])},
                     'lssti': {'weights': np.array([0.62, 0.38]),
                               'means': np.array([1.97, 2.45]),
                               'covars': np.array([0.08, 0.22])},
                     'lsstz': {'weights': np.array([1.]),
                               'means': np.array([2.65]),
                               'covars': np.array([0.14])},
                     'lssty': {'weights': np.array([1.]),
                               'means': np.array([3.23]),
                               'covars': np.array([0.15])}}

        # Calculate the new uncertainty levels for each passband
        # Initialize the variable to store the mean and standard deviation
        lognormal_parameters = np.zeros((len(aug_obj_data), 2))  # mean, std

        filter_set = self.dataset.filter_set
        for pb in filter_set:  # go through each passband
            is_pb = aug_obj_data['filter'] == pb
            aug_obj_data_pb = aug_obj_data[is_pb]

            # Generate new uncertainty levels for each observation
            try:
                pb_unc_vals = pb_noises[pb]
            except KeyError:
                raise ValueError(f'The noise properties of the passband {pb} '
                                 f'are not known. Add them to '
                                 f'`GPAugment._compute_obs_uncertainty`.')
            # For each event, choose from which gaussian the unc. will come
            number_comps = len(pb_unc_vals['weights'])
            number_obs_pb = len(aug_obj_data_pb)
            gauss_choice = self._rs.choice(number_comps, size=number_obs_pb,
                                           p=pb_unc_vals['weights'])

            # Store the lognormal parameters to later use
            means = pb_unc_vals['means'][gauss_choice]
            covars = pb_unc_vals['covars'][gauss_choice]
            lognormal_parameters[is_pb, 0] = means
            lognormal_parameters[is_pb, 1] = np.sqrt(covars)

        # Combine the flux uncertainty of the augmented events predicted by
        # the GP in quadrature with a value drawn from the flux uncertainty
        # distribution of the test set
        add_stds = self._rs.lognormal(lognormal_parameters[:, 0],
                                      lognormal_parameters[:, 1])
        noise_add = self._rs.normal(loc=0.0, scale=add_stds)
        aug_obj_data['flux'] += noise_add  # add noise to increase variability
        aug_obj_data['flux_error'] = np.sqrt(aug_obj_data['flux_error'] ** 2 + 
                                             add_stds ** 2)
        return aug_obj_data


class NorollV20SubsetWFDAugment(GPAugment):
    """Augment the Wide-Fast-Deep (WFD) events in the LSST-like dataset
    corresponding to the years 1.5 to 3 in the `noroll_v2.0_10yrs.db` cadence.
    The augmentation uses Gaussian Process extrapolation method (see
    `snaugment.GPAugment`).
    """

    def __init__(self, dataset, path_saved_gps, objs_number_to_aug=None,
                 z_table=None, max_duration=None,
                 cosmology=FlatLambdaCDM(**{"H0": 70, "Om0": 0.3,
                                            "Tcmb0": 2.725}),
                 random_seed=None, **kwargs):
        """Class enclosing the Gaussian Process augmentation of WFD events.

        This class augments the Wide-Fast-Deep (WFD) events in the
        LSST-like dataset corresponding to the `noroll_v2.0_10yrs.db`
        cadence.

        Parameters
        ----------
        dataset : Dataset object (`sndata` class)
            Dataset to augment.
        path_saved_gps: str
            Path to the Gaussian Process files.
        objs_number_to_aug : {`None`, 'all', dict}, optional
            Specify which events to augment and by how much. If `None`, the
            dataset it not augmented. If `all`, all the events are augmented
            10 times. If a dictionary is provided, it should be in the form of:
                event: number of times to augment that event.
        z_table : {None, pandas.DataFrame}, optional
            Dataset of the spectroscopic and photometric redshift, and
            photometric redshift error of events. This table is used to
            generate the photometric redshift and respective error for the
            augmented events. If `None`, this table is generated from the
            events in the original dataset.
        max_duration : {None, float}, optional
            Maximum duration of the augmented light curves. If `None`, it is
            set to the length of the longest event in `dataset`.
        cosmology : astropy.cosmology.core.COSMOLOGY, optional
            Cosmology from `astropy` with the cosmologic parameters already
            defined. By default it assumes Flat LambdaCDM with parameters
            `H0 = 70`, `Om0 = 0.3` and `T_cmb0 = 2.725`.
        random_seed : int, optional
            Random seed used. Saving this seed allows reproducible results.
            If given, it must be between 0 and 2**32 - 1.
        **kwargs : dict, optional
            Optional keywords to pass arguments into
            `snamchine.gps.compute_gps`.

        Notes
        -----
        This augmentation is based on [1]_.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        super().__init__(dataset=dataset, path_saved_gps=path_saved_gps,
                         objs_number_to_aug=objs_number_to_aug,
                         choose_z=choose_z_wfd_basev2, z_table=z_table,
                         max_duration=max_duration, cosmology=cosmology,
                         random_seed=random_seed, **kwargs)

        self._aug_method = 'GP augmentation; subset noroll_v2.0_10yrs WFD'

    def create_aug_obj_metadata(self, aug_obj, obj_metadata):
        """Create metadata for the WFD augmented event.

        The new metadata is based based on the metadata of the original event.

        Parameters
        ----------
        aug_obj : str
            Name of the augmented event in the form
                `[original event name]_[number of the augmentation]`.
        obj_metadata: pandas.DataFrame
            Metadata of the original event.

        Returns
        -------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        """
        aug_obj_metadata = obj_metadata.copy()
        aug_obj_metadata.name = aug_obj
        aug_obj_metadata.object_id = aug_obj
        aug_obj_metadata['augmented'] = True
        aug_obj_metadata['original_event'] = aug_obj.split('_')[0]

        z_spec = self.choose_z(obj_metadata['hostgal_specz'], **self._kwargs)
        z_photo, z_photo_error = self.compute_new_z_photo(z_spec)
        aug_obj_metadata['hostgal_specz'] = z_spec
        aug_obj_metadata['hostgal_photoz'] = z_photo
        aug_obj_metadata['hostgal_photoz_err'] = z_photo_error

        # The new event will be in WFD regardless of the original event; we
        # can always degrade DDF events to simulate a WFD event
        aug_obj_metadata['ddf'] = False

        return aug_obj_metadata

    def _choose_target_number_obs(self, aug_obj_metadata):
        """Randomly choose the target number of light curve observations.

        Using Gaussian mixture models, we model the number of observations in
        the test set events simulated on the Wide-Fast-Deep (WFD).

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.

        Returns
        -------
        target_number_obs : int
            The target number of observations in the new light curve.

        Notes
        -----
        This function is adapted from the code developed in [1]_. In
        particular, the funtion
        `PlasticcAugmentor._choose_target_observation_count` of
        `avocado/plasticc.py`.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        # Estimate the distribution of number of observations in WFD
        # with a mixture of 4 gaussian distributions.
        gauss_choice = self._rs.choice(4, p=[0.289, 0.215, 0.181, 0.315])
        if gauss_choice == 0:
            mean = 31.93
            var = np.sqrt(30.23)
        elif gauss_choice == 1:
            mean = 18.07
            var = np.sqrt(42.76)
        elif gauss_choice == 2:
            mean = 59.65
            var = np.sqrt(75.83)
        elif gauss_choice == 3:
            mean = 45.35
            var = np.sqrt(34.96)
        target_number_obs = int(
            np.clip(self._rs.normal(mean, var), 3, None))

        return target_number_obs

    def _compute_obs_uncertainty(self, aug_obj_data, aug_obj_metadata):
        """Compute and add uncertainty to the light curve observations.

        Following [1]_, we estimate the flux uncertainties for each
        passband with a lognormal distribution for the Wide-Fast-Deep (WFD)
        survey. Each passband was modeled individually with test set events.
        The flux uncertanty of the augmented events is the combination of the
        flux uncertainty of the augmented events predicted by the GP in
        quadrature with a value drawn from the flux uncertainty distribution
        described above.

        Parameters
        ----------
        aug_obj_metadata : pandas.DataFrame
            Metadata of the augmented event.
        obj_data : pandas.DataFrame
            Observations of the original event.

        Returns
        -------
        aug_obj_data : pandas.DataFrame
            Observations of the augmented event.

        Notes
        -----
        This function is adapted from the code developed in [1]_. In
        particular, the funtion
        `PlasticcAugmentor._simulate_light_curve_uncertainties` of
        `avocado/plasticc.py`.

        References
        ----------
        .. [1] Boone, Kyle. "Avocado: Photometric classification of
        astronomical transients with gaussian process augmentation." The
        Astronomical Journal 158.6 (2019): 257.
        """
        # Make a copy of the original data to avoid modifying it
        aug_obj_data = aug_obj_data.copy()

        # Skip this function if there are no observations
        if len(aug_obj_data) == 0:
            return aug_obj_data

        # The uncertainty levels of the observations in each passband can be
        # modeled with the logaritm of a Gaussian mixture model.
        # Lognormal parameters from GMM:
        pb_noises = {'lsstu': {'weights': np.array([0.25, 0.75]),
                               'means': np.array([2.29, 1.95]),
                               'covars': np.array([0.45, 0.10])},
                     'lsstg': {'weights': np.array([0.11, 0.89]),
                               'means': np.array([2.13, 1.27]),
                               'covars': np.array([1.08, 0.12])},
                     'lsstr': {'weights': np.array([0.76, 0.24]),
                               'means': np.array([1.59, 1.92]),
                               'covars': np.array([0.09, 0.40])},
                     'lssti': {'weights': np.array([0.63, 0.37]),
                               'means': np.array([1.98, 2.44]),
                               'covars': np.array([0.08, 0.23])},
                     'lsstz': {'weights': np.array([1.]),
                               'means': np.array([2.65]),
                               'covars': np.array([0.14])},
                     'lssty': {'weights': np.array([1.]),
                               'means': np.array([3.22]),
                               'covars': np.array([0.14])}}

        # Calculate the new uncertainty levels for each passband
        # Initialize the variable to store the mean and standard deviation
        lognormal_parameters = np.zeros((len(aug_obj_data), 2))  # mean, std

        filter_set = self.dataset.filter_set
        for pb in filter_set:  # go through each passband
            is_pb = aug_obj_data['filter'] == pb
            aug_obj_data_pb = aug_obj_data[is_pb]

            # Generate new uncertainty levels for each observation
            try:
                pb_unc_vals = pb_noises[pb]
            except KeyError:
                raise ValueError(f'The noise properties of the passband {pb} '
                                 f'are not known. Add them to '
                                 f'`GPAugment._compute_obs_uncertainty`.')
            # For each event, choose from which gaussian the unc. will come
            number_comps = len(pb_unc_vals['weights'])
            number_obs_pb = len(aug_obj_data_pb)
            gauss_choice = self._rs.choice(number_comps, size=number_obs_pb,
                                           p=pb_unc_vals['weights'])

            # Store the lognormal parameters to later use
            means = pb_unc_vals['means'][gauss_choice]
            covars = pb_unc_vals['covars'][gauss_choice]
            lognormal_parameters[is_pb, 0] = means
            lognormal_parameters[is_pb, 1] = np.sqrt(covars)

        # Combine the flux uncertainty of the augmented events predicted by
        # the GP in quadrature with a value drawn from the flux uncertainty
        # distribution of the test set
        add_stds = self._rs.lognormal(lognormal_parameters[:, 0],
                                      lognormal_parameters[:, 1])
        noise_add = self._rs.normal(loc=0.0, scale=add_stds)
        aug_obj_data['flux'] += noise_add  # add noise to increase variability
        aug_obj_data['flux_error'] = np.sqrt(aug_obj_data['flux_error'] ** 2 + 
                                             add_stds ** 2)
        return aug_obj_data
