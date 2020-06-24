"""
Module handling the data augmentation of a snmachine dataset.
"""

import copy
import os
import pickle
import time

import george
import numpy as np
import pandas as pd
import scipy.optimize as op

from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM
from collections import Counter
from functools import partial  # TODO: erase when old sndata is deprecated
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE
from scipy.special import erf
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from snmachine import gps, snfeatures


# Functions to choose spectroscopic redshift for `GPAugment`.
# They are inputed as the parameter `choose_z` and their arguments as `*kwargs`
def choose_new_z_spec(z_ori, pb_wavelengths):
    '''Choose a new spec-z for the event based on the original spec-z'''

    z_min = max(10**(-6), (1 + z_ori) * (2 - pb_wavelengths['lsstg']
                                         / pb_wavelengths['lsstu']) - 1)
    z_max = ((1 + z_ori)
             * (2 - pb_wavelengths['lsstz']/pb_wavelengths['lssty']) - 1)
    log_z_star = np.random.uniform(low=np.log(z_min), high=np.log(z_max))
    z_new = - np.exp(log_z_star) + z_min + z_max

    assert (z_new > z_min) and (z_new < z_max)

    return z_new


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
        pass

    def extract_proxy_features(self, peak_filter='desr', nproc=1,
                               fit_salt2=False, salt2feats=None,
                               return_features=False, fix_redshift=False,
                               which_redshifts='header', sampler='leastsq'):
        """Extracts the 2D proxy features from raw light curves, e.g.,
        redshift and peak logflux in a certain band. There are plenty of
        options for how to get these values, if you should be so inclined.
        For the peak flux, we take either the maximum flux amongst the
        observations in the specified band (quick and dirty), or we perform
        SALT2 fits to the data and extract the peak flux from there (proper
        and slow). For the redshift, we take either the redshift specified in
        the header or the fitted SALT2 parameter.

        Parameters:
        ----------
        peak_filter : str, optional (default: 'desr')
            Name of the filter whose peak flux will be used as second column.
        nproc : int, optional (default: 1)
            Number of processes for salt2 feature extraction
        fit_salt2 : boolean, optional (default: False)
            If True, we compute the peak flux from SALT2 fits; if False, we
            return the brightest observation
        salt2feats : astropy.table.Table, optional (default: None)
            If you already have the features precomputed and do not want to
            recompute, you can hand them over here.
        return_features : bool, optional (default: False)
            If you want to store fitted SALT2 features, you can set this flag
            to return them
        fix_redshift : bool, optional (default: False)
            If True, we fix the redshift in the SALT2 fits to the value found
            in the table headers; if False, we leave the parameter free for
            sampling
        which_redshifts : str, optional (default: 'header')
            If 'salt2', the first column of proxy features will be the
            SALT2-fitted redshift; if 'header', we take the redshift from the
            header, if 'headerfill', we take the header redshift where
            available and fill with fitted values for the objects without
            valid redshift.
        sampler : str, optional (default: 'leastsq')
            Which sampler do we use to perform the SALT2 fit? 'leastsq' is a
            simple least squares fit, 'nested' uses nested sampling and
            requires MultiNest and PyMultiNest to be installed.
        Returns:
        -------
        proxy_features : np.array
            Nobj x 2 table with the extracted (z,peakflux) proxy features
        salt2feats : astropy.table.Table
            fitted SALT2 features for further applications
        """

        # Consistency check: is the specified filter actually in the dataset?
        if peak_filter not in self.dataset.filter_set:
            raise RuntimeError('Filter %s not amongst the filters in the'
                               'dataset!')

        # Consistency check: if we want to return salt2-fitted redshifts, do we actually have features?
        if which_redshifts == 'headerfill' and all([(isinstance(z, float)) & (z >= 0) for z in self.dataset.get_redshift()]):
            # Corner case: if 'headerfill' this check should only complain if there are actually invalid z values to fill in
            which_redshifts = 'header'
        if not fit_salt2 and which_redshifts in ['salt2', 'headerfill']:
            print('We need SALT2 features in order to return fitted redshifts!'
                  'Setting which_redshifts to "header".')
            which_redshifts = 'header'

        # Consistency check: to return features, we need to have features
        if return_features and not fit_salt2 and salt2feats is None:
            print('We need SALT2 features to return features - either provide'
                  'some or compute them! Setting return_features to False.')
            return_features = False

        # Fitting new features
        if fit_salt2:
            # We use a fit to SALT2 model to extract the r-band peak magnitudes
            tf = snfeatures.TemplateFeatures(sampler=sampler)
            if salt2feats is None:
                salt2feats = tf.extract_features(self.dataset,
                                                 number_processes=nproc,
                                                 use_redshift=fix_redshift)

            # Fit models and extract r-peakmags
            peaklogflux = []
            for i in range(len(self.dataset.object_names)):
                model = tf.fit_sn(self.dataset.data[self.dataset.object_names[i]], salt2feats)
                model = model[model['filter'] == peak_filter]
                if len(model) > 0:
                    peaklogflux = np.append(peaklogflux,
                                            np.log10(np.nanmax(model['flux'])))
                else:
                    # Band is missing: do something better than this
                    peaklogflux = np.append(peaklogflux, -42)
        else:
            peaklogflux = []
            for o in self.dataset.object_names:
                model = self.dataset.data[o]
                model = model[model['filter'] == peak_filter]
                if len(model) > 0:
                    peaklogflux = np.append(peaklogflux,
                                            np.log10(np.nanmax(model['flux'])))
                else:
                    # Band is missing: do something better
                    peaklogflux = np.append(peaklogflux, -42)

        # Extracting redshifts
        if which_redshifts == 'header':
            z = self.dataset.get_redshift()
        elif which_redshifts == 'salt2':
            z = [float(salt2feats[salt2feats['Object'] == o]['[Ia]z']) for o in self.dataset.object_names]
        elif which_redshifts == 'headerfill':
            z = self.dataset.get_redshift().astype(float)
            for i in range(len(z)):
                if not isinstance(z[i], float) or z[i] < 0:
                    z[i] = float(salt2feats['[Ia]z'][salt2feats['Object'] == self.dataset.object_names[i]])
        else:
            raise RuntimeError('Unknown value %s for argument which_redshifts!' % which_redshifts)

        proxy_features = np.array([z, peaklogflux]).transpose()

        if return_features:
            return proxy_features, salt2feats
        else:
            return proxy_features

    def compute_propensity_scores(self, train_names, algo='logistic', **kwargs):
        """Wherein we fit a model for the propensity score (the probability of
        an object to be in the training set) in the proxy-feature parameter
        space. We then evaluate the model on the full dataset and return the
        values.

        Parameters:
        ----------
        train_names : np.array of str
            Names of all objects in the training set
        algo : str, optional
            Name of the model to fit. 'logistic' is logistic regression,
            'network' is a simple ANN with two nodes in one hidden layer.
        kwargs : optional
            All of these will be handed to `self.extract_proxy_features`. See
            there for more info.

        Returns:
        -------
        propensity_scores : np.array
             array of all fitted scores
        """
        retval = self.extract_proxy_features(**kwargs)
        if len(retval) == 2 and 'return_features' in kwargs.keys() and kwargs['return_features']:
            proxy_features = retval[0]
            salt2_features = retval[1]
        else:
            proxy_features = retval
        # Logistic regression on proxy_features
        is_in_training_set = [1 if o in train_names else 0 for o in self.dataset.object_names]
        if algo == 'logistic':
            regr = LogisticRegression()
        elif algo == 'network':
            regr = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                 hidden_layer_sizes=(2,))
        regr.fit(proxy_features, is_in_training_set)
        propensity_scores = regr.predict_proba(proxy_features)
        if len(retval) == 2 and 'return_features' in kwargs.keys() and kwargs['return_features']:
            return propensity_scores[:, 0], salt2_features
        else:
            return propensity_scores[:, 0]

    def divide_into_propensity_percentiles(self, train_names, nclasses,
                                           **kwargs):
        """Wherein we fit the propensity scores and divide into
        equal-percentile classes from lowest to hightest.

        Parameters:
        ----------
        train_names : np.array of str
            Names of objects in the training set
        nclasses : int
            Number of classes
        kwargs : optional
            All of these will be handed to self.extract_proxy_features. See
            there for more info.
        Returns:
        -------
        classes : np.array of int
            classes from 0, ..., nclasses-1, in the same order as
            `self.dataset.object_names`.
        """
        retval = self.compute_propensity_scores(train_names, **kwargs)
        if len(retval) == 2 and 'return_features' in kwargs.keys() and kwargs['return_features']:
            prop = retval[0]
            salt2_features = retval[1]
        else:
            prop = retval
        sorted_indices = np.argsort(prop)
        N = len(self.dataset.object_names)
        classes = np.array([-42] * N)
        for c in range(len(self.dataset.object_names)):
            thisclass = (c * nclasses) // N
            classes[sorted_indices[c]] = thisclass
        if len(retval) == 2 and 'return_features' in kwargs.keys() and kwargs['return_features']:
            return classes, salt2_features
        else:
            return classes


class NNAugment(SNAugment):
    """
    Derived class that encapsulates data augmentation via Nearest Neighbour
    inspired algorithms such as SMOTE, ADASYN etc.
    """

    def __init__(self, X, y, method):
        self.X = X
        self.y = y
        self.method = method
    # TODO : allow for inheritance from SNAugment's constructor
    # def __init__(self, data, method):
    #         super().__init__(data)
    #         self.method = method
    # TODO : Update docstrings, add references to Notes section on methods
    """Make directories that will be used for analysis

    Parameters
    ----------
    X : numpy.ndarray
        Collection of data containing features
    y : numpy.ndarray
        True labels for data
    method : string
        Augmentation method one would like to resample the given data with.
        List of possible choices include:
            ['SMOTE', 'ADASYN', 'SVMSMOTE', 'SMOTEENN', 'SMOTETomek']

    Notes
    -------


    """
    _METHODS = [
        'SMOTE',
        'ADASYN',
        'SVMSMOTE',
        'SMOTEENN',
        'SMOTETomek'
    ]

    @classmethod
    def methods(cls):
        return cls._METHODS

    @staticmethod
    def augment(X, y, method):

        print(NNAugment.methods())
        if method not in NNAugment.methods():
            raise ValueError(F"{method} not a possible augmentation method in `snmachine`")

        print(F"Before resampling: {sorted(Counter(y).items())}")

        X_resampled, y_resampled = eval(method)().fit_resample(X, y)
        print(F"After resampling: {sorted(Counter(y_resampled).items())}")

        return X_resampled, y_resampled


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
        dataset : Dataset object (sndata class)
            Dataset to augment.
        path_saved_gps: str
            Path to the Gaussian Process files.
        objs_number_to_aug: {`None`, 'all', dict}, optional
            Specify which events to augment and by how much. If `None`, the
            dataset it not augmented. If `all`, all the events are augmented
            10 times. If a dictionary is provided, it should be in the form of:
                event: number of times to augment that event.
        choose_z: {None, function}, optional
            Function used to choose the new true redshift of the augmented
            events. If `None`, the new events have the same redshift as the
            original event. If a function is provided, arguments can be
            included as `**kwargs`.
        z_table: {None, pandas.DataFrame}, optional
            Dataset of the spectroscopic and photometric redshift and
            photometric redshift error of events. This table is used to
            generate the photometric redshift and respective error for the
            augmented events. If `None`, this table is generated from the
            events in the original dataset.
        max_duration: {None, float}, optional
            Maximum duration of the lightcurve. If `None`, it is set to the
            maximum lenght of an event in `dataset`.
        cosmology: astropy.cosmology.core.COSMOLOGY, optional
            Cosmology from `astropy` with the cosmologic parameters already
            defined. By default it assumes Flat LambdaCDM with parameters
            `H0 = 70`, `Om0 = 0.3` and `T_cmb0 = 2.725`.
        random_seed: int, optional
            Random seed used. Saving this seed allows reproducible results.
            If given, it must be between 0 and 2**32 - 1.
        **kwargs: dict, optional
            Optional keywords to pass arguments into `choose_z` and into
            `snamchine.gps.compute_gps`.
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
        self._kwargs = dict(kwargs, pb_wavelengths=self.dataset.pb_wavelengths)
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

        time_spent = pd.to_timedelta(int(time.time()-initial_time), unit='s')
        print('Time spent augmenting: {}.'.format(time_spent))

    def _add_augment_objs_to_dataset(self, aug_objs_data, aug_objs_metadata):
        """Add the new events to the dataset.

        There are two new datasets created:
            - aug_dataset: original + augmented events
            - only_new_dataset: augmented events

        Parameters
        ----------
        aug_objs_data: list of pandas.DataFrame
            List containing the observations of each augmentation of each
            event.
        aug_objs_metadata: list of pandas.DataFrame
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
                aug_dataset.set_inner_metadata(obj)
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
        obj: str
            Name of the original event.

        Returns
        -------
        aug_objs_data: list of pandas.DataFrame
            Ordered list containing the observations of each augmentation of
            `obj`.
        aug_objs_metadata: list of pandas.DataFrame
            Ordered list containing the metadata of each augmentation of `obj`.
        """
        obj_data = self.dataset.data[obj].to_pandas()
        obj_metadata = self.dataset.metadata.loc[obj]
        z_obj = obj_metadata['hostgal_specz']

        number_aug = self.objs_number_to_aug[obj]  # # of new events
        number_tries = 15  # # tries to generate an augmented event
        aug_objs_data = []
        aug_objs_metadata = []
        for i in np.arange(number_aug):
            aug_obj = '{}_aug{}'.format(obj, i)
            j = 0
            while j < number_tries:  # TODO: not efficient
                aug_obj_metadata = self.create_aug_obj_metadata(aug_obj,
                                                                obj_metadata)
                aug_obj_data = self.create_aug_obj_obs(aug_obj_metadata,
                                                       obj_data, z_obj)
                if len(aug_obj_data) == 0:
                    j += 1
                else:
                    j = number_tries  # finish the loop
                    aug_objs_data.append(aug_obj_data)
                    aug_objs_metadata.append(aug_obj_metadata)
        return aug_objs_data, aug_objs_metadata

    def _choose_obs_times(self, aug_obj_metadata, obj_data, z_ori):
        """TODO: avocado
        """
        z_aug = aug_obj_metadata['hostgal_specz']
        # Figure out the target number of observations to have for the new
        # lightcurve. #TODO: avocado
        target_number_obs = self._choose_target_observation_count(
            aug_obj_metadata)

        aug_obj_data = obj_data.copy()
        aug_obj_data['object_id'] = aug_obj_metadata['object_id']
        aug_obj_data['ref_mjd'] = aug_obj_data['mjd'].copy()

        # Adjust the observation times
        z_scale = (1 + z_ori) / (1 + z_aug)
        # Shift relative to an approximation of the peak flux time so that
        # we generally keep the interesting part of the light curve in the
        # frame.
        ref_peak_time = obj_data['mjd'].iloc[
            np.argmax(obj_data['flux'].values)]
        aug_obj_data['mjd'] = ref_peak_time + z_scale**-1 * (
            aug_obj_data['ref_mjd'] - ref_peak_time)

        # TODO: avocado
        max_time_shift = 30
        aug_obj_data['mjd'] += self._rs.uniform(-max_time_shift,
                                                max_time_shift)
        is_not_seen = aug_obj_data['mjd'] < 0
        aug_obj_data = aug_obj_data[~is_not_seen]  # before 0
        aug_obj_data = self.trim_obj(aug_obj_data, self.max_duration)  # after

        # Make sure that we have some observations left at this point. If not,
        # return an empty observations list. TODO: avocado
        if len(aug_obj_data) == 0:
            print('obj {} failed.'.format(aug_obj_metadata['object_id']))
            return None

        # At high redshifts, we need to fill in the light curve to account for
        # the fact that there is a lower observation density compared to lower
        # redshifts. TODO: avocado
        num_fill = int(target_number_obs * (z_scale**-1 - 1))
        if num_fill > 0:
            new_indices = self._rs.choice(aug_obj_data.index, num_fill,
                                          replace=True)
            new_rows = aug_obj_data.loc[new_indices]

            # Choose new bands randomly.
            obj_pbs = np.unique(aug_obj_data['filter'])
            new_rows['filter'] = np.random.choice(obj_pbs, num_fill,
                                                  replace=True)

            aug_obj_data = pd.concat([aug_obj_data, new_rows])

        # Drop back down to the target number of observations. Having too few
        # observations is fine, but having too many is not. We always drop at
        # least 10% of observations to get some shakeup of the light curve.
        # TODO: avocado
        drop_fraction = 0.1
        number_drop = int(max(
            len(aug_obj_data) - target_number_obs,
            drop_fraction * len(aug_obj_data)))
        drop_indices = np.random.choice(aug_obj_data.index, number_drop,
                                        replace=False)
        aug_obj_data = aug_obj_data.drop(drop_indices).copy()

        aug_obj_data.reset_index(inplace=True, drop=True)
        return aug_obj_data

    def create_aug_obj_obs(self, aug_obj_metadata, obj_data, z_ori):
        """Create observations for the augmented event.

        The new observations are based on the observations of the original
        event.

        Parameters
        ----------
        aug_obj_metadata: pandas.DataFrame
            Metadata of the augmented event.
        obj_data: pandas.DataFrame
            Observations of the original event.
        z_ori: float
            Redshift of the original event.

        Returns
        -------
        aug_obj_data: pandas.DataFrame
            Observations of the augmented event.
        """
        z_aug = aug_obj_metadata['hostgal_specz']
        ori_obj = str(obj_data['object_id'][0])

        aug_obj_data = self._choose_obs_times(aug_obj_metadata, obj_data,
                                              z_ori)
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
        dist_scale = (self.cosmology.distmod(z_ori)
                      / self.cosmology.distmod(z_aug))**2
        aug_obj_data['flux'] = flux_pred * z_scale * dist_scale
        aug_obj_data['flux_error'] = flux_pred_error * z_scale * dist_scale

        # Add in light curve noise. This is survey specific and must be
        # implemented in subclasses.
        aug_obj_data = self._simulate_light_curve_uncertainties(
            aug_obj_data, aug_obj_metadata)

        # Simulate detection
        aug_obj_data, pass_detection = self._simulate_detection(
            aug_obj_data, aug_obj_metadata)
        # If our light curve passes detection thresholds, we're done!
        if pass_detection:
            return aug_obj_data

        return []  # failed attempt

    def load_gp(self, obj):
        """Load the Gaussian Process predict object.

        Parameters
        ----------
        obj: str
            Name of the original event.

        Returns
        -------
        gp_predict: functools.partial with bound method GP.predict
            Function to predict the Gaussian Process flux and uncertainty at
            any time and wavelength.
        """
        # The name format of the saved Gaussian Processes is hard coded
        path_saved_obj_gp = os.path.join(self.path_saved_gps,
                                         'used_gp_'+obj+'.pckl')
        with open(path_saved_obj_gp, 'rb') as input:
            gp_predict = pickle.load(input)
        try:  # old format - TODO: deprecate the old sndata format
            obj_flux = self.dataset.data[obj]['flux']
            gp_predict = partial(gp_predict.predict, obj_flux)
        except AttributeError:
            pass
        return gp_predict

    def create_aug_obj_metadata(self, aug_obj, obj_metadata):
        """Create metadata for the augmented event.

        The new metadata is based based on the metadata of the original event.

        Parameters
        ----------
        aug_obj: str
            Name of the augmented event in the form
                `[original event name]_[number of the augmentation]`.
        obj_metadata: pandas.DataFrame
            Metadata of the original event.

        Returns
        -------
        aug_obj_metadata: pandas.DataFrame
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

        # Choose whether the new object will be in the DDF or not. TODO: avo
        if obj_metadata["ddf"]:
            # Most observations are WFD observations, so generate more of
            # those. The DDF and WFD samples are effectively completely
            # different, so this ratio doesn't really matter.
            aug_obj_metadata["ddf"] = self._rs.rand() > 0.8
        else:
            # If the reference wasn't a DDF observation, can't simulate a DDF
            # observation.
            aug_obj_metadata["ddf"] = False

        return aug_obj_metadata

    def compute_new_z_photo(self, z_spec):
        """Compute a new photometric redshift and error.

        The new values are randomly withdrawn from a redshift table containing
        differences between spectroscopic and photometric redshifts and the
        respective photometric redshift error.

        Parameters
        ----------
        z_spec: float
            Spectroscopic redshift of the augmented event.
        i: int, optional
            Number of the current iteration. This acts as a stopping criteria.

        Returns
        -------
        z_photo: float
            Photometric redshift of the augmented event.
        z_photo_error: float
            Photometric redshift error of the augmented event.

        Raises
        ------
        ValueError
            If none of the generated `z_photo` or `z_photo_error` are positive.
        """
        z_table = self.z_table
        number_tries = 100  # # of tries to get positive z values
        rd_zs_triple = z_table.sample(random_state=self._rs, n=number_tries)
        zs_diff = rd_zs_triple['z_diff']
        zs_photo = z_spec + (self._rs.choice([-1, 1], size=number_tries)
                             * zs_diff)
        zs_photo_error = (rd_zs_triple['hostgal_photoz_err']
                          * self._rs.normal(1, .05, size=number_tries))

        are_zs_pos = (zs_photo > 0) & (zs_photo_error > 0)
        try:  # choose the first appropriate value
            z_photo = zs_photo[are_zs_pos][0]
            z_photo_error = zs_photo_error[are_zs_pos][0]
        except (KeyError, IndexError):
            raise ValueError('The new redshift and respective error must be '
                             'positive and they were not after 100 tries so '
                             'something is wrong.')
        return z_photo, z_photo_error

    def fit_gps(self, path_to_save_gps, **kwargs):
        """Fit Gaussian Processes to the augmented events.

        Parameters
        ----------
        path_to_save_gps: str
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
    def dataset(self):
        """Return the original dataset.

        Returns
        -------
        Dataset object (sndata class)
            Dataset to augment.
        """
        return self._dataset

    @property
    def max_duration(self):
        """Return the maximum duration any lightcurve can have.

        All the events in the original dataset must be shorter than this value.

        Returns
        -------
        float
            Maximum duration any lightcurve can have.
        """
        return self._max_duration

    @max_duration.setter
    def max_duration(self, value):
        """Set the maximum duration any lightcurve can have.

        Parameters
        ----------
        value: {None, float}, optional
            Maximum duration of the lightcurve. If `None`, it is set to the
            maximum lenght of an event in `dataset`.

        Raises
        ------
        ValueError
            If any event in the original dataset is longer than the maximum
            duration any lightcurve can have.
        """
        max_duration_ori = self.dataset.get_max_length()
        if value is None:
            duration = max_duration_ori
        elif max_duration_ori < value:
            raise ValueError('All the events in the original dataset must be '
                             'shorter than the required maximum duration any '
                             'lightcurve. At the moment the maximum duration '
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
        obj_data: pandas.DataFrame
            Observations of an event.
        max_duration: float
            Maximum duration of the lightcurve.

        Returns
        -------
        obj_data: pandas.DataFrame
            Trimmed observations of an event.
        """
        obj_duration = np.max(obj_data['mjd'])
        while obj_duration > max_duration:
            diff_duration = max_duration - obj_duration
            number_days_remove = np.max([5., diff_duration/3])  # >= 5 days

            index_initial_obs = obj_data['mjd'] < number_days_remove
            mean_initial_flux = np.mean(obj_data['flux'][index_initial_obs])

            index_final_obs = obj_data['mjd'] < (obj_duration
                                                 - number_days_remove)
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
        """Compute the new observations wavelenght at the original redshift.

        The observation flux is measured at specific wavelengths. This
        function calculates the wavelength of the new event as seen by an
        observer at redshift `z_ori`.

        Parameters
        ----------
        z_ori: float
            Redshift of the original event.
        z_new: float
            Redshift of the new event.
        obj_data: pandas.DataFrame
            Observations of the original event.

        Returns
        -------
        wavelength_new: list-like
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
        value: {None, pandas.DataFrame}, optional
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
            z_table['z_diff'] = (z_table['hostgal_photoz']
                                 - z_table['hostgal_specz'])
        else:
            z_table = self._standardise_z_table(value)
        self._z_table = z_table

    @staticmethod
    def _standardise_z_table(z_table):
        """Standardise redshift table to have redshift difference and error.

        Returns
        -------
        z_table: pandas.DataFrame
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
        z_table['z_diff'] = (z_table['hostgal_photoz']
                             - z_table['hostgal_specz'])
        return z_table

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
        """Return the number and which events used to augment.

        Returns
        -------
        dict
            The events to augment and how many of each in the form of:
                event: number of times to augment that event.
        """
        return self._objs_number_to_aug

    @objs_number_to_aug.setter
    def objs_number_to_aug(self, value):
        """Set the number and which events used to augment.

        Parameters
        ----------
        value: {`None`, 'all', dict}, optional
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
        objs_number_to_aug: {`None`, 'all', dict}, optional
            Specify which events to augment and by how much. If `None`, the
            dataset it not augmented. If `all`, all the events are augmented
            10 times. If a dictionary is provided, it should be in the form of:
                event: number of times to augment that event.

        Returns
        -------
        objs_number_to_aug: dict
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
        self._rs = np.random.RandomState(value)  # initialise the random state
        self._random_seed = value

    # avocado functions
    def _choose_target_observation_count(self, augmented_metadata):
        """Choose the target number of observations for a new augmented light
        curve.
        We use a functional form that roughly maps out the number of
        observations in the PLAsTiCC test dataset for each of the DDF and WFD
        samples.
        Parameters
        ----------
        augmented_metadata : dict
            The augmented metadata
        Returns
        -------
        target_number_obs : int
            The target number of observations in the new light curve.
        """
        if augmented_metadata["ddf"]:
            # I estimate the distribution of number of observations in the
            # WFD regions with a mixture of 2 gaussian distributions.
            gauss_choice = np.random.choice(2, p=[0.34393457, 0.65606543])
            if gauss_choice == 0:
                mean = 57.36015146
                var = np.sqrt(271.58889272)
            elif gauss_choice == 1:
                mean = 92.7741619
                var = np.sqrt(338.53085446)
            target_number_obs = int(
                np.clip(self._rs.normal(mean, var), 20, None))
        else:  # WFD event -> at least 3 observations
            target_number_obs = (
                self._rs.normal(24.5006006, np.sqrt(72.5106613)))
            target_number_obs = int(np.clip(target_number_obs, 3, None))
        return target_number_obs

    def _simulate_light_curve_uncertainties(self, aug_obj_data,
                                            aug_obj_metadata):
        """Simulate the observation-related noise and detections for a light
        curve.
        For the PLAsTiCC dataset, we estimate the measurement uncertainties for
        each band with a lognormal distribution for both the WFD and DDF
        surveys. Those measurement uncertainties are added to the simulated
        observations.
        Parameters
        ----------
        observations : pandas.DataFrame
            The augmented observations that have been sampled from a Gaussian
            Process. These observations have model flux uncertainties listed
            that should be included in the final uncertainties.
        augmented_metadata : dict
            The augmented metadata
        Returns
        -------
        observations : pandas.DataFrame
            The observations with uncertainties added.
        """
        # Make a copy so that we don't modify the original array.
        aug_obj_data = aug_obj_data.copy()

        if len(aug_obj_data) == 0:
            # No data, skip
            return aug_obj_data

        if aug_obj_metadata["ddf"]:
            band_noises = {
                "lsstu": (0.68, 0.26),
                "lsstg": (0.25, 0.50),
                "lsstr": (0.16, 0.36),
                "lssti": (0.53, 0.27),
                "lsstz": (0.88, 0.22),
                "lssty": (1.76, 0.23),
            }
        else:
            band_noises = {
                "lsstu": (2.34, 0.43),
                "lsstg": (0.94, 0.41),
                "lsstr": (1.30, 0.41),
                "lssti": (1.82, 0.42),
                "lsstz": (2.56, 0.36),
                "lssty": (3.33, 0.37),
            }

        # Calculate the new noise levels using a lognormal distribution for
        # each band.
        lognormal_parameters = []
        for pb in aug_obj_data['filter']:
            try:
                lognormal_parameters.append(band_noises[pb])
            except KeyError:
                raise ValueError(
                    'Noise properties of passband {} not known, add them in '
                    'PlasticcAugmentor._simulate_light_curve_uncertainties.'
                    ''.format(pb))
        lognormal_parameters = np.array(lognormal_parameters)

        add_stds = self._rs.lognormal(
            lognormal_parameters[:, 0], lognormal_parameters[:, 1])

        noise_add = self._rs.normal(loc=0.0, scale=add_stds)
        aug_obj_data['flux'] += noise_add
        aug_obj_data['flux_error'] = np.sqrt(aug_obj_data['flux_error'] ** 2
                                             + add_stds ** 2)
        return aug_obj_data

    def _simulate_detection(self, aug_obj_data, aug_obj_metadata):
        """Simulate the detection process for a light curve.
        We model the PLAsTiCC detection probabilities with an error function.
        I'm not entirely sure why this isn't deterministic. The full light
        curve is considered to be detected if there are at least 2 individual
        detected observations.
        Parameters
        ==========
        observations : pandas.DataFrame
            The augmented observations that have been sampled from a Gaussian
            Process.
        augmented_metadata : dict
            The augmented metadata
        Returns
        =======
        observations : pandas.DataFrame
            The observations with the detected flag set.
        pass_detection : bool
            Whether or not the full light curve passes the detection thresholds
            used for the full sample.
        """
        s2n = np.abs(aug_obj_data["flux"]) / aug_obj_data["flux_error"]
        prob_detected = (erf((s2n - 5.5) / 2) + 1) / 2.0
        aug_obj_data["detected"] = self._rs.rand(len(s2n)) < prob_detected

        pass_detection = np.sum(aug_obj_data["detected"]) >= 2

        return aug_obj_data, pass_detection


class SimpleGPAugment(SNAugment):
    """Derived class that encapsulates data augmentation via Gaussian Processes.
    """

    def __init__(self, dataset, stencils=None, cadence_stencils=None,
                 stencil_weights=None, cadence_stencil_weights=None):
        """Class constructor.

        Parameters:
        ----------
        dataset : Dataset object (sndata class)
            Dataset.
        stencils : list of strings
            If the stencils argument is given (as a list of object names that
            are in the data set), then the augmentation step will take these
            light curves to train the GPs on. If not, then every object in the
            data set is considered fair game.
        cadence_stencils : list of strings
            If given, the augmentation will sample the cadence for the new
            light curves from these objects. If not, every object is fair game.
        stencil_weights : np.array or list of float
            If given, these weights are the probabilities with which the
            respective stencils will be picked in the augmentation step. If
            not given, we will use uniform weights.
        cadence_stencil_weights : np.array or list of float
            Like stencil_weights, but for the cadence stencils.
        """

        self.dataset = dataset
        self.meta = {}
        self.meta['trained_gp'] = {}
        self.augmentation_method = 'GP augmentation'
        if stencils is None:
            self.stencils = dataset.object_names.copy()
        else:
            self.stencils = stencils
        if cadence_stencils is None:
            self.cadence_stencils = dataset.object_names.copy()
        else:
            self.cadence_stencils = cadence_stencils

        if stencil_weights is not None:
            assert np.all(stencil_weights >= 0.), 'Stencil weights need to be larger than zero!'
            stencil_weights = np.array(stencil_weights)
            self.stencil_weights = stencil_weights / sum(stencil_weights)
        else:
            self.stencil_weights = np.ones(len(self.stencils)) / len(self.stencils)

        if cadence_stencil_weights is not None:
            assert np.all(cadence_stencil_weights >= 0.), 'Cadence stencil weights need to be larger than zero!'
            cadence_stencil_weights = np.array(cadence_stencil_weights)
            self.cadence_stencil_weights = cadence_stencil_weights / sum(cadence_stencil_weights)
        else:
            self.cadence_stencil_weights = np.ones(len(self.cadence_stencils)) / len(self.cadence_stencils)

        self.rng = np.random.RandomState()
        self.random_seed = self.rng.get_state()

        self.original_object_names = dataset.object_names.copy()

    def train_filter(self, x, y, yerr, initheta=[100, 20]):
        """Train one Gaussian process on the data from one band. We use the
        squared-exponential kernel, and we optimise its hyperparameters

        Parameters:
        -----------
        x : numpy array
            `mjd` values for the cadence
        y : numpy array
            Flux values
        yerr : numpy array
            Errors on the flux
        initheta : list, optional
            Initial values for the hyperparameters. They should roughly
            correspond to the spread in y and x direction.

        Returns:
        -------
        g : george.GP
            the trained GP object
        """
        def nll(p):
            g.set_parameter_vector(p)
            ll = g.log_likelihood(y, quiet=True)
            return -ll if np.isfinite(ll) else 1.e25

        def grad_nll(p):
            g.set_parameter_vector(p)
            return -g.grad_log_likelihood(y, quiet=True)

        g = george.GP(initheta[0] ** 2 * george.kernels.ExpSquaredKernel(metric=initheta[1]**2))
        if len(x) == 0:
            return g
        g.compute(x, yerr)
        p0 = g.get_parameter_vector()
        results = op.minimize(nll, p0, jac=grad_nll, method='L-BFGS-B')
        g.set_parameter_vector(results.x)
        return g

    def sample_cadence_filter(self, g, cadence, y, yerr,
                              add_measurement_noise=True):
        """Given a trained GP, and a cadence of mjd values, produce a sample from
        the distribution defined by the GP, on that cadence. The error bars are
        set to the spread of the GP distribution at the given mjd value.

        Parameters:
        -----------
        g : george.GP
            The trained Gaussian process object
        cadence : numpy.array
            The cadence of mjd values.
        y : numpy array
            The flux values of the data that the GP has been trained on.
        yerr : numpy array
            ??? It was never described here.
        add_measurement_noise : bool, optional
            Usually, the data is modelled as y_i = f(t_i) + sigma_i*eps_i,
            where f is a gaussianly-distributed function, and where eps_i are
            iid Normal RVs, and sigma_i are the measurement error bars. If this
            flag is unset, we return a sample from the GP f and its stddev. If
            it is set, we return y*_j including the measurement noise (also in
            the error bar). If this is unclear, please consult
            Rasmussen/Williams chapter 2.2.

        Returns:
        --------
        flux : numpy array
            Flux values for the new sample
        fluxerr : numpy array
            Error bars on the flux for the new sample
        """
        if len(cadence) == 0:
            flux = np.array([])
            fluxerr = np.array([])
        else:
            mu, cov = g.predict(y, cadence)
            flux = self.rng.multivariate_normal(mu, cov)
            fluxerr = np.sqrt(np.diag(cov))
        # Adding measurement error - Cat no need for this comment
        if add_measurement_noise:
            flux += self.rng.randn(len(y)) * yerr
            fluxerr = np.sqrt(fluxerr**2 + yerr**2)
        return flux, fluxerr

    def produce_new_lc(self, obj, cadence=None, savegp=True, samplez=True,
                       name='dummy', add_measurement_noise=True):
        """Assemble a new light curve from a stencil. If the stencil already has
        been used and the resulting GPs have been saved, then we use those. If
        not, we train a new GP.

        Parameters:
        -----------
        obj : str or astropy.table.Table
            The object (or name thereof) that we use as a stencil to train the
            GP on.
        cadence : str or dict of type {string:numpy.array}, optional.
            The cadence for the new light curve, defined either by object name
            or by {filter:mjds}. If none is given, then we pull the cadence of
            the stencil.
        savegp : bool, optional
            Do we save the trained GP in self.meta? This results in a speedup,
            but costs memory.
        samplez : bool, optional
            Do we give the new light curve a random redshift value drawn from a
            Gaussian of location and width defined by the stencil? If not, we
            just take the value of the stencil.
        name : str, optional
            Object name of the new light curve.
        add_measurement_noise : bool, optional
            Usually, the data is modelled as y_i = f(t_i) + sigma_i*eps_i,
            where f is a gaussianly-distributed function, and where eps_i are
            iid Normal RVs, and sigma_i are the measurement error bars. If this
            flag is unset, we return a sample from the GP f and its stddev. If
            it is set, we return y*_j including the measurement noise (also in
            the error bar). If this is unclear, please consult
            Rasmussen/Williams chapter 2.2.

        Returns:
        --------
        new_lc: astropy.table.Table
            The new light curve
        """
        if type(obj) is Table:
            obj_table = obj
            obj_name = obj.meta['name']
        elif type(obj) in [str, np.str_]:
            obj_table = self.dataset.data[obj]
            obj_name = str(obj)
        else:
            print('obj: type %s not recognised in produce_new_lc()!' % type(obj))
            # TODO: actually throw an error

        if cadence is None:
            cadence_dict = self.extract_cadence(obj)
        else:
            if add_measurement_noise:
                print('warning: GP sampling does NOT include measurement noise, since sampling is performed on a different cadence!')
                add_measurement_noise = False
            if type(cadence) in [str, np.str_]:
                cadence_dict = self.extract_cadence(cadence)
            elif type(cadence) is dict:
                cadence_dict = cadence
            else:
                print('cadence: type %s not recognised in produce_new_lc()!' % type(cadence))
                # TODO: actually throw an error

        # Either train a new set of GP on the stencil obj, or pull from metadata
        if obj_name in self.meta['trained_gp'].keys():
            print('fetching')
            all_g = self.meta['trained_gp'][obj_name]
        else:
            print('training')
            all_g = {}
            for f in self.dataset.filter_set:
                obj_f = obj_table[obj_table['filter'] == f]
                x = np.array(obj_f['mjd'])
                y = np.array(obj_f['flux'])
                yerr = np.array(obj_f['flux_error'])
                g = self.train_filter(x, y, yerr)
                all_g[f] = g
            if savegp:
                self.meta['trained_gp'][obj_name] = all_g

        # Produce new LC based on the set of GP
        if samplez and 'z_err' in obj_table.meta.keys():
            newz = obj_table.meta['z'] + obj_table.meta['z_err'] * self.rng.randn()
        else:
            newz = obj_table.meta['z']
        new_lc_meta = obj_table.meta.copy()
        modified_meta = {'name': name, 'z': newz, 'stencil': obj_name,
                         'augment_algo': self.augmentation_method}
        new_lc_meta.update(modified_meta)
        new_lc = Table(names=['mjd', 'filter', 'flux', 'flux_error'],
                       dtype=['f', 'U', 'f', 'f'], meta=new_lc_meta)
        for f in self.dataset.filter_set:
            obj_f = obj_table[obj_table['filter'] == f]
            y = obj_f['flux']
            yerr = obj_f['flux_error']
            flux, fluxerr = self.sample_cadence_filter(
                                all_g[f], cadence_dict[f], y, yerr,
                                add_measurement_noise=add_measurement_noise)
            filter_col = [str(f)] * len(cadence_dict[f])
            dummy_table = Table((cadence_dict[f], filter_col, flux, fluxerr),
                                names=['mjd', 'filter', 'flux', 'flux_error'],
                                dtype=['f', 'U', 'f', 'f'])
            new_lc = vstack([new_lc, dummy_table])

        new_lc.sort('mjd')  # Sort by cadence, for cosmetics
        return new_lc

    def extract_cadence(self, obj):
        """Given a light curve, we extract the cadence in a format that we can
        insert into produce_lc and sample_cadence_filter.

        Parameters:
        -----------
        obj : str or astropy.table.Table
            (Name of) the object

        Returns:
        --------
        cadence : dict of type {str:numpy.array}
            The cadence, in the format {filter1:mjd1, filter2:mjd2, ...}
        """
        if type(obj) in [str, np.str_]:
            table = self.dataset.data[obj]
        elif type(obj) is Table:
            table = obj
        else:
            print('obj: type %s not recognised in extract_cadence()!' % type(obj))
        cadence = {flt: np.array(table[table['filter'] == flt]['mjd']) for flt in self.dataset.filter_set}
        return cadence

    def augment(self, numbers, return_names=False, **kwargs):
        """High-level wrapper of GP augmentation: The dataset will be
        augmented to the numbers by type.

        Parameters:
        ----------
        numbers : dict of type int:int
            The numbers to which the data set will be augmented, by type. Keys
            are the types, values are the numbers of light curves pertaining to
            a type after augmentation. Types that do not appear in this dict
            will not be touched.
        return_names : bool
            If True, we return a list of names of the objects that have been
            added into the data set
        kwargs :
            Additional arguments that will be handed verbatim to
            `produce_new_lc`. Interesting choices include `savegp=False` and
            `samplez=True`.

        Returns:
        --------
        newobjects : list of str
            The new object names that have been created by augmentation.
        """
        dataset_types = self.dataset.get_types()
        # dataset_types['Type'][dataset_types['Type']>10]=dataset_types['Type'][dataset_types['Type']>10]//10#NB: this is specific to a particular remapping of indices!!

        types = np.unique(dataset_types['Type'])

        newnumbers = dict()
        newobjects = []
        for t in types:
            thistype_oldnumbers = len(dataset_types['Type'][dataset_types['Type'] == t])
            newnumbers[t] = numbers[t] - thistype_oldnumbers
            thistype_stencils = [dataset_types['Object'][i] for i in range(len(dataset_types)) if dataset_types['Object'][i] in self.stencils and dataset_types['Type'][i] == t]
            thistype_stencil_weights = [self.stencil_weights[i] for i in range(len(dataset_types)) if dataset_types['Object'][i] in self.stencils and dataset_types['Type'][i] == t]

            if newnumbers[t] < 0:
                print('There are already %d objects of type %d in the original data set, cannot augment to %d!' % (thistype_oldnumbers, t, numbers[t]))
                continue
            elif newnumbers[t] == 0:
                continue
            else:
                # print('now dealing with type: '+str(t))
                # print('stencils: '+str(thistype_stencils))
                for i in range(newnumbers[t]):
                    # Pick random stencil
                    # thisstencil=thistype_stencils[self.rng.randint(0,len(thistype_stencils))]
                    thisstencil = self.rng.choice(thistype_stencils, p=thistype_stencil_weights / np.sum(thistype_stencil_weights))
                    # Pick random cadence
                    # thiscadence_stencil=self.cadence_stencils[self.rng.randint(0,len(self.cadence_stencils))]
                    # thiscadence_stencil=thisstencil

                    # cadence=self.extract_cadence(thiscadence_stencil)
                    # new_name='augm_t%d_%d_'%(t,i) + thisstencil + '_' + thiscadence_stencil + '.DAT'
                    new_name = 'augm_t%d_%d_' % (t, i) + thisstencil  # + '.DAT'
                    new_lightcurve = self.produce_new_lc(obj=thisstencil, name=new_name, **kwargs)  # ,cadence=cadence)
                    self.dataset.insert_lightcurve(new_lightcurve)
                    # print('types: '+str(new_lightcurve.meta['type'])+' '+str(self.dataset.data[thisstencil].meta['type']))
                    newobjects = np.append(newobjects, new_name)
        return newobjects
