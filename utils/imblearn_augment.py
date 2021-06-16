"""
File to save possible future augmentations.
These classes/functions should be implemented in `snaugment`.

We do not find this code to work for our imbalanced problem, but it might be
useful for someone else.
"""

import copy
import os
import pickle
import sys
import time
import warnings

import george
import numpy as np
import pandas as pd
import scipy.optimize as op

# Solve imblearn problems introduced with sklearn version 0.24
import sklearn
import sklearn.neighbors, sklearn.utils, sklearn.ensemble
from sklearn.utils._testing import ignore_warnings
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
sys.modules['sklearn.utils.safe_indexing'] = sklearn.utils._safe_indexing
sys.modules['sklearn.utils.testing'] = sklearn.utils._testing
sys.modules['sklearn.ensemble.bagging'] = sklearn.ensemble._bagging
sys.modules['sklearn.ensemble.base'] = sklearn.ensemble._base
sys.modules['sklearn.ensemble.forest'] = sklearn.ensemble._forest
sys.modules['sklearn.metrics.classification'] = sklearn.metrics._classification

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE


class ImblearnAugment(SNAugment):
    """
    Derived class that encapsulates data augmentation via Nearest Neighbour
    inspired algorithms such as SMOTE, ADASYN etc.
    """

    def __init__(self, dataset, features, method, random_seed=None,
                 output_root=None, **kwargs):
        self._dataset = dataset
        self.aug_method = method
        self.random_seed = random_seed
        self.features = features
        self.output_root = output_root
        self._kwargs = kwargs

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
    _METHODS = ['SMOTE', 'ADASYN', 'SVMSMOTE', 'SMOTEENN', 'SMOTETomek']

    def augment(self):
        """Augment the dataset."""
        print('Augmenting the dataset...')
        initial_time = time.time()

        method = self.aug_method
        labels = np.array(self.dataset.labels, dtype=str)
        features = self._join_features()
        self._ori_columns = features.columns

        print(f"Before resampling: {sorted(Counter(labels).items())}")
        try:
            sampling_strategy = self._kwargs['sampling_strategy']
        except KeyError:
            sampling_strategy = 'auto'
        sm = eval(method)(random_state=self._rs,
                          sampling_strategy=sampling_strategy)
        aug_features, aug_labels = sm.fit_resample(features, labels)
        print(f"After resampling: {sorted(Counter(aug_labels).items())}")

        self._create_aug_metadata(aug_features, aug_labels)
        self._save_aug_feature_space()

        time_spent = pd.to_timedelta(int(time.time()-initial_time), unit='s')
        print('Time spent augmenting: {}.'.format(time_spent))

    def _save_aug_feature_space(self):
        output_root = self.output_root
        if output_root is not None:
            path_save_features = os.path.join(output_root,
                                              'aug_feature_space.pckl')
            with open(path_save_features, 'wb') as f:
                pickle.dump(self.aug_features, f, pickle.HIGHEST_PROTOCOL)

    def reconstruct_real_space(self):
        """Go to real space to generate augmented light curves."""
        # TODO: do it
        aug_dataset = copy.deepcopy(self.dataset)
        aug_dataset.metadata = self.aug_metadata
        aug_dataset.object_names = self.aug_metadata.object_id
        # objs = aug_dataset.object_names
        self.aug_dataset = aug_dataset

    def _create_aug_metadata(self, aug_features, aug_labels):
        """d"""
        self.aug_labels = aug_labels
        aug_features = self._format_features(aug_features)

        metadata = self.dataset.metadata
        all_cols = metadata.columns
        cols = all_cols.drop(['hostgal_photoz', 'hostgal_photoz_err', 'target',
                              'object_id'])
        aug_metadata = pd.DataFrame(aug_features[['hostgal_photoz',
                                                  'hostgal_photoz_err']])
        aug_metadata[cols] = metadata[cols]
        aug_metadata['target'] = aug_labels
        aug_metadata['object_id'] = aug_metadata.index
        aug_metadata = aug_metadata[all_cols]
        self.aug_metadata = aug_metadata
        self.aug_features = aug_features.drop(columns=['hostgal_photoz',
                                                       'hostgal_photoz_err'])

    def _join_features(self):
        """Join redshift features to the wavelet features."""
        features = self.features.copy()
        metadata = self.dataset.metadata
        photoz = metadata.hostgal_photoz.values.astype(float)
        photoz_err = metadata.hostgal_photoz_err.values.astype(float)
        features['hostgal_photoz'] = photoz
        features['hostgal_photoz_err'] = photoz_err
        return features

    def _format_features(self, aug_features):
        """Format the new features into a Dataframe."""
        ori_features = self.features
        ori_index = ori_features.index

        aug_features = pd.DataFrame(aug_features)
        aug_features['object_id'] = None
        aug_features['object_id'][:len(ori_index)] = ori_index
        aug_objs_ids = [f'aug_{j}' for j in np.arange(0, (len(aug_features)
                                                          - len(ori_index)))]
        aug_features['object_id'][len(ori_index):] = aug_objs_ids
        aug_features.set_index('object_id', inplace=True)
        aug_features.columns = self._ori_columns
        return aug_features

    @classmethod
    def methods(cls):
        return cls._METHODS

    @property
    def aug_method(self):
        """Return the augmentation method.

        Returns
        -------
        str
            Name of the augmentation method.
        """
        return self._aug_method

    @aug_method.setter
    def aug_method(self, method):
        """Set the augmentation method.

        Parameters
        ----------
        method : str
            Name of the augmentation method.
        """
        if method not in ImblearnAugment.methods():
            error_message = ('{} is not a possible augmentation method in '
                             '`snmachine`.'.format(method))
            raise ValueError(error_message)
        else:
            self._aug_method = method
