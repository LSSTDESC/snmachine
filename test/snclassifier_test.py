"""
Unit tests for testing the module `snclassifier`.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from astropy.table import Table
from snmachine import example_data, sndata, snclassifier

# Path to the dataset
test_data_path = os.path.join(example_data, 'SPCC_SUBSET', '')
precomp_features_path = os.path.join(example_data, 'output_spcc_no_z',
                                     'features', 'spcc_all_wavelets.dat')

# Separate the slower classifiers
slow_classifiers = ['boost_dt', 'random_forest', 'boost_rf']


def setup_module(module):
    # UNPACKING TEST DATA IF NOT ALREADY UNPACKED #
    if not os.path.exists(test_data_path):
        print('Unpacking example data')
        subprocess.call(['tar', '-zxf',
                         os.path.join('..', 'examples', 'SPCC_SUBSET.tar.gz'),
                         '-C', os.path.join('..', 'examples', '')])


@pytest.fixture(scope='module')
def check_avail_classifiers(request):
    avail_classifiers = snclassifier.choice_of_classifiers
    return avail_classifiers


@pytest.fixture(scope='module')
def load_full_testdata(request):
    d_full = sndata.Dataset(test_data_path)
    precomp_features = Table.read(precomp_features_path, format='ascii')
    types = d_full.get_types()
    types['Type'][np.floor(types['Type']/10) == 2] = 2
    types['Type'][np.floor(types['Type']/10) == 3] = 3
    return d_full, precomp_features, types


def test_module_loading():
    """test-loading snmachine modules"""
    modules = sys.modules.keys()
    assert 'snmachine.sndata' in modules, 'module sndata could not be loaded'
    assert 'snmachine.snfeatures' in modules, ('module snfeatures could not be'
                                               ' loaded')
    assert 'snmachine.snclassifier' in modules, ('module snclassifier could '
                                                 'not be loaded')


def classification_test(classifiers_list, featz, types):
    out_dir = os.path.join('classifications', '')
    if not os.path.exists(out_dir):
        subprocess.call(['mkdir', out_dir])

    # Transform the astropy tables into pandas
    features = featz.to_pandas()
    features.set_index('Object', inplace=True)
    data_labels = types.to_pandas()
    data_labels.set_index('Object', inplace=True)
    data_labels = data_labels['Type']

    # Run the classifiers
    which_column = 0  # column that corresponds to SN Ia in this dataset
    snclassifier.run_several_classifiers(
        classifier_list=classifiers_list, features=features,
        labels=data_labels, param_grid={'lgbm': {'num_leaves': [10, 30, 50]}},
        scoring='accuracy', train_set=.3, scale_features=True,
        which_column=which_column, output_root=out_dir, random_seed=42,
        **{'plot_roc_curve': False, 'number_processes': 4})

    # True AUC values
    auc_truth = {'nb': 0.5841146038130118,
                 'svm': 0.9233858736560651,
                 'knn': 0.9538666569300424,
                 'decision_tree': 0.7431595533136354,
                 'random_forest': 0.9597353136129243,
                 'boost_dt': 0.9296758265832312,
                 'boost_rf': 0.9615539090674187,
                 'neural_network': 0.9217849479277842,
                 'lgbm': 0.94423863}

    # Check if the classifiers reproduce the true AUC values within tolerance
    for classifier in classifiers_list:
        auc = np.load(os.path.join('classifications', f'auc_{classifier}.npy'))
        np.testing.assert_allclose(auc, auc_truth[classifier], rtol=0.05)


@pytest.mark.slow
def test_classification_slow(check_avail_classifiers, load_full_testdata):
    avail_classifiers = check_avail_classifiers
    d, featz, types = load_full_testdata

    my_slow_classifiers = list(set(slow_classifiers) & set(avail_classifiers))

    classification_test(my_slow_classifiers, featz, types)


def test_classification_fast(check_avail_classifiers, load_full_testdata):
    avail_classifiers = check_avail_classifiers
    d, featz, types = load_full_testdata

    fast_classifiers = list(set(avail_classifiers) - set(slow_classifiers))

    classification_test(fast_classifiers, featz, types)
