import pytest
import sys, os, subprocess
from astropy.table import Table
from snmachine import sndata, snfeatures, snclassifier, example_data
import numpy as np

test_data_path=os.path.join(example_data, 'SPCC_SUBSET', '')
precomp_features_path=os.path.join(example_data, 'output_spcc_no_z', 'features', 'spcc_all_wavelets.dat')
slow_classifiers=['boost_dt', 'random_forest', 'boost_rf']

def setup_module(module):
    ### UNPACKING TEST DATA IF NOT ALREADY UNPACKED ###
    if not os.path.exists(test_data_path):
        print('Unpacking example data')
        subprocess.call(['tar', '-zxf', os.path.join('..', 'examples', 'SPCC_SUBSET.tar.gz'), '-C', os.path.join('..', 'examples', '')])

@pytest.fixture(scope='module')
def check_avail_classifiers(request):
    avail_classifiers=snclassifier.choice_of_classifiers
    return avail_classifiers


@pytest.fixture(scope='module')
def load_full_testdata(request):
    d_full=sndata.Dataset(test_data_path)
    precomp_features=Table.read(precomp_features_path, format='ascii')
    types=d_full.get_types()
    types['Type'][np.floor(types['Type']/10)==2]=2
    types['Type'][np.floor(types['Type']/10)==3]=3
    return d_full, precomp_features, types

def test_module_loading():
    """test-loading snmachine modules"""
    modules=sys.modules.keys()
    assert 'snmachine.sndata' in modules, 'module sndata could not be loaded'
    assert 'snmachine.snfeatures' in modules, 'module snfeatures could not be loaded'
    assert 'snmachine.snclassifier' in modules, 'module snclassifier could not be loaded'

def classification_test(cls, featz, types):
    out_dir=os.path.join('classifications', '')
    if not os.path.exists(out_dir):
        subprocess.call(['mkdir',out_dir])

    snclassifier.run_pipeline(featz, types, classifiers=cls, nprocesses=4, plot_roc_curve=False, output_name=out_dir)

    auc_truth={'nb':5.498296484233418102e-01, 'svm': 9.607832585029829620e-01, 'knn':8.683540372670807139e-01, 'random_forest': 9.794267790146994335e-01, 'decision_tree':9.046528076757488490e-01, 'boost_dt': 9.597607478934744307e-01, 'boost_rf': 9.791576972753551766e-01, 'neural_network': 9.637969739836398375e-01}

    for classifier in cls:
        auc=np.loadtxt(os.path.join('classifications', classifier+'.auc'))
        np.testing.assert_allclose(auc, auc_truth[classifier], rtol=0.25)

@pytest.mark.slow
def test_classification_slow(check_avail_classifiers, load_full_testdata):
    avail_classifiers=check_avail_classifiers
    d, featz, types=load_full_testdata

    my_slow_classifiers=list(set(slow_classifiers) & set(avail_classifiers))

    classification_test(my_slow_classifiers, featz, types)


def test_classification_fast(check_avail_classifiers, load_full_testdata):
    avail_classifiers=check_avail_classifiers
    d, featz, types=load_full_testdata

    fast_classifiers=list(set(avail_classifiers) - set(slow_classifiers))

    classification_test(fast_classifiers, featz, types)
