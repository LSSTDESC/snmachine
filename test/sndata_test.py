import pytest

import os, sys, subprocess
from snmachine import (sndata,
                       example_data)
import numpy as np

import matplotlib.pyplot as plt
#from matplotlib.testing.compare import compare_images
#from matplotlib.testing.noseclasses import ImageComparisonFailure


### THESE GLOBAL VARIABLES DEFINE THE LOCATION OF THE TEST DATA SET AND THE SPECIFIC EXAMPLE LC WE USE FOR TESTING ###

test_data_path=os.path.join(example_data, 'SPCC_SUBSET', '')
example_name='DES_SN001695.DAT'

@pytest.fixture(scope='module')
def load_example_lightcurve(request):
    d=sndata.Dataset(test_data_path, subset=[example_name])
    test_lc=d.data[example_name]
    return test_lc

def test_module_loading():
    """test-loading snmachine modules"""
    #this is probably paranoid, but why not.
    modules=sys.modules.keys()
    assert 'snmachine.sndata' in modules, 'module sndata could not be loaded'

    # if not os.path.exists(test_data_path):
    #    print('Unpacking example data')
    #    subprocess.call(['tar', '-zxf', os.path.join('..', 'examples', 'SPCC_SUBSET.tar.gz'), '-C', os.path.join('..', 'examples', '')])

def test_load_example_data():
    """ test-loading example dataset"""
    d=sndata.Dataset(test_data_path)
    assert len(d.data)==2000, 'test dataset: light curves could not be read in'
    assert len(d.object_names)==2000, 'test dataset: object names could not be read in'
    np.testing.assert_almost_equal( d.get_max_length(), 174.907, err_msg='test dataset: max length not correct')

### TEST sndata ROUTINES BY PICKING ONE REPRESENTATIVE LIGHTCURVE ###

def test_single_lightcurve_meta(load_example_lightcurve):
    """test that lc metadata has been properly parsed"""
    test_lc=load_example_lightcurve
    np.testing.assert_allclose(test_lc.meta['initial_observation_time'], 56177.172, err_msg='test lightcurve metadata: initial observation time false')
    assert test_lc.meta['name']==example_name, 'test lightcurve metadata: object name false'
    assert test_lc.meta['type']==2, 'test lightcurve metadata: type false'
    np.testing.assert_allclose(test_lc.meta['z'], 0.5022, err_msg='test lightcurve metadata: redshift false')
    np.testing.assert_allclose(test_lc.meta['z_err'], 0.0466, err_msg='test lightcurve metadata: redshift error false')

@pytest.mark.mpl_image_compare
@pytest.mark.plots
def test_single_lightcurve_plot(load_example_lightcurve):
    """testing plotting routine on single light curve: this is without a model fit, in snfeatures we redo with test-extracted features"""
    test_lc=load_example_lightcurve
    fig=plt.figure()
    sndata.plot_lc(test_lc)
    plt.savefig('raw_lc_test.png')

    assert os.path.getsize('raw_lc_test.png')>0

    return fig
#    plt.savefig('raw_lc_test.png')

#    err=compare_images('raw_lc_test.png', 'raw_lc_truth.png', tol=0.1)

#    if err:
#        raise ImageComparisonFailure(err)
