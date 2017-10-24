import pytest
from snmachine import sndata, snfeatures, tsne_plot
import sys, os, subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
from matplotlib.testing.noseclasses import ImageComparisonFailure
from astropy.table import join, Table


test_data_path=os.path.join('..', 'examples', 'SPCC_SUBSET', '')
precomp_features_path=os.path.join('..', 'examples', 'output_spcc_no_z', 'features', 'spcc_all_templates.dat')
example_name='DES_SN001695.DAT'
rtol=0.05

print('')
try:
    import pymultinest
    from pymultinest.analyse import Analyzer
    has_multinest=True
    print ('module pymultinest found')
except ImportError:
    has_multinest=False
    print ('module pymultinest not found, skipping tests with pymultinest')
try:
    import emcee
    has_emcee=True
    print ('module emcee found')
except ImportError:
    has_emcee=False
    print ('module emcee not found, skipping tests with emcee')

try:
    import george
    has_george=True
    print ('module george found')
except ImportError:
    has_george=False
    print ('module george not found, skipping tests with george')
print('')


def setup_module(module):
    ### UNPACKING TEST DATA IF NOT ALREADY UNPACKED ###
    if not os.path.exists(test_data_path):
        print('Unpacking example data')
        subprocess.call(['tar', '-zxf', os.path.join('..', 'examples', 'SPCC_SUBSET.tar.gz'), '-C', os.path.join('..', 'examples', '')])


@pytest.fixture(scope='module')
def load_example_data(request):
    example_name='DES_SN001695.DAT'
    d=sndata.Dataset(test_data_path, subset=[example_name])
    return d

def fit_templates(d, sampler='leastsq', use_redshift=False, nprocesses=1):
    temp_featz=snfeatures.TemplateFeatures(sampler=sampler)
    extr_features=temp_featz.extract_features(d, save_chains=False, use_redshift=use_redshift, nprocesses=nprocesses, seed=42)
    d.set_model(temp_featz.fit_sn, extr_features)
    gof=temp_featz.goodness_of_fit(d)
    gof=np.array([gof[f] for f in d.filter_set]).T
    return gof[0]
#    return gof[example_name]

def fit_parametric(model_choice, d, sampler='leastsq', nprocesses=1):
    parametric_featz=snfeatures.ParametricFeatures(model_choice=model_choice, sampler=sampler)
    extr_features=parametric_featz.extract_features(d, nprocesses=nprocesses, seed=42)
    d.set_model(parametric_featz.fit_sn, extr_features)
    gof=parametric_featz.goodness_of_fit(d)
    gof=np.array([gof[f] for f in d.filter_set]).T
    return gof[0]

def fit_gp(d, gpalgo='gapp', nprocesses=1):
    ###NB: THIS TESTS ONLY THE GP EXTRACTION, NOT WAVELET TRAFO OR PCA###
    gp_featz=snfeatures.WaveletFeatures(gpalgo=gpalgo)
    gp_featz.extract_GP(d, ngp=gp_featz.ngp, xmin=0, xmax=d.get_max_length(), initheta=[500,20], save_output='none', output_root='features', nprocesses=nprocesses)
    gof=gp_featz.goodness_of_fit(d)
    gof=np.array([gof[f] for f in d.filter_set]).T
    return gof[0]

def test_module_loading():
    """test-loading snmachine modules"""
    #this is probably paranoid, but why not.
    modules=sys.modules.keys()
    assert 'snmachine.sndata' in modules, 'module sndata could not be loaded'
    assert 'snmachine.snfeatures' in modules, 'module snfeatures could not be loaded'
    assert 'snmachine.tsne_plot' in modules, 'module snfeatures could not be loaded'

    if not os.path.exists(test_data_path):
        print('Unpacking example data')
        subprocess.call(['tar', '-zxf', os.path.join('..', 'examples', 'SPCC_SUBSET.tar.gz'), '-C', os.path.join('..', 'examples', '')])

### HERE WE ASSEMBLE ALL THE DIFFERENT CONFIGURATIONS THAT WE WILL TEST THE FEATURE EXTRACTION METHODS ON ###
samplers=['leastsq']
#gpalgos=['gapp']
gpalgos=[]
parallel_cores=[1]

if has_emcee:
    samplers+=['mcmc']
if has_multinest:
    samplers+=['nested']
if has_george:
    gpalgos+=['george']

def test_templates_leastsq(load_example_data):
    d=load_example_data
    for nproc in parallel_cores:
        gof=fit_templates(d, sampler='leastsq', use_redshift=False, nprocesses=nproc)
        np.testing.assert_allclose(gof, [  8.40916682,   41.41075389,   24.17508901,  14.13583212], rtol=rtol)
    for nproc in parallel_cores:
        gof=fit_templates(d, sampler='leastsq', use_redshift=True, nprocesses=nproc)
        np.testing.assert_allclose(gof, [ 4.18634996,  2.73534527,  2.77111264,  2.51829497], rtol=rtol)


#TODO: Something is random in the State of Denmark. FIND!!

@pytest.mark.skipif('mcmc' not in samplers, reason='emcee not found')
def test_templates_mcmc(load_example_data):
    d=load_example_data
    for nproc in parallel_cores:
        gof=fit_templates(d, sampler='mcmc', use_redshift=False, nprocesses=nproc)
        print gof
#        np.testing.assert_almost_equal(gof, 2.38963686123)
#        np.testing.assert_almost_equal(gof, [ 2.38403233,  1.72647827,  2.13395609,  3.60967734])
    for nproc in parallel_cores:
        gof=fit_templates(d, sampler='mcmc', use_redshift=True, nprocesses=nproc)
        print gof
#        np.testing.assert_almost_equal(gof, 2.38870548308)
#        np.testing.assert_almost_equal(gof, [ 3.79908039,  2.10689115,  3.25059276,  2.74976956])


@pytest.mark.skipif('nested' not in samplers, reason='(py)multinest not found')
@pytest.mark.slow
def test_templates_nested(load_example_data):
    d=load_example_data
    for nproc in parallel_cores:
        gof=fit_templates(d, sampler='nested', use_redshift=False, nprocesses=nproc)
#        print gof
        np.testing.assert_allclose(np.sort(gof), np.sort([ 2.4059210272, 2.0797560142, 1.7905944939, 3.57346633979]), rtol=rtol)
    for nproc in parallel_cores:
        gof=fit_templates(d, sampler='nested', use_redshift=True, nprocesses=nproc)
#        print gof
        np.testing.assert_allclose(np.sort(gof), np.sort([ 4.17113262208, 2.73807890918, 2.79493934974, 2.50869418371]), rtol=rtol)

def test_newling_leastsq(load_example_data):
    d=load_example_data
    for nproc in parallel_cores:
        gof=fit_parametric('newling', d, sampler='leastsq', nprocesses=nproc)
        np.testing.assert_allclose(gof, [ 0.83526717,  0.51772027,  1.1398396,   1.11812427], rtol=rtol)

def test_karpenka_leastsq(load_example_data):
    d=load_example_data
    for nproc in parallel_cores:
        gof=fit_parametric('karpenka', d, sampler='leastsq', nprocesses=nproc)
        np.testing.assert_allclose(gof, [ 0.82915774,  0.45095918,  1.00380726,  1.14341116], rtol=rtol)

@pytest.mark.skipif('nested' not in samplers, reason='(py)multinest not found')
@pytest.mark.slow
def test_newling_nested(load_example_data):
    d=load_example_data
    for nproc in parallel_cores:
        gof=fit_parametric('newling', d, sampler='nested', nprocesses=nproc)
        np.testing.assert_allclose(gof, [ 0.83587807,  0.52186865,  1.1369503,   1.12273183], rtol=rtol)

@pytest.mark.skipif('nested' not in samplers, reason='(py)multinest not found')
@pytest.mark.slow
def test_karpenka_nested(load_example_data):
    d=load_example_data
    for nproc in parallel_cores:
        gof=fit_parametric('karpenka', d, sampler='nested', nprocesses=nproc)
        np.testing.assert_allclose(gof, [ 0.8307367,   0.4544247,   1.04103566,  1.14468516], rtol=rtol)

def test_gp_extraction(load_example_data):
    d=load_example_data
    gof_truth={'gapp':[ 0.76875293,  0.4266906,   0.7617092,   0.8427292 ], 'george':[ 0.76875293,  0.4266906,   0.7617092,   0.8427292 ]}#TODO george truth
    for gpalgo in gpalgos:
        for nproc in parallel_cores:
            gof=fit_gp(d, gpalgo=gpalgo, nprocesses=nproc)
	    print gof
            np.testing.assert_allclose(gof, gof_truth[gpalgo], rtol=rtol)

@pytest.fixture(scope='module')
def load_full_testdata(request):
    d_full=sndata.Dataset(test_data_path)
    precomp_features=Table.read(precomp_features_path, format='ascii')
    types=d_full.get_types()
    types['Type'][np.floor(types['Type']/10)==2]=2
    types['Type'][np.floor(types['Type']/10)==3]=3
    return d_full, precomp_features, types


def test_tsne(load_full_testdata):
    d_full, precomp_features, types=load_full_testdata
    plt.figure()
    tsne_plot.plot(precomp_features, join(precomp_features, types)['Type'], seed=42)
    plt.savefig('tsne_plot_test.png')

    err=compare_images('tsne_plot_test.png', 'tsne_plot_truth.png', tol=1.e-3)

    if err:
        raise ImageComparisonFailure(err)
