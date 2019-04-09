import pytest
from snmachine import sndata, snfeatures, tsne_plot, example_data
import sys
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.testing.compare import compare_images
# from matplotlib.testing.noseclasses import ImageComparisonFailure
from astropy.table import join, Table
import sncosmo

test_data_path = os.path.join(example_data, 'SPCC_SUBSET', '')
precomp_features_path = os.path.join(example_data, 'output_spcc_no_z', 'features', 'spcc_all_templates.dat')
# example_name='DES_SN001695.DAT'
# example_name='DES_SN084250.DAT'
example_name='DES_SN013866.DAT'
rtol=0.25

print('')
try:
    import pymultinest
    from pymultinest.analyse import Analyzer
    has_multinest = True
    print('Module pymultinest found')
except (ImportError, SystemExit) as exception:
    print(exception)
    if str(exception) == "No module named 'pymultinest'":
        errmsg = """
                PyMultinest not found. If you would like to use, please install
                Mulitnest with 'sh install/multinest_install.sh; source install/setup.sh'
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
    print('module emcee found')
except ImportError:
    has_emcee = False
    print('module emcee not found, skipping tests with emcee')

try:
    import george
    has_george = True
    print('module george found')
except ImportError:
    has_george = False
    print('module george not found, skipping GP tests with george')
print('')


def setup_module(module):
    # UNPACKING TEST DATA IF NOT ALREADY UNPACKED #
    if not os.path.exists(test_data_path):
        print('Unpacking example data')
        subprocess.call(['tar', '-zxf', os.path.join('..', 'examples', 'SPCC_SUBSET.tar.gz'), '-C', os.path.join('..', 'examples', '')])


@pytest.fixture(scope='module')
def load_example_data(request):
    # example_name='DES_SN001695.DAT'
    d = sndata.Dataset(test_data_path, subset=[example_name])
    return d


def fit_templates(d, sampler='leastsq', use_redshift=False, nprocesses=1):
    temp_featz = snfeatures.TemplateFeatures(sampler=sampler)
    extr_features = temp_featz.extract_features(d, use_redshift=use_redshift,
                                                nprocesses=nprocesses, seed=42)
    d.set_model(temp_featz.fit_sn, extr_features)
    gof = temp_featz.goodness_of_fit(d)
    gof = np.array([gof[f] for f in d.filter_set]).T
    return gof[0]
#    return gof[example_name]


def fit_parametric(model_choice, d, sampler='leastsq', nprocesses=1):
    parametric_featz = snfeatures.ParametricFeatures(model_choice=model_choice, sampler=sampler)
    extr_features = parametric_featz.extract_features(d, nprocesses=nprocesses, seed=42)
    d.set_model(parametric_featz.fit_sn, extr_features)
    gof = parametric_featz.goodness_of_fit(d)
    gof = np.array([gof[f] for f in d.filter_set]).T
    return gof[0]


def test_module_loading():
    """test-loading snmachine modules"""
    #this is probably paranoid, but why not.
    modules=sys.modules.keys()
    assert 'snmachine.sndata' in modules, 'module sndata could not be loaded'
    assert 'snmachine.snfeatures' in modules, 'module snfeatures could not be loaded'
    assert 'snmachine.tsne_plot' in modules, 'module tsne_plot could not be loaded'

#    if not os.path.exists(test_data_path):
#        print('Unpacking example data')
#        subprocess.call(['tar', '-zxf', os.path.join('..', 'examples', 'SPCC_SUBSET.tar.gz'), '-C', os.path.join('..', 'examples', '')])


# HERE WE ASSEMBLE ALL THE DIFFERENT CONFIGURATIONS THAT WE WILL TEST THE FEATURE EXTRACTION METHODS ON #
samplers = ['leastsq']
# gpalgos=['gapp']
gpalgos = []
parallel_cores = [1]

if has_emcee:
    samplers += ['mcmc']
if has_multinest:
    samplers += ['nested']
if has_george:
    gpalgos += ['george']


def test_templates_leastsq(load_example_data):
    d = load_example_data
    for nproc in parallel_cores:
        gof = fit_templates(d, sampler='leastsq', use_redshift=False, nprocesses=nproc)

        # This case distinction is necessary, since from sncosmo-1.4 to sncosmo-1.5 there have been
        # significant changes in the salt2 templates that result in different fits.

        if sncosmo.__version__ < '1.5.0':
            gof_truth = [6.19486141,  18.22896966,   6.11967201,   1.06182221]
        else:
            gof_truth = [6.15794175,  18.22484842,   6.47569171,   2.2642403]
        np.testing.assert_allclose(gof, gof_truth, rtol=rtol)

    for nproc in parallel_cores:
        gof = fit_templates(d, sampler='leastsq', use_redshift=True, nprocesses=nproc)
        if sncosmo.__version__ < '1.5.0':
            gof_truth = [6.21906514,  18.35383076,   6.08646565,   1.0458849]
        else:
            gof_truth = [6.23329476,  18.5004063,    6.35119046,   2.21491234]
        np.testing.assert_allclose(gof, gof_truth, rtol=rtol)


"""
@pytest.mark.skipif('mcmc' not in samplers, reason='emcee not found')
def test_templates_mcmc(load_example_data):
    d=load_example_data
    for nproc in parallel_cores:
        gof=fit_templates(d, sampler='mcmc', use_redshift=False, nprocesses=nproc)
        if sncosmo.__version__<'1.5.0':
	    gof_truth=[  6.16634571,  17.82825731,   6.23085278,   1.11415153]
	else:
	    gof_truth=[  6.09118872,  18.24212791,   6.6823486,    1.29993102]
        np.testing.assert_allclose(gof, gof_truth, rtol=rtol)
    for nproc in parallel_cores:
        gof=fit_templates(d, sampler='mcmc', use_redshift=True, nprocesses=nproc)
	if sncosmo.__version__<'1.5.0':
	    gof_truth=[  5.34752278,  18.87138068,   6.98768329,   1.84565766]
	else:
	    gof_truth=[  6.0600575,   18.18278172,   6.66374311,   1.29601296]
        np.testing.assert_allclose(gof, gof_truth, rtol=rtol)
"""

@pytest.mark.skipif('nested' not in samplers, reason='(py)multinest not found')
@pytest.mark.slow
def test_templates_nested(load_example_data):
    d = load_example_data
    for nproc in parallel_cores:
        gof = fit_templates(d, sampler='nested', use_redshift=False, nprocesses=nproc)
        if sncosmo.__version__ < '1.5.0':
            gof_truth = [6.14752938,  18.26134481,   6.12642616,   1.06306042]
        else:
            gof_truth = [6.10687986,  18.16491907,   6.48794317,   2.26138874]
        np.testing.assert_allclose(np.sort(gof), np.sort(gof_truth), rtol=rtol)
#        np.testing.assert_allclose(np.sort(gof), np.sort([ 2.4059210272, 2.0797560142, 1.7905944939, 3.57346633979]), rtol=rtol)
    for nproc in parallel_cores:
        gof=fit_templates(d, sampler='nested', use_redshift=True, nprocesses=nproc)
        if sncosmo.__version__ < '1.5.0':
            gof_truth = [6.27339226,  18.63956378,   6.16584135,   1.05712933]
        else:
            gof_truth = [7.49051438,  23.41279761,   7.80852619,   2.49817101]

        np.testing.assert_allclose(np.sort(gof), np.sort(gof_truth), rtol=rtol)
#        np.testing.assert_allclose(np.sort(gof), np.sort([ 4.17113262208, 2.73807890918, 2.79493934974, 2.50869418371]), rtol=rtol)


def test_newling_leastsq(load_example_data):
    d = load_example_data
    for nproc in parallel_cores:
        gof = fit_parametric('newling', d, sampler='leastsq', nprocesses=nproc)
        np.testing.assert_allclose(gof, [6.00072104,  22.03567143,   7.2070583,    1.28674332], rtol=rtol)
#        np.testing.assert_allclose(gof, [ 0.83526717,  0.51772027,  1.1398396,   1.11812427], rtol=rtol)

def test_karpenka_leastsq(load_example_data):
    d=load_example_data
    for nproc in parallel_cores:
        gof=fit_parametric('karpenka', d, sampler='leastsq', nprocesses=nproc)
        np.testing.assert_allclose(gof, [  5.24617927,  23.03744351,   7.82406324,   0.88721942], rtol=rtol)

"""
@pytest.mark.skipif('mcmc' not in samplers, reason='emcee not found')
def test_newling_mcmc(load_example_data):
    d=load_example_data
    for nproc in parallel_cores:
        gof=fit_parametric('newling', d, sampler='mcmc', nprocesses=nproc)
	print gof
#        np.testing.assert_allclose(gof, , rtol=rtol)
#        np.testing.assert_allclose(gof, [ 0.83526717,  0.51772027,  1.1398396,   1.11812427], rtol=rtol)

@pytest.mark.skipif('mcmc' not in samplers, reason='emcee not found')
def test_karpenka_mcmc(load_example_data):
    d=load_example_data
    for nproc in parallel_cores:
        gof=fit_parametric('karpenka', d, sampler='mcmc', nprocesses=nproc)
	print gof
#        np.testing.assert_allclose(gof, , rtol=rtol)
#"""


@pytest.mark.skipif('nested' not in samplers, reason='(py)multinest not found')
@pytest.mark.slow
def test_newling_nested(load_example_data):
    d = load_example_data
    for nproc in parallel_cores:
        gof = fit_parametric('newling', d, sampler='nested', nprocesses=nproc)
        np.testing.assert_allclose(gof, [5.83656883,  21.81049531,   7.21428601,   1.29572207], rtol=rtol)


@pytest.mark.skipif('nested' not in samplers, reason='(py)multinest not found')
@pytest.mark.slow
def test_karpenka_nested(load_example_data):
    d = load_example_data
    for nproc in parallel_cores:
        gof = fit_parametric('karpenka', d, sampler='nested', nprocesses=nproc)
        np.testing.assert_allclose(gof, [5.10496956,  29.83861575,   6.50170389,   0.89942577], rtol=rtol)


@pytest.fixture(scope='module')
def load_full_testdata(request):
    d_full = sndata.Dataset(test_data_path)
    precomp_features = Table.read(precomp_features_path, format='ascii')
    types = d_full.get_types()
    types['Type'][np.floor(types['Type']/10) == 2] = 2
    types['Type'][np.floor(types['Type']/10) == 3] = 3
    return d_full, precomp_features, types


@pytest.mark.mpl_image_compare
@pytest.mark.plots
def test_tsne(load_full_testdata):
    d_full, precomp_features, types = load_full_testdata
    fig = plt.figure()
    tsne_plot.plot(precomp_features, join(precomp_features, types)['Type'], seed=42)
    plt.savefig('tsne_plot_test.png')

    assert os.path.getsize('tsne_plot_test.png') > 0

    return fig
 #   err=compare_images('tsne_plot_test.png', 'tsne_plot_truth.png', tol=1.e-3)

 #   if err:
 #       raise ImageComparisonFailure(err)
