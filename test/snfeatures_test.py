"""
Module to test the features in `snfeatures.py`.
"""

import os
import sys
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import sncosmo

from astropy.table import join, Table
from snmachine import gps, sndata, snfeatures, tsne_plot, example_data

try:
    import pymultinest
    from pymultinest.analyse import Analyzer
    has_multinest = True
    print('Module pymultinest found')
except (ImportError, SystemExit) as exception:
    print(exception)
    if str(exception) == "No module named 'pymultinest'":
        errmsg = """PyMultinest not found. If you would like to use, please
                    install Mulitnest with 'sh install/multinest_install.sh;
                    source install/setup.sh'"""
        print(errmsg)
        has_multinest = False
    else:
        errmsg = """Multinest installed but not linked. Please ensure
                    $LD_LIBRARY_PATH set correctly with:
                        source install/setup.sh"""
        raise OSError(errmsg) from exception


test_data_path = os.path.join(example_data, 'SPCC_SUBSET', '')
precomp_features_path = os.path.join(example_data, 'output_spcc_no_z',
                                     'features', 'spcc_all_templates.dat')
example_name = 'DES_SN013866.DAT'
rtol = 0.25

example_data_path = os.path.join(example_data, 'example_data_for_tests.pckl')
ex_data = pd.read_pickle(example_data_path)


def setup_module(module):
    # UNPACKING TEST DATA IF NOT ALREADY UNPACKED #
    if not os.path.exists(test_data_path):
        print('Unpacking example data')
        subprocess.call(['tar', '-zxf', os.path.join('..', 'examples',
                                                     'SPCC_SUBSET.tar.gz'),
                         '-C', os.path.join('..', 'examples', '')])


@pytest.fixture(scope='module')
def load_example_data(request):
    d = sndata.Dataset(test_data_path, subset=[example_name])
    return d


def fit_templates(d, sampler='leastsq', use_redshift=False,
                  number_processes=1):
    temp_featz = snfeatures.TemplateFeatures(sampler=sampler)
    extr_features = temp_featz.extract_features(
        d, use_redshift=use_redshift, number_processes=number_processes,
        seed=42)
    d.set_model(temp_featz.fit_sn, extr_features)
    gof = temp_featz.goodness_of_fit(d)
    gof = np.array([gof[f] for f in d.filter_set]).T
    return gof[0]


def fit_parametric(model_choice, d, sampler='leastsq', number_processes=1):
    parametric_featz = snfeatures.ParametricFeatures(model_choice=model_choice,
                                                     sampler=sampler)
    extr_features = parametric_featz.extract_features(
        d, number_processes=number_processes, seed=42)
    d.set_model(parametric_featz.fit_sn, extr_features)
    gof = parametric_featz.goodness_of_fit(d)
    gof = np.array([gof[f] for f in d.filter_set]).T
    return gof[0]


def test_module_loading():
    """test-loading snmachine modules"""
    # this is probably paranoid, but why not.
    modules = sys.modules.keys()
    message = 'module {} could not be loaded'
    assert 'snmachine.sndata' in modules, message.format('sndata')
    assert 'snmachine.snfeatures' in modules, message.format('snfeatures')
    assert 'snmachine.tsne_plot' in modules, message.format('tsne_plot')


# HERE WE ASSEMBLE ALL THE DIFFERENT CONFIGURATIONS THAT WE WILL TEST THE
# FEATURE EXTRACTION METHODS ON #
samplers = ['leastsq']
parallel_cores = [1]

if has_multinest:
    samplers += ['nested']


""" TODO: this tests fails so we need to discover what changed in the
# parametric fits with the packages' update
def test_templates_leastsq(load_example_data):
    d = load_example_data
    for nproc in parallel_cores:
        gof = fit_templates(d, sampler='leastsq', use_redshift=False,
                            number_processes=nproc)

        # This case distinction is necessary, since from sncosmo-1.4 to
        # sncosmo-1.5 there have been significant changes in the salt2
        # templates that result in different fits. Same for the 2.0 version
        if sncosmo.__version__ < '1.5.0':
            gof_truth = [6.19486141,    18.22896966,    6.11967201, 1.06182221]
        elif sncosmo.__version__ < '2.0.0':
            gof_truth = [6.15794175,    18.22484842,    6.47569171, 2.2642403]
        else:
            gof_truth = [5.02820836,    17.2957761,     7.386472,   2.89795602]
        np.testing.assert_allclose(gof, gof_truth, rtol=rtol)

    for nproc in parallel_cores:
        gof = fit_templates(d, sampler='leastsq', use_redshift=True,
                            number_processes=nproc)
        if sncosmo.__version__ < '1.5.0':
            gof_truth = [6.21906514,  18.35383076,   6.08646565,   1.0458849]
        else:
            gof_truth = [6.23329476,  18.5004063,    6.35119046,   2.21491234]
        np.testing.assert_allclose(gof, gof_truth, rtol=rtol)
"""


@pytest.mark.skipif('nested' not in samplers, reason='(py)multinest not found')
@pytest.mark.slow
def test_templates_nested(load_example_data):
    d = load_example_data
    for nproc in parallel_cores:
        gof = fit_templates(d, sampler='nested', use_redshift=False,
                            number_processes=nproc)
        if sncosmo.__version__ < '1.5.0':
            gof_truth = [6.14752938,  18.26134481,   6.12642616,   1.06306042]
        else:
            gof_truth = [6.10687986,  18.16491907,   6.48794317,   2.26138874]
        np.testing.assert_allclose(np.sort(gof), np.sort(gof_truth), rtol=rtol)
    for nproc in parallel_cores:
        gof = fit_templates(d, sampler='nested', use_redshift=True,
                            number_processes=nproc)
        if sncosmo.__version__ < '1.5.0':
            gof_truth = [6.27339226,  18.63956378,   6.16584135,   1.05712933]
        else:
            gof_truth = [7.49051438,  23.41279761,   7.80852619,   2.49817101]

        np.testing.assert_allclose(np.sort(gof), np.sort(gof_truth), rtol=rtol)


""" TODO: this tests fails so we need to discover what changed in the
# parametric fits with the packages' update
def test_newling_leastsq(load_example_data):
    d = load_example_data
    for nproc in parallel_cores:
        gof = fit_parametric('newling', d, sampler='leastsq',
                             number_processes=nproc)
        gof_truth = [6.00072104,    22.03567143,    7.2070583,  1.28674332]
        np.testing.assert_allclose(gof, gof_truth, rtol=rtol)
"""


""" TODO: this tests fails so we need to discover what changed in the
# parametric fits with the packages' update
def test_karpenka_leastsq(load_example_data):
    d = load_example_data
    for nproc in parallel_cores:
        gof = fit_parametric('karpenka', d, sampler='leastsq',
                             number_processes=nproc)
        gof_true = [5.24617927,  23.03744351,   7.82406324,   0.88721942]
        np.testing.assert_allclose(gof, gof_true, rtol=rtol)
"""


@pytest.mark.skipif('nested' not in samplers, reason='(py)multinest not found')
@pytest.mark.slow
def test_newling_nested(load_example_data):
    d = load_example_data
    for nproc in parallel_cores:
        gof = fit_parametric('newling', d, sampler='nested',
                             number_processes=nproc)
        gof_true = [5.83656883,  21.81049531,   7.21428601,   1.29572207]
        np.testing.assert_allclose(gof, gof_true, rtol=rtol)


@pytest.mark.skipif('nested' not in samplers, reason='(py)multinest not found')
@pytest.mark.slow
def test_karpenka_nested(load_example_data):
    d = load_example_data
    for nproc in parallel_cores:
        gof = fit_parametric('karpenka', d, sampler='nested',
                             number_processes=nproc)
        gof_true = [5.10496956,  29.83861575,   6.50170389,   0.89942577]
        np.testing.assert_allclose(gof, gof_true, rtol=rtol)


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
    tsne_plot.plot(precomp_features, join(precomp_features, types)['Type'],
                   seed=42)
    plt.savefig('tsne_plot_test.png')
    assert os.path.getsize('tsne_plot_test.png') > 0
    return fig


def compute_gps(dataset, path_saved_gp_files):
    gps.compute_gps(dataset, number_gp=100, t_min=0, t_max=880,
                    output_root='.')


@pytest.mark.wavelets
def test_wavelet_pipeline(dataset=ex_data):
    path_saved_gp_files = '.'
    compute_gps(dataset, path_saved_gp_files)

    number_comps = 3
    wf = snfeatures.WaveletFeatures(output_root='.')
    reduced_features = wf.compute_reduced_features(
        dataset, number_comps, **{'path_saved_gp_files': path_saved_gp_files})

    true_reduced = np.array([[-8207.05949144,  140.50960821,    94.07240438],
                             [1789.04336137,  -426.96616221, -1389.04208587],
                             [1819.48086379,  -733.65719381,  -893.23425132],
                             [2467.74920243,  3199.94277009,   652.12629742],
                             [2130.78606385, -2179.82902228,  1536.07763539]])

    assert np.allclose(reduced_features, true_reduced)


@pytest.mark.wavelets
def test_reconstruction(dataset=ex_data):
    path_saved_gp_files = '.'
    output_root = path_saved_gp_files
    compute_gps(dataset, path_saved_gp_files)

    wavelet_name = 'sym2'
    number_comps = 5
    wf = snfeatures.WaveletFeatures(output_root)
    reduced_features = wf.compute_reduced_features(
        dataset, number_comps, **{'path_saved_gp_files': path_saved_gp_files,
                                  'wavelet_name': wavelet_name})
    rec_space = wf.reconstruct_feature_space(reduced_features, '.')

    reconstruct_error = wf.compute_reconstruct_error(
        dataset, **{'feature_space': rec_space, 'wavelet_name': wavelet_name})

    assert np.allclose(reconstruct_error.chisq_over_pts, 0)
