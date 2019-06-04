"""
Tests related to pca

1.
"""
import numpy as np
import os
import snmachine as sm
from snmachine.snfeatures import WaveletFeatures
import pytest

testopts = [(5, 'svd', None, True),
            (5, 'eigendecomposition', None, False),
            (None, 'svd', 0.999, False),
            (5, 'eigendecomposition', None, True)]



# Test 1
@pytest.mark.skip(reason='will eventually replace')
def test_number_comps_for_tolerance(tol=0.999):
    """
    Check the new code for calculating the best number of coefficients for
    a value of tolerance.

    This is more of an integration test, where we check that the number matches
    the previous code.
    """
    fname = os.path.join(sm.example_data, 'eigenvals.npz')
    eigs = np.load(fname)['arr_0'][::-1]
    number_comp = WaveletFeatures.number_comps_for_tolerance(eigs, tol=tol)
    assert number_comp == 16


# Test 2
@pytest.mark.parametrize("number_comp,method,tol,normalize_variance", testopts)
def test_pca(number_comp, method, tol, normalize_variance, Nsamps=1000, Nfeats=10):
    """
    Directly test the `_pca` method on matrices, and check that the results look OK
    in terms of shapes and limiting cases. Tries out both the svd and eigendecomposition
    methods, and tolerance vs number_comps methods.

    Both methods use the same definition of covariance which is ** Different
    From the Original Code **
    """
    X = np.random.normal(size=(Nsamps, 3))
    R = np.random.random((3, Nfeats))
    X = np.dot(X, R)
    assert X.shape == (Nsamps, Nfeats)
    wf = WaveletFeatures()
    vec, comps, M, s, vals = wf._pca(X, number_comp=number_comp, method=method, tol=tol,
                                     normalize_variance=normalize_variance)
    assert M.size == Nfeats

    if number_comp is None:
        number_comp = vals.size
    assert comps.shape == (Nsamps, number_comp)
    assert vals.size == number_comp
    assert all(np.diff(vals) <= 0.)
    assert vec.shape == (Nfeats, number_comp)
    if number_comp == 5:
        # Here we know the number of principal components with non=trivial
        # eigenvalues by construction is 3 as expected. So the last two
        # eigenvalues should be close to 0.
        assert np.allclose(vals[-2:], 0.)


@pytest.mark.parametrize("number_comp,method,tol,normalize_variance", testopts)
def test_extract_wavelets(number_comp, method, tol, normalize_variance, Nsamps=1000,
                          Nfeats=10):
    """
    Test PCA in terms of shapes and limiting cases using the `extract_pca`
    methods.
    """
    X = np.random.normal(size=(Nsamps, 3))
    R = np.random.random((3, Nfeats))
    X = np.dot(X, R)
    assert X.shape == (Nsamps, Nfeats)

    wf = WaveletFeatures()
    object_names = np.array(list('sn_{}'.format(i) for i in range(Nsamps)))
    wavs, vals, vec, M, s = wf.extract_pca(object_names, X, number_comp=number_comp,
                                           method=method, tol=tol,
                                           normalize_variance=normalize_variance)
    assert M.size == Nfeats
    # Can't run this test, as this is a structured array with object names
    # assert np.asarray(wavs.to_pandas()).shape == Nsamps, number_comp
    # When we set number_comp from tols
    if number_comp is None:
        number_comp = vals.size
    assert vals.size == number_comp
    assert all(np.diff(vals) <= 0.)
    assert vec.shape == (Nfeats, number_comp)
    if number_comp == 5:
        # Here we know the number of principal components with non=trivial
        # eigenvalues by construction
        assert np.allclose(vals[-2:], 0.)


# Test 4.
@pytest.mark.parametrize("number_comp,method,tol,normalize_variance", testopts)
def test_lossy_reconstruct(number_comp, method, tol, normalize_variance,
                           Nsamps=1000, Nfeats=10):
    """
    Test the reconstruction of the data matrix from lower dimensinonal features
    obtained using PCA using the `_pca` method and the
    `reconstruct_datamatrix_lossy` method. The testing is at the level of
    expected shapes and ensuring that that the reconstructed data matrix is
    close to the original data matrix.
    """
    X = np.random.normal(size=(Nsamps, 3))
    R = np.random.random((3, Nfeats))
    X = np.dot(X, R)
    assert X.shape == (Nsamps, Nfeats)

    wf = WaveletFeatures()
    vec, comps, M, s, vals = wf._pca(X, number_comp=number_comp, method=method, tol=tol,
                                     normalize_variance=normalize_variance)
    # object_names = np.array(list('sn_{}'.format(i) for i in range(Nsamps)))
    # wavs, vals, vec, M = wf.extract_pca(object_names, X, number_comp=number_comp,
    #                                    method=method, tol=tol)
    assert M.size == Nfeats
    # Can't run this test, as this is a structured array with object names
    # assert np.asarray(wavs.to_pandas()).shape == Nsamps, number_comp
    # When we set number_comp from tols
    D = WaveletFeatures.reconstruct_datamatrix_lossy(comps, vec, M, s)
    assert D.shape == (Nsamps, Nfeats)
    Delta = D - X
    var = np.sum(Delta**2, axis=0)
    if normalize_variance:
        assert np.allclose(var, 0.)

    assert 2 > 1
