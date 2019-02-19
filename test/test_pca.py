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
def test_best_coeffs(tol=0.999):
    """
    Check the new code for calculating the best number of coefficients for
    a value of tolerance.

    This is more of an integration test, where we check that the number matches
    the previous code. We make it match the previous code, and we should remove the ad-hoc
    addition of 1., ie the number should be 1 less than what it is.
    """
    fname = os.path.join(sm.example_data, 'eigenvals.npz')
    eigs = np.load(fname)['arr_0'][::-1]
    ncomp = WaveletFeatures.ncompsForTolerance(eigs, tol=tol)
    assert ncomp == 16
    ncomp = WaveletFeatures.best_coeffs(eigs, tol=tol)
    assert ncomp == 16


# Test 2
@pytest.mark.parametrize("ncomp,method,tol,normalize_variance", testopts)
def test_pca(ncomp, method, tol, normalize_variance, Nsamps=10000, Nfeats=10):
    """
    Directly test the pca on matrices, and check that the results look OK
    in terms of shapes and limiting cases. Tries out both the svd and eigendecomposition
    methods, and tolerance vs ncomps methods.

    Both methods use the same definition of covariance which is ** Different
    From the Original Code **
    """
    X = np.random.normal(size=(Nsamps, 3))
    R = np.random.random((3, Nfeats))
    X = np.dot(X, R)
    assert X.shape == (Nsamps, Nfeats) 
    wf = WaveletFeatures()
    vec, comps, M, s, vals = wf._pca(X, ncomp=ncomp, method=method, tol=tol,
                                     normalize_variance=normalize_variance)
    assert M.size == Nfeats

    if ncomp is None:
        ncomp = vals.size
    assert comps.shape == (Nsamps, ncomp)
    assert vals.size == ncomp
    assert all(np.diff(vals) <= 0.)
    assert vec.shape == (Nfeats, ncomp)
    if ncomp == 5:
        # Here we know the number of principal components with non=trivial
        # eigenvalues by construction
        assert np.allclose(vals[-2:], 0.)


@pytest.mark.parametrize("ncomp,method,tol,normalize_variance", testopts)
def test_extract_wavelets(ncomp, method, tol, normalize_variance, Nsamps=10000,
                          Nfeats=10):
    X = np.random.normal(size=(Nsamps, 3))
    R = np.random.random((3, Nfeats))
    X = np.dot(X, R)
    assert X.shape == (Nsamps, Nfeats) 

    wf = WaveletFeatures()
    object_names = np.array(list('sn_{}'.format(i) for i in range(Nsamps)))
    wavs, vals, vec, M, s = wf.extract_pca(object_names, X, ncomp=ncomp,
                                           method=method, tol=tol,
                                           normalize_variance=normalize_variance)
    assert M.size == Nfeats
    # Can't run this test, as this is a structured array with object names
    # assert np.asarray(wavs.to_pandas()).shape == Nsamps, ncomp
    # When we set ncomp from tols 
    if ncomp is None:
        ncomp = vals.size
    assert vals.size == ncomp
    assert all(np.diff(vals) <= 0.)
    assert vec.shape == (Nfeats, ncomp)
    if ncomp == 5:
        # Here we know the number of principal components with non=trivial
        # eigenvalues by construction
        assert np.allclose(vals[-2:], 0.)


# Test 4.
@pytest.mark.parametrize("ncomp,method,tol,normalize_variance", testopts)
def test_lossy_reconstruct(ncomp, method, tol, normalize_variance,
                           Nsamps=10000, Nfeats=10):
    X = np.random.normal(size=(Nsamps, 3))
    R = np.random.random((3, Nfeats))
    X = np.dot(X, R)
    assert X.shape == (Nsamps, Nfeats) 

    wf = WaveletFeatures()
    vec, comps, M, s, vals = wf._pca(X, ncomp=ncomp, method=method, tol=tol,
                                     normalize_variance=normalize_variance)
    # object_names = np.array(list('sn_{}'.format(i) for i in range(Nsamps)))
    # wavs, vals, vec, M = wf.extract_pca(object_names, X, ncomp=ncomp,
    #                                    method=method, tol=tol)
    assert M.size == Nfeats
    # Can't run this test, as this is a structured array with object names
    # assert np.asarray(wavs.to_pandas()).shape == Nsamps, ncomp
    # When we set ncomp from tols 
    D = WaveletFeatures.reconstruct_datamatrix_lossy(comps, vec, M, s)
    assert D.shape == (Nsamps, Nfeats)
    Delta = D - X
    var = np.sum(Delta**2, axis=0)
    if normalize_variance:
        assert np.allclose(var, 0.)

    assert 2 > 1
