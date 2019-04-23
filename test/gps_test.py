"""
Tests related to GPs.
"""

import os
import pickle
import sys

import numpy as np
import pytest
from snmachine import sndata, snfeatures, tsne_plot, example_data, gps


# Import test data
example_data_path = os.path.join(example_data, 'example_data_for_tests.pckl')
with open(example_data_path, 'rb') as input:
    example_data = pickle.load(input)


# Start the tests
@pytest.mark.gp
def test_gps_reduced_chi2():
    """
    Test if the GPs are returning the expected values of reduced X^2.
    """
    reduced_chi2_true_values = {
        '615': 5695.3046266894344, '713': 1.067914355545299, '730': 0.83081248341774283, '745': 1.0721592615480273, '1124': 0.77149815252112608
        } # These are the objects in example_data. They are the first 5 PLAsTiCC's objects.
    gps.extract_GP(example_data, ngp=100, t_min=0, t_max=1100, initheta=[100., 400.], output_root=None, nprocesses=1)
    reduced_chi2_example_data = example_data.reduced_chi_squared()
    for obj in reduced_chi2_example_data.keys():
        np.testing.assert_allclose(reduced_chi2_example_data[obj], reduced_chi2_true_values[obj])
