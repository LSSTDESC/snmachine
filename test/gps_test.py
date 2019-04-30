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
def test_gps_chisq_over_datapoints():
    """
    Test if the GPs are returning the expected values of X^2/number of datapoints.
    """
    chisq_over_datapoints_true_values = {
        '615': 5695.7282978660587, '713': 1.0678220723794234, '730': 0.83081717132128197, '745': 1.0721589469244539, 
        '1124': 0.77145060447736791
        } # These are the objects in example_data. They are the first 5 PLAsTiCC's objects.
    gps.compute_gps(example_data, number_gp=100, t_min=0, t_max=1100, kernel_param=[500., 20.], output_root=None, 
                    number_processes=1)
    chisq_over_datapoints_example_data = example_data.compute_chisq_over_datapoints()
    for obj in chisq_over_datapoints_example_data.keys():
        np.testing.assert_allclose(chisq_over_datapoints_example_data[obj], chisq_over_datapoints_true_values[obj])
