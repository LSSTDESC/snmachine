"""
Tests related to Gaussian Processes module.
"""

import os

import numpy as np
import pytest
import pandas as pd

from snmachine import example_data, gps


# Import test data
example_data_path = os.path.join(example_data, 'example_data_for_tests.pckl')
example_data = pd.read_pickle(example_data_path)


# Start the tests
@pytest.mark.gp
def test_1d_gps():
    """Test the 1 dimensional Gaussian Processes.

    Run the Gaussian Processes for `example_data` (the first 5 PLAsTiCC's
    objects) and verify the flux mean of each event is correct.
    """
    mean_flux_true_values = {
        '615': -23.22402573951826, '713': -1.4359755792364706,
        '730': 2.844743959137723, '745': 21.13420610513882,
        '1124': 16.099748055475427
        }  # These are the objects in example_data.

    gps.compute_gps(example_data, number_gp=100, t_min=0, t_max=1100,
                    kernel_param=[500., 20.], output_root=None,
                    number_processes=1, gp_dim=1)

    for obj in example_data.object_names:
        mean_flux_obj = np.mean(example_data.models[obj]['flux'])
        np.testing.assert_allclose(mean_flux_obj,
                                   mean_flux_true_values[obj])


@pytest.mark.gp
def test_2d_gps():
    """Test the 2 dimensional Gaussian Processes.

    Run the Gaussian Processes for `example_data` (the first 5 PLAsTiCC's
    objects) and verify the flux mean of each event is correct.
    """
    mean_flux_true_values = {
        '615': -45.486487005741296, '713': -0.9183076088586439,
        '730': 3.673929979995239, '745': 29.142299327650303,
        '1124': 17.134494730329973
        }  # These are the objects in example_data.

    gps.compute_gps(example_data, number_gp=100, t_min=0, t_max=1100,
                    kernel_param=[500., 20.], output_root=None,
                    number_processes=1, gp_dim=2)

    for obj in example_data.object_names:
        mean_flux_obj = np.mean(example_data.models[obj]['flux'])
        np.testing.assert_allclose(mean_flux_obj,
                                   mean_flux_true_values[obj])


@pytest.mark.gp
def test_gps_chisq_over_pts():
    """Test the X^2/number of datapoints values.

    Run the Gaussian Processes for `example_data` (the first 5 PLAsTiCC's
    objects) and verify it returns the expected values of X^2/number of
    datapoints.
    """
    chisq_over_pts_true_values = {
        '615': 5695.7282978660587, '713': 1.0678220723794234,
        '730': 0.83081717132128197, '745': 1.0721589469244539,
        '1124': 0.77145060447736791
        }  # These are the objects in example_data.

    gps.compute_gps(example_data, number_gp=100, t_min=0, t_max=1100,
                    kernel_param=[500., 20.], output_root=None,
                    number_processes=1, gp_dim=1)
    chisq_over_pts = example_data.compute_chisq_over_pts()

    for obj in chisq_over_pts.keys():
        np.testing.assert_allclose(chisq_over_pts[obj],
                                   chisq_over_pts_true_values[obj])
