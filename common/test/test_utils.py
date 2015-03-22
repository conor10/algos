import datetime
import unittest

import numpy as np
from numpy.testing import utils as np_utils

import utils


class TestUtils(unittest.TestCase):
    RETURNS = np.array([0.1, 0.2, -0.1, -0.2, 0.1, -0.2, 0.1, 0.1, 0.1, 0.1, 0.3])

    def test_create_date(self):
        self.assertEqual(datetime.datetime(2014, 7, 10),
                         utils.create_date('2014-07-10'))

    def test_day_count(self):
        start = datetime.datetime(2014, 7, 10)
        end = datetime.datetime(2014, 8, 10)
        self.assertEqual(31, utils.day_count(start, end))

    def test_work_day_count(self):
        start = datetime.datetime(2014, 7, 10)
        end = datetime.datetime(2014, 8, 10)
        self.assertEqual(22, utils.work_day_count(start, end))

    def test_work_day_delta(self):
        base_date = datetime.datetime(2014, 8, 10)
        expected = datetime.datetime(2014, 7, 10)
        delta = -22
        self.assertEquals(expected, utils.work_day_delta(base_date, delta))

    def test_get_max(self):
        data = np.arange(9).reshape(3, 3)
        np_utils.assert_equal(np.array([6, 7, 8]), utils.get_max_vector(data))

    def test_get_max_nan(self):
        data = np.array([[np.nan, 2, 3], [7, 8, 9], [4, 5, 6]])
        np_utils.assert_equal(np.array([7, 8, 9]), utils.get_max_vector(data))

    def test_ffill(self):
        data = np.array([0, np.nan, 1, np.nan, np.nan, -1, np.nan])
        np_utils.assert_array_equal(
            np.array([0, 0, 1, 1, 1, -1, -1]),
            utils.ffill(data))

    def test_lag(self):
        np_utils.assert_array_equal(
            np.array([0, 1, 2, 3, 4]),
            utils.lag(np.array([1, 2, 3, 4, 5])))

    def test_lag_nan(self):
        np_utils.assert_array_equal(
            np.array([np.NAN, 1, 2, 3, 4]),
            utils.lag(np.array([1., 2., 3., 4., 5.]), np.NAN))

    def test_calculate_returns(self):
        positions = np.array(
            [0., 15., 15., 27.5, 27.5, 17.5, 32.5, 32.5, 10., -7.5])

        returns = utils.calculate_returns(positions)
        np_utils.assert_allclose(
            [0., 0., 0., 0.83333333, 0., -0.36363636,
             0.85714286, 0., -0.69230769, -1.75],
            returns, rtol=1e-7)


    def test_calculate_log_returns(self):
        positions = np.array(
            [0., 15., 15., 27.5, 27.5, 17.5, 32.5, 32.5, 10., -7.5])
        returns = utils.calculate_log_returns(positions)
        np_utils.assert_allclose(
            [0., 0., 0., 0.6061358, 0., -0.45198512, 0.61903921, 0.,
             -1.178655, 0.],
            returns, rtol=1e-7)


    def test_calculate_sharpe_ratio(self):
        self.assertEqual(
            5.7752005312777319,
            utils.calculate_sharpe_ratio(self.RETURNS))

    def test_calculate_sortino_ratio(self):
        self.assertEqual(
            18.368136262344809,
            utils.calculate_sortino_ratio(self.RETURNS))

    def test_calculate_sortino_ratio_with_freq(self):
        self.assertEqual(
            -9.2268702784386836,
            utils.calculate_sortino_ratio_with_freq(self.RETURNS))

    def test_calculate_max_drawdown(self):
        compound_returns = (1. + self.RETURNS).cumprod() - 1.
        max_dd, max_duration, dd_idx, duration_idx, hwm_idx = \
            utils.calculate_max_drawdown(compound_returns)
        self.assertAlmostEqual(-0.3664, max_dd)
        self.assertEqual(8, max_duration)
        self.assertEqual(5, dd_idx)
        self.assertEqual(9, duration_idx)
        self.assertEqual(1, hwm_idx)

    def test_calculate_max_drawdown_log(self):
        compound_returns = np.log(1 + self.RETURNS).cumsum()
        max_dd, max_duration, dd_idx, duration_idx, hwm_idx = \
            utils.calculate_max_drawdown_log(compound_returns)
        self.assertAlmostEqual(-0.4563374, max_dd)
        self.assertEqual(8, max_duration)
        self.assertEqual(5, dd_idx)
        self.assertEqual(9, duration_idx)
        self.assertEqual(1, hwm_idx)

if __name__ == '__main__':
    unittest.main()
