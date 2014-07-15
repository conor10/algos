import datetime
import unittest

import numpy as np
from numpy.testing import utils as np_utils

import utils

class TestUtils(unittest.TestCase):
    def test_create_date(self):
        self.assertEqual(datetime.datetime(2014, 7, 10),
                         utils.create_date('2014-07-10'))

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

if __name__ == '__main__':
    unittest.main()
