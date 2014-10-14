import unittest

import numpy as np
from numpy import random
from numpy.testing import utils as np_utils

from option import Type
import price_series


class PriceSeriesTest(unittest.TestCase):
    def test_random_walk(self):
        random.seed(1)
        series1 = price_series.generate_random_walks(5, 2, 70, 1)
        random.seed(1)
        series2 = price_series.generate_random_walks_alt(5, 2, 70, 1)
        print(series1)
        print(series2)

    def test_generate_random_price_series(self):
        random.seed(1)
        series = price_series.generate_random_price_series(70, 0.35, 5)
        print(series)
        np_utils.assert_array_equal(
            np.array([70.56852088, 69.78588526, 69.81513989, 69.62446098, 70.30289267]),
            series)


    def test_itm_probability(self):
        price = 70.0
        sigma = 3.5  # 5% volatility
        strikes = np.arange(60, 80, 0.1)
        itm_probability = price_series.calc_itm_probability(
            strikes, price, sigma, 100, 1000, Type.PUT)

        print(itm_probability)


if __name__ == '__main__':
    unittest.main()
