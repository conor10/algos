import unittest

import numpy as np

import price_distribution_model as pdm


class TestPriceDistributionModel(unittest.TestCase):
    def test_normal_distribution(self):
        x = np.array([-3, -2, -1, 0, 1, 2, 3])
        print(pdm.normal_dist(x, (0, 1)))


if __name__ == '__main__':
    unittest.main()
