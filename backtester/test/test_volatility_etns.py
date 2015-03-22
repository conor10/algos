import unittest

import numpy as np

from numpy import testing

import volatility_etns as etns


class TestVolatilityEtns(unittest.TestCase):
    def test_calc_initial_adjustments(self):
        qty = np.array([0, 0])
        prices = np.array([20.0, 60.0])
        target_alloc = np.array([-0.3, 0.7])
        cash = 10000

        adjustments = etns.calc_adjustments(
            qty, prices, target_alloc, cash)
        testing.assert_array_equal(
            ((target_alloc * cash) / prices).astype(int),
            adjustments)

    # TODO: Create a long long ratio

    def test_calc_subsequent_adjustments_no_cash(self):
        initial_prices = np.array([20.0, 60.0])
        target_alloc = np.array([-0.3, 0.7])
        initial_cash = 10000.0

        qty = np.round((target_alloc * initial_cash) / initial_prices)
        updated_prices = initial_prices * np.array([1.1, 0.9])

        adjustments = etns.calc_adjustments(qty, updated_prices,
                                            target_alloc, 0)
        testing.assert_array_equal(np.array([18., 7.]), adjustments)

    def test_calc_subsequent_adjustments_with_cash(self):
        prices = np.array([10.0, 10.0])
        target_alloc = np.array([-0.5, 0.5])
        initial_cash = 10000.0

        correct_alloc = np.round((target_alloc * initial_cash)
                                 / prices)
        qty = correct_alloc

        adjustments = etns.calc_adjustments(qty, prices,
                                            target_alloc,
                                            100)
        testing.assert_array_equal(np.array([-5., 5.]), adjustments)




if __name__ == '__main__':
    unittest.main()
