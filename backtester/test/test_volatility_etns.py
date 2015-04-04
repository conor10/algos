import datetime as dt
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
        testing.assert_array_equal(np.array([19., 7.]), adjustments)

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

    def test_calc_adjustment_ratio_sign_change(self):
        # We go from being long to short one of our trades
        # We may want to modify this such that the best we can
        # do is go flat & spend leftover cash
        positions = np.array([10., 10])
        prices = np.array([5., 5.])
        target_alloc = np.array([-0.5, 0.5])
        cash = 0

        adjustments = etns.calc_adjustments(
            positions, prices, target_alloc, cash)

        testing.assert_array_equal(np.array([-20., 0.]), adjustments)

    def test_calc_cash_delta_long_short(self):
        orders = np.array([5, -5])
        prices = np.array([2., 2.])
        positions = np.array([10, -10])

        delta = etns.calc_cash_delta(orders, prices, positions)
        testing.assert_array_equal(-20., delta)

    def test_calc_cash_delta_new_positions(self):
        orders = np.array([5, -5])
        prices = np.array([2., 2.])
        positions = np.array([0, 0])

        delta = etns.calc_cash_delta(orders, prices, positions)
        testing.assert_array_equal(-20., delta)

    def test_calc_cash_delta_switch_sign_negative(self):
        orders = np.array([-20, 0])
        prices = np.array([2., 2.])
        positions = np.array([10, -10])

        delta = etns.calc_cash_delta(orders, prices, positions)
        testing.assert_array_equal(0., delta)

    def test_calc_cash_delta_switch_sign_positive(self):
        orders = np.array([20, 0])
        prices = np.array([2., 2.])
        positions = np.array([-10, 10])

        delta = etns.calc_cash_delta(orders, prices, positions)
        testing.assert_array_equal(0., delta)

    def test_get_expiry_date_for_month(self):
        self.assertEqual(
            dt.date(2014, 12, 17),
            etns.get_expiry_date_for_month(dt.date(2014, 12, 1)))
        self.assertEqual(
            dt.date(2015, 1, 21),
            etns.get_expiry_date_for_month(dt.date(2015, 1, 1)))
        self.assertEqual(
            dt.date(2015, 2, 18),
            etns.get_expiry_date_for_month(dt.date(2015, 2, 1)))

    def test_get_next_expiry_date(self):
        self.assertEqual(
            dt.date(2014, 12, 17),
            etns.get_next_expiry_date(dt.date(2014, 12, 16)))
        self.assertEqual(
            dt.date(2015, 1, 21),
            etns.get_next_expiry_date(dt.date(2014, 12, 17)))
        self.assertEqual(
            dt.date(2015, 1, 21),
            etns.get_next_expiry_date(dt.date(2015, 1, 1)))
        self.assertEqual(
            dt.date(2015, 2, 18),
            etns.get_next_expiry_date(dt.date(2015, 2, 1)))


if __name__ == '__main__':
    unittest.main()
