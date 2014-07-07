import numpy as np
import unittest

from numpy.testing import utils

import backtester as bt

PRICES = np.array([10.0, 15.0, 17.5, 12.5, 7.5, 10.0, 15.0, 20.0, 22.5, 17.5])
SIGNALS1 = np.array([0, 1, 0, 1, 0, -1, 1, 0, -1, -1])
SIGNALS2 = signals2 = np.array([1, 1, 0, 1, 0, -1, 1, 0, -1, 0])


class TestBacktester(unittest.TestCase):

    def test_break_upwards(self):
        self.assertTrue(bt.break_upwards([10.0, 15.0], [11.0, 14.0], 1))
        self.assertTrue(bt.break_upwards([10.0, 15.0], [10.0, 14.0], 1))
        self.assertFalse(bt.break_upwards([10.0, 15.0], [11.0, 16.0], 1))

    def test_break_downwards(self):
        self.assertTrue(bt.break_downwards([15.0, 10.0], [14.0, 11.0], 1))
        self.assertTrue(bt.break_downwards([15.0, 10.0], [15.0, 11.0], 1))
        self.assertFalse(bt.break_downwards([15.0, 10.0], [14.0, 9.0], 1))

    def test_calculate_pnl(self):
        self.assertEqual(-7.5, bt.calculate_pnl(PRICES, SIGNALS1)[-1])
        self.assertEqual(20, bt.calculate_pnl(PRICES, SIGNALS2)[-1])

    def test_calculate_returns(self):
        returns = bt.calculate_returns(self._generate_pnl())
        utils.assert_allclose(
            [0., 0., 0., 0.83333333, 0., -0.36363636,
             0.85714286, 0., -0.69230769, -1.75],
            returns, rtol=1e-7)

    def test_calculate_sharpe_ratio(self):
        pnl = self._generate_pnl()
        returns = bt.calculate_returns(pnl)
        self.assertEqual(-2.5095619671562686,
                         bt.calculate_sharpe_ratio(returns))

    def test_order_results_desc(self):
        self.assertEqual(
            [('T3', 10), ('T4', 7), ('T', 5), ('T2', 3)],
            bt.order_results_desc(
                [('T', 5), ('T2', 3), ('T3', 10), ('T4', 7)]))

    def test_ffill(self):
        data = np.array([0, np.nan, 1, np.nan, np.nan, -1, np.nan])
        utils.assert_array_equal(
            np.array([0, 0, 1, 1, 1, -1, -1]),
            bt.ffill(data))

    def test_lag(self):
        utils.assert_array_equal(
            np.array([0, 1, 2, 3, 4]),
            bt.lag(np.array([1, 2, 3, 4, 5])))

    def _generate_pnl(self):
        return bt.calculate_pnl(PRICES, SIGNALS1)


if __name__ == '__main__':
    unittest.main()
