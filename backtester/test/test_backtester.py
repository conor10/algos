import numpy as np
import unittest

from numpy.testing import utils

import backtester
from backtester import MovingAverageBacktest

PRICES = np.array([10.0, 15.0, 17.5, 12.5, 7.5, 10.0, 15.0, 20.0, 22.5, 17.5])
SIGNALS1 = np.array([0, 1, 0, 1, 0, -1, 1, 0, -1, -1])
SIGNALS2 = signals2 = np.array([1, 1, 0, 1, 0, -1, 1, 0, -1, 0])


class TestBacktester(unittest.TestCase):

    def test_calculate_returns(self):
        returns = backtester._calculate_returns(self._generate_pnl())
        utils.assert_allclose(
            [0., 0., 0., 0.83333333, 0., -0.36363636,
             0.85714286, 0., -0.69230769, -1.75],
            returns, rtol=1e-7)

    def test_calculate_sharpe_ratio(self):
        pnl = self._generate_pnl()
        returns = backtester._calculate_returns(pnl)
        self.assertEqual(-2.5095619671562686,
                         backtester._calculate_sharpe_ratio(returns))

    def _generate_pnl(self):
        return MovingAverageBacktest._calculate_positions(PRICES, SIGNALS1)


class TestMovingAverageBacktest(unittest.TestCase):

    def test_break_upwards(self):
        self.assertTrue(MovingAverageBacktest._break_upwards(
            [10.0, 15.0], [11.0, 14.0], 1))
        self.assertTrue(MovingAverageBacktest._break_upwards(
            [10.0, 15.0], [10.0, 14.0], 1))
        self.assertFalse(MovingAverageBacktest._break_upwards(
            [10.0, 15.0], [11.0, 16.0], 1))

    def test_break_downwards(self):
        self.assertTrue(MovingAverageBacktest._break_downwards(
            [15.0, 10.0], [14.0, 11.0], 1))
        self.assertTrue(MovingAverageBacktest._break_downwards(
            [15.0, 10.0], [15.0, 11.0], 1))
        self.assertFalse(MovingAverageBacktest._break_downwards(
            [15.0, 10.0], [14.0, 9.0], 1))

    def test_calculate_pnl(self):
        self.assertEqual(-7.5, MovingAverageBacktest._calculate_positions(
            PRICES, SIGNALS1)[-1])
        self.assertEqual(20, MovingAverageBacktest._calculate_positions(
            PRICES, SIGNALS2)[-1])


if __name__ == '__main__':
    unittest.main()
