import unittest

import numpy as np

from market_making import Fill, Positions, Position, Side
import market_making as mm


class TestMarketMaking(unittest.TestCase):
    def test_add_buy_position(self):
        positions = Positions()

        p1 = Position(Side.BUY, 99.0, 10.0)
        p2 = Position(Side.BUY, 98.0, 9.0)
        p3 = Position(Side.BUY, 97.0, 8.0)

        positions.add_position(p1)
        positions.add_position(p2)
        positions.add_position(p3)

        self.assertListEqual([p1, p2, p3], positions.buy_positions)

        positions.remove_position(Position(Side.BUY, 98.0, 9.0))
        self.assertListEqual([p1, p3], positions.buy_positions)

        positions.remove_position(Position(Side.BUY, 97.0, 8.0))
        self.assertListEqual([p1], positions.buy_positions)

        positions.remove_position(Position(Side.BUY, 99.0, 10.0))
        self.assertListEqual([], positions.buy_positions)

    def test_add_sell_position(self):
        positions = Positions()

        p1 = Position(Side.SELL, 99.0, 10.0)
        p2 = Position(Side.SELL, 98.0, 9.0)
        p3 = Position(Side.SELL, 97.0, 8.0)

        positions.add_position(p1)
        positions.add_position(p2)
        positions.add_position(p3)

        self.assertListEqual([p1, p2, p3], positions.sell_positions)

        positions.remove_position(Position(Side.SELL, 98.0, 9.0))
        self.assertListEqual([p1, p3], positions.sell_positions)

        positions.remove_position(Position(Side.SELL, 97.0, 8.0))
        self.assertListEqual([p1], positions.sell_positions)

        positions.remove_position(Position(Side.SELL, 99.0, 10.0))
        self.assertListEqual([], positions.sell_positions)

    def test_create_fill(self):
        np.random.seed(10)
        self.assertEqual(101.0, mm.create_fill(100.0, 1.0).price)
        self.assertEqual(99.0, mm.create_fill(100.0, 1.0).price)
        self.assertEqual(101.0, mm.create_fill(100.0, 1.0).price)


if __name__ == '__main__':
    unittest.main()
