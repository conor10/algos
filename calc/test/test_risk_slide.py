import unittest

import risk_slide as rs


class RiskSlideTest(unittest.TestCase):
    def test_load_positions(self):
        positions = rs.load_positions('data/positions.csv')
        self.assertEqual(8, len(positions))

    def test_risk_slide(self):
        pass
