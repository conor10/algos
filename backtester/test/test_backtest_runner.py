import unittest

import backtest_runner

class TestBacktestRunner(unittest.TestCase):

    def test_order_results_desc(self):
        self.assertEqual(
            [('T3', 10), ('T4', 7), ('T', 5), ('T2', 3)],
            backtest_runner.order_results_desc(
                [('T', 5), ('T2', 3), ('T3', 10), ('T4', 7)]))


if __name__ == '__main__':
    unittest.main()
