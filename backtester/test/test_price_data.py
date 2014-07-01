import unittest

from pandas import Timestamp

import data_loader
from price_data import PriceData
from test.test_sources import DATA_DIR, SYMBOL_LIST


class TestPriceData(unittest.TestCase):
    def test_get_price_data(self):

        price_data = PriceData(self._get_test_data())
        result = price_data.get_price_data(['TEST', 'TEST2'], 'Close')

        # float('NaN') != float('NaN')
        result = result.fillna(0)

        self.assertEqual(2, len(result.columns))
        self.assertDictEqual(
            {Timestamp('2012-08-28 00:00:00'): 13102.99,
             Timestamp('2012-08-29 00:00:00'): 13107.48,
             Timestamp('2012-08-30 00:00:00'): 13000.709999999999},
            result["TEST"].to_dict())
        self.assertDictEqual(
            {Timestamp('2012-08-28 00:00:00'): 0,
             Timestamp('2012-08-29 00:00:00'): 1410.49,
             Timestamp('2012-08-30 00:00:00'): 1399.48},
            result["TEST2"].to_dict())

    def _get_test_data(self):
        return data_loader.load_price_data(DATA_DIR, SYMBOL_LIST)

if __name__ == '__main__':
    unittest.main()
