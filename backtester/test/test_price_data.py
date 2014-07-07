import unittest

from pandas import Timestamp

import data_loader as dl
from price_data import PriceData
from test.test_sources import DATA_DIR, SYMBOL_LIST
import utils
from utils import DATE_FORMAT


DATE1 = '2012-08-28'
DATE2 = '2012-08-29'
DATE3 = '2012-08-30'


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

    def test_get_price_data_by_date_range(self):

        price_data = PriceData(self._get_test_data())
        start = utils.create_date(DATE1)
        end = utils.create_date(DATE2)

        result = price_data.get_price_data(['TEST'], 'Close', start, end)

        self.assertEqual(2, len(result))
        self.assertEqual(DATE1, result.index[0].strftime(DATE_FORMAT))
        self.assertEqual(DATE2, result.index[-1].strftime(DATE_FORMAT))

    def test_get_price_data_by_open_date_range(self):
        price_data = PriceData(self._get_test_data())
        start = utils.create_date(DATE2)

        result = price_data.get_price_data(['TEST'], 'Close', start)

        self.assertEqual(2, len(result))
        self.assertEqual(DATE2, result.index[0].strftime(DATE_FORMAT))
        self.assertEqual(DATE3, result.index[-1].strftime(DATE_FORMAT))

    def _get_test_data(self):
        return dl.load_price_data(DATA_DIR, SYMBOL_LIST)


if __name__ == '__main__':
    unittest.main()
