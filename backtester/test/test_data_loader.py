import unittest

import pandas as pd
from pandas import Timestamp

import data_loader as dl

from test.test_sources import DATA_DIR, SYMBOL_LIST, SYMBOL_LIST_FILE, \
    TEST_CSV_FILE, TEST_REVERSED_CSV_FILE


class TestDataLoader(unittest.TestCase):
    def test_load_datafile(self):
        self.assertListEqual(SYMBOL_LIST, self._get_symbol_list())

    def test_load_invalid_datafile(self):
        self.assertListEqual([], dl.load_symbol_list('invalid'))

    def test_get_price_file(self):
        self.assertEqual('someDir/test.csv',
                         dl._get_price_file('test', 'someDir'))

    def test_load_symbol_data(self):
        df = dl._load_symbol_data(TEST_CSV_FILE, 'Date')
        # we don't won't index to be returned with each result
        self.assertDictEqual(
            {'Adj Close': 13107.48,
              'Close': 13107.48,
              'High': 13144.81,
              'Low': 13081.27,
              'Open': 13103.46,
              'Volume': 446252369.0},
            df.ix['2012-08-29'].to_dict())
        self.assertDictEqual(
            {Timestamp('2012-08-28'): 13102.99,
             Timestamp('2012-08-29'): 13107.48,
             Timestamp('2012-08-30'): 13000.709999999999},
            df['Adj Close'].to_dict())

    def test_load_symbol_data_reversed(self):
        test = dl._load_symbol_data(TEST_CSV_FILE)
        test_reversed = dl._load_symbol_data(TEST_REVERSED_CSV_FILE)
        self.assertEquals(test.ix[0]['Close'], test_reversed.ix[0]['Close'])

    def test_load_price_data(self):
        price_data = dl.load_price_data(DATA_DIR, self._get_symbol_list())
        self.assertEqual(3, len(price_data))
        self.assertEqual(pd.DataFrame, type(price_data['TEST']))
        self.assertEqual(pd.DataFrame, type(price_data['TEST2']))
        self.assertEqual(pd.DataFrame, type(price_data['TEST3']))

    def _get_symbol_list(self):
        return dl.load_symbol_list(SYMBOL_LIST_FILE)


if __name__ == '__main__':
    unittest.main()