import unittest

import pandas as pd
from pandas import Timestamp
from pandas.util.testing import assert_frame_equal

import data_loader as dl
from test_sources import CHAINS_DIR, DATA_DIR, FUTURES_DIR, INTRADAY_DIR, \
    INTRADAY_SP500_DIR, SYMBOL_LIST, SYMBOL_LIST_FILE, TEST_CSV_FILE, \
    TEST_REVERSED_CSV_FILE


class TestDataLoader(unittest.TestCase):
    def test_load_datafile(self):
        self.assertListEqual(SYMBOL_LIST, self._get_symbol_list())

    def test_load_invalid_datafile(self):
        self.assertListEqual([], dl.load_symbol_list('invalid'))

    def test_get_price_file(self):
        self.assertEqual('someDir/test.csv',
                         dl._get_price_file('test', 'someDir'))

    def test_load_symbol_data(self):
        df = dl.load_symbol_data(TEST_CSV_FILE, 'Date')
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
        test = dl.load_symbol_data(TEST_CSV_FILE)
        test_reversed = dl.load_symbol_data(TEST_REVERSED_CSV_FILE)
        self.assertEquals(test.ix[0]['Close'], test_reversed.ix[0]['Close'])

    def test_load_price_data(self):
        price_data = dl.load_price_data(DATA_DIR, self._get_symbol_list())
        self.assertEqual(3, len(price_data))
        self.assertEqual(pd.DataFrame, type(price_data['TEST']))
        self.assertEqual(pd.DataFrame, type(price_data['TEST2']))
        self.assertEqual(pd.DataFrame, type(price_data['TEST3']))

    def test_load_option_data(self):
        symbols = ['AAPL', 'MSFT']
        option_data = dl.load_option_data('SP500', CHAINS_DIR, symbols)
        print(option_data)
        self.assertEqual(2, len(option_data))

        aapl_call = option_data['AAPL']['20140908']['20140912']['C']
        aapl_put = option_data['AAPL']['20140908']['20140912']['P']
        self.assertEqual(49, len(aapl_call))
        self.assertEqual(49, len(aapl_put))
        self.assert_frame_different(aapl_call, aapl_put)

        msft_call = option_data['MSFT']['20140909']['20160115']['C']
        msft_put = option_data['MSFT']['20140909']['20160115']['P']
        self.assertEqual(17, len(msft_call))
        self.assertEqual(17, len(msft_put))
        self.assert_frame_different(msft_call, msft_put)

    def assert_frame_different(self, left, right):
        try:
            assert_frame_equal(left, right)
            raise TestException
        except AssertionError:
            pass
        except TestException:
            self.fail()

    def _get_symbol_list(self):
        return dl.load_symbol_list(SYMBOL_LIST_FILE)

    def test_load_intraday_data(self):
        data = dl.load_intraday_data('SP500', INTRADAY_DIR, ['AA', 'AAPL'])
        self.assertTrue('AA' in data)
        self.assertTrue('AAPL' in data)

        self.assertTrue('20140811' in data['AA'])
        self.assertTrue('20140811' in data['AAPL'])

        self.assertTrue('20140818' in data['AA'])
        self.assertTrue('20140818' in data['AAPL'])

        self.assertEqual(6, len(data['AA']))
        self.assertEqual(6, len(data['AA']))

        self.assertEqual(391, len(data['AA']['20140811']))
        self.assertEqual(391, len(data['AAPL']['20140818']))

    def test_get_dates(self):
        self.assertListEqual(
            ['20140811', '20140812', '20140813', '20140814', '20140815',
             '20140818'],
            dl.get_dates(INTRADAY_SP500_DIR))

        self.assertListEqual(
            ['20140813', '20140814', '20140815', '20140818'],
            dl.get_dates(INTRADAY_SP500_DIR, '20140813'))

        self.assertListEqual(
            ['20140811', '20140812', '20140813'],
            dl.get_dates(INTRADAY_SP500_DIR, end_date='20140813'))

        self.assertListEqual(
            ['20140812', '20140813', '20140814', '20140815'],
            dl.get_dates(INTRADAY_SP500_DIR, '20140812', '20140815'))

    def test_load_vix_futures_prices(self):
        data = dl.load_vix_futures_prices(FUTURES_DIR)
        self.assertTrue(2014 in data)
        self.assertTrue(2010 in data)
        self.assertEqual(156, len(data[2010][0]))
        self.assertEqual(188, len(data[2014][11]))


class TestException(Exception):
    pass


if __name__ == '__main__':
    unittest.main()
