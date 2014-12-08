import unittest

import numpy.testing as ntest
import pandas as pd

import volatility_models as vol


FILENAME = 'data/VolatilityCones_corrected.xls'


class TestVolatilityModels(unittest.TestCase):

    def setUp(self):
        self.xls_file = pd.ExcelFile(FILENAME)

    def test_population_std_dev(self):
        """
        Row 474 of sheet needed 30 day calculation filled in
        """
        data = self.xls_file.parse('raw volatility cone',
                               header=1, index_col=0, parse_cols='B:H')
        data.sort_index(inplace=True)

        volatility = vol.population_std_dev(data['Adj. Close'], 30)
        ntest.assert_array_almost_equal(data['30 day vol'], volatility)

    def test_parkinson_std_dev(self):
        data = self.xls_file.parse('Parkinson volatility cone',
                               header=1, index_col=0, parse_cols='B:K')
        data.sort_index(inplace=True)

        volatility = vol.parkinson_std_dev(data['Adj High'],
                                           data['Adj Low'], 30)
        ntest.assert_array_almost_equal(data['30 day vol'].values, volatility)

    def test_sinclair_garman_klass_std_dev(self):
        data = self.xls_file.parse('Garman Klass volatility cone',
                              header=1, index_col=0, parse_cols='B:R')
        data.sort_index(inplace=True)

        volatility = vol.sinclair_garman_klass_std_dev(data['Adj Close'],
                                                       data['Adj Open'],
                                                       data['Adj High'],
                                                       data['Adj Low'],
                                                       30)
        ntest.assert_array_almost_equal(data['30 day vol'].values,
                                        volatility,
                                        decimal=5)

    def test_rogers_satchell_std_dev(self):
        data = self.xls_file.parse('Rogers Satchell volatility cone',
                                   header=1, index_col=0, parse_cols='B:R')
        data.sort_index(inplace=True)

        volatility = vol.rogers_satchell_std_dev(data['Adj Close'],
                                                 data['Adj Open'],
                                                 data['Adj High'],
                                                 data['Adj Low'],
                                                 30)
        ntest.assert_array_almost_equal(data['30 day vol'].values[29:],
                                        (volatility[29:]))

        volatility_sinclair = vol.sinclair_rogers_satchell_std_dev(
            data['Adj Close'],
            data['Adj Open'],
            data['Adj High'],
            data['Adj Low'],
            30)
        ntest.assert_array_almost_equal(data['30 day vol'].values[29:],
                                        (volatility_sinclair[29:]))


    def test_sinclair_yang_zhang_std_dev(self):
        """
        Mistakes:
        1. LogCC uses regular instead of adjusted close price for 30 day close
        2. 30 day open uses logOC, 30 day close uses logCC => inconsistent?
        3. k is incorrectly set to 0.164333, should be 0.141139421701
        4. 30 day RS is divided by N-1, it should be N
        """

        data = self.xls_file.parse('Yang Zhang volatility cone',
                                   header=1, index_col=0, parse_cols='B:AH')
        data.sort_index(inplace=True)

        volatility = vol.sinclair_yang_zhang_std_dev(data['Adj Close'],
                                                     data['Adj Open'],
                                                     data['Adj High'],
                                                     data['Adj Low'],
                                                     30)

        ntest.assert_array_almost_equal(data['30 day Yang Zhang'].values[30:],
                                        (volatility[30:]))


if __name__ == '__main__':
    unittest.main()
