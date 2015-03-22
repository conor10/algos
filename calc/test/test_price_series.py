import math
import unittest

import numpy as np
from numpy import random
from numpy.testing import utils as np_utils

from option import OptionType
import price_series
import utils

import matplotlib.pyplot as plt


class PriceSeriesTest(unittest.TestCase):
    def test_random_walk(self):
        random.seed(1)
        series1 = price_series.generate_random_walks(5, 2, 70, 1)
        random.seed(1)
        series2 = price_series.generate_random_walks_alt(5, 2, 70, 1)
        print(series1)
        print(series2)

    def test_generate_random_price_series(self):
        random.seed(1)
        series = price_series._generate_random_price_series(70, 0.35, 5)
        print(series)
        np_utils.assert_array_almost_equal(
            np.array([70.56852088, 69.78588526, 69.81513989, 69.62446098, 70.30289267]),
            series)


    def test_itm_probability(self):
        price = 70.0
        sigma = 3.5  # 5% volatility
        strikes = np.arange(60, 80, 0.1)
        itm_probability = price_series.calc_itm_probability(
            strikes, price, sigma, 100, 1000, OptionType.PUT)

        print(itm_probability)

    def test_generate_random_walk(self):
        iterations = 1
        periods = 200
        price = 70.0
        mu = 0.05
        sigma = 0.01
        delta = 1.0
        # tom_prices = price_series.generate_random_walk(100, price, sigma)
        #
        # tom_returns = utils.calculate_returns(tom_prices)
        # print('Return std dev: {}, expected: {}'
        #       .format(tom_returns.std() * math.sqrt(252.0), sigma))

        # self.assertEqual(price, prices.mean())
        self.run_price_simulation(price_series.generate_bm_prices,
                                  utils.calculate_returns,
                                  iterations, periods, price, mu, sigma, delta)
        self.run_price_simulation(price_series.generate_gbm_prices,
                                  utils.calculate_log_returns,
                                  iterations, periods, price, mu, sigma, delta)


    def run_price_simulation(self, sim_func, ret_func, simulation_count,
                             periods, price, mu, sigma, delta):
        np.random.seed(1)
        annualised_days = 252.0
        sigmas = []
        mus = []
        for i in range(0, simulation_count):
            prices = sim_func(periods, price, mu, sigma, delta)
            returns = ret_func(prices)
            mus.append(returns.mean() * annualised_days) # These are log returns
            sigmas.append(returns.std() * math.sqrt(annualised_days))

        mus = np.array(mus)
        sigmas = np.array(sigmas)


        # plt.scatter(sigmas, mus)
        # plt.subplot(211)
        # plt.hist(mus)
        # plt.subplot(212)
        # plt.hist(sigmas)
        # plt.show()

        mu_hat = mus.mean()
        sigma_hat = sigmas.mean()

        mu_hat_se = mus.std() / math.sqrt(periods)
        sigma_hat_se = sigmas.std() / math.sqrt(periods)

        print('Return mean: {}, expected: {}, error: +/-{}'
              .format(mu_hat, mu, mu_hat_se))
        print('Return std dev: {}, expected: {}, error: +/-{}'
              .format(sigma_hat, sigma, sigma_hat_se))
        # self.assertAlmostEqual(mu, mu_hat, delta=mu_hat_se)
        # self.assertAlmostEqual(sigma, sigma_hat, delta=sigma_hat_se)

    def test_geometric_models(self):
        np.random.seed(10)

        periods = 1000
        price = 70.0
        mu = 0.0
        sigma = 0.3
        period_duration = 1.0

        np.random.seed(10)
        iterative_ret = price_series.generate_gbm_prices(
            periods, price, mu, sigma, period_duration)
        np.random.seed(10)
        vectorised_ret = price_series.generate_gbm_prices_vec(
            periods, price, mu, sigma, period_duration)

        # Equal to 6 decimal places
        np_utils.assert_array_almost_equal(iterative_ret, vectorised_ret)


    def test_real_models(self):
        np.random.seed(10)

        periods = 1000
        price = 70.0
        mu = 0.0
        sigma = 0.3
        period_duration = 1.0

        np.random.seed(10)
        iterative_ret = price_series.generate_bm_prices(
            periods, price, mu, sigma, period_duration)
        np.random.seed(10)
        vectorised_ret = price_series.generate_real_returns_vec(
            periods, price, mu, sigma, period_duration)

        # Equal to 6 decimal places
        np_utils.assert_array_almost_equal(iterative_ret, vectorised_ret)



if __name__ == '__main__':
    unittest.main()
