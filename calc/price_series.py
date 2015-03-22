import math

import numpy as np
from scikits.statsmodels.tsa import arima_process

from option import OptionType


class Series:
    RANDOM_WALK = 0
    MOMENTUM = 1
    MEAN_REVERTING = 2


NORMAL = lambda periods: np.random.normal(0, 1, periods)


def generate(type, start_price, iterations=1000):
    if type == Series.MOMENTUM:
        pass
    elif type == Series.MOMENTUM:
        pass
    else:
        pass



def _generate_random_return(iterations, mu=0.0, sigma=1):
    return np.cumsum(
        _generate_random_price_series(mu, sigma, iterations))

def _generate_random_price_series(mu, sigma, iterations):
    #return np.random.randn(iterations) * sigma + mu
    return np.random.normal(mu, sigma, iterations)

def _generate_ar_return(iterations, ar, phi):
    arima_process.arma_generate_sample([ar], [phi], iterations, sigma=1)

# create cointegrated price series

def generate_random_walk(periods, start_price, sigma, model=NORMAL):
    return np.cumsum(model(periods) * sigma) + start_price

def generate_random_walk_revised(periods, start_price, sigma, days):
    return start_price + \
           start_price * (np.cumsum(np.random.normal(0, 1, periods)) *
                          sigma * np.sqrt(days / 252.0))

def generate_random_walks(periods, simulation_count, start_price, sigma,
                          model=NORMAL):
    return [np.cumsum(model(periods) * sigma) + start_price
            for i in range(simulation_count)]


DAYS_PER_YEAR = 252.0

def generate_bm_prices(periods, start_price, mu, sigma, delta):
    t = delta / DAYS_PER_YEAR
    prices = np.zeros(periods)
    epsilon_sigma_t = np.random.normal(0, 1, periods) * sigma * np.sqrt(t)
    prices[0] = start_price
    for i in range(1, len(prices)):
        prices[i] = prices[i-1] * mu * t + \
                    prices[i-1] * epsilon_sigma_t[i-1] + \
                    prices[i-1]
    return prices


def generate_gbm_prices(periods, start_price, mu, sigma, delta):
    t = delta / DAYS_PER_YEAR
    prices = np.zeros(periods)
    epsilon_sigma_t = np.random.normal(0, 1, periods-1) * sigma * np.sqrt(t)
    prices[0] = start_price
    for i in range(1, len(prices)):
        prices[i] = prices[i-1] * \
                    np.exp((mu - 0.5 * sigma**2) * t +
                           epsilon_sigma_t[i-1])
    return prices


def generate_gbm_prices_vec(periods, start_price, mu, sigma, delta):
    epsilon = np.random.normal(0, 1, periods-1)

    t = np.linspace(0, periods-1, periods) / DAYS_PER_YEAR
    t[0] = 0.0

    W = np.insert(np.cumsum(epsilon), 0, 0.0) / math.sqrt(DAYS_PER_YEAR)
    t = t * delta
    W = W * math.sqrt(delta)

    return start_price * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)



def calc_itm_probability(strikes, price, sigma, periods, simulation_count, type):
    outcomes = generate_random_walks(periods, simulation_count, price, sigma)
    final_prices = np.transpose(np.array(outcomes))[-1]

    results = np.column_stack((strikes, np.zeros(strikes.shape)))

    for idx in np.ndindex(strikes.shape):
        # Can this be vectorised?
        if type is OptionType.CALL:
            results[idx][1] = \
                sum(final_prices > strikes[idx]) / float(simulation_count)
        elif type is OptionType.PUT:
            results[idx][1] = \
                sum(final_prices < strikes[idx]) / float(simulation_count)

    return results
