import numpy as np
from option import Type

from scikits.statsmodels.tsa import arima_process

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


def generate_random_walks(periods, simulation_count, start_price, sigma,
                          model=NORMAL):
    return [np.cumsum(model(periods) * sigma) + start_price
            for i in range(simulation_count)]

# Why are these not the same?
def generate_random_walks_alt(periods, count, mu=0, sigma=1):
    return [np.random.normal(mu, sigma, periods) for i in range(count)]


def calc_itm_probability(strikes, price, sigma, periods, simulation_count, type):
    outcomes = generate_random_walks(periods, simulation_count, price, sigma)
    final_prices = np.transpose(np.array(outcomes))[-1]

    results = np.column_stack((strikes, np.zeros(strikes.shape)))

    for idx in np.ndindex(strikes.shape):
        # Can this be vectorised?
        if type is Type.CALL:
            results[idx][1] = \
                sum(final_prices > strikes[idx]) / float(simulation_count)
        elif type is Type.PUT:
            results[idx][1] = \
                sum(final_prices < strikes[idx]) / float(simulation_count)

    return results
