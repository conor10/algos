from math import gamma
from numpy import log, sqrt
import numpy as np

import utils

ANNUALISER = sqrt(252.0)


def annualise(data):
    return data * ANNUALISER


def sample_variance(returns):
    N = len(returns)
    return (1 / N) * sum(returns**2)


def population_variance(returns):
    N = len(returns)
    return (N / (N - 1)) * sample_variance(returns)


def population_std_dev(close_prices, lookback, unbiased=False):
    N = float(lookback)

    prices = log(close_prices / utils.lag(close_prices))
    results = np.zeros(np.size(prices))
    results[:] = np.NAN
    for i in range(lookback, len(prices)):
        bounds = range(i-(lookback-1), i+1)
        results[i] = sqrt(
            ((prices[bounds] - prices[bounds].sum() / N)**2).sum() / (N - 1))
    if unbiased:
        results = unbias_std_dev(results, N)

    return annualise(results)


def unbias_std_dev(std_dev):
    N = float(len(std_dev))
    bias = sqrt(2 / N) * (gamma(N / 2) / gamma((N - 1) / 2))
    return bias * std_dev


def parkinson_std_dev(high_prices, low_prices, lookback):
    """
    Requires high and low prices during trading period
    """
    N = float(lookback)

    prices = log(high_prices / low_prices)**2
    results = np.zeros(np.size(prices))
    results[:] = np.NAN
    for i in range(lookback-1, len(high_prices)):
        bounds = range(i-(lookback-1), i+1)
        results[i] = sqrt((1 / (4 * N * log(2))) *
                          (prices[bounds]).sum())
    return annualise(results)

    # Sinclair's implementation produces the same result
    # prices = (1 / (4 * log(2))) * log(high_prices / low_prices)**2
    # results = np.zeros(np.size(prices))
    # results[:] = np.NAN
    # for i in range(lookback-1, len(high_prices)):
    #     bounds = range(i-(lookback-1), i+1)
    #     results[i] = sqrt((prices[bounds]).sum() / N)
    # return annualise(results)


def garman_klass_std_dev(close_prices, open_prices, high_prices, low_prices, lookback):
    N = float(lookback)
    # lagged_close_prices = utils.lag(close_prices)

    mids = 0.5 * log(high_prices / low_prices)**2
    # The original paper uses open prices, not lagged close
    closes = (2 * log(2) - 1) * log(close_prices / open_prices)**2
    # closes = (2 * log(2) - 1) * log(close_prices / lagged_close_prices)**2

    results = np.zeros(np.size(close_prices))
    results[:] = np.NAN

    for i in range(lookback-1, len(high_prices)):
        bounds = range(i-(lookback-1), i+1)
        results[i] = sqrt((mids[bounds].sum() / N) -
                          (closes[bounds].sum() / N))
    return annualise(results)


def sinclair_garman_klass_std_dev(close_prices, open_prices, high_prices,
                                  low_prices, lookback):
    N = float(lookback)
    lagged_close_prices = utils.lag(close_prices, 0.)

    logHL = 0.5 * log(high_prices / low_prices)**2
    logCO = (2 * log(2) - 1) * log(close_prices / open_prices)**2
    # Sinclair's implementation includes overnight price changes as a factor
    logOC = log(open_prices / lagged_close_prices)**2

    results = np.zeros(np.size(close_prices))
    results[:] = np.NAN

    # Start at lookback, as we require a drift term
    for i in range(lookback, len(high_prices)):
        bounds = range(i-(lookback-1), i+1)
        results[i] = sqrt((logOC[bounds].sum() +
                          logHL[bounds].sum() -
                          logCO[bounds].sum()) / N)
    return annualise(results)


def rogers_satchell_std_dev(close_prices, open_prices, high_prices,
                    low_prices, lookback):
    N = float(lookback)

    logHC = log(high_prices / close_prices)
    logHO = log(high_prices / open_prices)
    logLC = log(low_prices / close_prices)
    logLO = log(low_prices / open_prices)

    results = np.zeros(np.size(close_prices))
    results[:] = np.NAN

    for i in range(lookback-1, len(results)):
        bounds = range(i-(lookback-1), i+1)
        results[i] = sqrt(
            ((logHC[bounds] * logHO[bounds]) +
            (logLC[bounds] * logLO[bounds])).sum() / N)
    return annualise(results)


"""Alternative, but valid implementation"""
def sinclair_rogers_satchell_std_dev(close_prices, open_prices, high_prices,
                            low_prices, lookback):
    N = float(lookback)

    logHO = log(high_prices / open_prices)
    logLO = log(low_prices / open_prices)
    logCO = log(close_prices / open_prices)

    results = np.zeros(np.size(close_prices))
    results[:] = np.NAN

    for i in range(lookback-1, len(results)):
        bounds = range(i-(lookback-1), i+1)
        results[i] = sqrt(
            ((logHO[bounds] *
              (logHO[bounds] - logCO[bounds])) +
             (logLO[bounds] *
              (logLO[bounds] - logCO[bounds]))).sum()
            / N)
    return annualise(results)


def yang_zhang_std_dev(close_prices, open_prices, high_prices,
               low_prices, lookback):
    """
    Implementation as per Yhang Zang 2000
    """
    N = float(lookback)

    k = 0.34 / (1.34 + ((N + 1) / (N - 1)))

    logOC = log(open_prices / utils.lag(close_prices))
    logCO = log(close_prices / open_prices)
    logHC = log(high_prices / close_prices)
    logHO = log(high_prices / open_prices)
    logLC = log(low_prices / close_prices)
    logLO = log(low_prices / open_prices)

    results = np.zeros(np.size(close_prices))
    results[:] = np.NAN

    # Start at lookback, as we require a lag term
    for i in range(lookback, len(results)):
        bounds = range(i - (lookback - 1), i + 1)
        open_var = ((logOC[bounds] - (logOC[bounds].sum() / N))**2).sum() / \
                   (N - 1)
        close_var = ((logCO[bounds] - (logCO[bounds].sum() / N))**2).sum() / \
                    (N - 1)
        rs_var = ((logHC[bounds] * logHO[bounds]) +
                  (logLC[bounds] * logLO[bounds])).sum() / N

        results[i] = sqrt(open_var + k * close_var + (1 - k) * rs_var)

    return annualise(results)


def sinclair_yang_zhang_std_dev(close_prices, open_prices, high_prices,
                       low_prices, lookback):
    """
    Sinclair implementation uses 0 as our mean for std_dev calculations
    Additionally uses open/close for open variance, & close/close for close
    std_dev, as opposed to the same for both
    """
    N = float(lookback)

    k = 0.34 / (1.34 + ((N + 1) / (N - 1)))
    print(k)

    logOC = log(open_prices / utils.lag(close_prices))
    logCC = log(close_prices / utils.lag(close_prices))
    logHC = log(high_prices / close_prices)
    logHO = log(high_prices / open_prices)
    logLC = log(low_prices / close_prices)
    logLO = log(low_prices / open_prices)

    results = np.zeros(np.size(close_prices))
    results[:] = np.NAN

    # Start at lookback, as we require a lag term
    for i in range(lookback, len(results)):
        bounds = range(i - (lookback - 1), i + 1)
        open_var = (logOC[bounds]**2).sum() / (N - 1)
        close_var = (logCC[bounds]**2).sum() / (N - 1)
        rs_var = ((logHC[bounds] * logHO[bounds]) +
                  (logLC[bounds] * logLO[bounds])).sum() / N

        results[i] = sqrt(open_var + k * close_var + (1 - k) * rs_var)

    return annualise(results)


def first_exit(prices, delta, lookback):
    """
    :param delta: our observation_barrier:
    """
    # Our exit_times
    exit_times = []

    results = np.zeros(np.size(prices))
    results[:] = np.NAN

    for i in range(lookback-1, len(results)):
        bounds = range(i-(lookback-1), i+1)

        upper_barrier = prices[0] + delta
        lower_barrier = prices[0] - delta
        for i in range(1, len(prices)):
            price = prices[i]
            if price >= upper_barrier or price <= lower_barrier:
                exit_times.append(i)
                upper_barrier = price + delta
                lower_barrier = price - delta

        tau = np.array(exit_times)
        results[i] = delta / (sqrt(tau / len(tau)))