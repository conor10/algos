import datetime as dt
from numpy import exp, linspace, pi, sqrt
import numpy as np
from pylab import plot, show
from scipy.optimize import fmin

import data_loader
import init_logger
import price_series
import utils

# Given a price series, determine the most appropriate distribution to
# use for modelling future prices

def main():

    run_date = dt.datetime(2014, 9, 19)

    # symbols = \
    #     data_loader.load_symbol_list(
    #         '/Users/Conor/code/python/conor10.tickdata/symbols/'
    #         'SP500/20140907/symbols.txt')
    symbols = ['AAPL']

    all_stock_prices = data_loader.load_price_data(
        '/Users/Conor/code/python/conor10.tickdata/daily_prices/SP500',
        symbols)

    for symbol in symbols:
        stock_prices = all_stock_prices[symbol]
        underlying_price = \
            stock_prices['Close'][[run_date]].values[0]
        print(underlying_price)

        stock_prices['Adj Returns'] = \
            utils.calculate_returns(stock_prices['Adj Close'].values)

        # extract a training set
        training_set = stock_prices['Adj Returns'][-2500:-200].values
        test_set = stock_prices['Adj Returns'][-200:].values

        sigma = training_set.std()
        mu = training_set.mean()
        print('training set mu: {}, sigma: {}'.format(mu, sigma))

        training_hist, training_edges = np.histogram(training_set, 120)

        training_hist = training_hist / float(len(training_set))

        training_centres = 0.5 * (training_edges[1:] + training_edges[:-1])

        fit_mu, fit_sigma = calc_fit(
            normal_dist, np.random.rand(2), training_centres, training_hist)

        # normal_dist = generate_normal_dist(mu, sigma, 100000)
        # normal_bars = get_bars(normal_dist)

        # normalise our data - not applicable here

        # interpolate - how?
        xx = linspace(min(training_centres), max(training_centres),
                      len(training_centres))
        yy = normal_dist(xx, fit_mu, fit_sigma)
        plot(training_centres, training_hist, 'bo', xx, yy, 'r')
        show()


def calc_fit(func, p0, ret, freq):
    min_func = lambda p, x, y: ((func(x, p[0], p[1]) - y)**2).sum()

    mu, sigma = fmin(min_func, p0, args=(ret, freq), ftol=0.00001, xtol=0.00001)
    print('fitted mu: {}, sigma: {}'.format(mu, sigma))
    return mu, sigma


def normal_dist(x, mu, sigma):
    # return exp(-((x - mu)**2 / (2 * sigma**2))) / (sigma * sqrt(2 * pi))
    result = exp(-((x - mu)**2 / (2 * sigma**2))) / (sigma * sqrt(2 * pi))
    return result / sum(result)


def t_dist():
    pass


def generate_normal_dist(mu, sigma, size):
    np.random.normal(mu, sigma, size)


def generate_t_dist():
    pass


if __name__ == '__main__':
    init_logger.setup()
    main()