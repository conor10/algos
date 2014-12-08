import numpy as np
from numpy import exp, pi, sqrt
from pylab import plot, show
from scipy.optimize import fmin
from scipy.stats import t

import datetime as dt
import data_loader
import init_logger
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

        # TODO: Plot these
        # TODO: Fix log return calculations
        stock_prices['Adj Returns'] = \
            utils.calculate_log_returns(stock_prices['Adj Close'].values)

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
        print('fitted mu: {}, sigma: {}'.format(fit_mu, fit_sigma))

        plot_fit(normal_dist, (fit_mu, fit_sigma), training_centres,
                 training_hist)

        # interpolate - how?

        fit_nu, fit_mu, fit_sigma = calc_fit(
            t_dist, np.random.rand(3), training_centres, training_hist)
        print('fitted nu: {}, mu: {}, sigma: {}'.format(
            fit_nu, fit_mu, fit_sigma))
        plot_fit(t_dist, (fit_nu, fit_mu, fit_sigma), training_centres,
                 training_hist)


        fit_log_mu, fit_log_sigma = calc_fit(
            log_normal_dist, np.random.rand(2), training_centres, training_hist)
        print('fitted log mu: {}, log sigma: {}'
              .format(fit_log_mu, fit_log_sigma))
        plot_fit(log_normal_dist, (fit_log_mu, fit_log_sigma),
                 training_centres, training_hist)

        fit_lambda = calc_fit(exp_weighted_moving_average, np.random.rand(1),
                              training_centres, training_hist)
        print('fitted lambda: {}'.format(fit_lambda))
        plot_fit(exp_weighted_moving_average, fit_lambda, training_centres,
                 training_hist)


def calc_fit(func, p0, ret, freq):
    min_func = lambda p, x, y: ((func(x, p) - y)**2).sum()

    return fmin(min_func, p0, args=(ret, freq), ftol=0.00001, xtol=0.00001)


def plot_fit(func, params, training_centres, training_hist):
    xx = linspace(min(training_centres), max(training_centres),
                  len(training_centres))
    yy = func(xx, params)
    plot(training_centres, training_hist, 'bo', xx, yy, 'r')
    show()


def normal_dist(x, params):
    mu = params[0]
    sigma = params[1]
    result = exp(-((x - mu)**2 / (2 * sigma**2))) / (sigma * sqrt(2 * pi))
    return result / sum(result)


def log_normal_dist(x, params):
    # mu = log(params[0]**2 / (sqrt(params[1]**2 + params[0]**2)))
    # sigma = sqrt(log(1 + (params[1]**2 / params[0]**2)))

    # mu = exp(params[0] + (params[1]**2 / 2))
    # sigma = sqrt((exp(params[1]**2) - 1) * exp(2 * params[0] + params[1]**2))
    mu = params[0]
    sigma = params[1]

    # result = exp(-((log(x) - mu)**2 / (2 * sigma**2)))
    #       / (x * sigma * sqrt(2 * pi))
    result = exp(-((x - mu)**2 / (2 * sigma**2))) \
            / (exp(x) * sigma * sqrt(2 * pi))
    return result / sum(result)


# def t_dist(x, nu, gamma):
#     return gamma((nu + 1) / 2) / \
#         (sqrt(pi * nu) * gamma(nu / 2) * (1 + x**2 / nu)**((nu + 1) / 2))
#
# def gamma():
#     pass

def t_dist(x, params):
    nu = params[0]
    mu = params[1]
    sigma = params[2]
    result = t.pdf(x, df=nu, loc=mu, scale=sigma)
    return result / sum(result)


def poisson_dist(x, params):
    lam = params[0]



def exp_weighted_moving_average(x, params):
    lam = params[0]
    result = np.zeros(len(x))
    result[0] = x[0]**2
    for i in range(1, len(x)):
        result[i] = (1 - lam) * x[i-1]**2 + lam * result[i-1]**2
    return sqrt(result) / sum(sqrt(result))


if __name__ == '__main__':
    init_logger.setup()
    main()