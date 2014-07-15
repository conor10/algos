import logging
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from scipy.stats import uniform

import backtester
import data_loader
import data_sources
import init_logger
from price_data import PriceData
import utils


def run():
    #symbols = data_loader.load_symbol_list(data_sources.SP_500_2012)
    symbols = ['ACN']

    start = utils.create_date('2012-01-01')
    end = utils.create_date('2012-12-31')

    close_price_data = _get_price_data(symbols, start, end)

    perform_backtests(symbols, close_price_data)
    perform_monte_carlo_simulation(symbols, close_price_data)


def _get_price_data(symbols, start_date=None, end_date=None):
    symbol_data = data_loader.load_price_data(data_sources.DATA_DIR, symbols)

    price_data = PriceData(symbol_data)
    return price_data.get_price_data_np(symbols, 'Adj Close', start_date,
                                        end_date)


def perform_backtests(symbols, close_price_data):

    sharpe_ratios = []

    for symbol in symbols:
        np_close = close_price_data[symbol]
        sharpe_ratio = run_backtest(np_close)
        sharpe_ratios.append((symbol, sharpe_ratio))

    sharpe_ratios_sorted = order_results_desc(sharpe_ratios)
    display_results(sharpe_ratios_sorted)


def run_backtest(np_close):

    iterative_backtest = backtester.MovingAverageBacktest()
    sharpe_ratio = iterative_backtest.run(np_close)

    chan_backtest = backtester.ChanBacktest()
    chan_backtest.run(np_close)

    return sharpe_ratio


def perform_monte_carlo_simulation(symbols, close_price_data):

    for symbol in symbols:
        np_close = close_price_data[symbol]
        run_monte_carlo_simulation(np_close)


def run_monte_carlo_simulation(np_close, simulation_count=1000):

    lookback_dist = uniform(2, 90)
    entry_z_score_dist = uniform(0, 5)
    exit_z_score_dist = uniform(-5, 5)

    results = np.zeros((simulation_count, 4))

    for i in range(0, simulation_count):
        lookback = lookback_dist.rvs()
        entry_z_score = entry_z_score_dist.rvs()
        exit_z_score = exit_z_score_dist.rvs()

        try:
            backtest = backtester.MovingAverageBacktest(
                lookback=lookback,
                entry_z_score=entry_z_score,
                exit_z_score=exit_z_score)
        except backtester.ParameterException as e:
            logging.debug(e)
            continue

        sharpe_ratio = backtest.run(np_close)

        results[i] = [sharpe_ratio, lookback, entry_z_score, exit_z_score]

    optimal = utils.get_max_vector(results)

    plot_3d_params(
        ['Sharpe Ratio', 'Lookback', 'Entry Z Score', 'Exit Z Score'],
        results)

    logging.info("Optimial result - Sharpe Ratio={} [lookback={}, "
                  "entry_z_score={}, exit_z_score={}]".format(*optimal))

    return results


def plot_3d_params(params, results):
    fig = plt.figure(figsize=(14,6))

    # TODO: How do we translate into a surface plot?
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(results[:, 1], results[:, 2], results[:, 0],
                   linewidth=1)

    ax.set_xlabel(params[1])
    ax.set_ylabel(params[2])
    ax.set_zlabel(params[0])

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(results[:, 1], results[:, 3], results[:, 0],
                   linewidth=1)

    ax.set_xlabel(params[1])
    ax.set_ylabel(params[3])
    ax.set_zlabel(params[0])

    plt.show()


def order_results_desc(result_set):
    return sorted(result_set, reverse=True, key=lambda tup: tup[1])


def display_results(results):
    for (sym, value) in results:
        logging.info("{}: {}".format(sym, value))


if __name__ == '__main__':
    init_logger.setup()
    run()
