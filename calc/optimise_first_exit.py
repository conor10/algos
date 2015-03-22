import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

import data_loader as dl
import price_series as ps
import utils
import volatility_models as vm


def main():
    aapl_first_exit()
    # random_first_exit()


def aapl_first_exit():
    aapl_intraday_prices = get_intraday_data()
    aapl_returns = vm.intraday_returns(aapl_intraday_prices)
    day_count = len(aapl_intraday_prices)
    print(day_count)
    first_exit_vol = vm.first_exit(aapl_returns, 0.01, 20, day_count)
    print(first_exit_vol)

    close_prices = get_close_prices(aapl_intraday_prices)
    rolling_std = pd.rolling_std(aapl_returns, window=391 * 20) * np.sqrt(252.0)
    std_dev = []
    for i in range(0, len(rolling_std), 391):
        std_dev.append(rolling_std[i:i+391].mean())
    print(std_dev)

    plot(close_prices, first_exit_vol, std_dev)


def random_first_exit():
    iterations = 1
    mu = 95.
    sigma = 0.3

    start = dt.datetime(2014, 8, 11)
    end = dt.datetime(2014, 12, 05)
    day_count = utils.work_day_count(start, end)

    intraday_returns = get_random_intraday_returns(mu, sigma, day_count)
    first_exit_vol = vm.first_exit(intraday_returns, 0.01, 20, day_count)

    print(first_exit_vol)

    # TODO: Implement a resampling parameter?
    rolling_std = pd.rolling_std(intraday_returns, window=391 * 20) * np.sqrt(252.0)
    std_dev = []
    for i in range(0, len(rolling_std), 391):
        std_dev.append(rolling_std[i:i+391].mean())

    plot(intraday_returns,
         first_exit_vol,
         std_dev)


def plot(close_prices, first_exit_vol, std_dev):

    fig = plt.figure()
    fig.add_subplot(2, 1, 1)
    plt.plot(close_prices)
    # plt.plot(intraday_returns)

    fig.add_subplot(2, 1, 2)
    plt.plot(first_exit_vol, label='first exit')
    plt.plot(std_dev, label='std dev')
    plt.legend()
    plt.show()



def get_intraday_data():
    data = dl.load_intraday_data('SP500', '/Users/Conor/trading/prices', ['AAPL'],
                                 '20140811', '20141205')
    return data['AAPL']


def get_random_intraday_returns(mu, sigma, day_count):
    # prices = np.random.normal(mu, sigma, 391 * day_count)
    prices = np.cumsum(np.random.normal(0, 1, 391 * day_count) * sigma) + mu
    log_returns = (np.log(prices / utils.lag(prices, np.NaN)))
    log_returns[0] = 0.
    return np.cumsum(log_returns)


def get_close_prices(data):
    close_prices = []
    for date in sorted(data.keys()):
        close_prices.append(data[date]['close'][-1])
    return np.array(close_prices)


if __name__ == '__main__':
    main()
