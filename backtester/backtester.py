import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import talib
from talib import MA_Type

import data_loader
import data_sources
import init_logger
from price_data import PriceData


class Side(object):
    BUY = 1
    SELL = -1
    NONE = 0

# We want to enable printing of full numpy arrays
np.set_printoptions(threshold=np.nan)


def perform_backtests():
    #symbols = data_loader.load_symbol_list(data_sources.SP_500_2012)
    symbols = ['BLK']
    symbol_data = data_loader.load_price_data(data_sources.DATA_DIR, symbols)

    price_data = PriceData(symbol_data)
    close_price_data = price_data.get_price_data(symbols, 'Adj Close')

    sharpe_ratios = []

    for symbol in symbols:
        sharpe_ratio = run_backtest(symbol, close_price_data)
        sharpe_ratios.append((symbol, sharpe_ratio))

    sharpe_ratios_sorted = order_results_desc(sharpe_ratios)
    display_results(sharpe_ratios_sorted)


def run_backtest(symbol, close_price_data):
    # TODO: What should be done with NaN values? Ignore of ffill/bfill? ffill
    np_close = close_price_data[symbol].fillna(0).as_matrix()

    # signals = generate_moving_average_signals(np_close)
    # pnl = calculate_pnl(np_close, signals)
    # returns = calculate_returns(pnl)
    # sharpe_ratio = caclulate_sharpe_ratio(pnl)

    signals = generate_moving_average_signals_chan(np_close)
    pnl = calculate_pnl_chan(np_close, signals)
    sharpe_ratio = caclulate_sharpe_ratio(pnl)

    logging.debug('PnL account: [Close, PnL]\n{}'
                  .format(np.matrix([np_close, pnl]).T))
    logging.debug("Annualised Sharpe Ratio: {}".format(sharpe_ratio))

    return sharpe_ratio


def generate_moving_average_signals(close):

    lookback = 5

    signals = np.zeros(close.shape, dtype=np.float)

    upper, middle, lower = talib.BBANDS(close, timeperiod=lookback,
                                        matype=MA_Type.T3)

    """
    Upper and lower bands are price targets
    If we cross a band, we buy/sell and hold this position until we break
    through the middle band

    Price touches the lower band => Buy
    Sell when we touch the middle band

    Price touches the upper band => Sell
    Buy when we touch the middle band
    """
    up_trend = False
    down_trend = False

    for i in range(1, len(close)):

        if not(up_trend and down_trend):
            if break_upwards(close, upper, i):
                signals[i] = Side.SELL
                down_trend = True

            elif break_downwards(close, lower, i):
                signals[i] = Side.BUY
                up_trend = True

        elif down_trend:
            if break_downwards(close, middle, i):
                signals[i] = Side.BUY
                down_trend = False

        elif up_trend:
            if break_upwards(close, middle, i):
                signals[i] = Side.SELL
                up_trend = False

    #plot_series(close, upper, middle, lower, signals)
    return signals


def break_upwards(close, band, index):
    return close[index-1] <= band[index-1] and close[index] > band[index]


def break_downwards(close, band, index):
    return close[index-1] >= band[index-1] and close[index] < band[index]


def generate_moving_average_signals_chan(close):

    lookback = 5
    entry_z_score = 2
    exit_z_score = 0

    upper, middle, lower = talib.BBANDS(close,
                                        timeperiod=lookback,
                                        nbdevup=entry_z_score,
                                        nbdevdn=entry_z_score,
                                        matype=MA_Type.T3)

    moving_std = (upper - middle) / entry_z_score
    z_score = (close - middle) / moving_std

    long_entry = z_score < -entry_z_score
    long_exit = z_score >= -exit_z_score

    short_entry = z_score > entry_z_score
    short_exit = z_score <= exit_z_score

    long_signals = np.empty(close.shape, dtype=np.float)
    long_signals[:] = np.nan
    long_signals[0] = 0.
    short_signals = np.empty(close.shape, dtype=np.float)
    short_signals[:] = np.nan
    short_signals[0] = 0.

    long_signals[long_entry] = Side.BUY
    long_signals[long_exit] = Side.NONE
    long_signals = ffill(long_signals)

    short_signals[short_entry] = Side.SELL
    short_signals[short_exit] = Side.NONE
    short_signals = ffill(short_signals)

    positions = long_signals + short_signals
    plot_series(close, upper, middle, lower, positions)
    return positions


def ffill(data):
    for i in range(1, len(data)):
        if np.isnan(data[i]):
            data[i] = data[i-1]
    return data


def lag(data):
    lag = np.roll(data, 1)
    lag[0] = 0.
    return lag


def plot_series(close, upper, middle, lower, signals):
    plt.clf()
    plt.plot(close, label='Close', color='black')
    plt.plot(upper, label='Upper', color='red')
    plt.plot(middle, label='Middle', color='blue')
    plt.plot(lower, label='Lower', color='green')

    buy = np.copy(close)
    buy[signals != Side.BUY] = np.nan
    sell = np.copy(close)
    sell[signals != Side.SELL] = np.nan
    exit = np.copy(close)
    exit[get_exit_signals(signals) != Side.NONE] = np.nan
    plt.plot(buy, 'ro', label='Buy', color='orange')
    plt.plot(sell, 'ro', label='Sell', color='purple')
    plt.plot(exit, 'ro', label='Exit', color='black')

    plt.show()


def get_exit_signals(signals):
    for i in range(1, len(signals)):
        # We use 2 to differentiate between 0, -1, 1 values in signals
        if (signals[i-1] == Side.BUY or signals[i-1] == Side.SELL) \
                and signals[i] == Side.NONE:
            signals[i] = 2

    signals[signals != 2] = np.nan
    signals[signals == 2] = Side.NONE
    return signals


def calculate_pnl(close, signals):
    return np.cumsum(close * signals)


def calculate_pnl_chan(close, signals):
    positions = close * signals
    lag_close = lag(close)

    lag_positions = lag(positions)

    daily_pnl = lag_positions * ((close - lag_close) / lag_close)
    daily_pnl[np.isnan(daily_pnl)] = 0.
    return daily_pnl


def calculate_returns(pnl):
    lagged_pnl = lag(pnl)
    returns = (pnl - lagged_pnl) / lagged_pnl

    # All values prior to our position opening in pnl will have a value of inf
    # this is due to division by 0.0
    returns[np.isinf(returns)] = 0.
    # Additionally, any values of 0 / 0 will produce NaN
    returns[np.isnan(returns)] = 0.
    return returns


def caclulate_sharpe_ratio(returns, duration=252):
    return math.sqrt(duration) * np.mean(returns) / np.std(returns)


def order_results_desc(result_set):
    return sorted(result_set, reverse=True, key=lambda tup: tup[1])


def display_results(results):
    for (sym, value) in results:
        logging.info("{}: {}".format(sym, value))


if __name__ == '__main__':
    init_logger.setup()
    perform_backtests()
