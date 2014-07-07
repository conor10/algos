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
import utils


_PLOT_SERIES = True

# We want to enable printing of full numpy arrays
np.set_printoptions(threshold=np.nan)


class Side(object):
    BUY = 1
    SELL = -1
    NONE = 0


def perform_backtests():
    #symbols = data_loader.load_symbol_list(data_sources.SP_500_2012)
    symbols = ['ACN']
    symbol_data = data_loader.load_price_data(data_sources.DATA_DIR, symbols)

    start = utils.create_date('2011-01-01')

    price_data = PriceData(symbol_data)
    close_price_data = price_data.get_price_data(symbols, 'Adj Close', start)

    sharpe_ratios = []

    for symbol in symbols:
        sharpe_ratio = run_backtest(symbol, close_price_data)
        sharpe_ratios.append((symbol, sharpe_ratio))

    sharpe_ratios_sorted = order_results_desc(sharpe_ratios)
    display_results(sharpe_ratios_sorted)


def run_backtest(symbol, close_price_data):
    pd_close = close_price_data[symbol]
    pd_close.fillna(method='ffill', inplace=True)
    pd_close.fillna(method='bfill', inplace=True)

    np_close = pd_close.as_matrix()

    # TODO: Use our optimiser here
    signals = generate_moving_average_signals(np_close)
    pnl = calculate_pnl(np_close, signals)
    returns = calculate_returns(pnl)
    sharpe_ratio = calculate_sharpe_ratio(returns, len(np_close))

    # The Chan implementation only holds positions for single days
    # i.e. For a multi-day position we must re-enter into it each day
    signals_chan = generate_moving_average_signals_chan(np_close)
    positions_chan = calculate_positions_chan(np_close, signals_chan)
    returns_chan = calculate_returns(positions_chan)
    sharpe_ratio_chan = calculate_sharpe_ratio(pnl)

    logging.debug('PnL account: [Close, PnL]\n{}'
                  .format(np.matrix([np_close, pnl]).T))
    logging.debug("Annualised Sharpe Ratio: {}".format(sharpe_ratio))

    return sharpe_ratio


def generate_moving_average_signals(close, lookback=20, entry_z_score=2,
                                    exit_z_score=0):
    """Limitations:
    1. Only a single trend can be entered into at a time
    2. Orders are treated as GTD
    """

    signals = np.zeros(close.shape, dtype=np.int)
    positions = np.empty(close.shape, dtype=np.float)
    positions[:] = np.nan

    upper, middle, lower = talib.BBANDS(close,
                                        timeperiod=lookback,
                                        nbdevup=entry_z_score,
                                        nbdevdn=entry_z_score,
                                        matype=MA_Type.SMA)

    z_score_upper = middle + exit_z_score * ((upper - middle) / entry_z_score)
    z_score_lower = middle + -exit_z_score * ((middle - lower) / entry_z_score)

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

        if not(up_trend or down_trend):
            if break_upwards(close, upper, i):
                signals[i] = Side.SELL
                positions[i] = Side.SELL
                down_trend = True

            elif break_downwards(close, lower, i):
                signals[i] = Side.BUY
                positions[i] = Side.BUY
                up_trend = True

        elif down_trend:
            if break_downwards(close, z_score_upper, i):
                signals[i] = Side.BUY
                positions[i] = Side.NONE
                down_trend = False
            else:
                positions[i] = Side.SELL

        elif up_trend:
            if break_upwards(close, z_score_lower, i):
                signals[i] = Side.SELL
                positions[i] = Side.NONE
                up_trend = False
            else:
                positions[i] = Side.BUY

    plot_series(close, upper, middle, lower, positions)
    return signals


def break_upwards(close, band, index):
    return close[index-1] <= band[index-1] and close[index] > band[index]


def break_downwards(close, band, index):
    return close[index-1] >= band[index-1] and close[index] < band[index]


def generate_moving_average_signals_chan(close, lookback=20, entry_z_score=2,
                                         exit_z_score=0):

    upper, middle, lower = talib.BBANDS(close,
                                        timeperiod=lookback,
                                        nbdevup=entry_z_score,
                                        nbdevdn=entry_z_score,
                                        matype=MA_Type.SMA)

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
    if not _PLOT_SERIES:
        return

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
    exit[signals != Side.NONE] = np.nan
    plt.plot(buy, 'ro', label='Buy', color='orange')
    plt.plot(sell, 'ro', label='Sell', color='purple')
    plt.plot(exit, 'ro', label='Exit', color='black')

    plt.show()


def calculate_pnl(close, signals):
    return np.cumsum(close * signals)


def calculate_positions_chan(close, signals):
    return close * signals


def calculate_returns(pnl):
    lagged_pnl = lag(pnl)
    returns = (pnl - lagged_pnl) / lagged_pnl

    # All values prior to our position opening in pnl will have a value of inf
    # this is due to division by 0.0
    returns[np.isinf(returns)] = 0.
    # Additionally, any values of 0 / 0 will produce NaN
    returns[np.isnan(returns)] = 0.
    return returns


def calculate_sharpe_ratio(returns, duration=252):
    return math.sqrt(duration) * np.mean(returns) / np.std(returns)


def order_results_desc(result_set):
    return sorted(result_set, reverse=True, key=lambda tup: tup[1])


def display_results(results):
    for (sym, value) in results:
        logging.info("{}: {}".format(sym, value))


if __name__ == '__main__':
    init_logger.setup()
    perform_backtests()
