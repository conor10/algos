from abc import abstractmethod
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import talib
from talib import MA_Type

import utils



# We want to enable printing of full numpy arrays
np.set_printoptions(threshold=np.nan)


_PLOT_SERIES = False


class Side(object):
    BUY = 1
    SELL = -1
    NONE = 0


def _calculate_sharpe_ratio(returns, duration=252):
    return math.sqrt(duration) * np.mean(returns) / np.std(returns)


def _plot_series(close, upper, middle, lower, signals):
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


def _print_results(np_close, positions, sharpe_ratio):
    logging.debug('Positions account: [Close, PnL]\n{}'
                 .format(np.matrix([np_close, positions]).T))
    logging.debug("Annualised Sharpe Ratio: {}".format(sharpe_ratio))


class SharpeBacktest(object):

    def __init__(self):
        pass

    def run(self, np_close):
        self._validate_params()
        signals = self._run_strategy(np_close)
        positions = self._calculate_positions(np_close, signals)
        returns = utils.calculate_returns(positions)
        sharpe_ratio = _calculate_sharpe_ratio(returns)
        _print_results(np_close, positions, sharpe_ratio)
        return sharpe_ratio

    @abstractmethod
    def _validate_params(self):
        pass

    @abstractmethod
    def _run_strategy(self, close):
        pass

    @abstractmethod
    def _calculate_positions(self, close, signals):
        pass


class ParameterException(Exception):
    """ Should any parameters passed to our constructors be invalid,
    we raise this
    """
    pass


class MovingAverageBacktest(SharpeBacktest):

    def __init__(self, lookback=20, entry_z_score=2,
                 exit_z_score=0):
        super(MovingAverageBacktest, self).__init__()

        self.lookback = lookback
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score

    def _validate_params(self):
        if not self._z_scores_valid(self.entry_z_score, self.exit_z_score):
            raise ParameterException(
                'Invalid z scores: entry_z_score={}, exit_z_score={}'
                .format(self.entry_z_score, self.exit_z_score))

    def _run_strategy(self, close):
        """Limitations:
        1. Only a single trend can be entered into at a time
        2. Orders are treated as GTD
        3. We don't support multiple moving averages for crossovers
        """

        signals = np.zeros(close.shape, dtype=np.int)
        positions = np.empty(close.shape, dtype=np.float)
        positions[:] = np.nan

        upper, middle, lower = talib.BBANDS(close,
                                            timeperiod=self.lookback,
                                            nbdevup=self.entry_z_score,
                                            nbdevdn=self.entry_z_score,
                                            matype=MA_Type.SMA)

        z_score_upper = middle + self.exit_z_score * \
                                  ((upper - middle) / self.entry_z_score)
        z_score_lower = middle + -self.exit_z_score * \
                                 ((middle - lower) / self.entry_z_score)

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
                if self._break_upwards(close, upper, i):
                    signals[i] = Side.SELL
                    positions[i] = Side.SELL
                    down_trend = True

                elif self._break_downwards(close, lower, i):
                    signals[i] = Side.BUY
                    positions[i] = Side.BUY
                    up_trend = True

            elif down_trend:
                if self._break_downwards(close, z_score_upper, i):
                    signals[i] = Side.BUY
                    positions[i] = Side.NONE
                    down_trend = False
                else:
                    positions[i] = Side.SELL

            elif up_trend:
                if self._break_upwards(close, z_score_lower, i):
                    signals[i] = Side.SELL
                    positions[i] = Side.NONE
                    up_trend = False
                else:
                    positions[i] = Side.BUY

        _plot_series(close, upper, middle, lower, positions)
        return signals

    @staticmethod
    def _z_scores_valid(entry_z_score, exit_z_score):
        return entry_z_score > exit_z_score

    @staticmethod
    def _break_upwards(close, band, index):
        return close[index-1] <= band[index-1] and close[index] > band[index]

    @staticmethod
    def _break_downwards(close, band, index):
        return close[index-1] >= band[index-1] and close[index] < band[index]

    @staticmethod
    def _calculate_positions(close, signals):
        return np.cumsum(close * signals)


class ChanBacktest(SharpeBacktest):
    """
    The Chan implementation only holds positions for single days
    i.e. For a multi-day position we must re-enter into it each day
    """
    def _run_strategy(self, close, lookback=20,
                                             entry_z_score=2, exit_z_score=0):

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
        long_signals = utils.ffill(long_signals)

        short_signals[short_entry] = Side.SELL
        short_signals[short_exit] = Side.NONE
        short_signals = utils.ffill(short_signals)

        positions = long_signals + short_signals
        _plot_series(close, upper, middle, lower, positions)
        return positions

    def _validate_params(self):
        pass

    @staticmethod
    def _calculate_positions(close, signals):
        return close * signals
