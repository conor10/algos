from datetime import datetime
from datetime import timedelta
import math

import numpy as np


DATE_FORMAT = '%Y-%m-%d'


def create_date(date):
    return datetime.strptime(date, DATE_FORMAT)


def day_count(start, end):
    delta = end - start
    return delta.days


def work_day_count(start, end):
    # daydiff = end.weekday() - start.weekday()
    # return ((end - start).days - daydiff) / 7 * 5 + min(daydiff, 5)
    return np.busday_count(start.date(), end.date())


def work_day_delta(base_date, day_count):
    one_day = timedelta(days=np.sign(day_count))
    result = base_date

    for i in range(0, abs(day_count)):
        result += one_day
        while result.weekday() > 4:
            result += one_day
    return result


def get_max_vector(array, index=0):
    return array[np.nanargmax(array[:, index])]


def ffill(data):
    for i in range(1, len(data)):
        if np.isnan(data[i]):
            data[i] = data[i-1]
    return data


def lag(data, empty_term=0.):
    lagged = np.roll(data, 1)
    lagged[0] = empty_term
    return lagged


def calculate_returns(prices):
    lagged_pnl = lag(prices)
    returns = (prices - lagged_pnl) / lagged_pnl

    # All values prior to our position opening in pnl will have a
    # value of inf. This is due to division by 0.0
    returns[np.isinf(returns)] = 0.
    # Additionally, any values of 0 / 0 will produce NaN
    returns[np.isnan(returns)] = 0.
    return returns


def calculate_short_returns(prices):
    return calculate_returns(prices) * -1.


def calculate_log_returns(prices):
    lagged_pnl = lag(prices)
    returns = np.log(prices / lagged_pnl)

    # All values prior to our position opening in pnl will have a
    # value of inf. This is due to division by 0.0
    returns[np.isinf(returns)] = 0.
    # Additionally, any values of 0 / 0 will produce NaN
    returns[np.isnan(returns)] = 0.
    return returns


def calculate_short_log_returns(prices):
    return calculate_log_returns(prices) * -1.


def calculate_sharpe_ratio(returns, annulisation_factor=252.0):
    return (np.mean(returns) / np.std(returns)) * \
           math.sqrt(annulisation_factor)

def calculate_sortino_ratio_with_freq(returns, annualisation_factor=252.0):
    """
    Modified Sortino ratio that takes into account the frequency in addition
    to the magnitude of below target returns. This was as per
    http://managed-futures-blog.attaincapital.com/
    2013/09/11/sortino-ratio-are-you-calculating-it-wrong/
    """
    def f(x):
        return x if x < 0. else 0.
    f = np.vectorize(f)
    return calculate_sharpe_ratio(f(returns), annualisation_factor)


def calculate_sortino_ratio(returns, annualisation_factor=252.0):
    return (np.mean(returns) / np.std(returns[np.where(returns < 0.)])) \
           * math.sqrt(annualisation_factor)


def calculate_max_drawdown(returns):
    size = len(returns)
    highwatermark = np.zeros(size) # Max return seen
    drawdown = np.zeros(size)
    dd_duration = np.zeros(size, dtype=int)

    for i in range(1, size):
        highwatermark[i] = max(highwatermark[i-1], returns[i])
        drawdown[i] = ((1.0 + returns[i]) / (1.0 + highwatermark[i])) - 1.0
        if drawdown[i] == 0.:
            dd_duration[i] = 0
        else:
            dd_duration[i] = dd_duration[i-1] + 1

    min_dd_idx = drawdown.argmin()
    return min(drawdown), max(dd_duration), \
           min_dd_idx, dd_duration.argmax(), \
           np.where(returns == highwatermark[min_dd_idx-1])[-1]


def calculate_max_drawdown_log(returns):
    size = len(returns)
    highwatermark = np.zeros(size) # Max return seen
    drawdown = np.zeros(size)
    dd_duration = np.zeros(size, dtype=int)

    for i in range(1, size):
        highwatermark[i] = max(highwatermark[i-1], returns[i])
        drawdown[i] = returns[i] - highwatermark[i]
        if drawdown[i] == 0.:
            dd_duration[i] = 0
        else:
            dd_duration[i] = dd_duration[i-1] + 1

    min_dd_idx = drawdown.argmin()
    return min(drawdown), max(dd_duration), \
           min_dd_idx, dd_duration.argmax(), \
           np.where(returns == highwatermark[min_dd_idx-1])[-1]

