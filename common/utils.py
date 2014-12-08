from datetime import datetime
from datetime import timedelta

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


def calculate_returns(pnl):
    lagged_pnl = lag(pnl)
    returns = (pnl - lagged_pnl) / lagged_pnl

    # All values prior to our position opening in pnl will have a
    # value of inf. This is due to division by 0.0
    returns[np.isinf(returns)] = 0.
    # Additionally, any values of 0 / 0 will produce NaN
    returns[np.isnan(returns)] = 0.
    return returns


def calculate_log_returns(pnl):
    lagged_pnl = lag(pnl)
    returns = np.log(pnl / lagged_pnl)

    # All values prior to our position opening in pnl will have a
    # value of inf. This is due to division by 0.0
    returns[np.isinf(returns)] = 0.
    # Additionally, any values of 0 / 0 will produce NaN
    returns[np.isnan(returns)] = 0.
    return returns