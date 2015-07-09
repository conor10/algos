import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from tabulate import tabulate

import data_loader as dl
import utils

FUTURES_DATA_DIR = '/Users/Conor/marketdata/CFE_VX'


def main():
    futures_prices = load_futures_prices()

    n_futures = [30., 60., 90., 120., 150., 180.]
    prices = {}

    # start = '21/10/2004'
    # Going earlier then this date results in missing data, as we don't
    # have all required prices for all periods
    start = '01/01/2009'
    # end = '30/12/2011'
    end = '30/12/2014'
    dates = pd.date_range(start, end, normalize=True)

    prices = pd.DataFrame(index=dates)

    expected_nan = 0

    for n in n_futures:
        result = load_vix_future(n, futures_prices, dates)
        if expected_nan is 0:
            expected_nan = np.count_nonzero(~np.isnan(result))
        else:
            nan_count = np.count_nonzero(~np.isnan(result))
            if expected_nan != nan_count:
                print('Different count of NaN values returned for period {}, '
                      'expected: {}, actual: {}'.format(
                    n, expected_nan, nan_count))
                utils.np_print_full(result)
                exit(1)
        prices[str(n)] = result

    prices = prices.dropna()

    returns = pd.DataFrame(index=prices.index)

    for key in prices:
        returns[key] = utils.calculate_returns(prices[key])

    prices.plot()
    plt.show()

    corr = returns.corr()
    print(corr)
    eig_val, eig_vec = np.linalg.eig(corr)
    prop_of_variation = eig_val / eig_val.sum()
    print(eig_val)
    print(eig_vec)
    print('Proportion of variation explained: {}'.format(prop_of_variation))


def load_futures_prices():
    return dl.load_vix_futures_prices(
        FUTURES_DATA_DIR, price='Settle', start_year=2005)


def load_vix_future(n, futures_prices, dates):
    # TODO: Remove non-trading days
    ratios = np.empty(len(dates))

    for i in range(0, len(ratios)):
        ratios[i] = get_vix_future_ndays(n, futures_prices, dates[i].date())

    return ratios


def get_vix_future_30(futures_prices, curr_date):

    expiry_date = get_next_expiry_date(curr_date)

    prices = get_futures_prices(futures_prices, curr_date, expiry_date, 2)
    near_date_maturity = (expiry_date - curr_date).days

    return (near_date_maturity / 30.) * prices[0] + \
           ((30. - near_date_maturity) / 30.) * prices[1]


def get_vix_future_ndays(n, futures_prices, curr_date):
    """
    Weighted future expiring n days from now.
    """
    # We use n-1 as we roll to the next month if our target date falls on an
    # expiry date, we want to that months contract, not the next
    target_date = curr_date + dt.timedelta(days=n-1)

    prev_expiry_date = get_next_expiry_date(curr_date)
    next_expiry_date = get_next_expiry_date(
        prev_expiry_date + dt.timedelta(days=1))

    while next_expiry_date < target_date:
        prev_expiry_date = next_expiry_date
        next_expiry_date = get_next_expiry_date(
            prev_expiry_date + dt.timedelta(days=1))

    prices = get_futures_prices(
        futures_prices, curr_date, prev_expiry_date, 2)
    near_date_maturity = (prev_expiry_date - curr_date).days

    # print('vxN {}, {}, {}: {}'.format(curr_date, prev_expiry_date,
    #                                    near_date_maturity, prices))

    if prices[0] is np.NaN or prices[1] is np.NaN:
        return np.NaN
    else:
        return (near_date_maturity / n) * prices[0] + \
               ((n - near_date_maturity) / n) * prices[1]


def get_futures_prices(futures_prices, curr_date, expiry_date, month_count):

    def get_price_value(prices, curr_date):
        if curr_date in prices:
            return prices[curr_date]
        else:
            return np.NaN

    year = expiry_date.year
    month = expiry_date.month

    prices = []

    prices.append(
        get_price_value(futures_prices[year][month - 1], curr_date))

    remaining = month_count - 1
    next_month = month
    next_year = year
    while remaining > 0:
        if next_month == 12:
            next_month = 1
            next_year += 1
        else:
            next_month += 1

        prices.append(
            get_price_value(
                futures_prices[next_year][next_month - 1], curr_date))

        remaining -= 1

    return prices


def get_next_expiry_date(curr_date):
    expiry_date = get_expiry_date_for_month(curr_date)

    # It must be less then the expiry date, as on expiry date we only have
    # a settlement price
    if curr_date < expiry_date:
        return expiry_date
    else:
        return get_expiry_date_for_month(curr_date + dt.timedelta(days=30))


def get_expiry_date_for_month(curr_date):
    """
    http://cfe.cboe.com/products/spec_vix.aspx

    TERMINATION OF TRADING:

    Trading hours for expiring VIX futures contracts end at 7:00 a.m. Chicago
    time on the final settlement date.

    FINAL SETTLEMENT DATE:

    The Wednesday that is thirty days prior to the third Friday of the
    calendar month immediately following the month in which the contract
    expires ("Final Settlement Date"). If the third Friday of the month
    subsequent to expiration of the applicable VIX futures contract is a
    CBOE holiday, the Final Settlement Date for the contract shall be thirty
    days prior to the CBOE business day immediately preceding that Friday.
    """
    # Date of third friday of the following month
    if curr_date.month == 12:
        third_friday_next_month = dt.date(curr_date.year + 1, 1, 15)
    else:
        third_friday_next_month = dt.date(curr_date.year,
                                          curr_date.month + 1, 15)

    one_day = dt.timedelta(days=1)
    thirty_days = dt.timedelta(days=30)
    while third_friday_next_month.weekday() != 4:
        # Using += results in a timedelta object
        third_friday_next_month = third_friday_next_month + one_day

    # TODO: Incorporate check that it's a trading day, if so move the 3rd
    # Friday back by one day before subtracting
    return third_friday_next_month - thirty_days


if __name__ == '__main__':
    main()
