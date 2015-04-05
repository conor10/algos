import datetime as dt
import os

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
import seaborn
from tabulate import tabulate

import data_loader as dl
import utils


DATA_DIR = '/Users/Conor/marketdata/VIX'
FUTURES_DATA_DIR = '/Users/Conor/marketdata/CFE_VX'
SYMBOL_FILE = os.path.join(DATA_DIR, 'symbols.txt')

PRICE = 'Adj Close'

DEBUG = False
PLOT_CHARTS = False


"""
Mojito v1.0 Strategy implementation

http://godotfinance.com/pdf/DynamicVIXFuturesImproved_Rev1.pdf
"""
def main():
    symbols = ['^VIX', '^VXV', 'VXX', 'VXZ']

    futures_prices = load_futures_prices()

    # plot(prices)
    # Cash needs to remain above 100k as adjusted VXX start price is 6914.91,
    # whereas VXZ is around 106.73
    start_cash = 100000.0

    # Mojito paper
    # start_date = '13/8/2010'
    # end_date = '29/6/2012'

    # Mojito 2 paper
    # start_date = '24/2/2009'
    # end_date = '14/5/2013'

    # Mojito 3 paper
    # start_date = '3/1/2011'
    # end_date = '16/12/2013'
    # start_date = '3/1/2009'
    # end_date = '16/12/2013'

    # VT Trading 2ed came out on April 1st 2013
    start_date = '1/2/2013'
    end_date = '1/1/2015'

    prices = load_prices(symbols, start_date, end_date)
    ivts = prices['^VIX'] / prices['^VXV']
    vix_future_30 = load_vix_future_30(futures_prices, prices.index)
    # vix_future_45 = load_vix_future_30(futures_prices, prices.index)

    vr_0_30_ts = prices['^VIX'] / vix_future_30
    # vr_0_45_ts = prices['^VIX'] / vix_future_45

    strategies = [
        ('Mojito', get_mojito_ratio, ivts),
        ('Mojito Aggressive', get_mojito_aggressive_ratio, ivts),
        ('Dynamic VIX', get_dynamic_vix_ratio, ivts),
        ('Fixed VIX', get_fixed_vix_ratio, ivts),
        ('Mojito 2.0 Medium', get_mojito_2_0_medium_ratio, ivts),
        ('Mojito 2.0 Aggressive', get_mojito_2_0_aggressive_ratio, ivts),
        ('Mojito 2.0 Medium 30TS', get_mojito_2_0_medium_30ts_ratio,
        vr_0_30_ts),
        ('Mojito 2.0 Aggressive 30TS', get_mojito_2_0_aggressive_30ts_ratio,
        vr_0_30_ts),
        ('Mojito 3.0 VIX/VXV', get_mojito_3_0_vix_vxv_ratio, ivts, True)
    ]

    results = []

    for strategy in strategies:
        name = strategy[0]
        ratio_func = strategy[1]
        ts = strategy[2]
        plot_chart = strategy[3] if len(strategy) == 4 else False
        # date_range = strategy[2]

        retuns, cash = run(prices, ts, ratio_func, start_cash)

        real_returns, final_return, sharpe_ratio, sortino_ratio, \
        max_dd, max_duration, max_dd_idx, max_duration_idx, hwm_idx = \
            calc_performance(retuns, start_cash)

        if PLOT_CHARTS or plot_chart:
            print_results(final_return, sharpe_ratio, sortino_ratio,
                          max_dd, max_duration)
            plot_results(name, real_returns, cash, max_dd, max_duration,
                         max_dd_idx, max_duration_idx, hwm_idx)

        results.append([name, final_return, sharpe_ratio, sortino_ratio,
                        max_dd * 100., max_duration])

    print tabulate(
        results,
        floatfmt=".2f",
        headers=['Strategy', 'Returns', 'Sharpe Ratio',
                 'Sortino Ratio', 'Max DD', 'Max DD Duration'])


def run(prices, ts, ratio_func, start_cash):

    count = len(ts)
    orders = np.zeros((count, 2), dtype=np.int)
    positions = np.zeros((count, 2), dtype=np.int)
    cash = np.zeros(count)
    cash[0] = start_cash
    cash_delta = np.zeros(count)
    vxx_ratio = np.zeros(count)
    vxz_ratio = np.zeros(count)

    for i in range(1, count):
        ratio = ratio_func(ts[i-1])
        vxx_ratio[i] = ratio[0]
        vxz_ratio[i] = ratio[1]

        orders[i] = calc_adjustments(
            positions[i-1],
            (prices['VXX'][i-1], prices['VXZ'][i-1]),
            ratio,
            cash[i-1])

        positions[i] = positions[i-1] + orders[i]
        cash_delta[i] = calc_cash_delta(
            orders[i],
            (prices['VXX'][i-1], prices['VXZ'][i-1]),
            positions[i-1])

        cash[i] = cash[i-1] + cash_delta[i]

    if DEBUG:
        print('vxx ratio, vxz ratio, vxx px, vxz px, vxx orders, vxz orders, '
              'vxx positions, vxz positions, cash, cash delta')
        for i in range(0, count):
            print('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
                vxx_ratio[i], vxz_ratio[i],
                prices['VXX'][i], prices['VXZ'][i],
                orders[i][0], orders[i][1],
                positions[i-1][0], positions[i-1][1],
                cash[i], cash_delta[i]))

    vxx_change = (prices['VXX'] - utils.lag(prices['VXX'])) * positions[:, 0]
    vxz_change = (prices['VXZ'] - utils.lag(prices['VXZ'])) * positions[:, 1]

    notional_value = (np.absolute(positions) * prices[['VXX', 'VXZ']]).sum(1) \
                     + cash
    returns = (vxx_change + vxz_change) / notional_value

    return returns, cash


def calc_performance(returns, start_cash):
    log_returns = (np.log(1.0 + returns)).cumsum()
    real_returns = start_cash * np.exp(log_returns)
    # normal_returns = start_cash * (1.0 + returns).cumprod()

    # final_return = (np.exp(log_returns[-1]) * 100.)
    final_return = (np.exp(log_returns[-1]) - 1.0) * 100.
    sharpe_ratio = utils.calculate_sharpe_ratio(returns)
    sortino_ratio = utils.calculate_sortino_ratio(returns.values)

    max_dd, max_duration, max_dd_idx, max_duration_idx, hwm_idx = \
        utils.calculate_max_drawdown_log(log_returns)

    return real_returns, final_return, sharpe_ratio, sortino_ratio, \
           max_dd, max_duration, max_dd_idx, max_duration_idx, hwm_idx


def print_results(final_return, sharpe_ratio, sortino_ratio, max_dd, max_duration):
    print('Sharpe Ratio: {0:.2f}'.format(sharpe_ratio))
    print('Sortino Ratio: {0:.2f}'.format(sortino_ratio))
    print('Returns: {:.2f}%'.format(final_return))
    print('Max drawdown: {:.2f}%, max duration: {} days'
          .format(max_dd * 100.0, max_duration))


def plot_results(name, real_returns, cash, max_dd,
                 max_duration, max_dd_idx, max_duration_idx, hwm_idx):

    plt.plot(real_returns, label='log_returns')
    # plt.ylim(ymin = 0.)
    # plt.plot(normal_returns, label='normal_returns')
    # real_returns.plot()

    plt.plot((hwm_idx, max_dd_idx),
             (real_returns[hwm_idx], real_returns[max_dd_idx]), color='black')
    plt.annotate('max dd ({0:.2f}%)'.format(max_dd * 100.0),
                 xy=(max_dd_idx, real_returns[max_dd_idx]),
                 xycoords='data', xytext=(0, -50),
                 textcoords='offset points',
                 arrowprops=dict(facecolor='black', shrink=0.05))

    max_duration_start_idx = max_duration_idx - max_duration
    max_duration_x1x2 = (max_duration_start_idx, max_duration_idx)
    max_duration_y1y2 = (real_returns[max_duration_start_idx],
                         real_returns[max_duration_start_idx])

    plt.plot(max_duration_x1x2, max_duration_y1y2, color='black')
    plt.annotate('max dd duration ({} days)'.format(max_duration),
                 xy=((max_duration_start_idx + max_duration_idx) / 2,
                     real_returns[max_duration_start_idx]),
                 xycoords='data',
                 xytext=(-100, 30), textcoords='offset points',
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.plot(cash, label='cash')

    plt.title(name)
    plt.legend()

    # Format x-axis with dates
    # dates = real_returns.index.to_pydatetime()
    dates = real_returns.index.map(lambda t: t.strftime('%Y-%m-%d'))
    plt.gca().xaxis.set_major_formatter(ticker.IndexFormatter(dates))
    plt.gcf().autofmt_xdate()

    plt.show()

    """
    TODO:
    Fix costs - they shouldn't decrease over the life of the strategy.
    1. Add transaction costs.
    2. Figure out why results differ slightly from papers - returns appear
       to jump after 250 days (~10/8/2011).
       This is best illustrated by plotting the fixed ratio results
       Calculating orders using price[i-1] instead of price[i] takes the
       profitability down closer in line with the paper's results.
    3. Plot target ratios of stocks.
    4. Figure out why Mojito 2.0 returns differ so much from paper.

    """


def load_prices(symbols, start_date, end_date):
    prices = dl.load_price_data(DATA_DIR, symbols)
    df = None

    for key, value in prices.iteritems():
        dataset = value[start_date:end_date]
        dataset.rename(columns={PRICE: key}, inplace=True)

        if df is None:
            df = pd.DataFrame(dataset[key])
        else:
            df = df.join(dataset[key])
    return df


def load_futures_prices(start_year=None, end_year=None):
    # TODO: Figure out if year bounds will work
    return dl.load_vix_futures_prices(FUTURES_DATA_DIR)


def load_vix_future_30(futures_prices, dates):
    ratios = np.empty(len(dates))

    for i in range(0, len(ratios)):
        ratios[i] = get_vix_future_30(futures_prices, dates[i].date())
    return ratios


def get_vix_future_30(futures_prices, curr_date):

    expiry_date = get_next_expiry_date(curr_date)

    prices = get_futures_prices(futures_prices, curr_date, expiry_date, 2)
    near_date_maturity = (expiry_date - curr_date).days

    return (near_date_maturity / 30.) * prices[0] + \
        ((30. - near_date_maturity) / 30.) * prices[1]


def get_vix_future_45(futures_prices, curr_date):
    # TODO: Implement
    expiry_date = get_next_expiry_date(curr_date)

    prices = get_futures_prices(futures_prices, curr_date, expiry_date, 3)
    near_date_maturity = (expiry_date - curr_date).days

    return (near_date_maturity / 30.) * prices[0] + \
           ((30. - near_date_maturity) / 30.) * prices[1]


def get_futures_prices(futures_prices, curr_date, expiry_date, month_count):
    year = expiry_date.year
    month = expiry_date.month

    prices = []
    prices.append(futures_prices[year][month - 1][curr_date])

    remaining = month_count - 1
    next_month = month
    next_year = year
    while remaining > 0:
        if next_month == 12:
            next_month = 1
            next_year += 1
        else:
            next_month += 1

        prices.append(futures_prices[next_year][next_month - 1][curr_date])
        remaining -= 1

    return prices


def get_next_expiry_date(curr_date):
    expiry_date = get_expiry_date_for_month(curr_date)

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


"""
| IVTS     | VXX Weight | VXZ Weight |
| <= 0.91  | -0.60      | 0.40       |
| <= 0.94  | -0.32      | 0.68       |
| <= 0.97  | -0.32      | 0.68       |
| <= 1.005 | -0.25      | 0.75       |
| > 1.005  | -0.10      | 0.90       |

:returns VXX Weight, VXZ Weight
"""
def get_mojito_ratio(ivts):
    if ivts <= 0.91:
        return np.array((-0.60, 0.40))
    elif ivts <= 0.94:
        return np.array((-0.32, 0.68))
    elif ivts <= 0.97:
        return np.array((-0.32, 0.68))
    elif ivts <= 1.005:
        return np.array((-0.25, 0.75))
    else:
        return np.array((-0.10, 0.90))


"""
| IVTS     | VXX Weight | VXZ Weight |
| <= 0.91  | -0.70      | 0.30       |
| <= 0.94  | -0.32      | 0.68       |
| <= 0.97  | -0.32      | 0.68       |
| <= 1.005 | -0.28      | 0.72       |
| > 1.005  | -0.00      | 1.00       |

:returns VXX Weight, VXZ Weight
"""
def get_mojito_aggressive_ratio(ivts):
    if ivts <= 0.91:
        return np.array((-0.70, 0.30))
    elif ivts <= 0.94:
        return np.array((-0.32, 0.68))
    elif ivts <= 0.97:
        return np.array((-0.32, 0.68))
    elif ivts <= 1.005:
        return np.array((-0.28, 0.72))
    else:
        return np.array((0.00, 1.00))


"""
Mojito medium aggressive from page 8 of
http://godotfinance.com/pdf/DynamicVIXFuturesVersion2.pdf

| IVTS     | VXX Weight | VXZ Weight |
| <= 0.92  | -0.70      | 0.30       |
| <= 0.94  | -0.46      | 0.54       |
| <= 1.005 | -0.36      | 0.64       |
| > 1.005  |  0.50      | 0.50       |

:returns VXX Weight, VXZ Weight
"""
def get_mojito_2_0_aggressive_ratio(ivts):
    if ivts <= 0.92:
        return np.array((-0.70, 0.30))
    elif ivts <= 0.94:
        return np.array((-0.46, 0.54))
    elif ivts <= 1.005:
        return np.array((-0.36, 0.64))
    else:
        return np.array((0.50, 0.50))


"""
Mojito medium from page 5 of
http://godotfinance.com/pdf/DynamicVIXFuturesVersion2.pdf

| IVTS     | VXX Weight | VXZ Weight |
| <= 0.92  | -0.60      | 0.40       |
| <= 0.94  | -0.46      | 0.54       |
| <= 1.005 | -0.36      | 0.64       |
| > 1.005  |  0.50      | 0.50       |

:returns VXX Weight, VXZ Weight
"""
def get_mojito_2_0_medium_ratio(ivts):
    if ivts <= 0.92:
        return np.array((-0.60, 0.40))
    elif ivts <= 0.94:
        return np.array((-0.46, 0.54))
    elif ivts <= 1.005:
        return np.array((-0.36, 0.64))
    else:
        return np.array((0.50, 0.50))

"""
Mojito medium VR_0_30TS from page 5 of
http://godotfinance.com/pdf/DynamicVIXFuturesVersion2.pdf

| IVTS     | VXX Weight | VXZ Weight |
| <= 0.92  | -0.60      | 0.40       |
| <= 0.94  | -0.46      | 0.54       |
| <= 1.00  | -0.36      | 0.64       |
| > 1.00   |  0.50      | 0.50       |

:returns VXX Weight, VXZ Weight
"""
def get_mojito_2_0_medium_30ts_ratio(ts):
    if ts <= 0.92:
        return np.array((-0.60, 0.40))
    elif ts <= 0.94:
        return np.array((-0.46, 0.54))
    elif ts <= 1.:
        return np.array((-0.36, 0.64))
    else:
        return np.array((0.50, 0.50))


"""
Mojito 2.0 aggressive VR_0_30TS from page 8 of
http://godotfinance.com/pdf/DynamicVIXFuturesVersion2.pdf

| IVTS     | VXX Weight | VXZ Weight |
| <= 0.92  | -0.70      | 0.30       |
| <= 0.94  | -0.46      | 0.54       |
| <= 1.00  | -0.36      | 0.64       |
| > 1.00   |  0.50      | 0.50       |

:returns VXX Weight, VXZ Weight
"""
def get_mojito_2_0_aggressive_30ts_ratio(ts):
    if ts <= 0.92:
        return np.array((-0.70, 0.30))
    elif ts <= 0.94:
        return np.array((-0.46, 0.54))
    elif ts <= 1.:
        return np.array((-0.36, 0.64))
    else:
        return np.array((0.50, 0.50))


def get_mojito_3_0_vix_vxv_ratio(ts):
    if ts <= 0.92:
        return np.array((-0.60, 0.40))
    elif ts <= 0.94:
        return np.array((-0.16, 0.84))
    elif ts <= 1.02:
        return np.array((-0.03, 0.97))
    else:
        return np.array((0.79, 0.21))


"""
| IVTS     | VXX Weight | VXZ Weight |
| < 0.90   | -0.30      | 0.70       |
| <= 1.00  | -0.20      | 0.80       |
| <= 1.05  | 0.00       | 1.00       |
| <= 1.15  | 0.25       | 0.75       |
| > 1.15   | 0.50       | 0.50       |

:returns VXX Weight, VXZ Weight
"""
def get_dynamic_vix_ratio(ivts):
    if ivts < 0.90:
        return np.array((-0.3, 0.7))
    elif ivts <= 1.0:
        return np.array((-0.2, 0.8))
    elif ivts <= 1.05:
        return np.array((0.0, 1.0))
    elif ivts <= 1.15:
        return np.array((0.25, 0.75))
    else:
        return np.array((0.5, 0.5))


def get_fixed_vix_ratio(ivts):
    return np.array((-0.32, 0.68))


def calc_adjustments(existing_qty, curr_prices, target_alloc, cash):
    curr_value = existing_qty * curr_prices
    # Portfolio is balanced using absolute weightings
    target_orders = target_alloc * (np.sum(np.absolute(curr_value))
                                    + max(0, cash)) / curr_prices

    # We round our target value down to ensure we can actually afford the
    # position
    adjustments = target_orders.astype(int) - existing_qty

    # TODO: We may need to refine this model to cater for when
    # a position goes from long to short or vice-versa, such that
    # we only adjust down to zero initially, & use whatever cash
    # we have to enter the new position partially. Then the following
    # day we perform the remainder of the adjustment. This will
    # depend on if our broker allows us to jump positions in this
    # manner
    return adjustments


def calc_cash_delta(orders, prices, positions):
    costs = 0.0
    for i in range(0, len(orders)):
        if positions[i] > 0: # long positions
            if (positions[i] + orders[i]) >= 0:
                # Overall direction doesn't change
                costs += orders[i] * prices[i]
            else:
                # Position is going from long to short in single trade
                costs += positions[i] * prices[i] * -1.
                costs += (positions[i] + orders[i]) * prices[i] * -1.
        elif positions[i] < 0: # short positions
            if (positions[i] + orders[i]) <= 0:
                # When we cover our shorts we make more funds available
                costs += (orders[i] * prices[i] * -1.)
            else:
                # Position is going from short to long in single trade
                costs += positions[i] * prices[i]
                costs += (positions[i] + orders[i]) * prices[i]
        else:
            costs += abs(orders[i] * prices[i])
    return -costs


def plot(prices):
    for key, value in prices.iteritems():
        plt.plot(value, label=key)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()