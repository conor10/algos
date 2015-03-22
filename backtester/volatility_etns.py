import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

import data_loader as dl
import utils


DATA_DIR = '/Users/Conor/marketdata/VIX'
SYMBOL_FILE = os.path.join(DATA_DIR, 'symbols.txt')

PRICE = 'Adj Close'

"""
Mojito v1.0 Strategy implementation

http://godotfinance.com/pdf/DynamicVIXFuturesImproved_Rev1.pdf
"""
def main():
    symbols = ['^VIX', '^VXV', 'VXX', 'VXZ']
    prices = load_prices(symbols)
    # plot(prices)

    ivts = prices['^VIX'] / prices['^VXV']

    count = len(ivts)
    orders = np.zeros((count, 2), dtype=np.int)
    positions = np.zeros((count, 2), dtype=np.int)
    cash = np.zeros(count)
    cash[0] = 10000.0
    cash_delta = np.zeros(count)

    # for i in range(1, length-1):
    for i in range(1, count):
        # ratio = get_mojito_ratio(ivts[i-1])
        ratio = get_mojito_aggressive_ratio(ivts[i-1])
        # ratio = get_dynamic_vix_ratio(ivts[i-1])
        # ratio = get_fixed_vix_ratio()
        # print('{}, {}'.format(ratio[0], ratio[1]))

        orders[i] = calc_adjustments(
            positions[i-1],
            (prices['VXX'][i], prices['VXZ'][i]),
            ratio,
            cash[i-1])

        positions[i] = positions[i-1] + orders[i]
        cash_delta[i] = calc_cash_delta(
            orders[i], (prices['VXX'][i], prices['VXZ'][i]), positions[i])

        cash[i] = cash[i-1] + cash_delta[i]

    # Exit positions on last day
    # orders[-1] = - positions[-2]
    # positions[-1] = positions[-2] + orders[-1]
    # cash_delta[-1] = calc_cash_delta(
    #     orders[-1], (prices['VXX'][-1], prices['VXZ'][-1]), positions[-2]) # We use positions[-2] so we can identify which positions are long versus short
    # cash[-1] = cash[-2] + cash_delta[-1]

    vxx_change = (prices['VXX'] - utils.lag(prices['VXX'])) * positions[:, 0]
    vxz_change = (prices['VXZ'] - utils.lag(prices['VXZ'])) * positions[:, 1]

    notional_value = (np.absolute(positions) * prices[['VXX', 'VXZ']]).sum(1) \
                     + cash
    returns = (vxx_change + vxz_change) / notional_value

    log_returns = (np.log(1.0 + returns)).cumsum()
    real_returns = 1000.0 * np.exp(log_returns)
    # normal_returns = 1000.0 * (1.0 + returns).cumprod()

    sharpe_ratio = utils.calculate_sharpe_ratio(returns)
    sortino_ratio = utils.calculate_sortino_ratio(returns.values)
    print('Sharpe Ratio: {0:.2f}'.format(sharpe_ratio))
    print('Sortino Ratio: {0:.2f}'.format(sortino_ratio))

    max_dd, max_duration, max_dd_idx, max_duration_idx, hwm_idx = \
        utils.calculate_max_drawdown_log(log_returns)
    print('Max drawdown: {:.2f}%, max duration: {} days'
          .format(max_dd * 100.0, max_duration))

    plt.plot(real_returns, label='log_returns')
    # plt.plot(normal_returns, label='normal_returns')

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

    plt.legend()
    plt.show()

    # for i in range(0, length):
    #     print('{}, {}, {}, {}, {}, {}, return: {}'.format(
    #         i, prices['VXX'][i], prices['VXZ'][i], orders[i], positions[i], cash[i],
    #         returns[i]))


    # TODO:
    # 1. Add transaction costs
    # 2. Run over multiple time periods


def load_prices(symbols):
    prices = dl.load_price_data(DATA_DIR, symbols)
    df = None

    for key, value in prices.iteritems():
        # dataset = value[-252:]
        dataset = value['13/8/2010':'29/6/2012']
        dataset.rename(columns={PRICE: key}, inplace=True)
        if df is None:
            df = pd.DataFrame(dataset[key])
        else:
            df = df.join(dataset[key])
    return df


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


def get_fixed_vix_ratio():
    return np.array((-0.32, 0.68))


def calc_adjustments(existing_qty, curr_prices, target_alloc, cash):
    curr_value = existing_qty * curr_prices
    # Portfolio is balanced using absolute weightings
    target_orders = target_alloc * (np.sum(np.absolute(curr_value))
                                    + max(0, cash)) / curr_prices

    # We round our target value down to ensure we can actually afford the
    # position
    adjustments = target_orders.astype(int) - existing_qty

    return adjustments


def calc_cash_delta(orders, prices, positions):
    costs = 0.0
    for i in range(0, len(orders)):
        if positions[i] > 0: # long positions
            costs += orders[i] * prices[i]
        elif positions[i] < 0: # short positions
            # When we cover our shorts we make more funds available
            costs += (orders[i] * prices[i] * -1.)
        else:
            if orders[i] > 0:
                costs += (orders[i] * prices[i] * -1.)
            else:
                costs += (orders[i] * prices[i])
    return -costs


def plot(prices):
    for key, value in prices.iteritems():
        plt.plot(value, label=key)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()