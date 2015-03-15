import math
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

    length = len(ivts)
    orders = np.zeros((length, 2), dtype=np.int)
    positions = np.zeros((length, 2), dtype=np.int)
    cash = np.zeros(length)
    cash[0] = 100000.0
    cash_delta = np.zeros(length)

    # for i in range(1, length-1):
    for i in range(1, length):
        # ratio = get_mojito_ratio(ivts[i-1])
        ratio = get_dynamic_vix_ratio(ivts[i-1])
        print('{}, {}'.format(ratio[0], ratio[1]))

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

    sharpe_ratio = (np.mean(returns) / np.std(returns)) * math.sqrt(252.0)
    print('Sharpe Ratio: {}'.format(sharpe_ratio))

    plt.plot(1000 + (returns.cumsum() * 1000))
    plt.show()

    for i in range(0, length):
    #     print('{}, {}, {}'.format(prices.index[i], prices['VXX'].ix[i], prices['VXZ'].ix[i]))
        print('{}, {}, {}, {}, {}, {}, return: {}'.format(
            i, prices['VXX'][i], prices['VXZ'][i], orders[i], positions[i], cash[i],
            returns[i]))


    # portfolio_value = (positions * prices[['VXX', 'VXZ']])
    # print(portfolio_value)

    # TODO:
    # 1. Calculate Sharpe & Sortino ratios
    # 2. Calculate maximum drawdown
    # 3. Plot portfolio value
    # 4. Add transaction costs
    # 5. Run over multiple time periods
    # 6. Fix cash calculation

    # print(value / (np.absolute(value['VXX']) + np.absolute(value['VXZ'])))


def load_prices(symbols):
    prices = dl.load_price_data(DATA_DIR, symbols)
    df = None

    for key, value in prices.iteritems():
        # dataset = value[-200:]
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


def calc_adjustments(existing_qty, curr_prices, target_alloc, cash):
    curr_value = existing_qty * curr_prices
    # Portfolio is balanced using absolute weightings
    target_orders = target_alloc * (np.sum(np.absolute(curr_value))
                                    + max(0, cash)) / curr_prices

    # We round our target value down to ensure we can actually afford the position
    adjustments = target_orders.astype(int) - existing_qty

    return adjustments


def calc_adjustments2(long_positions, short_positions, curr_prices, target_alloc, cash):
    curr_value_long = long_positions * curr_prices
    curr_value_short = short_positions * curr_prices
    # Portfolio is balanced using absolute weightings
    target_value = target_alloc * \
                   (np.sum(curr_value_short * -1.0 +
                          curr_value_long) + max(0, cash))

    # TODO: need to ensure we have enough cash to cover new ratios (cannot go negative)
    adjustments_long = ((target_value - curr_value_long) / curr_prices)
    adjustments_short = ((target_value - curr_value_short) / curr_prices)
    # We round orders down

    # Need to ensure enough cash for adjustments

    return adjustments_long.astype(int), adjustments_short.astype(int)


def calc_cash_delta(orders, prices, positions):
    costs = 0.0
    # TODO: What do we do when we have no positions? Perhaps seperate positions into long & short
    for i in range(0, len(orders)):
        if positions[i] > 0: # long positions
            costs += orders[i] * prices[i]
        elif positions[i] < 0: # short positions
            # When we cover our shorts we make more funds available
            costs += (orders[i] * prices[i] * -1)
    return -costs


def calc_sharpe_ratio(positions, prices):
    value = (positions * prices).sum(1)
    returns = utils.calculate_returns(value)
    return (returns.mean() * 252.0) / (returns.std() * math.sqrt(252.0))


def plot(prices):
    for key, value in prices.iteritems():
        plt.plot(value, label=key)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()