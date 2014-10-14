import datetime as dt
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import data_loader
import init_logger
from option import Type
import price_series

import utils


def main():

    run_date = dt.datetime(2014, 9, 19)

    # symbols = \
    #     data_loader.load_symbol_list(
    #         '/Users/Conor/code/python/conor10.tickdata/symbols/'
    #         'SP500/20140907/symbols.txt')
    symbols = ['AAPL']
    all_options_prices = data_loader.load_option_data(
        'SP500', '/Users/Conor/code/python/conor10.tickdata/chains', symbols,
        [run_date.strftime('%Y%m%d')])
    all_stock_prices = data_loader.load_price_data(
        '/Users/Conor/code/python/conor10.tickdata/daily_prices/SP500',
        symbols)

    for symbol in symbols:

        # Get all puts

        # if put is OTM, predict probability it will be ITM at expiry
        stock_prices = all_stock_prices[symbol]
        underlying_price = \
            stock_prices['Close'][[run_date]].values[0]
        print(underlying_price)

        stock_prices['Adj Returns'] = \
            utils.calculate_returns(stock_prices['Adj Close'].values)

        option_prices = all_options_prices[symbol]


        for capture_date in option_prices.keys():
            expiries = option_prices[capture_date]
            for expiry in expiries.keys():
                prices = expiries[expiry]
                puts = prices[Type.PUT]
                print(expiry)

                dt_expiry = dt.datetime.strptime(expiry, '%Y%m%d')
                # TODO: Calculate the exact value -
                # TODO: there's 9 trading holidays on NASDAQ in 2014
                days_in_year = np.busday_count('2014', '2015') - 9

                days_to_expiry = utils.work_day_count(run_date, dt_expiry)
                # TODO: Need to calculate trading days to expiry
                # TODO: workalendar may be useful

                #TODO Rename
                delta = days_to_expiry / days_in_year

                # TODO: Make model more sophisticated for sigma
                # lookback_period = days_to_expiry
                lookback_period = days_in_year
                sigma = calc_sigma(stock_prices, lookback_period, run_date,
                                   days_in_year)

                itm_probability = price_series.calc_itm_probability(
                    puts.index, underlying_price, sigma, days_to_expiry,
                    10000, Type.PUT)

#                print(itm_probability)

                # We're only interested in strikes that are OTM to start with
                otm_probabilities = itm_probability[np.where(
                    itm_probability[:, 0] < underlying_price)]
                # print(otm_probabilities)

                # select all options we have a last price for & assume this
                # selling cost

                results = []

                for strike, probability in otm_probabilities:
                    last_price_str = puts.ix[strike]['LAST_PRICE']
                    if last_price_str == '-':
                        continue
                    last_price = float(last_price_str)
                    # print('Strike: {}, P(ITM): {}, Last Price: {}'.format(
                    #     strike, probability, last_price))
                    results.append([strike, probability, last_price])


                data = np.asarray(results).T

                plot3D(data[0], data[1], data[2],
                       'Date: {}, Underlying: {}, Last price: {}, Expiry: {}'
                       .format(capture_date, symbol, underlying_price, expiry))

                # TODO: Plot some potential outcomes
                # TODO: What strategy to use to determine option to buy?

# Need to determine most appropriate distribution for price modelling
# bayesian inference prior to posterior distribution
# Monty hall problem, secretary problem & the hanged man problem


def get_returns():
    pass

def calc_sigma(returns, lookback_period, end_date, annual_day_count):
    start_date = utils.work_day_delta(end_date, -lookback_period)
    return returns[start_date:end_date]['Adj Returns'].std() \
        * math.sqrt(annual_day_count)


def plot3D(X, Y, Z, title):
    # fig = plt.figure ()
    # ax = Axes3D(fig, azim=-29, elev=50)
    figure = plt.figure(figsize=(8, 8), facecolor='w')
    ax = figure.gca(projection='3d')
    ax.plot(X, Y, Z, 'o')
    plt.xlabel('strike')
    plt.ylabel('probability')
    ax.set_zlabel('last price')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    init_logger.setup()
    main()
