import calendar
import datetime as dt
import init_logger

import data_loader
import math
import option
from option import Type

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

        stock_price = all_stock_prices[symbol]
        underlying_price = \
            stock_price['Adj Close'][[run_date]].values[0]
        print(underlying_price)

        stock_price['Adj Returns'] = \
            utils.calculate_returns(stock_price['Adj Close'].values)

        start_date = run_date - dt.timedelta(31)
        sigma = stock_price[start_date:run_date]['Adj Returns'].std() * math.sqrt(252.0)

        option_prices = all_options_prices[symbol]

        for capture_date in option_prices.keys():
            expiries = option_prices[capture_date]
            for expiry in expiries.keys():
                prices = expiries[expiry]
                puts = prices[Type.PUT]
                print(expiry)
                # print(puts)

                dt_expiry = dt.datetime.strptime(expiry, '%Y%m%d')
                days_in_year = 366 if calendar.isleap(dt_expiry.year) else 365
                days_to_expiry = dt_expiry - run_date

                delta = days_to_expiry.days / days_in_year

                for strike in puts.index:
                    last_price_str = puts.ix[strike]['LAST_PRICE']
                    if last_price_str == '-':
                        continue
                    last_price = float(last_price_str)
                    price = option.calc_option_price(
                        Type.PUT, strike, underlying_price, sigma, delta,
                        risk_free_rate=0.01)
                    implied_volatility = option.implied_volatility(
                        Type.PUT, strike, underlying_price, last_price, delta)
                    print('Strike: {}, calc price: {}, actual {}'.format(
                        strike, price, last_price))
                    print('Sigma: {}, implied sigma: {}'.format(
                        sigma, implied_volatility))


def calc_sigma(returns, start_date, expiry_date):
    pass


if __name__ == '__main__':
    init_logger.setup()
    main()