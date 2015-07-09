import logging
import glob
import os

import pandas as pd

from datatypes import OptionType, FUTURES_MONTHS


def load_price_data(data_dir, symbols, range=[]):
    price_data = {}

    for symbol in symbols:
        symbol_file = _get_price_file(symbol, data_dir)
        price_data[symbol] = load_symbol_data(symbol_file)

    return price_data


def load_symbol_list(filename):
    try:
        logging.debug('Loading symbol list from {}'.format(filename))
        with open(filename, 'r') as f:
            return f.read().splitlines()
    except IOError:
        logging.error('Unable to open file: {}'.format(filename))
        return []


def _get_price_file(name, data_dir):
    return os.path.join(data_dir, name + '.csv')


def load_symbol_data(filename, index=0, header_row=0, order_ascending=True):
    df = pd.read_csv(filename, index_col=index, header=header_row,
                     parse_dates=True)

    if order_ascending:
        # Ensure data is in ascending order by date
        if df.index[0] > df.index[1]:
            df.sort_index(inplace=True)
    return df


def load_intraday_data(index, directory, symbols, start_date=None,
                       end_date=None):
    basedir = os.path.join(directory, index)

    dates = get_dates(basedir, start_date, end_date)

    data = {}

    if len(symbols) > 0:
        for symbol in symbols:
            data[symbol] = {}

            for date in dates:
                data[symbol][date] = {}

                prices_dir = os.path.join(basedir, date,
                                          '{}.csv'.format(symbol))
                files = glob.glob(prices_dir)
                data[symbol][date] = _process_intraday_price_files(
                    symbol, files)

    else:
        raise LoadingException('No symbols specified')

    return data


def _process_intraday_price_files(symbol, files):
    for filename in files:
        return pd.read_csv(filename, index_col='timestamp', parse_dates=True)


def load_option_data(index, directory, symbols=[],
                     start_date=None, end_date=None):
    """
    Dictionary of dataframe price data in format
    data[<symbol>][<date>][<expiry>][P|C]
    """
    basedir = os.path.join(directory, index)
    dates = get_dates(basedir, start_date, end_date)

    data = {}

    if len(symbols) > 0:
        for symbol in symbols:
            data[symbol] = {}

            for date in dates:
                data[symbol][date] = {}

                prices_dir = os.path.join(basedir, date,
                                         '{}[0-9]*[PC].csv'.format(symbol))
                files = glob.glob(prices_dir)
                data[symbol][date] = _process_option_price_files(symbol, files)

    else:
        raise LoadingException('No symbols specified')

    return data


def _process_option_price_files(symbol, files):

    expiries = {}

    for filename in files:

        logging.debug('Loading price file: {}'.format(filename))
        detail = filename.lstrip(symbol).rstrip('.csv')
        type = detail[-1]
        expiry = detail[-9:-1]
        # Some files contain an extra column
        # For example, see line 41 of 20140908/COST20140920C.csv
        df = pd.read_csv(filename, index_col='STRIKE', thousands=',',
                         error_bad_lines=False, warn_bad_lines=True)

        if expiry not in expiries:
            expiries[expiry] = {}

        if type is OptionType.CALL or type is OptionType.PUT:
            expiries[expiry][type] = df

        else:
            raise LoadingException(
                'Unable to determine option type {}'.format(type))

    return expiries


def get_dates(directory, start_date=None, end_date=None):
    all_dates = os.listdir(directory)
    all_dates.sort()

    start_idx = 0
    end_idx = len(all_dates)

    if start_date in all_dates:
        start_idx = all_dates.index(start_date)
    if end_date in all_dates:
        # We want result to be inclusive of end_date
        end_idx = all_dates.index(end_date) + 1

    return all_dates[start_idx:end_idx]


def load_vix_futures_prices(source_dir, price='Close',
                            start_year=2005, end_year=2099):
    """
    Dictionary of dataframe price data in format
    CFE_[M][YY]_VX.csv where M is []

    start_year and end_year parameters refer to futures we are interested in,
    not dates we have price data for.

    :return data[YYYY][M] = dataframe
    Where YYYY is expiry year, M is expiry month in range [0, 11]
    """

    data = {}

    files = glob.glob(os.path.join(source_dir, 'CFE_*'))
    for f in files:
        filename = os.path.basename(f)
        month = FUTURES_MONTHS.index(filename[4])
        year = int('20' + filename[5] + filename[6])

        if year < start_year or year > end_year:
            continue

        try:
            df = load_symbol_data(f, index=0, header_row=0)
        except IndexError:
            df = load_symbol_data(f, index=0, header_row=1)

        if year not in data:
            data[year] = 12 * [None]
        data[year][month] = df[price]

    return data


class LoadingException(Exception):
    pass
