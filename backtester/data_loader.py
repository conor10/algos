import logging
import glob
import os

import pandas as pd

from option import Type


def load_price_data(data_dir, symbols):
    price_data = {}

    for symbol in symbols:
        symbol_file = _get_price_file(symbol, data_dir)
        price_data[symbol] = _load_symbol_data(symbol_file)

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


def _load_symbol_data(filename, index=0):
    df = pd.read_csv(filename, index_col=index, parse_dates=True)
    # Ensure data is ascending by date
    if df.index[0] > df.index[1]:
        logging.debug(
            'Input data in descending order for file {}, reversing'.format(
                filename))
        df.sort_index(inplace=True)
    return df


def load_option_data(index, directory, symbols=[], dates=[]):
    """
    Dictionary of dataframe price data in format
    data[<symbol>][<date>][<expiry>][P|C]
    """
    basedir = os.path.join(directory, index)
    if len(dates) == 0:
        dates = os.listdir(basedir)
        dates.sort()

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

        if type is Type.CALL or type is Type.PUT:
            expiries[expiry][type] = df

        else:
            raise LoadingException(
                'Unable to determine option type {}'.format(type))

    return expiries


class LoadingException(Exception):
    pass