import logging
import os.path

import pandas as pd


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
