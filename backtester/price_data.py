import pandas as pd

# TODO: we need to ensure dates are indexed correctly in the merge -
# TODO: i.e. no dates are missed out where to get this data from?
class PriceData(object):
    def __init__(self, price_data):
        self.price_data = price_data

    def get_price_data(self, symbols, price_type, start=None, end=None):
        result = pd.DataFrame()

        for symbol in symbols:
            if symbol in self.price_data:
                data = self.price_data[symbol][price_type]
                clean_data = self._fill_missing_values(data)
                result[symbol] = clean_data.ix[start:end]

        return result

    def get_price_data_np(self, symbols, price_type, start=None, end=None):
        price_data = self.get_price_data(symbols, price_type, start, end)

        np_price_data = {}

        for symbol in symbols:
            np_price_data[symbol] = price_data[symbol].as_matrix()

        return np_price_data

    @staticmethod
    def _fill_missing_values(df):
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        return df