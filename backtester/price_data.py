import pandas as pd


# TODO: we need to ensure dates are indexed correctly in the merge -
# TODO: where to get this data from?
class PriceData(object):
    def __init__(self, price_data):
        self.price_data = price_data

    def get_price_data(self, symbols, price_type, start=None, end=None):
        result = pd.DataFrame()

        for symbol in symbols:
            if symbol in self.price_data:
                data = self.price_data[symbol][price_type]
                result[symbol] = data.ix[start:end]

        return result

