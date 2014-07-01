import pandas as pd


# TODO: we need to ensure dates are indexed correctly in the merge -
#  where to get this data from?
class PriceData(object):
    def __init__(self, price_data):
        self.price_data = price_data

    def get_price_data(self, symbols, price_type):

        result = pd.DataFrame()

        for symbol in symbols:
            if symbol in self.price_data:
                result[symbol] = self.price_data[symbol][price_type]

        return result