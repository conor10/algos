import numpy as np

import option
from datatypes import OptionType
import price_series



def main():
    qty = 25.0
    get_revised_price(100.0)



def get_revised_price(underlying):
    opt = option.Option(OptionType.CALL, 100.0, underlying, 0.3, 1)
    print(opt.price * 100.0 * 25.0)
    print(opt.vega * 25.0)


def get_random_walk(start_price, sigma, count):
    price_series.generate_random_walks(100)

    np.cumsum(np.random.normal(0, sigma, 100)) + start_price


if __name__ == '__main__':
    main()