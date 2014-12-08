import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import volatility_models as models

import data_loader as dl


bounds = range(-300, -100)


def main():
    prices = get_AAPL_data()
    # prices = get_ss_data()
    close_data = prices['Close'].values
    open_data = prices['Open'].values
    high_data = prices['High'].values
    low_data = prices['Low'].values

    imp_vol = get_AAPL_imp()

    std_dev = models.population_std_dev(close_data, 30)
    # unbiased_std_dev = models.unbias_std_dev(std_dev)
    parkinson_std_dev = models.parkinson_std_dev(high_data, low_data, 30)
    garman_klass_std_dev = models.garman_klass_std_dev(close_data,
                                                        open_data,
                                                        high_data,
                                                        low_data,
                                                        30)
    sinclair_garman_klass_std_dev  = models.sinclair_garman_klass_std_dev(
        close_data,
        open_data,
        high_data,
        low_data,
        30)
    rogers_satchell_std_dev = models.rogers_satchell_std_dev(close_data,
                                                             open_data,
                                                             high_data,
                                                             low_data,
                                                             30)
    yang_zhang_std_dev = models.yang_zhang_std_dev(close_data,
                                                   open_data,
                                                   high_data,
                                                   low_data,
                                                   30)
    sinclair_yang_zhang_std_dev = models.sinclair_yang_zhang_std_dev(
        close_data,
        open_data,
        high_data,
        low_data,
        30)

    plt.plot(std_dev, label='std_dev')
    # plt.plot(unbiased_std_dev, label='unbiased')
    plt.plot(parkinson_std_dev, label='parkinson')
    plt.plot(garman_klass_std_dev, label='gk')
    plt.plot(sinclair_garman_klass_std_dev, label='s-gk')
    plt.plot(rogers_satchell_std_dev, label='rs')
    plt.plot(yang_zhang_std_dev, label='yz')
    plt.plot(sinclair_yang_zhang_std_dev, label='s-yz')
    plt.plot(imp_vol, label='imp_vol')
    plt.legend()
    plt.show()


def get_AAPL_data():
    return dl.load_symbol_data(
        '/Users/Conor/code/blog/python/cones/AAPL.csv')[-300:-100]


def get_AAPL_imp():
    imp_vol = pd.read_csv(
        '/Users/Conor/code/blog/python/cones/AAPL_IMP_VOL.csv',
        index_col=2, parse_dates=True)
    imp_vol.sort_index(inplace=True)
    return imp_vol['30d iv mean'][bounds].values


def get_ss_data():
    xls_file = pd.ExcelFile(
        '/Users/Conor/code/python/algos/calc/test/data/VolatilityCones_corrected.xls')
    data = xls_file.parse('yahoo finance data',
                   header=1, index_col=0, parse_cols='B:K')
    data.sort_index(inplace=True)
    return data#[bounds]


if __name__ == '__main__':
    main()