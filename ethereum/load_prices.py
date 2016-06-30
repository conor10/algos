import json
import math

import pandas as pd
from pandas.io.json import json_normalize
import requests
import seaborn

import common.utils as utils

def main():
    r = requests.get('http://etherchain.org/api/statistics/price')

    json_data = json.loads(str(r.text))
    data = json_normalize(json_data['data'])
    data['time'] = pd.to_datetime(data['time'])

    data = data.set_index('time')
    data.columns = ['USD/ETH']

    returns = utils.calculate_returns(data)
    annual_std_dev = returns.std() * math.sqrt(252. * 24.)

    data.plot()


if __name__ == '__main__':
    main()
