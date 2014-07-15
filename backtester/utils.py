from datetime import datetime

import numpy as np


DATE_FORMAT = '%Y-%m-%d'


def create_date(date):
    return datetime.strptime(date, DATE_FORMAT)


def get_max_vector(array, index=0):
    return array[np.nanargmax(array[:, index])]


def ffill(data):
    for i in range(1, len(data)):
        if np.isnan(data[i]):
            data[i] = data[i-1]
    return data


def lag(data):
    lag = np.roll(data, 1)
    lag[0] = 0.
    return lag