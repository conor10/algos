import numpy as np

class Series:
    RANDOM_WALK = 0
    MOMENTUM = 1
    MEAN_REVERTING = 2

def generate(type, start_price, iterations=1000):
    if type == Series.MOMENTUM:
        pass
    elif type == Series.MOMENTUM:
        pass
    else:
        pass



def _generate_random_return(iterations, mu=0.0, sigma=1.0):
    return np.cumsum(
        _generate_random_price_series(mu, sigma, iterations))

def _generate_random_price_series(mu, sigma, iterations):
    #return np.random.randn(iterations) * sigma + mu
    return np.random.normal(mu, sigma, iterations)

def _generate_ar_return(iterations, mu, phi):
    pass