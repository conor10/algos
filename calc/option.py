import math
from math import exp
from math import log
from math import sqrt

from scipy.stats import norm as N


class Type(object):
    CALL = 'C'
    PUT = 'P'


class Option(object):
    def __init__(self, type, strike_price, underlying_price, sigma,
                 days_to_expiry, start_time=0.0, risk_free_rate=0.0,
                 dividend_rate=0.0):
        # Ensure all values are floats to avoid integer conversions & rounding
        self.type = type
        self.K = float(strike_price)
        self.S = float(underlying_price)
        self.T = float(days_to_expiry)
        self.q = float(dividend_rate)
        self.r = float(risk_free_rate)
        self.sigma = float(sigma)
        self.t = float(start_time)

        self._d1 = None
        self._d2 = None
        self._price = None


    @property
    def price(self):
        if self._price is None:
            if self.q != 0.0:
                if self.type == Type.CALL:
                    self._price = self._call_price_with_dividends()
                elif self.type == Type.PUT:
                    self._price = self._put_price_with_dividends()
            else:
                if self.type == Type.CALL:
                    self._price = self._call_price_no_dividends()
                elif self.type == Type.PUT:
                    self._price = self._put_price_with_dividends()

        return self._price

    def _call_price_with_dividends(self):
        return self.S * exp(-self.q * (self.T - self.t)) * N.cdf(self.d1) - \
            self.K * exp(-self.r * (self.T - self.t)) * N.cdf(self.d2)

    def _put_price_with_dividends(self):
        return self.K * exp(-self.r * (self.T - self.t)) * N.cdf(-self.d2) - \
            self.S * math.exp(-self.q * (self.T - self.t)) * N.cdf(-self.d1)

    def _call_price_no_dividends(self):
        return self.S * N.cdf(self.d1) - \
            self.K * exp(-self.r * (self.T - self.t)) * N.cdf(self.d2)

    def _put_price_no_dividends(self):
        return self.K * exp(-self.r * (self.T - self.t)) * N.cdf(-self.d2) - \
            self.S * N.cdf(-self.d1)


    @property
    def d1(self):
        if self._d1 is None:
            self._d1 = (log(self.S / self.K) +
                    (self.r - self.q + (self.sigma ** 2) / 2 ) *
                    (self.T - self.t)) / \
                (self.sigma * sqrt(self.T - self.t))
        return self._d1

    @property
    def d2(self):
        if self._d2 is None:
            self._d2 = self.d1 - self.sigma * sqrt(self.T - self.t)
        return self._d2