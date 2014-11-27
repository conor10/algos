import math
from math import exp, log, pi, sqrt

from scipy.stats import norm as N
from scipy import optimize


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

        self._delta = None
        self._theta = None
        self._vega = None
        self._gamma = None
        self._rho = None


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
            try:
                self._d1 = (log(self.S / self.K) +
                        (self.r - self.q + (self.sigma ** 2) / 2) *
                        (self.T - self.t)) / \
                    (self.sigma * sqrt(self.T - self.t))
            except ZeroDivisionError:
                self._d1 = 0
        return self._d1

    @property
    def d2(self):
        if self._d2 is None:
            self._d2 = self.d1 - self.sigma * sqrt(self.T - self.t)
        return self._d2

    @property
    def delta(self):
        if self._delta is None:
            if self.type == Type.CALL:
                self._delta = self._delta_call()
            elif self.type == Type.PUT:
                self._delta = self._delta_put()
        return self._delta

    def _delta_call(self):
        return exp(-self.q * self.T) * N.cdf(self.d1)

    def _delta_put(self):
        return -exp(-self.q * self.T) * N.cdf(-self.d1)

    @property
    def gamma(self):
        if self._gamma is None:
            self._gamma = (exp(-self.q * self.T) /
                           (self.S * self.sigma * sqrt(self.T))) * \
                          ((1 / sqrt(2 * pi)) * exp(-self.d1**2 / 2))
        return self._gamma

    @property
    def rho(self):
        if self._rho is None:
            if self.type == Type.CALL:
                self._rho = self._rho_call()
            elif self.type == Type.PUT:
                self._rho = self._rho_put()
        return self._rho

    def _rho_call(self):
        return self.K * self.T * exp(-self.r * self.T) * N.cdf(self.d2)

    def _rho_put(self):
        return -self.K * self.T * exp(-self.r * self.T) * N.cdf(-self.d2)

    @property
    def theta(self):
        if self._theta is None:
            if self.type == Type.CALL:
                self._theta = self._theta_call()
            elif self.type == Type.PUT:
                self._theta = self._theta_put()
        return self._theta

    def _theta_call(self):
        return - ((self.S * self.sigma * exp(-self.q * self.T)) /
                  (2 * sqrt(2 * pi * self.T))) * exp(-self.d1**2 / 2) + \
               self.q * self.S * exp(-self.q * self.T) * N.cdf(self.d1) - \
               self.r * self.K * exp(-self.r * self.T) * N.cdf(self.d2)

    def _theta_put(self):
        return - ((self.S * self.sigma * exp(-self.q * self.T)) /
                  (2 * sqrt(2 * pi * self.T))) * exp(-self.d1**2 / 2) - \
               self.q * self.S * exp(-self.q * self.T) * N.cdf(-self.d1) + \
               self.r * self.K * exp(-self.r * self.T) * N.cdf(-self.d2)

    @property
    def vega(self):
        if self._vega is None:
            self._vega = self.S * exp(-self.q * self.T) * sqrt(self.T) * \
                ((1 / sqrt(2 * pi)) * exp(-self.d1**2 / 2))
        return self._vega


def calc_option_price(type, strike_price, underlying_price, volatility,
                      days_to_expiry, start_time=0.0, risk_free_rate=0.0,
                      dividend_rate=0.0):
    option = Option(type, strike_price, underlying_price, volatility,
                    days_to_expiry, start_time, risk_free_rate, dividend_rate)
    return option.price


def implied_volatility(type, strike_price, underlying_price, option_price,
                       days_to_expiry, start_time=0.0, risk_free_rate=0.0,
                       dividend_rate=0.0):

    f = lambda sigma : \
        calc_option_price(type, strike_price, underlying_price, sigma,
                          days_to_expiry, start_time, risk_free_rate,
                          dividend_rate) \
        - option_price

    try:
        result = optimize.brentq(f, 0.000, 5.0)
    except ValueError:
        print('Unable to determine solution for calculation: '
              '[strike_price: {}, underlying_price: {}, option_price: {}, '
              'days_to_exp: {}]'
              .format(strike_price, underlying_price, option_price,
                      days_to_expiry))
        result = -1
    return result


