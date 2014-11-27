import unittest

from option import Option
from option import Type
import option


class OptionTests(unittest.TestCase):
    # Taken from p104 of Stefanica MFE
    def test_call_price_with_dividend(self):
        option = Option(Type.CALL, 40.0, 42.0,
                        0.3, 0.5, risk_free_rate=0.05,
                        dividend_rate=0.03)
        self.assertEqual(0.3832055088531541, option.d1)
        self.assertEqual(0.17107347449718982, option.d2)
        self.assertEqual(4.7053274115128225, option.price)

    def test_put_price_with_dividend(self):
        option = Option(Type.PUT, 40.0, 42.0,
                        0.3, 0.5, risk_free_rate=0.05,
                        dividend_rate=0.03)
        self.assertEqual(0.3832055088531541, option.d1)
        self.assertEqual(0.17107347449718982, option.d2)
        self.assertEqual(2.3430224293174966, option.price)

    # Taken from p316 of Hull
    def test_call_price_no_dividend(self):
        option = Option(Type.CALL, 40.0, 42.0,
                        0.2, 0.5, risk_free_rate=0.1)
        self.assertEqual(0.7692626281060315, option.d1)
        self.assertEqual(0.627841271868722, option.d2)
        self.assertEqual(4.7594223928715316, option.price)

    def test_put_price_no_dividend(self):
        option = Option(Type.PUT, 40.0, 42.0,
                        0.2, 0.5, risk_free_rate=0.1)
        self.assertEqual(0.7692626281060315, option.d1)
        self.assertEqual(0.627841271868722, option.d2)
        self.assertEquals(0.80859937290009221, option.price)

    def test_implied_volatility_call_with_dividend(self):
        implied_volatility = \
            option.implied_volatility(Type.CALL, 40.0, 42.0, 4.71, 0.5,
                                      risk_free_rate=0.05, dividend_rate=0.03)
        self.assertAlmostEqual(0.3, implied_volatility, places=3)

    def test_implied_volatility_put_with_dividend(self):
        implied_volatility = \
            option.implied_volatility(Type.PUT, 40.0, 42.0, 2.34, 0.5,
                                      risk_free_rate=0.05, dividend_rate=0.03)
        self.assertAlmostEqual(0.3, implied_volatility, places=3)

    def test_implied_volatility_call_no_dividend(self):
        implied_volatility = \
            option.implied_volatility(Type.CALL, 40.0, 42.0, 4.76, 0.5,
                                      risk_free_rate=0.1)
        self.assertAlmostEqual(0.2, implied_volatility, places=3)

    def test_implied_volatility_put_no_dividend(self):
        implied_volatility = \
            option.implied_volatility(Type.PUT, 40.0, 42.0, 0.81, 0.5,
                                      risk_free_rate=0.1)
        self.assertAlmostEqual(0.2, implied_volatility, places=3)

    # taken from p92 of Joshi
    def test_greeks_joshi(self):
        call = Option(Type.CALL, 110.0, 100.0, 0.1, 1.0, 0.0, 0.05, 0.0)

        self.assertAlmostEqual(2.174, call.price, places=3)
        self.assertAlmostEqual(0.343, call.delta, places=3)
        # Book value is 36.77
        self.assertAlmostEqual(36.78, call.vega, places=2)
        self.assertAlmostEqual(0.0368, call.gamma, places=4)

        put = Option(Type.PUT, 110.0, 100.0, 0.1, 1.0, 0.0, 0.05, 0.0)

        self.assertAlmostEqual(6.81, put.price, places=2)
        self.assertAlmostEqual(-0.657, put.delta, places=3)
        # Book value is 36.77
        self.assertAlmostEqual(36.78, put.vega, places=2)
        self.assertAlmostEqual(0.0368, put.gamma, places=4)

        print('rho: {}'.format(call.rho))
        print('theta: {}'.format(call.theta))

    # taken from p383 onwards of Hull
    def test_greeks_hull(self):
        call = Option(Type.CALL, 50.0, 49.0, 0.2, 0.3846, 0.0, 0.05, 0.0)

        self.assertAlmostEqual(0.522, call.delta, places=3)
        self.assertAlmostEqual(-4.31, call.theta, places=2)
        self.assertAlmostEqual(0.066, call.gamma, places=2)
        self.assertAlmostEqual(12.1, call.vega, places=1)
        self.assertAlmostEqual(8.91, call.rho, places=2)

if __name__ == '__main__':
    unittest.main()
