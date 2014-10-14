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

if __name__ == '__main__':
    unittest.main()
