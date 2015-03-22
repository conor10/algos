import math
import pandas as pd

import option


def main():
    # calc_slide('test/data/positions.csv', 100.0, 0.4, 90.0 / 365.0, 0.02)
    # calc_slide('test/data/positions2.csv', 100.0, 0.5, 30.0 / 365.0, 0.02)
    # calc_slide('test/data/positions3.csv', 99.60, 0.18, 63.0 / 365.0, 0)
    calc_slide('test/data/positions_mcmillan_p841.csv', 60.0, 0.33, 90.0 / 365.0, 0.1)
    # calc_slide('test/data/positions_mcmillan_p836.csv', 48.0, 0.35, 90.0 / 365.0, 0.05)
    other_positions()


def other_positions():
    delta = 0.0
    gamma = 0.0
    vega = 0.0
    theta = 0.0
    position = 0.0

    time = 21.0 / 365.0
    rate = 0.0

    opt1 = option.Option('C', 125.0, 124.08, 0.412, time, risk_free_rate=rate)
    print('Option 1: {}'.format(opt1.price))
    delta += opt1.delta * -12.0
    gamma += opt1.gamma * -12.0
    theta += opt1.theta * -12.0
    vega += opt1.vega * -12.0

    position += opt1.price * -12.0 * 100.0
    position += 124.08 * 500.0

    opt2 = option.Option('C', 125.0, 122.27, 0.4085, time, risk_free_rate=rate)
    opt3 = option.Option('P', 125.0, 122.27, 0.4085, time, risk_free_rate=rate)
    print('Straddle: {}'.format(opt2.price + opt3.price))
    delta += (opt2.delta + opt3.delta) * -50.0
    gamma += (opt2.gamma + opt3.gamma) * -50.0
    theta += (opt2.theta + opt3.theta) * -50.0
    vega += (opt2.vega + opt3.vega) * -50.0

    position += (opt2.price + opt3.price) * -50.0 * 100.0
    position += 122.27 * -300.0

    print('Position: {}'.format(position))
    print('Delta: {}'.format(delta * 100.0))
    print('Gamma: {}'.format(gamma * 100.0))
    print('Theta: {}'.format((theta / 365.0) * 100.0))
    print('Vega: {}'.format(vega))



def calc_slide(positions_file, underlying, sigma, time, risk_free_rate):
    positions = load_positions(positions_file)

    delta = 0.0
    gamma = 0.0
    vega = 0.0
    theta = 0.0
    pnl = 0.0

    for strike, inst_type, qty in positions.values:
        if inst_type == 'S':
            pnl += strike * qty
            delta += qty
        else:
            opt = option.Option(inst_type, strike, underlying, sigma,
                                time, risk_free_rate=risk_free_rate)

            delta += ((opt.delta * 100.0) * qty)
            print('Delta adjust: {}'.format(opt.delta * 100.0 * qty))
            gamma += (opt.gamma * 100.0) * qty # Gamma is always positive
            vega += (opt.vega) * qty # See p80
            theta += (opt.theta) * qty # See p76

            pnl += opt.price * qty * 100.0

            print('Strike: {}, Type: {}, Qty: {}, Price: {}'.format(strike, inst_type, qty, opt.price))
            print('Delta: {}, Gamma: {}, Vega: {}, Theta: {}'.format(opt.delta, opt.gamma, opt.vega, opt.theta / 365.0))

    print('Delta:\t\t{}'.format(delta))
    print('Gamma:\t\t{}'.format(gamma))
    print('Vega:\t\t{}'.format(vega))
    print('Theta:\t\t{}'.format((theta / 365.0) * 100.0))
    print('Position Cost:\t\t{}'.format(pnl))
    print('Total PnL:\t\t{}'.format(pnl))


def load_positions(filename):
    return pd.read_csv(filename)


if __name__ == '__main__':
    main()
