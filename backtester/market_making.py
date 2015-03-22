import matplotlib.pyplot as plt
import numpy as np


class Side(object):
    BUY = -1
    SELL = 1


def main():
    final_positions = []

    for i in range(0, 10000):
        final_positions.append(run_simulation(1000))

    plt.hist(final_positions)
    plt.show()


def run_simulation(iterations):
    mid_price = 100.0
    tick = 1.0
    qty = 1.0

    positions = Positions()

    for i in range(0, iterations):

        buy_position = Position(Side.BUY, mid_price - tick, qty)
        sell_position = Position(Side.SELL, mid_price + tick, qty)

        positions.add_position(buy_position)
        positions.add_position(sell_position)

        fill = create_fill(mid_price, tick)
        positions.process_fill(fill)

        positions.remove_position(buy_position)
        positions.remove_position(sell_position)

        mid_price = fill.price

    # print('Final position: {}'.format(positions.pnl))
    return positions.pnl


def create_fill(mid_price, tick):
    side = np.sign(np.random.rand() - 0.5)
    price = mid_price + (side * tick)
    qty = 1.0
    return Fill(side, price, qty)


class Positions(object):
    def __init__(self):
        super(Positions, self).__init__()
        self.buy_positions = []
        self.sell_positions = []
        self.pnl = 0.0

    def add_position(self, position):
        if position.side == Side.BUY:
            self.buy_positions.append(position)
        elif position.side == Side.SELL:
            self.sell_positions.append(position)

    def remove_position(self, position):
        if position.side == Side.BUY:
            self.remove_by_price(self.buy_positions, position.price)
        elif position.side == Side.SELL:
            self.remove_by_price(self.sell_positions, position.price)

    @staticmethod
    def remove_by_price(positions, price):
        for position in positions:
            if position.price == price:
                positions.remove(position)

    def process_fill(self, fill):
        if fill.side == Side.BUY:
            self._update_position(self.buy_positions, fill)
        elif fill.side == Side.SELL:
            self._update_position(self.sell_positions, fill)

    def _update_position(self, positions, fill):
        for position in positions:
            if position.price == fill.price:
                if position.qty > fill.qty:
                    position.qty -= fill.qty
                else:
                    positions.remove(position)
                self._update_pnl(fill)

    def _update_pnl(self, fill):
        self.pnl += fill.side * fill.price * fill.qty



class Fill(object):
    def __init__(self, side, price, qty):
        super(Fill, self).__init__()
        self.side = side
        self.price = price
        self.qty = qty


class Position(object):
    def __init__(self, side, price, qty):
        super(Position, self).__init__()
        self.side = side
        self.price = price
        self.qty = qty


if __name__ == '__main__':
    main()