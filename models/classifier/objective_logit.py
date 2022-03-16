import numpy as np
from gym_trading_btc.gym_anytrading.envs.bitcoin_env import Actions

# best buying and selling days


def maxima_buy_sell(price):
    argsort_price = np.argsort(price)
    res = np.full_like(price, Actions.Stay.value)
    s_price = argsort_price.shape[0]
    tmp = int(s_price/2)
    sells = argsort_price[s_price - tmp:]
    buys = argsort_price[:tmp]
    res[buys] = Actions.Buy.value
    res[sells] = Actions.Sell.value
    return res

# Find when to sell and to buy in comparaison with the last day


def end_buy_sell(price):
    last_price = price[-1]
    res = np.full_like(price, Actions.Stay.value)
    res[res < last_price] = Actions.Buy.value
    res[res > last_price] = Actions.Sell.value
    return res

# We buy if the next price increase, and the contrary if decreasing


def local_buy_sell(price):
    res = np.full_like(price, Actions.Stay.value)
    diff = price[:-1] - price[1:]
    diff = np.concatenate([diff, [0]])
    res[diff > 0] = Actions.Sell.value
    res[diff < 0] = Actions.Buy.value
    return res
