import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--period", help="function period", default=0.2, required=False)
parser.add_argument("--amplitude", help="function amplitude", default=50, required=False)
parser.add_argument("--offset", help="function offset", default=900, required=False)
parser.add_argument("--noise", help="function noise", default=0, required=False)
args = parser.parse_args()

T = 2693389
l = float(args.period)
M = float(args.amplitude)
h = float(args.offset)
noise = float(args.noise)

column_names = ["unix","date","symbol","open","high","low","close","Volume BTC","Volume USD"]
columns_pd = np.zeros((T,len(column_names)))

signal = h + M*np.sin(l*0)
for t in tqdm(range(T)):
    newsignal = h + M*np.sin(l*t)

    unix = np.random.randint(100)
    date = np.random.randint(100)
    symbol = 0
    high = max(signal, newsignal)
    low = min(signal, newsignal)
    volumeBTC = 20*np.random.random()
    volumeUSD = signal*volumeBTC
    columns_pd[t] = np.array([unix, date, symbol, signal, high, newsignal, low, volumeBTC, volumeUSD])
    if noise:
        columns_pd[t] *= 1 - noise + 2*noise*np.random.random((9,))

    signal = newsignal

os.makedirs("gym_trading_btc/datasets/data/generated_data", exist_ok=True)
pd.DataFrame(columns_pd,columns=column_names).to_csv(f"gym_trading_btc/datasets/data/generated_data/sinus_l{args.period}_M{args.amplitude}_h{args.offset}_noise{args.noise}.csv",index=False)