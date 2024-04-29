import numpy as np
import pandas as pd


def calculate_rsi(data, simple=False):
    change = data.diff().dropna()
    change_up = change.copy()
    change_down = change.copy()
    change_up[change_up < 0] = 0
    change_down[change_down > 0] = 0
    if simple:  # egyszerű mozgóátlag alapján
        avg_gain = change_up.rolling(14, min_periods=14).mean()
        avg_loss = change_down.rolling(14, min_periods=14).mean().abs()
    else:  # exponenciálisan súlyozott mozgóátlag alapján
        avg_gain = change_up.ewm(span=14, min_periods=14).mean()
        avg_loss = change_down.ewm(span=14, min_periods=14).mean().abs()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[:] = np.select([avg_loss == 0, avg_gain == 0, True], [100, 0, rsi])
    return rsi


def calculate_mom(data):
    return data.shift(1) / data.shift(10)


def calculate_macd(data):
    fast = data.ewm(span=12, adjust=False).mean()
    slow = data.ewm(span=26, adjust=False).mean()
    diff = fast - slow
    dea = diff.ewm(span=9, adjust=False).mean()
    return diff - dea


def calculate_stochastic_k(close_prices_df, high_prices_df, low_prices_df, period=14):
    lowest_low = low_prices_df.shift(1).rolling(window=period, min_periods=period).min()
    highest_high = high_prices_df.shift(1).rolling(window=period, min_periods=period).max()
    k_line = ((close_prices_df.shift(1) - lowest_low) / (highest_high - lowest_low)) * 100
    return k_line


def calculate_highopen_features(high_prices_df, open_prices_df, low_prices_df, lags, symbol, features_list):
    data = pd.DataFrame()
    for i in range(lags):
        features_list.append(f'{symbol}_h/o_{i + 1}')
        data[f'{symbol}_h/o_{i + 1}'] = (high_prices_df.shift(i+1) - open_prices_df.shift(i+1)) / \
                                        open_prices_df.shift(i+1)
        features_list.append(f'{symbol}_l/o_{i + 1}')
        data[f'{symbol}_l/o_{i + 1}'] = (low_prices_df.shift(i+1) - open_prices_df.shift(i+1)) / \
                                        open_prices_df.shift(i+1)
    return data, features_list

