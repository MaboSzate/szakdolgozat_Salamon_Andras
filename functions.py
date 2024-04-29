import pandas as pd
import indicators as ind
import matplotlib.pyplot as plt
import seaborn as sns


def create_features(portfolio, symbol, df, return_lags=0, ho_lags=0):
    data = pd.DataFrame(index=portfolio.excess_returns_df.index)
    data['Target'] = df[symbol]  # a célváltozó maga a többlethozam
    features = []  # magyarázóváltozók neveit tartalmazó lista
    # Múltbeli többlethozam
    for i in range(return_lags):
        features.append(f'{symbol}_lag_{i + 1}')
        data[f'{symbol}_lag_{i + 1}'] = df[symbol].shift(i + 1)
    # Pozitív/negatív napi hozamok
    newdata, features = ind.calculate_highopen_features(portfolio.high_df[symbol], portfolio.open_df[symbol],
                                                        portfolio.low_df[symbol], ho_lags, symbol, features)
    data = data.join(newdata, how='outer').dropna()
    # Technikai indikátorok
    data["RSI"] = ind.calculate_rsi(portfolio.close_df[symbol].shift(1), simple=False)
    data["MOM"] = ind.calculate_mom(portfolio.close_df[symbol])
    data["MACD"] = ind.calculate_macd(portfolio.close_df[symbol].shift(1))
    data["%K"] = ind.calculate_stochastic_k(portfolio.close_df[symbol], portfolio.high_df[symbol],
                                            portfolio.low_df[symbol])
    features.extend(["RSI", "MOM", "MACD", "%K"])
    return data, features


def sliding_windows(portfolio, data):
    train_start = portfolio.train_start_date
    test_start = portfolio.test_start_date
    offset = pd.DateOffset(months=12)
    while test_start <= portfolio.test_end_date:
        upper_bound = min(portfolio.test_end_date, test_start + offset)  # ez a tesztidőszak vége
        # visszaadja a train_start és test_start közti adatokat (tanulóhalmaz),
        # és a test_start és tesztidőszak vége közti adatokat (teszthalmaz)
        yield (data[(data.index >= train_start) & (data.index < test_start)].dropna(),  # tanulóhalmaz
               data[(data.index >= test_start) & (data.index < upper_bound)].dropna())  # tesztahalmaz
        # a tanuló- és a teszthalmaz egyaránt egy évvel előre lép
        train_start += offset
        test_start += offset


def plot_pf_values(df1, df2, df3, df4):
    combined_df = pd.concat([
        df1.rename(columns={"lgb": 'portfolio_value'}).assign(modell='LGB+Sharpe'),
        df2.rename(columns={"sharpe": 'portfolio_value'}).assign(modell='Sharpe'),
        df3.rename(columns={"min_var": 'portfolio_value'}).assign(modell='Min. var.'),
        df4.rename(columns={"eq_weight": 'portfolio_value'}).assign(modell='Egyenlő súlyok')
    ], axis=0)
    combined_df.reset_index(inplace=True)
    sns.lineplot(x='Date', y='portfolio_value', hue='modell', data=combined_df)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend(title='Modell')
    plt.show()


def plot_weights(filepath):
    df = pd.read_excel(filepath, index_col=0)
    df = df.rolling(window=10).mean()  # 10 napos mozgóátlag
    df.index = [dt.date().strftime('%Y-%m') for dt in df.index]
    df.plot(kind="bar", stacked=True, width=1)
    locs, labels = plt.xticks()
    plt.xticks(ticks=locs[1:-1:42], labels=df.index[1:-1:42], rotation=45)  # X-tengely feliratainak manipulálása
    plt.show()

