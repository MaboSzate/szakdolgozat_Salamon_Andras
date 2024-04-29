import pandas as pd
import numpy as np
import scipy.optimize as sp
from sklearn.model_selection import TimeSeriesSplit
import pipeline as pl
import openpyxl
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
import functions as f
import lightgbm as lgb


class PortfolioOptimizer:
    def __init__(self, test_start_date, test_end_date, train_start_date, start_date="2007-01-01",
                 end_date="2023-12-31"):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.train_start_date = pd.to_datetime(train_start_date)
        self.test_start_date = pd.to_datetime(test_start_date)
        self.test_end_date = pd.to_datetime(test_end_date)
        self.prices_df = None
        self.returns_df = None
        self.excess_returns_df = pd.DataFrame()
        self.logreturns_df = pd.DataFrame()
        self.symbols = []
        self.noa = 0
        self.high_df = None
        self.low_df = None
        self.close_df = None
        self.open_df = None
        self.risk_free_df = None
        self.merged_prices_df = None
        self.split = TimeSeriesSplit(max_train_size=444, test_size=111, n_splits=5)

    def load_data(self, symbols, risk_free_symbol):
        merged_price_df = pd.DataFrame()
        merged_high_df = pd.DataFrame()
        merged_low_df = pd.DataFrame()
        merged_close_df = pd.DataFrame()
        merged_open_df = pd.DataFrame()
        for symbol in symbols:
            self.noa = self.noa + 1
            stock_data = yf.download(symbol, start=self.start_date, end=self.end_date)
            stock_data.index = pd.to_datetime(stock_data.index)
            high_data = stock_data['High']
            low_data = stock_data['Low']
            close_data = stock_data['Close']
            open_data = stock_data['Open']
            stock_data = stock_data['Adj Close']
            merged_price_df = merged_price_df.join(stock_data.rename(symbol), how='outer').dropna()
            merged_high_df = merged_high_df.join(high_data.rename(symbol), how='outer').dropna()
            merged_low_df = merged_low_df.join(low_data.rename(symbol), how='outer').dropna()
            merged_close_df = merged_close_df.join(close_data.rename(symbol), how='outer').dropna()
            merged_open_df = merged_open_df.join(open_data.rename(symbol), how='outer').dropna()
        risk_free_data = yf.download(risk_free_symbol, start=self.start_date, end=self.end_date)
        risk_free_data.index = pd.to_datetime(risk_free_data.index)
        risk_free_data = risk_free_data['Adj Close']
        risk_free_data = risk_free_data / 100 / 252
        risk_free_data = risk_free_data.rename("RiskFree")
        self.risk_free_df = risk_free_data
        merged_df = pd.merge(merged_price_df, risk_free_data, left_index=True, right_index=True, how='outer').dropna()
        self.prices_df = merged_df[symbols]
        self.high_df = merged_high_df
        self.low_df = merged_low_df
        self.close_df = merged_close_df
        self.open_df = merged_open_df
        self.returns_df = (self.prices_df - self.prices_df.shift(1)) / self.prices_df.shift(1)
        self.returns_df = self.returns_df.dropna()
        for symbol in symbols:
            self.logreturns_df[symbol] = np.log1p(self.returns_df[symbol])
            self.excess_returns_df[symbol] = self.returns_df[symbol] - merged_df["RiskFree"]
        self.excess_returns_df = self.excess_returns_df.dropna()
        self.logreturns_df = self.logreturns_df.dropna()
        self.symbols = symbols
        self.merged_prices_df = merged_price_df

    def descriptives(self, start, end):
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        df = self.excess_returns_df[start:end]
        print(df.describe(), df.skew(axis=0), df.kurt(axis=0))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap="BrBG", cbar=False)
        plt.show()

    def plot_prices(self, start, end):
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        df = self.logreturns_df[start:end]
        df = df.cumsum()
        df = np.exp(df)
        df.plot()
        plt.show()

    def pipeline_search(self, configs, symbol):
        best_config = None
        best_error = np.inf
        df = self.excess_returns_df.dropna()
        for config in configs:
            method, return_lags, ho_lags = (config[0], config[1], config[2])  # a konfigurációs lista kibontása
            data, features = f.create_features(self, symbol, df, return_lags, ho_lags)  # magyarázóváltozók
            sum_error = 0
            for train, test in f.sliding_windows(self, data):
                X_train = train[features]
                y_train = train["Target"]
                split = TimeSeriesSplit(max_train_size=444, test_size=111, n_splits=5)  # mozgó időablakos keresztval.
                model, error = pl.original_pipeline(X_train, y_train, split, method)  # hiperparaméter-hangolás
                sum_error += error  # az MSE-k összeadása
            if sum_error < best_error:  # keresem a legalacsonyabb MSE-t
                best_error = sum_error
                best_config = config
            print("Current", config, sum_error)
        print("Best", best_config, best_error)

    def predict_returns(self, method, symbol, window, return_lags=10, ho_lags=5, importance='gain'):
        df = self.excess_returns_df.dropna()
        # tesztidőszakhoz tartozó munkanapok megkeresése
        test_dates = []
        for date in pd.date_range(start=self.test_start_date, end=self.test_end_date):
            if date in df.index:
                test_dates.append(date)
        predicted_returns = pd.DataFrame(columns=[method])  # ide kerülnek majd a becsült többlethozamok
        real = df.loc[df.index.isin(test_dates)]  # tesztidőszaki valódi hozamok
        if method == "Sharpe":
            # Hagyományos Sharpe-modell szerinti "becslés"
            for date in test_dates:
                start_idx = df.index.get_loc(date) - window
                end_idx = df.index.get_loc(date)
                window_data = df.iloc[start_idx:end_idx]
                predicted_returns.loc[date, method] = window_data[symbol].mean()  # historikus átlag
        else:
            # Gépi tanulási módszerek általi becslés
            data, features = f.create_features(self, symbol, df, return_lags, ho_lags)  # magyarázóváltozók
            split = TimeSeriesSplit(max_train_size=444, test_size=111, n_splits=5)  # mozgó időablakos keresztval., n=5
            idx = 0
            importances_df = pd.DataFrame()
            for train, test in f.sliding_windows(self, data):  # 4 tanulási periódus, mindegyikben egy tanuló-teszt pár
                idx += 1  # számolja az időszakokat
                X_train = train[features]  # magyarázóváltozók a tanulóhalmazon
                y_train = train["Target"]  # célváltozó a tanulóhalmazon
                X_test = test[features]  # magyarázóváltozók a teszthalmazon
                model, error = pl.final_pipeline(X_train, y_train, split, importance)  # hiperparaméter-hangolás
                if importance is not None:  # fontosságokat tartalmazó excel file létrehozása
                    importances = model.named_steps['reg'].feature_importances_
                    flist = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "rh1", "rl1", "rh2", "rl2",
                             "rh3", "rl3", "rh4", "rl4", "rh5", "rl5", "RSI", "MoM", "MACD", "%K"]
                    model_importances_df = pd.DataFrame({'Feature': flist, 'Importance': importances})
                    model_importances_df['Asset'] = symbol
                    model_importances_df['Time'] = idx
                    importances_df = pd.concat([importances_df, model_importances_df], ignore_index=True)
                prediction = model.predict(X_test)  # becslés a tesztidőszakon
                temp_df = pd.DataFrame(prediction, columns=[method])
                predicted_returns = pd.concat([predicted_returns, temp_df])  # összerakás a korábbi becslésekkel
            predicted_returns.index = test_dates  # a tesztidőszak dátumai bekerülnek indexbe
            if importance is not None:
                importances_df.to_excel('feature_importances_' + importance + str(symbol) + ".xlsx", index=False)
        predicted_returns.to_excel(method + "_pred.xlsx")
        return predicted_returns, real

    def optimize_portfolio(self, method, window, return_lags=10, ho_lags=5, filepath=None, importance='gain'):
        df = self.excess_returns_df
        optimal_weights_df = pd.DataFrame(index=df.index, columns=df.columns)
        predictions = pd.DataFrame()
        for symbol in self.symbols:
            # becslések mindegyik eszközre
            if filepath == None:
                pred, real = self.predict_returns(method=method, symbol=symbol, return_lags=return_lags,
                                                  ho_lags=ho_lags, window=window, importance=importance)
                predictions[symbol] = pred
            else:
                # megadott fájl esetén az ott tárolt becsléseket tölti be
                predictions[symbol] = pd.read_excel(filepath, index_col=0)[method]
        for date in pd.date_range(start=self.test_start_date, end=self.test_end_date):
            if date in df.index:
                # végigmegy a tesztidőszak összes napján
                start_idx = df.index.get_loc(date) - window
                end_idx = df.index.get_loc(date)
                window_data = df.iloc[start_idx:end_idx]  # az M nagyságú ablakhoz tartozó többlethozamok
                prediction = predictions.loc[date, :]  # az adott napi becsült érték

                def objective(weights):  # Sharpe-maximalizáló célfüggvény
                    portfolio_return = np.sum(prediction * weights)  # becsült portfólió többlethozam
                    # historikus kovarianciamátrix alapján számolt szórás
                    portfolio_std = np.sqrt(np.dot(weights, np.dot(window_data.cov(), weights)))
                    SR = portfolio_return / portfolio_std
                    return -SR  # a Sharpe-ráta -1-szerese, mert minimalizálva lesz

                constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # súlyok összege 1
                bounds = tuple((0, 1) for _ in range(self.noa))  # súlyok 0 és 1 között (nincs shortolás)
                initial_weights = np.array([1 / self.noa] * self.noa)  # induló becslés az optimalizáló függvénynek
                result = sp.minimize(objective, initial_weights, method='SLSQP', bounds=bounds,
                                     constraints=constraints)  # Sharpe -1-szeresének minimalizálása
                optimal_weights_df.loc[date] = result.x  # optimális súlyvektor
                print(date)  # kiírja, hogy hol tart épp
        optimal_weights_df = optimal_weights_df.dropna()
        # portfólió hozama a súlyozott eszközhozamok, ennek veszem a logaritmusát a loghozamokhoz
        portfolio_logreturns = np.log1p((self.returns_df * optimal_weights_df).dropna().sum(axis=1).astype(float))
        cum_returns = portfolio_logreturns.cumsum()  # kumulált loghozamok
        portfolio_values = pd.DataFrame()
        # a kumulált loghozam visszaalakítva a kumulált portfólióértéket adja meg, egységnyi kezdőtőke mellett
        portfolio_values[method] = np.exp(cum_returns.astype(float))
        return optimal_weights_df, portfolio_values

    def optimize_portfolio_min_var(self, window=252 * 5):
        df = self.excess_returns_df
        optimal_weights_df = pd.DataFrame(index=df.index, columns=df.columns)
        for date in pd.date_range(start=self.test_start_date, end=self.test_end_date):
            if date in df.index:
                start_idx = df.index.get_loc(date) - window
                end_idx = df.index.get_loc(date)
                window_data = df.iloc[start_idx:end_idx]  # eddig az előzővel megegyezik

                def objective(weights):  # célfüggvény most a variancia
                    portfolio_var = np.dot(weights, np.dot(window_data.cov() * 252, weights))
                    return portfolio_var

                constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # korlátok az előbbi szerint
                bounds = tuple((0, 1) for _ in range(self.noa))
                initial_weights = np.array([1 / self.noa] * self.noa)
                result = sp.minimize(objective, initial_weights, method='SLSQP', bounds=bounds,
                                     constraints=constraints)  # variancia minimalizálása
                optimal_weights_df.loc[date] = result.x
                print(date)
        optimal_weights_df = optimal_weights_df.dropna()
        # innen ismét az előzővel megegyező
        portfolio_logreturns = np.log1p((self.returns_df * optimal_weights_df).dropna().sum(axis=1).astype(float))
        cum_returns = portfolio_logreturns.cumsum()
        portfolio_values = pd.DataFrame()
        portfolio_values["Min_var"] = np.exp(cum_returns.astype(float))
        return optimal_weights_df, portfolio_values

    def equal_weighted_portfolio(self):
        self.returns_df = self.returns_df[self.test_start_date:self.test_end_date]
        # itt a portfólió hozama az egyenlő súlyokat tartalmazó súlyvektorral súlyozva jön ki
        portfolio_logreturns = np.log1p(
            (self.returns_df * [0.2, 0.2, 0.2, 0.2, 0.2]).dropna().sum(axis=1).astype(float))
        cum_returns = portfolio_logreturns.cumsum()
        portfolio_values = pd.DataFrame()
        portfolio_values["eq_weight"] = np.exp(cum_returns.astype(float))
        return portfolio_values

    def calculate_portfolio_metrics(self, returns, var_conf=0.95):
        daily_returns = returns.pct_change().dropna()  # portfólióértékek visszaalakítva napi hozamra
        mean_return = np.mean(daily_returns) * 252  # átlagos évesített hozam
        std_dev = np.std(daily_returns) * np.sqrt(252)  # évesített szórás
        risk_free_rate = self.risk_free_df.loc[daily_returns.index]  # kockázatmentes hozamok
        excess_return = daily_returns.iloc[:, 0] - risk_free_rate  # többlethozamok
        sharpe_ratio = np.sqrt(252) * np.mean(excess_return) / np.std(excess_return)  # ex-post Sharpe-mutató
        # maximum drawdown
        comp_ret = (daily_returns + 1).cumprod()  # visszaalakítás portfólióértékre
        peak = comp_ret.expanding(min_periods=1).max()  # az adott napig számolt maximum
        dd = (comp_ret / peak) - 1  # drawdown: hozam a maximumhoz képest
        max_drawdown_since_last_peak = dd.min()  # minimum drawdown (mert negatív)
        # var
        q = (1 - var_conf) * 100  # nálam 5
        var = np.percentile(daily_returns, q)
        print(mean_return, std_dev[0], sharpe_ratio, max_drawdown_since_last_peak[0], var)
