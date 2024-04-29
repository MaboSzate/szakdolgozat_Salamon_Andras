import pandas as pd
import openpyxl
from PortfolioOptimizer import PortfolioOptimizer
import functions as f

step = 3
window = 252*5  # M paraméter (időablak nagysága, napokban)


def main(step):
    # 0.  PortfolioOptimizer objektum létrehozása, adatok betöltés
    portfolio = PortfolioOptimizer(train_start_date="2015-01-01", test_start_date='2019-01-01',
                                   test_end_date='2022-12-31')
    stock_symbols = ['IVV', 'AGG', 'GLD', 'USL', 'TIP']
    portfolio.load_data(stock_symbols, '^IRX') # a második paraméter a kockázatmentes eszköz szimbóluma

    # 1. Leíró statisztikák, korrelációs mátrix
    if step == 1:
        portfolio.descriptives("2009-01-02", "2018-12-31")

    # 2. Hiperparaméter-hangolás, modellválasztás
    if step == 2:
        for s in portfolio.symbols:
            portfolio.pipeline_search([["SVR", 10, 5], ["RF", 10, 5], ["XGB", 10, 5], ["LGB", 10, 5]], s)

    # 3. Portfólióoptimalizálás
    if step == 3:
        weights, values = portfolio.optimize_portfolio(method="LGB", return_lags=10, ho_lags=5, window=window,
                                                       importance='gain')
        # A fontosság kiszámítása beállítható ('gain' vagy 'split')
        values.to_excel("lgb_results.xlsx")  # kumulált portfólióértékek
        weights.to_excel("lgb_weights.xlsx")  # súlyvektorok

    # 4. Benchmarkok elkészítése
    if step == 4:
        # Sharpe-modell
        weights_sharpe, values_sharpe = portfolio.optimize_portfolio(method="Sharpe", window=window)
        values_sharpe.to_excel("sharpe_results.xlsx")
        weights_sharpe.to_excel("sharpe_weights.xlsx")
        # Minimum variancia modell
        values_mv = portfolio.optimize_portfolio_min_var(window=window)[1]
        values_mv.to_excel("min_var_results.xlsx")
        # Egyenlően súlyozott portfólió
        values_ew = portfolio.equal_weighted_portfolio()
        values_ew.to_excel("eq_weight_results.xlsx")

    # 5. Portfólióértékek és teljesítmény-mérőszámok összehasonlítása
    if step == 5:
        pfs = ["lgb", "sharpe", "min_var", "eq_weight"]
        dfs = []
        for pf in pfs:
            df = pd.read_excel(pf + "_results.xlsx", index_col=0)
            portfolio.calculate_portfolio_metrics(df)
            dfs.append(df)
        f.plot_pf_values(dfs[0], dfs[1], dfs[2], dfs[3])

    # 6. Súlyvektorok elemzése
    if step == 6:
        portfolio.plot_prices("2019-01-02", "2022-12-31")  # tesztidőszaki eszközértékek
        f.plot_weights("lgb_weights.xlsx")
        f.plot_weights("sharpe_weights.xlsx")




if __name__ == '__main__':
    main(step)
