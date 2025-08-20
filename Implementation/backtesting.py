import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from Implementation.stock_factorial_analysis import FactorAnalytics
from Implementation.stock_picking_heuristic import StockEvaluator  # suppose que ces classes existent


class Backtester:
    def __init__(self, database, n_components=3, quantile=0.2, start_date="2015-01-01", method="pca"):
        self.n_components = n_components
        self.quantile = quantile
        self.ticker_list = list(database.dp.tickers_manager.get_all_stocks_tickers()["Ticker"])
        self.factors_analyser = FactorAnalytics(method)
        self.feature_cols = features_cols = ['Momentum_1M', 'Momentum_3M', 'Volatility_1W', 'Volatility_1M',
                                             'Volume_5d', 'RelativePrice_1M', 'PER', 'Yield', 'floting_cap']
        self.pca_df = self.factors_analyser.get_completed_df(start_date)

    def backtest_score(self):
        df = self.pca_df.dropna(subset=['PCA_Score', 'Ret_1M_fwd']).copy()[['Date', 'Ticker', 'PCA_Score', 'Ret_1M_fwd']]
        df = df[df['Ret_1M_fwd'].between(-1, 1)]
        df['rank'] = df.groupby('Date')['PCA_Score'].rank(pct=True)
        df['strategy'] = 'Neutral'
        df.loc[df['rank'] >= 1 - self.quantile, 'strategy'] = 'Top'
        df.loc[df['rank'] <= self.quantile, 'strategy'] = 'Bottom'
        return df.groupby('strategy')['Ret_1M_fwd'].describe()

    def cumulative_returns(self):
        df = self.pca_df.dropna(subset=['PCA_Score', 'Ret_1M_fwd']).copy()
        df['rank'] = df.groupby('Date')['PCA_Score'].rank(pct=True)
        df['group'] = 'Neutral'
        df.loc[df['rank'] >= 1 - self.quantile, 'group'] = 'Top'
        df.loc[df['rank'] <= self.quantile, 'group'] = 'Bottom'
        df = df.groupby(['Date', 'group'])['Ret_1M_fwd'].mean().reset_index()
        pivot = df.pivot(index='Date', columns='group', values='Ret_1M_fwd').fillna(0)
        return (1 + pivot).cumprod()

    def plot_cumulative_returns(self):
        cumret_df = self.cumulative_returns()
        plt.figure(figsize=(10, 6))
        cumret_df.plot(ax=plt.gca())
        plt.title("Cumulative Returns by PCA Score Group")
        plt.ylabel("Cumulative Return")
        plt.xlabel("Date")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
