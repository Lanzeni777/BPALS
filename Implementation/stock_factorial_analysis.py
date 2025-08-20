import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from DataBase.DBUtilities import BRVMDatabase


class FactorAnalytics:
    def __init__(self, n_components=3, start_date="2018-01-01"):
        self.n_components = n_components
        self.feature_names = None
        self.scaler = None
        self.pca_model = None
        self.pca_df = None
        self.start_date = start_date
        self.database = BRVMDatabase("KAN.db")

    def build_features(self, start_date=None):
        start_date = self.start_date if start_date is None else start_date
        price_df = self.database.get_prices(start_date=start_date)
        fundamental_df = self.database.get_fundamentals()
        market_cap_df = self.database.get_market_cap()

        fundamental_df = self.database.dp.clean_and_convert(
            fundamental_df.drop(columns=["Ticker", "Volume"]), fillna=True
        )
        price_df = self.database.dp.reset_df_index(price_df)

        fund_mcap = fundamental_df.merge(market_cap_df, on=['Date', 'Ticker'], how='left')
        df = price_df.merge(fund_mcap, on=['Date', 'Ticker'], how='left')
        df = df.groupby('Ticker', group_keys=False).apply(lambda x: x.ffill().bfill()).fillna(0.0)
        df = df.sort_values(['Ticker', 'Date'])

        df['Momentum_1M'] = df.groupby('Ticker')['Close'].pct_change(21)
        df['Momentum_3M'] = df.groupby('Ticker')['Close'].pct_change(63)
        df['Volatility_1W'] = df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change().rolling(5).std())
        df['Volatility_1M'] = df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change().rolling(21).std())
        df['Volume_5d'] = df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(5).mean())
        df['RelativePrice_1M'] = df.groupby('Ticker')['Close'].transform(lambda x: x / x.rolling(30).mean())

        df['Ret_1M_fwd'] = df.groupby('Ticker')['Close'].pct_change(periods=21).shift(-21)
        df['Ret_3M_fwd'] = df.groupby('Ticker')['Close'].pct_change(periods=63).shift(-63)
        df['Ret_1M_fwd'] = df['Ret_1M_fwd'].clip(-1, 1)

        self.feature_names = [
            'Momentum_1M', 'Momentum_3M', 'Volatility_1W', 'Volatility_1M',
            'Volume_5d', 'RelativePrice_1M', 'PER', 'Yield', 'floting_cap'
        ]

        return df.reset_index(drop=True)

    def run_pca(self, df):
        features_df = df.set_index(['Date', 'Ticker'])[self.feature_names]
        features_df = self.database.dp.impute_missing(features_df, method="mean").dropna()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features_df)
        self.pca_model = PCA(n_components=self.n_components)
        components = self.pca_model.fit_transform(X_scaled)

        pca_df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(self.n_components)])
        pca_df[['Date', 'Ticker']] = features_df.reset_index()[['Date', 'Ticker']]
        pca_df['Ret_1M_fwd'] = df.set_index(['Date', 'Ticker']).loc[features_df.index, 'Ret_1M_fwd'].values
        self.pca_df = pca_df
        return pca_df

    def rolling_pca(self, df=None, window_days=252):
        results = []
        df = self.pca_df.sort_values("Date") if df is None else df.sort_values("Date")
        unique_dates = df['Date'].drop_duplicates().sort_values()

        for i in range(window_days, len(unique_dates)):
            window_start = unique_dates.iloc[i - window_days]
            window_end = unique_dates.iloc[i]
            window_df = df[(df['Date'] >= window_start) & (df['Date'] <= window_end)].copy()

            try:
                pca_result = self.run_pca(window_df)
                pca_result["Window_End"] = window_end
                results.append(pca_result)
            except Exception as e:
                print(f"Skipping window ending {window_end}: {e}")

        self.rolling_pca_df = pd.concat(results).reset_index(drop=True) if results else pd.DataFrame()
        return self.rolling_pca_df

    def compute_pca_score(self, method="auto", weights=None):
        rolling_scores = []
        self.rolling_pca()
        for date, group in self.rolling_pca_df.groupby("Window_End"):
            # calcul de score par composantes pondérées
            corr = [group[[f'PC{i+1}', 'Ret_1M_fwd']].corr().iloc[0, 1] for i in range(self.n_components)]
            corr = [c if pd.notna(c) else 0 for c in corr]
            group["Rolling_PCA_Score"] = sum(group[f"PC{i+1}"] * corr[i] for i in range(self.n_components))
            rolling_scores.append(group)

        self.pca_scored_df = pd.concat(rolling_scores)
        return self.pca_scored_df

    def compute_supervised_score(self, df):
        features_df = df.set_index(['Date', 'Ticker'])[self.feature_names]
        features_df = self.database.dp.impute_missing(features_df, method="mean").dropna()
        df = df.set_index(['Date', 'Ticker']).loc[features_df.index]

        X = features_df.values
        y = df['Ret_1M_fwd'].clip(-1, 1).values

        model = LinearRegression()
        model.fit(X, y)
        df['Supervised_Score'] = model.predict(X)
        return df.reset_index()

    def check_sample_size(self):
        if self.pca_df is None:
            print("PCA not yet computed.")
            return
        n_obs = len(self.pca_df)
        n_features = len(self.feature_names)
        print(f"Sample size: {n_obs} rows for {n_features} features → ratio = {n_obs / n_features:.2f}")
        if n_obs < 5 * n_features:
            print("⚠️ Attention: faible profondeur d'échantillon pour une PCA fiable.")
        else:
            print("✅ Assez de données pour une PCA robuste.")

    def explained_variance(self):
        return self.pca_model.explained_variance_ratio_ if self.pca_model else None

    def get_completed_df(self, start_date="2018-01-01", how="inner"):
        df = self.build_features(start_date=start_date)
        self.run_pca(df)
        self.compute_pca_score()
        return df.merge(self.pca_df, on=["Date", "Ticker"], how=how)

    def do_factors_analysis(self, show=True):
        df = self.build_features()
        self.run_pca(df)
        self.compute_pca_score()
        if show:
            self.plot_pca_loadings()
            self.plot_pca_scatter()
        return self.pca_df

    def plot_pca_loadings(self):
        if not self.pca_model:
            return None
        loadings = pd.DataFrame(self.pca_model.components_.T,
                                columns=[f'PC{i+1}' for i in range(self.n_components)],
                                index=self.feature_names)
        plt.figure(figsize=(10, 5))
        sns.heatmap(loadings, annot=True, cmap="coolwarm")
        plt.title("PCA Loadings")
        plt.tight_layout()
        plt.show()
        return loadings

    def plot_pca_scatter(self, hue_col='PCA_Score'):
        if self.pca_df is None or 'PC1' not in self.pca_df.columns or 'PC2' not in self.pca_df.columns:
            return
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=self.pca_df, x='PC1', y='PC2', hue=hue_col,
            palette='coolwarm', edgecolor='k', alpha=0.8
        )
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        plt.title("PCA Component Space: PC1 vs PC2")
        plt.tight_layout()
        plt.show()
        return
    
    
    def backtest_from_rolling(self, df: pd.DataFrame, score_column: str = "PCA_Score", quantile=0.3):
        if "Window_End" not in df.columns:
            raise ValueError("'Window_End' must be present in the DataFrame for rolling backtest.")

        df = df.copy()
        df = df.dropna(subset=[score_column, 'Ret_1M_fwd'])
        df['group'] = df.groupby('Window_End')[score_column].transform(
            lambda x: pd.qcut(x.rank(method="first"), q=[0, quantile, 1 - quantile, 1], labels=['Bottom', 'Neutral', 'Top'])
        )
        df['strategy'] = df['group'].astype(str)

        # Éliminer les dates où il n'y a pas assez de titres dans les groupes Top/Bottom
        counts = df.groupby(['Window_End', 'strategy']).size().unstack().fillna(0)
        valid_dates = counts[(counts['Top'] > 3) & (counts['Bottom'] > 3)].index
        df = df[df['Window_End'].isin(valid_dates)]

        # Calcul des performances et lissage
        perf = df.groupby(['Window_End', 'strategy'])['Ret_1M_fwd'].mean().unstack()
        perf = (1 + perf).cumprod() - 1
        perf = perf.rolling(3).mean()  # Lissage pour atténuer les pics isolés

        plt.figure(figsize=(12, 6))
        for col in ['Top', 'Neutral', 'Bottom']:
            if col in perf.columns:
                plt.plot(perf.index, perf[col], label=col)
        plt.title("Backtest Cumulative Returns by Group")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

        return perf


class SupervisedScorer:
    def __init__(self, feature_names=None, start_date="2018-01-01"):
        self.start_date = start_date
        self.feature_names = feature_names
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.database = BRVMDatabase("KAN.db")

    def build_features(self, start_date=None):
        start_date = self.start_date if start_date is None else start_date
        price_df = self.database.get_prices(start_date=start_date)
        fundamental_df = self.database.get_fundamentals()
        market_cap_df = self.database.get_market_cap()

        fundamental_df = self.database.dp.clean_and_convert(
            fundamental_df.drop(columns=["Ticker", "Volume"]), fillna=True
        )
        price_df = self.database.dp.reset_df_index(price_df)

        fund_mcap = fundamental_df.merge(market_cap_df, on=['Date', 'Ticker'], how='left')
        df = price_df.merge(fund_mcap, on=['Date', 'Ticker'], how='left')
        df = df.groupby('Ticker', group_keys=False).apply(lambda x: x.ffill().bfill()).fillna(0.0)
        df = df.sort_values(['Ticker', 'Date'])

        df['Momentum_1M'] = df.groupby('Ticker')['Close'].pct_change(21)
        df['Momentum_3M'] = df.groupby('Ticker')['Close'].pct_change(63)
        df['Volatility_1W'] = df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change().rolling(5).std())
        df['Volatility_1M'] = df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change().rolling(21).std())
        df['Volume_5d'] = df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(5).mean())
        df['RelativePrice_1M'] = df.groupby('Ticker')['Close'].transform(lambda x: x / x.rolling(30).mean())

        df['Ret_1M_fwd'] = df.groupby('Ticker')['Close'].pct_change(periods=21).shift(-21)
        df['Ret_1M_fwd'] = df['Ret_1M_fwd'].clip(-1, 1)

        self.feature_names = [
            'Momentum_1M', 'Momentum_3M', 'Volatility_1W', 'Volatility_1M',
            'Volume_5d', 'RelativePrice_1M', 'PER', 'Yield', 'floting_cap'
        ]

        return df.reset_index(drop=True)

    def fit_predict(self, df=None):
        if df is None:
            df = self.build_features()

        df = df.dropna(subset=self.feature_names + ['Ret_1M_fwd'])
        X = df[self.feature_names]
        y = df['Ret_1M_fwd'].clip(-1, 1)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        df['Supervised_Score'] = self.model.predict(X_scaled)
        return df

    def backtest(self, df=None, quantile=0.3):
        if df is None:
            df = self.fit_predict()

        df = df.dropna(subset=['Supervised_Score', 'Ret_1M_fwd'])
        df = df.sort_values('Date')

        df['group'] = df.groupby('Date')['Supervised_Score'].transform(
            lambda x: pd.qcut(x.rank(method="first"), q=[0, quantile, 1 - quantile, 1], labels=['Bottom', 'Neutral', 'Top'])
        )
        df['strategy'] = df['group'].astype(str)

        perf = df.groupby(['Date', 'strategy'])['Ret_1M_fwd'].mean().unstack()
        perf = perf.clip(-0.25, 0.25)
        perf = (1 + perf).cumprod() - 1
        perf = perf.rolling(3).mean()

        plt.figure(figsize=(12, 6))
        for col in ['Top', 'Neutral', 'Bottom']:
            if col in perf.columns:
                plt.plot(perf.index, perf[col], label=col)
        plt.title("Supervised Score Backtest")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

        return perf
