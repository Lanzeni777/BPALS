import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from DataBase.DBUtilities import BRVMDatabase


class FactorAnalytics:
    def __init__(self, n_components=3):
        self.pca_df = None
        self.feature_names = None
        self.available_data = None
        self.n_components = n_components
        self.pca_model = None
        self.scaler = None
        self.database = BRVMDatabase("KAN.db")

    def build_features(self, prices: pd.DataFrame = None, fundamentals: pd.DataFrame = None,
                       market_cap: pd.DataFrame = None, start_date="2018-01-01"):
        price_df = prices if not prices is None else self.database.get_prices(start_date=start_date)
        fundamental_df = fundamentals if not fundamentals is None else self.database.get_fundamentals()
        market_cap_df = market_cap if not market_cap is None else self.database.get_market_cap()
        # market_cap_df = market_Cap[[""]]
        fundamental_df = self.database.dp.clean_and_convert(fundamental_df.drop(["Ticker", "Volume"], axis=1),
                    fillna=True)  # self.database.dp.reset_df_index(fundamental_df) # .sort_values(['Ticker', 'Date'])
        price_df = self.database.dp.reset_df_index(price_df)
        # fundamental_df = fundamental_df.groupby('Ticker').ffill()
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

        feature_cols = ['Ticker', 'Momentum_1M', 'Momentum_3M', 'Volatility_1W', 'Volatility_1M',
                        'Volume_5d', 'RelativePrice_1M', 'PER', 'Yield', 'floting_cap']
        self.available_data = df

        return df.reset_index(drop=True), feature_cols

    def prepare_data(self, features_df):
        df = features_df.dropna()
        X = features_df.copy()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        nb_nan = pd.DataFrame(X_scaled).isna().sum()
        nb_inf = np.isinf(pd.DataFrame(X_scaled)).sum()
        return X_scaled, df

    def run_pca(self, df, feature_cols, fillna_method="mean"):
        self.feature_names = feature_cols
        features_df = df.set_index(['Date', 'Ticker'], inplace=False)[feature_cols]
        check = features_df.dtypes
        features_df = self.database.dp.impute_missing(features_df, fillna_method).dropna()
        df_nan = features_df[features_df.isna().any(axis=1)]
        X_scaled, df_clean = self.prepare_data(features_df)
        self.pca_model = PCA(n_components=self.n_components)
        principal_components = self.pca_model.fit_transform(X_scaled)
        pc_df = pd.DataFrame(data=principal_components,
                             columns=[f'PC{i + 1}' for i in range(self.n_components)])
        return pd.concat([df_clean.reset_index(drop=False), pc_df], axis=1)

    def explained_variance(self):
        if self.pca_model:
            return self.pca_model.explained_variance_ratio_
        return None

    def plot_pca_loadings(self):
        if self.pca_model is None:
            print("PCA model not trained.")
            return

        loadings = pd.DataFrame(self.pca_model.components_.T,
                                columns=[f'PC{i + 1}' for i in range(self.n_components)],
                                index=self.feature_names)
        plt.figure(figsize=(10, 5))
        sns.heatmap(loadings, annot=True, cmap="coolwarm")
        plt.title("PCA Loadings")
        plt.tight_layout()
        plt.show()
        return loadings

    def do_factors_analysis(self, feat_cols, show=True):
        df, col = self.build_features()
        pca_df = self.run_pca(df, feat_cols)
        if show:
            self.explained_variance()
            self.plot_pca_loadings()
        return pca_df

    def _auto_score_direction(self):
        corr = []
        for i in range(self.n_components):
            pc = f'PC{i + 1}'
            c = self.pca_df[[pc, 'Ret_1M_fwd']].dropna().corr().iloc[0, 1]
            corr.append(c if pd.notna(c) else 0)

        score = sum(self.pca_df[f'PC{i + 1}'] * corr[i] for i in range(self.n_components))
        self.pca_df['PCA_Score'] = score

    def compute_pca_score(self, pca_df=None, weights=None, method="auto"):
        if method.lower() != "auto":
            pca_df = self.pca_df if pca_df is None else pca_df
            if weights is None:
                weights = [1 + 0.5*(self.n_components - 1)] + [-0.5] * (self.n_components - 1)  # default: favor PC1, penalize PC2, PC3
            score = sum(pca_df[f'PC{i + 1}'] * weights[i] for i in range(self.n_components))
            pca_df['PCA_Score'] = score
            return pca_df
        self._auto_score_direction()
        return self.pca_df

    def rank_latest_scores(self, features_cols, pca_df=None):
        self.pca_df = self.do_factors_analysis(features_cols, False) if pca_df is None else pca_df
        pca_score = self.compute_pca_score(self.pca_df)
        latest = pca_score.sort_values('Date').groupby('Ticker').tail(1)
        return latest.sort_values('PCA_Score', ascending=False)  # [["Ticker", "PC1", "PC2", "PC3", "PCA_Score"]]

    def get_completed_df(self, feat_cols, start_date="2015-01-01", how="inner"):
        df, _ = self.build_features(start_date=start_date)
        self.pca_df = self.do_factors_analysis(feat_cols, False)
        pca_score = self.compute_pca_score()
        return df.merge(pca_score, on=["Date", "Ticker"], how=how)

    def plot_pca_scatter(self, pca_df=None, hue_col='PCA_Score'):
        pca_df = self.rank_latest_scores(self.feature_names) if pca_df is None else pca_df
        if 'PC1' not in pca_df.columns or 'PC2' not in pca_df.columns:
            print("PC1 and PC2 not available for plotting.")
            return

        plt.figure(figsize=(10, 7))
        scatter = sns.scatterplot(
            data=pca_df, x='PC1', y='PC2', hue=hue_col,
            palette='coolwarm', edgecolor='k', alpha=0.8
        )
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        plt.title("PCA Component Space: PC1 vs PC2")
        plt.tight_layout()
        plt.show()




##############################################################################

import sqlite3
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import os, sys
import urllib3
import time
import unicodedata
from IPython.display import display


# DATA PREPROCESSOR
def get_prices_freq(self, intervalle=15, nb_time=20,
                    end_time=datetime.now().replace(hour=16, minute=30, second=0, microsecond=0)):
    if datetime.now() > end_time:
        print(datetime.today().strftime('%Y-%m-%d  :  %I:%M:%Ss PM'), " -  It's past 4:30 PM.")
        return
    nb = 0
    while datetime.now() <= end_time and nb < nb_time:
        df_price, df_cap, df_per = self.get_data_today()
        d = datetime.today().strftime('%Y-%m-%d-%Ih%Mm%Ss')
        df_price.to_csv(f'../Stocks_Prices/Continous_Prices/brvm_last_prices_{d}.csv')
        df_per.to_csv(f'../Stocks_Prices/Continous_PER/brvm_last_per_{d}.csv')
        df_cap.to_csv(f'../Stocks_Prices/Continous_Cap/brvm_last_cap_{d}.csv')

        time.sleep(int(1))  # 15 minutes in seconds
        nb += 1


def insertion_format(self, df: pd.DataFrame, data_type="Price"):
    if data_type.upper() == "PER":
        df.columns = ["Stock_Description", "Nombre_titres_echanges", "Valeur_echangee",
                      "PER", "Valeur_globale_echangee_pct", "Date"]
    elif data_type.upper() == "CAPITALISATION":
        df.columns = ["Stock_Description", "Nombre_de_titres", "Cours_du_jour", "Capitalisation_flottante",
                      "Capitalisation_globale", "Capitalisation_globale_pct", "Date"]
    else:
        df["Ticker"] = list(df.index)
        # df = df[["Date", "Price"]]
        # df = df_insertion[["Date", "Ticker", "Price"]]

    df_insertion = df.reset_index(inplace=False, drop=False)  # if df.index.name in ["Date", "Ticker"] else df
    display(df_insertion)

    return df_insertion

# BRVM DATA BASE


def DEPRECATED_load_available_data(self):
    df_p = pd.read_excel(f"{self.root}/../Stocks_Prices/Global_Historic/Cours_titres.xlsx")
    df_sorted = df_p.sort_index(ascending=False)
    df_sorted.columns = ["Date"] + list(df_sorted.columns[1:])
    df_melt = df_sorted.melt(id_vars='Date', var_name='Ticker', value_name='Price')
    # df_sorted = df_sorted.drop("Date/Société", axis=1)
    # Insertion
    self._connect()
    try:
        df_melt.to_sql("stock_prices", self.conn, if_exists="append", index=False)
    except:
        pass
    self.close()

def DEPRECATED_insert_today_price(self):
    old_df = self.get_prices()
    data = self.dp.get_data_today()
    name = ["Price", "capitalisation", "PER"]
    for i, t in enumerate(self.tables):
        try:
            self.insert_dataframe(self.dp.insertion_format(data[i], name[i]), t)
        except:
            print(f"Data in the table {t} has been already updated today")
    df = self.get_prices()
    return df[~df.isin(old_df)].dropna(how='all')
