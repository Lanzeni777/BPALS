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
