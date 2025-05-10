import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import os
import urllib3
import time
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

#import investpy
# Liste des tickers que tu veux surveiller
tickers = ['ORGT', 'PALC', 'STAC', 'CFAC', 'ABJC', 'SIBC', 'ETIT', 'BOAB', 'UNLC']

# URL de base pour les données BRVM
url_price = 'https://www.brvm.org/fr/cours-actions/0'
url_per = 'https://www.brvm.org/fr/volumes/0'
url_capitalisation = 'https://www.brvm.org/fr/capitalisations/0'
# Faire la requête HTTP

def get_info(url):
    r = requests.get(url, verify=False)
    soup = BeautifulSoup(r.content, 'html.parser')
    rows = soup.select('table tbody tr')
    return rows

def get_col_from_rows(rows, col_id, date=None, exclude=["Valeur des transactions", "Capitalisation Actions", "Capitalisation des obligations"]):
    data = []
    date = datetime.today().strftime('%Y-%m-%d') if date is None else date
    for row in rows:
        cols = row.find_all('td')
        #print(cols, len(cols))
        if len(cols) < 5:
            continue
        code = cols[0].text.strip()
        if code in exclude:
            continue
        d = {"Ticker":code}
        for c in col_id:
            val = cols[c[1]].text.strip()
            val = val if c[2].lower() != 'numeric' else float(val.replace(',', '').replace(' ', ''))
            d[c[0]] = val
        d["Date"] = date
        data.append(d)
    # Créer le DataFrame
    df = pd.DataFrame(data)
    df.set_index('Ticker', inplace=True)
    return df


def get_prices_freq(intervalle=15, nb_time=20, end_time=datetime.now().replace(hour=16, minute=30, second=0, microsecond=0)):
    
    if datetime.now() > end_time:
        print(datetime.today().strftime('%Y-%m-%d  :  %I:%M:%Ss PM'), " -  It's past 4:30 PM.")
        return
    nb = 1
    while datetime.now() <= end_time and nb < 20 :
        rows = get_info(url_price)
        columns = [('Stock Description', 1, 'string'), ('Price J-1', 3, 'numeric'), ('Price J', 5, 'numeric')]
        df = get_col_from_rows(rows, columns)
        d = datetime.today().strftime('%Y-%m-%d-%Ih%Mm%Ss')
        df.to_csv(f'../Stocks_Prices/Continous_Prices/brvm_last_prices_{d}.csv')
        time.sleep(int(15.5 * 60))  # 15 minutes in seconds
        nb += 1

get_prices_freq(end_time = datetime.now().replace(hour=16, minute=45, second=0, microsecond=0))
input("\n Tape to end the programme\n")