import sqlite3
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import os, sys
import urllib3
import time
#!pip install PyPDF2

class BRVMDatabase:
    def __init__(self, db_name="KLBrvm_BataBase.db", root_path="C:\\Users\\charl\\Documents\\KLDocs\\Investment\\DataBase"):
        self.root = root_path
        self.db_name = f"{self.root}\\{db_name}"
        self._init_db()
        self._connect()

    def _connect(self):
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        #return self.conn, self.cursor

    def _init_db(self, force=False):

        if os.path.exists(self.db_name) and not force:
            return 0
        
        self._connect()

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            Date TEXT NOT NULL,
            Ticker TEXT NOT NULL,
            Price REAL,
            PRIMARY KEY (Date, Ticker)
        )
        """)

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS capitalisations (
                Ticker TEXT,
                Stock_Description TEXT,
                Nombre_de_titres REAL,
                Cours_du_jour REAL,
                Capitalisation_flottante REAL,
                Capitalisation_globale REAL,
                Capitalisation_globale_pct REAL,
                Date TEXT,
                PRIMARY KEY (Ticker, Date)
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS per_data (
                Ticker TEXT,
                Stock_Description TEXT,
                Nombre_titres_echanges REAL,
                Valeur_echangee REAL,
                PER REAL,
                Valeur_globale_echangee_pct REAL,
                Date TEXT,
                PRIMARY KEY (Ticker, Date)
            )
        ''')

        self.conn.commit()
        self.close()
        

    def validate_dataframe(self, df: pd.DataFrame, table: str):
        required_fields = {
            'stock_prices': {'Date', 'Ticker', 'Price'},
            'capitalisations': {'Date', 'Ticker', 'Stock_Description'},
            'per_data': {'Date', 'Ticker', 'Stock_Description'}
        }
        if table not in required_fields:
            raise ValueError("Table inconnue")
        df_cols = set(df.columns)
        missing = required_fields[table] - df_cols
        if missing:
            raise ValueError(f"Colonnes manquantes pour {table}: {missing}")

        if 'Date' in df.columns:
            try:
                pd.to_datetime(df['Date'])
            except Exception as e:
                raise ValueError("Date non convertible: " + str(e))

    def insert_dataframe(self, df: pd.DataFrame, table: str):
        self.validate_dataframe(df, table)
        df = df.copy()
        df.reset_index(inplace=True, drop=False) if df.index.name in ["Date", "Ticker"] else None
        self._connect()
        try:
            df.to_sql(table, self.conn, if_exists='append', index=False)
        except Exception as e:
            raise RuntimeError(f"Erreur d'insertion: {e}")
        finally:
            self.close()
        self.close()
        
    def insert_dataframe2(self, df: pd.DataFrame, table_name: str):
        if df.empty:
            print("[WARNING] DataFrame is empty.")
            return
        self._connect()
        required_cols = set(self._get_table_columns(table_name))
        df = df.reset_index(drop=True)  # ⚠️ Important pour éviter l'insertion d'index
        incoming_cols = set(df.columns)

        if not required_cols.issubset(incoming_cols):
            raise ValueError(f"[ERROR] Incoming columns do not match table '{table_name}' schema. Required: {required_cols}, Got: {incoming_cols}")
        df = df[list(required_cols)]  # Assure l'ordre et filtre si besoin
        df.to_sql(table_name, self.conn, if_exists='append', index=False)
        self.close()

    def get_prices(self, tickers=None, start_date=None, end_date=None):
        query = "SELECT * FROM stock_prices WHERE 1=1"
        params = []
        if tickers:
            query += " AND Ticker IN ({})".format(",".join(["?"] * len(tickers)))
            params.extend(tickers)
        if start_date:
            query += " AND Date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND Date <= ?"
            params.append(end_date)

        self._connect()
        df = pd.read_sql_query(query, self.conn, params=params, parse_dates=['Date'])
        df.sort_values(by="Date", ascending=False)
        self.close()
        return df

    def clear_table(self, table):
        self._connect()
        self.cursor.execute(f"DELETE FROM {table}")
        self.conn.commit()
        self.close()

    def status(self, tables = ['stock_prices', 'capitalisations', 'per_data']):
        self._connect()
        print("Table status (rows):")
        for t in tables:
            self.cursor.execute(f"SELECT COUNT(*) FROM {t}")
            print(f"{t} : {self.cursor.fetchone()[0]} rows")
        self.close()
        
    def get_table_columns(self, table_name):
        db_test._connect()
        q = f"PRAGMA table_info({table_name});"#"SELECT * FROM stock_prices"
        #db_test.get_prices("ABJC", "2024-04-18", "2025-04-21")
        res_ = db_test.cursor.execute(q)
        res = [e for e in res]
        db_test.close()
        return res
    
    def delete_last_insert(self, table):
        self._connect()
        self.cursor.execute(f"SELECT MAX(Date) FROM {table}")
        last_date = self.cursor.fetchone()[0]
        if last_date:
            self.cursor.execute(f"DELETE FROM {table} WHERE Date = ?", (last_date,))
            print(f"Supprimée: {table} pour la date {last_date}")
        self.conn.commit()
        self.conn.close()
        
    def execute_query(self, query: str, params: tuple = ()):
        self._connect()
        res = self.cursor.execute(query, params)
        val = [r for r in res]
        self.conn.commit()
        self.close()
        return val

    def get_value(self, table: str, columns='*', condition: str = '', order_by: str = '', limit: int = None):
        self._connect()
        query = f"SELECT {columns} FROM {table}"
        if condition:
            query += f" WHERE {condition}"
        if order_by:
            query += f" ORDER BY {order_by}"
        if limit:
            query += f" LIMIT {limit}"
        return pd.read_sql_query(query, self.conn)
        self.close()

    def get_all_tables(self):
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        self._connect()
        res = [row[0] for row in self.cursor.execute(query)]
        self.close()
        return res

    def set_value(self, table: str, set_clause: str, condition: str):
        query = f"UPDATE {table} SET {set_clause} WHERE {condition}"
        self.execute_query(query)

    def delete_value(self, table: str, condition: str):
        query = f"DELETE FROM {table} WHERE {condition}"
        self.execute_query(query)

    def delete_and_create(self, table: str = None):
        tables = self.get_all_tables() if table is None else [table]
        self._connect()
        for t in tables:
            self.cursor.execute(f"DROP TABLE IF EXISTS {t}")
        self.conn.commit()
        self._init_db()
        self.close()

    def _get_table_columns(self, table: str):
        self._connect()
        self.cursor.execute(f"PRAGMA table_info({table})")
        res = [row[1] for row in self.cursor.fetchall()]
        self.close()
        return res

    def status(self):
        tables = self.get_all_tables()
        self._connect()
        print("[DATABASE STATUS]")
        for t in tables:
            self.cursor.execute(f"SELECT COUNT(*) FROM {t}")
            count = self.cursor.fetchone()[0]
            print(f"Table '{t}': {count} rows")

    def close(self):
        self.conn.close()
        
    def load_available_data(self):
        df_p = pd.read_excel(f"{self.root}/../Stocks_Prices/Global_Historic/Cours_titres.xlsx")
        df_sorted = df_p.sort_index(ascending=False)
        df_sorted.columns = ["Date"] + list(df_sorted.columns[1:])
        df_melt = df_sorted.melt(id_vars='Date', var_name='Ticker', value_name='Price')
        #df_sorted = df_sorted.drop("Date/Société", axis=1)
        # Insertion
        self._connect()
        try:
            df_melt.to_sql("stock_prices", self.conn, if_exists="append", index=False)
        except:
            pass
        self.close()
        
    def insert_today_price(self):
        dp = DataPreProcessor()
        data = dp.get_data_today()
        tables = ["stock_prices", "capitalisations", "per_data"]
        name = ["Price", "capitalisation", "PER"]
        for i, t in enumerate(tables):
            self.insert_dataframe(dp.insertion_format(data[i], name[i]), t)

            
class DataPreProcessor :
    
    def __init__(self):
        # Liste des tickers que tu veux surveiller
        self.tickers = ['ORGT', 'PALC', 'STAC', 'CFAC', 'ABJC', 'SIBC', 'ETIT', 'BOAB', 'UNLC']
        # URL de base pour les données BRVM
        self.url_price = 'https://www.brvm.org/fr/cours-actions/0'
        self.url_per = 'https://www.brvm.org/fr/volumes/0'
        self.url_capitalisation = 'https://www.brvm.org/fr/capitalisations/0'
        # Faire la requête HTTP

    def get_info(self, url):
        r = requests.get(url, verify=False)
        soup = BeautifulSoup(r.content, 'html.parser')
        #print(soup)
        rows = soup.select('table tbody tr')
        return rows

    def get_col_from_rows(self, rows, col_id, date=None,
                          exclude=["Valeur des transactions", "Capitalisation Actions", "Capitalisation des obligations"]):
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
                try:
                    val = val if c[2].lower() != 'numeric' else float(val.replace(',', '.').replace(' ', ''))
                except:
                    #print(code, val)
                    pass
                d[c[0]] = val
            d["Date"] = date
            data.append(d)
        # Créer le DataFrame
        df = pd.DataFrame(data)
        df.set_index('Ticker', inplace=True)
        return df

    def get_data_today(self):
        rows = self.get_info(self.url_price)
        columns = [('Stock Description', 1, 'string'), ('Price J-1', 3, 'numeric'), ('Price', 5, 'numeric')]
        df = self.get_col_from_rows(rows, columns)

        rows2 = self.get_info(self.url_capitalisation)
        columns2 = [('Stock Description', 1, 'string'), ("Nombre de titres", 2, "numeric"),
                ("Cours du jour", 3, "numeric"), ("Capitalisation flottante", 4, "numeric"),
                ("Capitalisation globale", 5, "numeric"), ("Capitalisation globale (%)", 6, "numeric")]
        df2 = self.get_col_from_rows(rows2, columns2)

        rows3 = self.get_info(self.url_per)
        columns3 = [('Stock Description', 1, 'string'), ("Nombre de titres échangés", 2, "numeric"),
                ("Valeur échangée", 3, "numeric"), ("PER", 4, "numeric"),
                ("valeur globale échangée %", 5, "numeric")]
        df3 = self.get_col_from_rows(rows3, columns3)

        return df, df2, df3

    def get_prices_freq(self, intervalle=15, nb_time=20, end_time=datetime.now().replace(hour=16, minute=30, second=0, microsecond=0)):

        if datetime.now() > end_time:
            print(datetime.today().strftime('%Y-%m-%d  :  %I:%M:%Ss PM'), " -  It's past 4:30 PM.")
            return
        nb = 0
        while datetime.now() <= end_time and nb < nb_time :
            df_price, df_cap, df_per = self.get_data_today()
            d = datetime.today().strftime('%Y-%m-%d-%Ih%Mm%Ss')
            df.to_csv(f'../Stocks_Prices/Continous_Prices/brvm_last_prices_{d}.csv')
            df.to_csv(f'../Stocks_Prices/Continous_PER/brvm_last_per_{d}.csv')
            df.to_csv(f'../Stocks_Prices/Continous_Cap/brvm_last_cap_{d}.csv')

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
            df = df[["Date", "Price"]]
            #df = df_insertion[["Date", "Ticker", "Price"]]
        
        df_insertion = df.reset_index(inplace=False, drop=False) #if df.index.name in ["Date", "Ticker"] else df

        return df_insertion
    
