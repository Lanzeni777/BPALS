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


class BRVMDatabase:
    def __init__(self, db_name="KLBrvm_DataBase.db",
                 root_path="C:\\Users\\charl\\Documents\\KLDocs\\BPALS\\DataBase"):
        self.root = root_path
        self.db_name = f"{self.root}\\{db_name}"
        self._init_db()
        self._connect()
        self.dp = DataPreProcessor()
        self.tables = ["stock_prices", "capitalisations", "per_data"]

    def _init_db(self, force=False):

        # if os.path.exists(self.db_name) and not force:
        #    return 0
        col = [('Date', 'TEXT NOT NULL'), ('Ticker', 'TEXT NOT NULL'), ('Description', 'TEXT NOT NULL'),
               ('Open', 'REAL'), ('High', 'REAL'), ('Low', 'REAL'), ('Close', 'REAL'), ('Volume', 'REAL')]
        self.create_table("stock_prices", col, ["Date", "Ticker"])
        self.create_table("indices_performances", col, ["Date", "Ticker"])

        col = [('Ticker', 'TEXT NOT NULL'), ('Date', 'TEXT NOT NULL'), ('stock_description', 'REAL'),
               ('stock_number', 'REAL'), ('floting_cap', 'REAL'), ('globale_cap', 'REAL'), ('globale_cap_pct', 'REAL'),
               ('trade_number', 'REAL'), ('trade_value', 'REAL'), ('globale_trade_value_pct', 'REAL'), ('PER', 'REAL')]
        self.create_table("market_cap", col, ["Date", "Ticker"])

        col = [('Ticker', 'TEXT NOT NULL'), ('Stock_Description', 'REAL'), ('Nombre_titres_echanges', 'REAL'),
               ('Valeur_echangee', 'REAL'), ('PER', 'REAL'), ('Valeur_globale_echangee_pct', 'REAL'),
               ('Date', 'TEXT NOT NULL')]
        self.create_table("per_data", col, ["Date", "Ticker"])

        col = [('Ticker', 'TEXT NOT NULL'), ('Volume', 'REAL'), ('Next Earnings Date', 'REAL'),
               ('Market Cap', 'REAL'), ('Revenue', 'REAL'), ("Average_3m Volume", 'REAL'), ('EPS', 'REAL'),
               ('P_E Ratio', 'REAL'), ('Beta', 'REAL'), ('Dividend', 'REAL'), ('Yield', 'REAL'),
               ('Daily Trend', 'REAL'), ('Weekly Trend', 'REAL'), ('Monthly Trend', 'REAL'), ('Day perf', 'REAL'),
               ('Week perf', 'REAL'), ('Month perf', 'REAL'), ('YTD', 'REAL'), ('Year perf', 'REAL'),
               ('Three_Years perf', 'REAL'), ('Date', 'TEXT NOT NULL')]
        self.create_table("fundamental_data", col, ["Date", "Ticker"])

    """##################################  DATA BASE STATE #################################"""

    def _connect(self):
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        # return self.conn, self.cursor

    def create_table(self, table_name: str, columns: list, primary_key: list = None):
        """
        Crée une table SQLite avec les colonnes spécifiées.

        Args:
            table_name (str): nom de la table à créer
            columns (list of tuple): liste de paires (nom_colonne, type_sqlite)
        """
        self._connect()
        cols_def = ",\n    ".join([f"{name} {dtype}" for name, dtype in columns])
        pk_clause = f",\n    PRIMARY KEY ({', '.join(primary_key)})" if primary_key else ""
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n    {cols_def}{pk_clause}\n);"
        if table_name not in self.get_all_tables():
            try:
                self.cursor.execute(sql)
                self.conn.commit()
                print(f"✅ Table '{table_name}' successfully  created.")
            except Exception as e:
                print(f"❌ Could not create the table '{table_name}':", e)
            finally:
                self.close()

    def status(self, tables=None):
        print("Table status (rows):")
        tables = self.get_all_tables() if tables is None else tables
        res = {}
        for t in tables:
            self._connect()
            self.cursor.execute(f"SELECT COUNT(*) FROM {t}")
            fetch = f"{self.cursor.fetchone()[0]} rows of {self._get_table_columns(t)}"
            print(f"{t} : {fetch}")
            res[t] = fetch
        self.close()
        return res

    def check_status(self):
        r = self.status()
        if int(r["stock_prices"].split()[0]) == 0:
            self.insert_3m_histo_data()
            print("\n\nData Base fill with the last three month data as it was empty\n\n")
            r = self.status()
        return r

    def check_ticker_missing(self, date1, date2):
        f = self.execute_query(f"SELECT Date,Ticker,Close FROM stock_prices WHERE Date > '2025-05-30'")
        t1 = list(f[f["Date"] == "2025-05-30 00:00:00"]["Ticker"])
        t2 = list(f[f["Date"] == "2025-06-02 00:00:00"]["Ticker"])
        res = []
        for e in t1:
            if not e in t2:
                res.append(e)
        print(res)
        return res

    def execute_query(self, query: str, params: tuple = ()):
        self._connect()
        res = pd.read_sql(query, self.conn) if query.split()[0].upper() == "SELECT" else self.cursor.execute(query,
                                                                                                             params)
        res = self.dp.clean_and_convert(res) if query.split()[0].upper() == "SELECT" else res
        # val = [r for r in res]
        # res = pd.read_sql(query, self.conn)
        self.conn.commit()
        self.close()
        return res

    def find_duplicated(self):
        df = self.execute_query(
            "SELECT Date, Ticker, COUNT(*) FROM stock_prices GROUP BY date, ticker HAVING COUNT(*) > 1;")
        if not df.empty:
            return df
        print("No duplicated row found")
        return df

    def close(self):
        self.conn.close()

    def validate_dataframe(self, df: pd.DataFrame, table: str):
        tables = self.get_all_tables()
        if table not in tables:
            raise ValueError("Table inconnue")
        df_cols, db_table_cols = set(df.columns), set(self._get_table_columns(table))

        missing1 = db_table_cols - df_cols
        if missing1:
            print(f"Columns missing in {table}: {missing1}")
        missing2 = df_cols - db_table_cols
        if missing2:
            print(f"Unknown columns in {table}: {missing2}")

        if 'Date' in df.columns:
            try:
                pd.to_datetime(df['Date'])
            except Exception as e:
                print("Date non convertible: " + str(e))
        return not (missing1 or missing2)

    def update(self):
        print("=====================Before daily update:=============================\n")
        self.status()
        print("\n=====================Begin daily update:=============================\n")
        self.insert_daily_per()
        self.insert_daily_prices()
        self.insert_daily_indices()
        self.insert_daily_market_cap()
        self.insert_weekly_fundamental_data()

        print("\n\n=============================After daily update:=======================\n")
        self.status()

    """##################################  DELETE/UNDO #################################"""

    def _remove_last_commit(self):
        self._connect()
        self.conn.rollback()
        self.close()

    def clear_table(self, table):
        self._connect()
        self.cursor.execute(f"DELETE FROM {table}")
        self.conn.commit()
        self.close()

    def delete_table(self, table):
        confirm = input(f"You are about to permanantly delete table {table} \n Which contains"
                        f"{self.cursor.execute(f'SELECT COUNT(*) FROM {table}')} rows of {self._get_table_columns(table)}")
        if confirm.lower() in ["yes", "oui", "y"]:
            self.execute_query(f"DROP TABLE {table}")

    def delete_last_insert(self, table):
        self._connect()
        self.cursor.execute(f"SELECT MAX(Date) FROM {table}")
        last_date = self.cursor.fetchone()[0]
        if last_date:
            self.cursor.execute(f"DELETE FROM {table} WHERE Date = ?", (last_date,))
            print(f"Supprimée: {table} pour la date {last_date}")
        self.conn.commit()
        self.conn.close()

    def delete_and_create(self, table: str = None, force: bool = False):
        if self.find_duplicated().empty and not force:
            return None
        tables = self.get_all_tables() if table is None else [table]
        for table in tables:
            self.execute_query(f"CREATE TABLE IF NOT EXISTS table_clean AS SELECT DISTINCT * FROM {table}")
            self.execute_query(f"DROP TABLE {table}")
            self.execute_query(f"ALTER TABLE table_clean RENAME TO {table}")
        # self._init_db()
        self.close()
        return tables

    def delete_value(self, table: str, condition: str):
        query = f"DELETE FROM {table} WHERE {condition}"
        self.execute_query(query)

    """##################################  SETTER/INSERTION #################################"""

    def set_value(self, table: str, set_clause: str, condition: str):
        query = f"UPDATE {table} SET {set_clause} WHERE {condition}"
        self.execute_query(query)

    def insert_dataframe(self, df: pd.DataFrame, table: str):
        df = self.get_new_data_from_df(df, table)
        if df.empty:
            print(u'\N{check mark}' + f" No new daily data to insert into <<{table}>> table. \
                                All entries already exist.")
            return None

        df = df[self._get_table_columns(table)]
        df = self.dp.reset_df_index(df) if df.index.name in ["Date", "Ticker"] else df
        df = df.drop("index", axis=1) if "index" in df else df
        # df = df.drop("Variation", axis=1) if "Variation" in df else df
        if self.validate_dataframe(df, table):
            df = df.copy()
            self._connect()
            try:
                df.to_sql(table, self.conn, if_exists='append', index=False)
                print('✅' + f" {len(df)} new rows inserted into table {table}.\n")
            except Exception as e:
                print(
                    u'\N{cross mark}' f" Failed to insert data {df.columns}in database table {table}, maybe data already exist")
                print("oups exception : ", e)
            finally:
                self.close()
            self.close()
        else:
            print(f"Failed to insert data in database table {table}")
            print(f"please check the format and try again : {df.columns}")

    def insert_daily_prices(self):
        df = self.get_last_prices()
        if not df.empty:
            self.insert_dataframe(df, 'stock_prices')
        else:
            print(u'\N{check mark}' + f" No new daily data to insert into <<stock_prices tables>>.\
             All entries already exist.")

        self.close()
        self.execute_query(f"DELETE FROM stock_prices WHERE Ticker LIKE '%BRVM%'")

    def insert_daily_indices(self):
        df = self.get_last_prices("indices")
        df["Volume"] = 1
        # for col in ["Ticker", "Description"]:
        #    df[col] = (df[col].astype(str).str.replace('-', '_', regex=False))
        if not df.empty:
            self.insert_dataframe(df, "indices_performances")
        else:
            print(u'\N{check mark}' + " No new daily data to insert into <<indices_performances>> table.\
             All entries already exist.")

        self.close()

    def insert_daily_per(self):
        per = self.dp.get_table_from_url("brvm", "per")
        per["Ticker"] = list(per.index)
        per["Date"] = self.dp.make_date(self.dp.get_last_business_day())
        per = per[["Ticker"] + list(per.columns)[:-1]]
        per.columns = self._get_table_columns("per_data")
        per = self.get_new_data_from_df(per, "per_data")
        if not per.empty:
            self.insert_dataframe(per, "per_data")
        else:
            print(
                u'\N{check mark}' + " No new daily data to insert into <<per_data>> table. All entries already exist.")

    def insert_daily_market_cap(self):
        ldf = self.dp.get_today_data_from_brvm()
        col = ['Date', 'Stock Description', 'Nombre de titres', 'Capitalisation flottante', 'Capitalisation globale',
               'Capitalisation globale (%)', 'Nombre de titres échangés', 'Valeur échangée',
               'valeur globale échangée %', 'PER']
        df_m = ldf[1].merge(ldf[2], on=["Stock Description", "Date", "Ticker"])[col]
        df_m.columns = ['Date', 'stock_description', 'stock_number', 'floting_cap', 'globale_cap', 'globale_cap_pct',
                        'trade_number', 'trade_value', 'globale_trade_value_pct', 'PER']
        df_m["Date"] = self.dp.make_date(df_m['Date'])
        df_m.index.name = "Ticker"
        df_m = self.dp.reset_df_index(df_m)
        df_m = self.get_new_data_from_df(df_m, "market_cap")
        if not df_m.empty:
            self.insert_dataframe(df_m, "market_cap")
        else:
            print(u'\N{check mark}' + " No new daily data to insert into <<market_cap>> table. \
                    All entries already exist.")

    def insert_3m_histo_data(self, df=None):
        # === 1. Connect to SQLite DB ===
        merged_df = self.merge_dfs(self.dp.get_3m_hist_data()) if df is None else df
        # === 2. Load existing date+ticker pairs to avoid duplicates ===
        merged_df = self.get_new_data_from_df(merged_df).drop(columns=['Volume(FCFA)'])
        self.validate_dataframe(merged_df, "stock_prices")
        if "Variation" in merged_df:
            merged_df = merged_df.drop("Variation", axis=1)
        # === 5. Insert into 'prices' table ===
        if not merged_df.empty:
            self._connect()
            merged_df.to_sql('stock_prices', self.conn, if_exists='append', index=False)
            print(f"Inserted {len(merged_df)} new rows into 'prices' table.")
        else:
            print("No new data to insert in <<stock_prices>>. All entries already exist.")

        self.close()

    def insert_histo_data(self):
        dfs = self.dp.get_historical_data()
        df_merged = self.merge_dfs(dfs)
        # display(df_merged)
        self.insert_dataframe(self.dp.reset_df_index(df_merged), "stock_prices")

    def insert_weekly_fundamental_data(self):
        fdata = FundamentalData()
        df = self.dp.clean_and_convert(fdata.df)
        df["Date"] = self.dp.make_date(df["Date"])
        df = self.get_new_data_from_df(df, "fundamental_data")
        if not df.empty:
            self.insert_dataframe(df, "fundamental_data")
        else:
            print(u'\N{check mark}' + " No new daily data to insert into <<fundamental_data>> table. \
                                All entries already exist.")

    """##################################  DATA GETTER #################################"""

    def _get_table_columns(self, table: str):
        self._connect()
        self.cursor.execute(f"PRAGMA table_info({table})")
        res = [row[1] for row in self.cursor.fetchall()]
        self.close()
        return res

    def get_most_recent_data(self, table):
        q = f"SELECT * FROM {table} WHERE Date = (SELECT MAX(Date) FROM {table});"
        return self.execute_query(q)

    def get_value(self, table: str, columns='*', condition: str = '', order_by: str = '', limit: int = None):
        query = f"SELECT {columns} FROM {table}"
        if condition:
            query += f" WHERE {condition}"
        if order_by:
            query += f" ORDER BY {order_by}"
        if limit:
            query += f" LIMIT {limit}"
        return self.execute_query(query)

    def get_all_tables(self):
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        self._connect()
        res = [row[0] for row in self.cursor.execute(query)]
        self.close()
        return res

    def get_last_prices(self, type="stock"):
        daily_df1, daily_df2 = self.dp.get_today_data_from_sikafin()

        daily_df1["Volume"] = daily_df2["Volume"]
        daily_df1["Ticker"] = daily_df1.index
        df = daily_df1.fillna("0.0")
        df["Date"] = self.dp.make_date(df["Date"])
        tickers = list(self.dp.tickers_manager.get_all_stocks_tickers()["Ticker"]) if type.lower() == "stock" \
            else list(self.dp.tickers_manager.get_brvm_indices()["Ticker"])
        df.index = df["Ticker"]
        df = df.loc[[t for t in tickers if t in df["Ticker"]]]
        table = "stock_prices" if type.lower() == "stock" else "indices_performances"
        # Avoid duplicates
        df = self.get_new_data_from_df(df, table).drop(columns=['Volume(FCFA)', 'Variation'])
        df = self.dp.clean_and_convert(df)
        return df

    def get_data_at_specific_date(self, table=None, date=None):
        dfs = []
        if not table is None:
            if date is None:
                return self.get_most_recent_data(table)
            query_condition = f" WHERE Date = '{date}'" if not date is None else ""
            df = pd.DataFrame(
                self.execute_query(f"SELECT * FROM {table}{query_condition}"),
                columns=self._get_table_columns(table))
            df.index = df["Date"]
            df.drop('Date', axis=1)
            dfs.append(df)
        else:
            for t_name in self.tables:
                if date is None:
                    df = self.get_most_recent_data(t_name)
                else:
                    df = pd.DataFrame(self.execute_query(
                        f"SELECT * FROM {t_name + ' WHERE Date = ' + date if not date is None else t_name}"),
                        columns=self._get_table_columns(t_name))
                df.index = df["Ticker"]
                df = df.drop('Ticker', axis=1)
                dfs.append(df)
        return dfs

    def merge_dfs(self, dfs):
        merged = []
        for ticker, df in dfs.items():
            df = df.copy()
            df['Ticker'] = ticker
            df['Description'] = self.dp.tickers_manager.get_description(ticker)
            merged.append(df)

        merged_df = pd.concat(merged)
        merged_df = self.dp.reset_df_index(merged_df)
        merged_df.set_index(['Date', 'Ticker'], inplace=True)
        return merged_df

    def get_new_data_from_df(self, df, table="stock_prices"):
        # === 2. Load existing date+ticker pairs to avoid duplicates ===
        self._connect()
        existing = pd.read_sql(f"SELECT Date, Ticker FROM {table}", self.conn)
        if existing.empty:
            return df
        existing['Date'] = self.dp.make_date(existing['Date'])
        # === 3. Assume merged_df already exists (from prior merge) ===
        # Ensure index is reset and column names are clean
        df = self.dp.reset_df_index(df)  # .reset_index()

        # === 4. Filter out already existing entries ===
        df['Date'] = self.dp.make_date(df['Date'])
        check_df = df.merge(existing, on=['Date', 'Ticker'], how='left', indicator=True)
        # display(check_df.head())
        df = self.dp.clean_and_convert(check_df[check_df['_merge'] == 'left_only'].drop(columns=['_merge']))
        self.close()
        return df

    def get_prices(self, tickers=None, start_date=None, end_date=None):
        query = "SELECT * FROM stock_prices WHERE 1=1"
        if tickers:
            query += " AND Ticker IN ({})".format(",".join([f"'{t}'" for t in tickers]))
        if start_date:
            query += f" AND Date >= '{self.dp.make_date(start_date)}'"
        if end_date:
            query += f" AND Date <= '{self.dp.make_date(end_date)}'"
        query += " ORDER BY Date DESC"
        self._connect()
        df = self.execute_query(query)
        df.index = df["Ticker"]
        # df.sort_values(by="Date", ascending=False)
        self.close()
        return df

    def get_per(self, ticker=None, date=None, col=['Date', 'Ticker', 'PER']):
        if ticker is not None:
            ticker = [ticker] if type(ticker) is str else ticker
            ticker = [f"'{t}'" for t in ticker]
        query = f"SELECT {','.join(col)} FROM per_data"
        query = query + f" WHERE Ticker in ({','.join(ticker)})" if ticker is not None else query
        query = query + f" {'WHERE' if ticker is None else 'AND'} Date ='{self.dp.make_date(date)}'" if date is not None else query
        query += " ORDER BY Date DESC"
        return self.execute_query(query)

    def get_close_matrix(self, tickers, startdate, enddate):
        df = self.get_prices(tickers, startdate, enddate)
        return df.pivot(index='Date', columns='Ticker', values='Close')

    def get_fundamental(self, ticker):
        ticker_list = [ticker] if type(ticker) is str else ticker
        tls = ','.join([f"'{t}'" for t in ticker_list])
        df = self.execute_query(f"SELECT * FROM fundamental_data WHERE Ticker in ({tls})")
        df.index = df["Ticker"]
        return df

    def get_market_cap(self, ticker):
        ticker_list = [ticker] if type(ticker) is str else ticker
        tls = ','.join([f"'{t}'" for t in ticker_list])
        df = self.execute_query(f"SELECT * FROM market_cap WHERE Ticker in ({tls})")
        df.index = df["Ticker"]
        return df


class DataPreProcessor:

    def __init__(self):
        # Liste des tickers que tu veux surveiller
        self.tickers_manager = BRVMTICKERS()
        # URL de base pour les données BRVM
        self.url_price = 'https://www.brvm.org/fr/cours-actions/0'
        self.url_per = 'https://www.brvm.org/fr/volumes/0'
        self.url_capitalisation = 'https://www.brvm.org/fr/capitalisations/0'
        # self.conn = connexion

        # Faire la requête HTTP

    def url_generator(self, source="brvm"):
        if "brvm" in source.lower():
            core = "https://www.brvm.org/fr"
            return {"price": (
                f"{core}/cours-actions/0",
                ["Ticker", "Description", "Volume", "Close J-1", "Open", "Close", "Variation"]),
                "per": (
                    f"{core}/volumes/0", ["Ticker", "Description", "Volume", "Trade Value", "PER", "Pct Trade Value"]),
                "capitalisation": (
                    f"{core}/capitalisations/0", ["Ticker", "Description", "Total Stock", "Close", "Floating Cap",
                                                  "Global Cap Pct"])}
        if "sikafinance" in source.lower():
            core = "https://www.sikafinance.com/marches"
            return {"price": (
                f"{core}/aaz", ["Description", "Open", "High", "Low", "Volume", "Volume(FCFA)", "Close", "Variation"]),
                "single stock": (core + "/historiques/{}.{}",
                                 ["Date", "Close", "Low", "High", "Open", "Volume", "Volume(FCFA)", "Variation"]),
                "variation": (f"{core}/palmares", ["Description", "High", "Low", "Close", "Volume", "Variation"])}
        return {"Default": source}

    def get_info(self, url):
        r = requests.get(url, verify=False)
        soup = BeautifulSoup(r.content, 'html.parser')
        # print(soup)
        rows = soup.select('table tbody tr')
        return rows

    def get_table_from_url(self, source, table, columns=None, date=None, stock=None, print_url=False):
        url_gen = self.url_generator(source)
        if table.lower() in url_gen:
            url = url_gen[table.lower()][0]
            if table.lower() == "single stock":
                if not stock is None:
                    url = url_gen[table.lower()][0].format(stock, self.tickers_manager.get_country(stock))
                    url = url.replace(f"{stock}.", stock) if stock in list(
                        self.tickers_manager.get_brvm_indices()["Ticker"]) else url
                else:
                    print("Stock None type incompatible with table single stock")
                    return None
            data = []
            rows = self.get_info(url)
            col_num = [i for i in range(len(url_gen[table.lower()][1]))] if columns is None else columns
            output_col = [url_gen[table.lower()][1][i] for i in col_num]
            for r_nb, row in enumerate(rows):
                d = {}
                cols = row.find_all('td')
                cols_val = [v.text.strip() for v in cols]
                cols_val = cols_val if (r_nb > 21 or table.lower() != "price") else cols_val[:4] + ["00",
                                                                                                    "00"] + cols_val[4:]
                if len(cols_val) < len(output_col):
                    continue
                for c, name in zip(col_num, output_col):
                    val = cols_val[c]
                    try:
                        val = float(val.replace(',', '.').replace(' ', ''))
                    except:
                        pass
                    # print(name, val)
                    d[name] = val
                if not "Date" in d:
                    d["Date"] = self.get_last_business_day() if date is None else date
                data.append(d)
                # print(data)
            # Créer le DataFrame
            df = pd.DataFrame(data)
            # display(df)
            if print_url:
                print(url)  # , stock, list(self.tickers_manager.get_brvm_indices()["Ticker"]))
                print("df columns : ", df.columns)
                print("expected ouput : ", output_col)
            if "Description" in df:
                if "Ticker" in df:
                    df.index = df["Ticker"]
                    df = df.drop("Ticker", axis=1)
                else:
                    df.index = [self.tickers_manager.get_ticker(des, print_missing=print_url) for des in
                                df["Description"]]
            else:
                df.set_index(output_col[0], inplace=True)
            df = self.clean_and_convert(df)
            return df.drop([None]) if None in df.index else df
        else:
            rows = self.get_info(source)
            return self.get_col_from_rows(rows, [i for i in range(len(str(rows[0]).split("</td>")) - 1)])

    def get_col_from_rows(self, rows, col_id, date=None,
                          exclude=("Valeur des transactions", "Capitalisation Actions",
                                   "Capitalisation des obligations")):
        data = []
        date = self.get_last_business_day() if date is None else date
        for row in rows:
            cols = row.find_all('td')
            # print(cols, len(cols))
            if len(cols) < 5:
                continue
            code = cols[0].text.strip()
            if code in exclude:
                continue
            d = {"Ticker": code}
            for c in col_id:
                val = cols[c[1]].text.strip()
                try:
                    val = val if c[2].lower() != 'numeric' else float(val.replace(',', '.').replace(' ', ''))
                except:
                    # print(code, val)
                    pass
                d[c[0]] = val
            d["Date"] = date
            data.append(d)
        # Créer le DataFrame
        df = pd.DataFrame(data)
        df.set_index('Ticker', inplace=True)
        df = self.clean_and_convert(df)
        return df

    def get_3m_hist_data(self, list_tickers="all", class_type="stock"):
        res = {}
        all_tickers = list(self.tickers_manager.get_all_stocks_tickers()["Ticker"]) if class_type == "stock" else \
            list(self.tickers_manager.get_brvm_indices()["Ticker"])
        tickers = all_tickers if list_tickers == "all" else list_tickers
        for ticker in tickers:
            try:
                df = self.get_table_from_url("sikafinance", "single stock", stock=ticker)
                if not df.empty:
                    res[ticker] = df
                else:
                    print(f"No data found for {ticker}")
            except:
                print(f"Failed to load data for {ticker}")
        return res

    def get_today_data_from_sikafin(self):
        return self.get_table_from_url("sikafinance", "price"), self.get_table_from_url("sikafinance", "variation")

    def get_today_data_from_brvm(self):
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

    def get_last_business_day(self):
        last_business_day = pd.Timestamp.today().normalize()
        last_business_day -= pd.Timedelta(days=1)
        while last_business_day.weekday() > 4:  # 0 = Monday, 6 = Sunday
            last_business_day -= pd.Timedelta(days=1)
        return last_business_day.strftime('%Y-%m-%d')

    def clean_and_convert(self, df, col_to_exclude=["Date", "Ticker", "Description", "Variation"]):
        df_clean, dd = df.copy(), None
        if "Date" in df_clean:
            dd = df_clean["Date"].astype(str)  # self.make_date(df_clean["Date"])
        for col in df_clean.columns:
            if df_clean[col].dtype == object and not col in col_to_exclude:
                # Supprime espaces insécables, normaux, et convertit en int si possible
                df_clean[col] = (
                    df_clean[col]
                    .astype(str)
                    .str.replace('\xa0', '', regex=False)
                    .str.replace(' ', '', regex=False)
                    .str.replace('-', '0.0', regex=False)
                    .str.replace(',', '', regex=False)
                    .str.strip()
                )
                try:
                    if "%" in df_clean[col][0]:
                        df_clean[col] = [float(e.replace("%", "")) / 100.0 for e in list(df_clean[col])]
                except:
                    pass
                # Essaie de convertir en entier si possible
                try:
                    df_clean[col] = df_clean[col].astype(int)
                except ValueError:
                    pass  # Ignore si conversion impossible
        if not dd is None:
            df_clean["Date"] = dd
        return df_clean

    def make_date(self, date_str):
        return pd.to_datetime(date_str, dayfirst=True, errors='coerce')

    def reset_df_index(self, df):
        if df.index.name in df:
            df = df.drop(df.index.name, axis=1)
        df.reset_index(inplace=True)
        return df

    def get_historical_data(self):
        d = self.tickers_manager.get_all_stocks_tickers()
        d.index = d["Ticker"]
        st, ref, dfs = [], list(d["Ticker"]), {}
        for file in os.listdir("../DataBase/XLSX"):
            df = pd.read_csv(f"../DataBase/XLSX/{file}")
            tck = file.split()[0]
            df["Ticker"] = tck
            df["Description"] = self.tickers_manager.get_description(tck)
            st.append(tck)

            # clean the df
            df.columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'Change', 'Ticker', 'Description']
            df = df.drop("Change", axis=1)
            df = df.fillna("0.0")
            df["Volume"] = [int(float(str(df["Volume"][j]).replace("K", "").replace("M", ""))) * 1000 for j in
                            range(len(list(df["Volume"])))]
            df["Date"] = self.make_date(
                [f"{dd.split('/')[-1]}-{dd.split('/')[0]}-{dd.split('/')[1]}" for dd in df["Date"]])
            df.index = df["Date"]
            dfs[tck] = self.clean_and_convert(df)
            if not tck in ref:
                print("to remove  ", tck)
        for e in ref:
            if not e in st:
                print("to add ", e, d.loc[e, "Description"])
        return dfs


class BRVMTICKERS:

    def __init__(self):
        # Liste des tickers actions BRVM
        self.actions_data = [('ABJC', 'SERVAIR ABIDJAN CI', 'CI'),
                             ('BICB', 'BANQUE INTERNATIONALE POUR LE COMMERCE DU BENIN', 'BJ'),
                             ('BICC', 'BICICI', 'CI'),
                             ('BNBC', "BERNABE CI", 'CI'),
                             ('BOAB', 'BANK OF AFRICA BENIN', 'BJ'),
                             ('BOABF', 'BANK OF AFRICA BURKINA FASO', 'BF'),
                             ('BOAC', "BANK OF AFRICA CI", 'CI'),
                             ('BOAM', 'BANK OF AFRICA MALI', 'ML'),
                             ('BOAN', 'BANK OF AFRICA NIGER', 'NE'),
                             ('BOAS', 'BANK OF AFRICA SENEGAL', 'SN'),
                             ('CABC', 'SICABLE CI', 'CI'),
                             ('CBIBF', 'CORIS BANK INTERNATIONAL BF', 'BF'),
                             ('CFAC', "CFAO CI", 'CI'),
                             ('CIEC', "CIE CI", 'CI'),
                             ('ECOC', "ECOBANK CI", 'CI'),
                             ('ETIT', 'ETI TG', 'TG'),
                             ('FTSC', 'FILTISAC CI', 'CI'),
                             ('LNBB', 'LOTERIE NATIONALE DU BENIN', 'BJ'),
                             ('NEIC', 'NEI CEDA CI', 'CI'),
                             ('NSBC', "NSIA BANQUE CI", 'CI'),
                             ('NTLC', "NESTLE CI", 'CI'),
                             ('ONTBF', 'ONATEL BF', 'BF'),
                             ('ORAC', "ORANGE CI", 'CI'),
                             ('ORGT', 'ORAGROUP TOGO', 'TG'),
                             ('PALC', "PALMCI", 'CI'),
                             ('SAFC', 'SAFCA CI', 'CI'),
                             ('SCRC', 'SUCRIVOIRE', 'CI'),
                             ('SDCC', 'SODECI', 'CI'),
                             ('SDSC', 'AFRICA GLOBAL LOGISTICS', 'CI'),
                             # ('SEMC', 'CROWN SIEM', 'CI'),
                             ('SGBC', 'SGBCI', 'CI'),
                             ('SHEC', 'VIVO ENERGY CI', 'CI'),
                             ('SIBC', 'SOCIETE IVOIRIENNE DE BANQUE CI', 'CI'),
                             ('SICC', 'SICOR', 'CI'),
                             ('SIVC', 'AIR LIQUIDE CI', 'CI'),
                             ('SLBC', 'SOLIBRA CI', 'CI'),
                             ('SMBC', 'SMB CI', 'CI'),
                             ('SNTS', 'SONATEL SENEGAL', 'SN'),
                             ('SOGC', 'SOGB', 'CI'),
                             ('SPHC', "SAPH CI", 'CI'),
                             ('STAC', "SETAO CI", 'CI'),
                             ('STBC', "SITAB CI", 'CI'),
                             # ('SVOC', "MOVIS CI", 'CI'),
                             ('TTLC', "TOTAL CI", 'CI'),
                             ('TTLS', 'TOTAL SENEGAL', 'SN'),
                             ('PRSC', "TRACTAFRIC MOTORS CI", 'CI'),
                             ('UNLC', "UNILEVER CI", 'CI'),
                             ('UNXC', "UNIWAX CI", 'CI')]
        # Liste des indices BRVM
        self.indices_data = [('BRVMAG', 'BRVM - AGRICULTURE', ''),
                             ('BRVMAS', 'BRVM - AUTRES SECTEURS', ''),
                             ('BRVM-CB', 'BRVM - CONSOMMATION DE BASE', ''),
                             ('BRVM-CD', 'BRVM - CONSOMMATION DISCRETIONNAIRE', ''),
                             ('BRVMDI', 'BRVM - DISTRIBUTION', ''),
                             ('BRVM-EN', 'BRVM - ENERGIE', ''),
                             ('BRVMFI', 'BRVM - FINANCE', ''),
                             ('BRVMIN', 'BRVM - INDUSTRIE', ''),
                             ('BRVM-IN', 'BRVM - INDUSTRIELS', ''),
                             ('BRVMPR', 'BRVM - PRESTIGE', ''),
                             ('BRVMPA', 'BRVM - PRINCIPAL', ''),
                             ('BRVM-SF', 'BRVM - SERVICES FINANCIERS', ''),
                             ('BRVMSP', 'BRVM - SERVICES PUBLICS', ''),
                             ('BRVM-SP', 'BRVM - SERVICES PUBLICS', ''),
                             ('BRVM-TEL', 'BRVM - TELECOMMUNICATIONS', ''),
                             ('BRVMTR', 'BRVM - TRANSPORT', ''),
                             ('BRVM30', 'BRVM 30', ''),
                             ('BRVMC', 'BRVM COMPOSITE', ''),
                             ('CAPIBRVM', 'Capitalisation BRVM', '')]

    def remove_accents(self, description):
        return ''.join(c for c in unicodedata.normalize('NFD', description) if unicodedata.category(c) != 'Mn')

    # Création du DataFrame
    def get_all_brvm_tickers(self):
        return pd.DataFrame(self.indices_data + self.actions_data, columns=["Ticker", "Description", "Country_Code"])

    def get_all_stocks_tickers(self):
        return pd.DataFrame(self.actions_data, columns=["Ticker", "Description", "Country_Code"])

    def get_brvm_indices(self):
        return pd.DataFrame(self.indices_data, columns=["Ticker", "Description", "Country_Code"])

    def get_ticker(self, des, pos=[0], print_missing=False):
        df = self.get_all_brvm_tickers()
        # pos_ = pos[0] if len(pos) == 1 else pos
        try:
            # des_modified = ' '.join(v.split()[:-1])
            res = list(df[df["Description"].str.contains(des, case=False, na=False)]["Ticker"])
            res = [res[i] for i in pos] if len(pos) > 1 else res[0]
            return res
        except:
            if print_missing:
                print(f"No Ticker correspond to description : {des}")
            return None

    def get_country(self, stock, pos=[0]):
        df = self.get_all_brvm_tickers()
        try:
            return list(df[df["Ticker"] == stock]["Country_Code"])[0]
        except:
            print(f"Country for stock {stock} not found")

    def get_description(self, stock, pos=[0]):
        df = self.get_all_brvm_tickers()
        try:
            return list(df[df["Ticker"] == stock]["Description"])[0]
        except:
            print(f"Description for stock {stock} not found")


class FundamentalData:
    def __init__(self, fundamentals_df=None):
        """
        Must contain columns: ['ticker', 'PER', 'PB', 'ROE', 'NetMargin', 'Growth']
        """
        self.df = fundamentals_df.set_index('Ticker') if fundamentals_df is not None else self.get_fundamentals()

    def get_fundamentals(self):
        file = os.listdir("../DataBase/FundamentalData")[-1]
        date = file.split(".")[0].split("_")[-1]
        date = f"{date[-4:]}-{date[:2]}-{date[2:4]}"
        df = pd.read_csv(f"../DataBase/FundamentalData/{file}")
        df.index, df.index.name = [e.split(".")[0] for e in df["Symbol"]], "Ticker"
        df["Date"] = date  # f"{datetime.today().year}-{df['Time'][0].split('/')[1]}-{df['Time'][0].split('/')[0]}"
        df = df.drop(["Name", "Symbol", "Exchange", 'Bid', 'Ask',
                      'Extended Hours', 'Extended Hours (%)', '5 Minutes',
                      '15 Minutes', '30 Minutes', 'Hourly', '5 Hours', 'Time',
                      'Last', 'Open', 'Prev.', 'High', 'Low', 'Chg.', 'Chg. %'], axis=1)
        df = (df
              .replace("Strong Sell", -1)
              .replace("Strong Buy", 1)
              .replace("Neutral", 0)
              .replace("Sell", -0.5)
              .replace("Buy", 0.5)
              )
        # df["Ticker"] = list(df.index)
        df.columns = ['Volume', 'Next Earnings Date', 'Market Cap', 'Revenue',
                      'Average 3m Volume', 'EPS', 'P/E Ratio', 'Beta', 'Dividend', 'Yield',
                      'Daily Trend', 'Weekly Trend', 'Monthly Trend', '1 Day perf', '1 Week perf', '1 Month perf',
                      'YTD',
                      '1 Year perf', '3 Years perf', 'Date']
        return df.reset_index(inplace=False)

    def get_per(self):
        return self.df['PER']

    def get_pb(self):
        return self.df['PB']

    def get_roe(self):
        return self.df['ROE']

    def get_margin(self):
        return self.df['NetMargin']

    def get_growth(self):
        return self.df['Growth']
