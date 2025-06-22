import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import os, sys
import urllib3
import time

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("."), '..')))
from DataBase.DBUtilities import BRVMDatabase, DataPreProcessor
from Implementation.stock_picking_heuristic import Stock, StockEvaluator
from Implementation.stock_factorial_analysis import FactorAnalytics


def update_data(db: BRVMDatabase, do=False):
    db.update() if do else None


def rank_all_stock(db):
    ticker_list = list(db.dp.tickers_manager.get_all_stocks_tickers()["Ticker"])
    stock_eval = StockEvaluator(db)
    print(stock_eval.rank_stocks(ticker_list))
    print(stock_eval.relative_rank_stocks(ticker_list))
    print(stock_eval.value_rank_stocks(ticker_list))
    print(stock_eval.hybrid_rank_stocks(ticker_list))


def do_picking_analysis():
    stock_eval = StockEvaluator(BRVMDatabase("KAN.db"))
    return stock_eval.do_picking_analysis()


def do_factors_analysis(show_graph=False):
    fact_an = FactorAnalytics()
    df, col = fact_an.build_features()
    # df.set_index(['Date', 'Ticker'], inplace=True)
    res = fact_an.run_pca(df, col[1:])
    fact_an.explained_variance()
    fact_an.plot_pca_scatter()
    return fact_an.rank_latest_scores(col[1:])
    # return res


if __name__ == '__main__':
    # kdb = BRVMDatabase("KAN.db")
    # update_data(kdb, True)
    # rank_all_stock(kdb)
    df1 = do_picking_analysis()
    df2 = do_factors_analysis()

    input("End")


def main_1306_getter():
    kl_db = BRVMDatabase("KAN.db")
    status = kl_db.check_status()
    tables_names = kl_db.get_all_tables()
    dfs = kl_db.dp.get_historical_data()
    dfs_merged = kl_db.merge_dfs(dfs)
    kl_db.insert_dataframe(dfs_merged, "stock_prices")
    per = kl_db.dp.get_table_from_url("brvm", "per")
    prices = kl_db.dp.get_table_from_url("sikafinance", "price")
    l1 = kl_db.dp.get_today_data_from_sikafin()
    l2 = kl_db.dp.get_today_data_from_brvm()
    status2 = kl_db.check_status()
    input("Tape any touch to end the programme")


def delete():
    kl_db = BRVMDatabase("KAN.db")
    dsd = kl_db.get_data_at_specific_date(date=kl_db.dp.make_date("2025-06-06"), table="stock_prices")
    df_check = kl_db.execute_query(f"SELECT * FROM stock_prices WHERE Date = {kl_db.dp.make_date('2025-05-23')}")
    kl_db.execute_query(f"DELETE FROM stock_prices WHERE Ticker LIKE '%BRVM%'")
    kl_db.insert_dataframe(df_check, "indices_performances")
    kl_db.execute_query("DROP TABLE IF EXISTS indices_performances;")
    kl_db.execute_query("DROP TABLE IF EXISTS indices_prices;")

    kl_db._connect()
    df_check.to_sql("indices_performances", kl_db.conn, index=False)
    kl_db.close()
    last_prices = kl_db.dp.get_last_prices()
    kl_db.delete_and_create("stock_prices")
    kl_db.insert_daily_prices()
    kl_db.insert_3m_histo_data()
