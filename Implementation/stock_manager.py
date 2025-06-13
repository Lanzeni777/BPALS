import pandas as pd
import numpy as np

class StockEvaluator:
    def __init__(self, database, criteria=None):
        self.database = database
        self.criteria = criteria or {
            "momentum": 0.3,
            "relative_price": 0.2,
            "volatility": 0.2,
            "volume": 0.15,
            "per": 0.15,
            "cap_flo": 0.0
        }

    def compute_score(self, stock, relative=False):
        score = 0
        for key, weight in self.criteria.items():
            val = stock.get_indicator(key)
            if val is not None:
                norm = self.normalize(key, val, stock)
                score += weight * norm
        return score

    def normalize(self, key, val, stock):
        # Logics inspired by real market interpretation
        if key == "momentum":
            return np.tanh(val * 5)  # saturate high momentum
        elif key == "relative_price":
            return 1 - abs(val - 1)  # best if close to 1
        elif key == "volatility":
            momentum = stock.get_indicator("momentum") or 0
            penalty = val / (1 + abs(momentum))
            return 1 / (1 + penalty)  # penalize erratic volatility when momentum is weak
        elif key == "volume":
            return np.tanh(val / 1e5)  # cap impact of raw volume
        elif key == "per":
            return 1 / (1 + val) if val > 0 else 0
        elif key == "cap_flo":
            return 1 / np.log1p(val) if val > 0 else 0
        else:
            return 0  # fallback

    def relative_normalize(self, stock_list):
        """
        Computes normalized indicators across all stocks for comparative ranking.
        Returns a dict of {ticker: {indicator: normalized_value}}
        """
        all_scores = {key: [] for key in self.criteria.keys()}
        for stock in stock_list:
            for key in self.criteria:
                val = stock.get_indicator(key)
                all_scores[key].append(val if val is not None else np.nan)

        norm_results = {}
        for key in self.criteria:
            values = np.array(all_scores[key], dtype=np.float64)
            if np.all(np.isnan(values)):
                norm_results[key] = np.full(len(values), 0.0)
            else:
                min_val = np.nanmin(values)
                max_val = np.nanmax(values)
                span = max_val - min_val if max_val > min_val else 1
                norm_results[key] = (values - min_val) / span

        normalized_dict = {}
        for i, stock in enumerate(stock_list):
            norm_indicators = {
                key: float(norm_results[key][i]) if not np.isnan(norm_results[key][i]) else 0.0
                for key in self.criteria
            }
            normalized_dict[stock.ticker] = norm_indicators

        return normalized_dict

    def rank_stocks(self, ticker_list, start_date="2020-01-01"):
        stock_prices = self.database.get_prices(ticker_list, start_date)  # , "2025-05-30")
        stock_fd = self.database.get_fundamental(ticker_list)
        stock_fd["PER"], stock_fd["Capitalisation_flottante"] = self.database.get_per(ticker_list)["PER"], 1000000
        stock_list = [Stock(tck, stock_prices.loc[tck], stock_fd.loc[tck]) for tck in ticker_list \
                      if tck in stock_fd.index and tck in stock_prices.index]
        scored = [(st.ticker, self.compute_score(st)) for st in stock_list]
        return pd.DataFrame(sorted(scored, key=lambda x: x[1], reverse=True), columns=["Ticker", "Score"])

    def relative_rank_stocks(self, ticker_list, start_date="2020-01-01"):
        stock_prices = self.database.get_prices(ticker_list, start_date)  # , "2025-05-30")
        stock_fd = self.database.get_fundamental(ticker_list)
        stock_fd["PER"], stock_fd["Capitalisation_flottante"] = self.database.get_per(ticker_list)["PER"], 1000000
        stock_list = [Stock(tck, stock_prices.loc[tck], stock_fd.loc[tck]) for tck in ticker_list\
                  if tck in stock_fd.index and tck in stock_prices.index]
        # scored = [(st.ticker, self.compute_score(st)) for st in stock_list]
        normalized = self.relative_normalize(stock_list)
        scored = []
        for stock in stock_list:
            score = sum(
                self.criteria[key] * normalized[stock.ticker].get(key, 0.0)
                for key in self.criteria
            )
            scored.append((stock, score))
        return pd.DataFrame(sorted(scored, key=lambda x: x[1], reverse=True), columns=["Ticker", "RelativeScore"])

    def build_stocks_from_data(self, price_df: pd.DataFrame, fundamental_df: pd.DataFrame):
        grouped = price_df.groupby("Ticker")
        stocks = []
        for ticker, df in grouped:
            fund_row = fundamental_df[fundamental_df['Ticker'] == ticker]
            stock = Stock(ticker, df.copy(), fund_row)
            stocks.append(stock)
        return stocks


class Stock:
    def __init__(self, ticker, price_df: pd.DataFrame, fund_row: pd.DataFrame):
        self.ticker = ticker
        self.price_df = price_df.sort_values("Date")
        self.indicators = {}
        self._compute_indicators(fund_row)

    def _compute_indicators(self, fund_row):
        if len(self.price_df) >= 21:
            self.indicators["momentum"] = self.price_df["Close"].iloc[-1] / self.price_df["Close"].iloc[-21] - 1
        if len(self.price_df) >= 5:
            self.indicators["volatility"] = self.price_df["Close"].pct_change().rolling(5).std().iloc[-1]
            self.indicators["volume"] = self.price_df["Volume"].rolling(5).mean().iloc[-1]
        if len(self.price_df) >= 30:
            self.indicators["relative_price"] = self.price_df["Close"].iloc[-1] / self.price_df["Close"].rolling(30).mean().iloc[-1]

        if not fund_row.empty:
            for col in ['PER', 'Yield', 'Capitalisation_flottante']:
                val = fund_row[col]
                if pd.notna(val):
                    self.indicators[col.lower() if col != 'Capitalisation_flottante' else 'cap_flo'] = val

    def get_indicator(self, name):
        return self.indicators.get(name)

    def get_all_indicators(self):
        return self.indicators

    def __repr__(self):
        return f"<Stock {self.ticker}>"
