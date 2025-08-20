import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from DataBase.DBUtilities import BRVMDatabase
from matplotlib import pyplot as plt
from DataBase.DBUtilities import BRVMDatabase
from sklearn.linear_model import LinearRegression
import math
from scipy.spatial import ConvexHull, convex_hull_plot_2d



class PriceDrivenStrategy:
    def __init__(self):
        self.database = BRVMDatabase("KAN.db")
        self.df = self._prepare()

    def _prepare(self, start_date="2018-01-01"):
        df = self.database.get_prices(start_date=start_date)
        df = self.database.dp.reset_df_index(df)
        df = df.sort_values(['Date', 'Ticker'])
        df['MA_90'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(90).mean())
        df['Deviation'] = (df['Close'] - df['MA_90']) / df['MA_90']
        df['Trend'] = df.groupby('Ticker')['Close'].transform(lambda x: x.diff().apply(np.sign))
        df['Trend_Streak'] = df.groupby('Ticker')['Trend'].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        df['Abnormal'] = (df['Deviation'].abs() > 0.3)
        return df

    def plot_price_with_signal(self, ticker):
        data = self.df[self.df['Ticker'] == ticker]
        plt.figure(figsize=(12, 5))
        plt.plot(data['Date'], data['Close'], label="Close Price")
        plt.plot(data['Date'], data['MA_90'], label="MA 90")
        plt.fill_between(data['Date'], data['Close'], data['MA_90'],
                         where=(data['Abnormal']), color='red', alpha=0.3, label="Abnormal")
        plt.title(f"Price vs MA with abnormal signals - {ticker}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def get_signals_summary(self):
        summary = self.df.groupby('Ticker').agg({
            'Deviation': ['mean', 'std'],
            'Trend_Streak': 'max',
            'Abnormal': 'sum'
        })
        return summary

    def forward_strategy(self, threshold=0.3, forward_days=21, backtest_window=60):
        df = self.df.copy()
        split_date = df['Date'].unique()[-backtest_window]
        train_df = df[df['Date'] < split_date].copy()
        test_df = df[df['Date'] == split_date].copy()

        train_df['Signal'] = (train_df['Deviation'] < -threshold).astype(int)
        last_signals = train_df.groupby('Ticker').apply(
            lambda x: x.loc[x['Signal'] == 1].sort_values('Date').tail(1)).reset_index(drop=True)

        test_df = test_df.merge(last_signals[['Ticker', 'Date']], on='Ticker', how='left', suffixes=('', '_lastsignal'))
        test_df['Action'] = 'keep'
        test_df.loc[test_df['Date'] == test_df['Date_lastsignal'], 'Action'] = 'buy'

        test_df['PriceToBuy'] = test_df['Close']
        test_df['PriceToSell'] = \
        self.df[self.df['Date'] == self.df['Date'].unique()[-backtest_window + forward_days]].set_index('Ticker')[
            'Close']
        test_df['PriceToSell'] = test_df['PriceToSell'].reindex(test_df['Ticker'].values).fillna(
            test_df['PriceToBuy']).values

        test_df['ExpectedReturn'] = (test_df['PriceToSell'] - test_df['PriceToBuy']) / test_df['PriceToBuy']
        test_df['Return'] = 0.0
        test_df.loc[test_df['Action'] == 'buy', 'Return'] = test_df['ExpectedReturn']

        return test_df[['Date', 'Ticker', 'Action', 'PriceToBuy', 'PriceToSell', 'Return']].sort_values(
            ['Date', 'Ticker'])

    def rolling_forward_strategy(self, threshold=0.3, forward_days=21, backtest_window=60, step=5):
        df = self.df.copy()
        df = df.sort_values(['Date', 'Ticker'])
        dates = sorted(df['Date'].unique())
        results = []

        for i in range(backtest_window, len(dates) - forward_days, step):
            split_date = dates[i]
            train_df = df[df['Date'] < split_date].copy()
            test_df = df[df['Date'] == split_date].copy()

            train_df['Signal'] = (train_df['Deviation'] < -threshold).astype(int)
            last_signals = train_df.groupby('Ticker').apply(
                lambda x: x.loc[x['Signal'] == 1].sort_values('Date').tail(1)).reset_index(drop=True)

            test_df = test_df.merge(last_signals[['Ticker', 'Date']], on='Ticker', how='left',
                                    suffixes=('', '_lastsignal'))
            test_df['Action'] = 'keep'
            test_df.loc[test_df['Date'] == test_df['Date_lastsignal'], 'Action'] = 'buy'

            test_df['PriceToBuy'] = test_df['Close']
            price_to_sell = df[df['Date'] == dates[i + forward_days]].set_index('Ticker')['Close']
            test_df['PriceToSell'] = price_to_sell.reindex(test_df['Ticker'].values).fillna(
                test_df['PriceToBuy']).values

            test_df['ExpectedReturn'] = (test_df['PriceToSell'] - test_df['PriceToBuy']) / test_df['PriceToBuy']
            test_df['Return'] = 0.0
            test_df.loc[test_df['Action'] == 'buy', 'Return'] = test_df['ExpectedReturn']

            selected = test_df[['Date', 'Ticker', 'Action', 'PriceToBuy', 'PriceToSell', 'Return']]
            results.append(selected)

        return pd.concat(results).sort_values(['Date', 'Ticker'])

    def average_vs_current_signal(self, window=90):
        df = self.df.copy()
        df = df.sort_values(['Date'])
        latest_date = df['Date'].max()
        current_df = df[df['Date'] == latest_date].copy()
        mean_df = df.groupby('Ticker').apply(lambda x: x.tail(window)['Close'].mean()).reset_index()
        mean_df.columns = ['Ticker', 'HistoricalAverage']

        merged = current_df[['Ticker', 'Close']].merge(mean_df, on='Ticker')
        merged = merged.sort_values(by="Ticker")
        merged = merged.set_index('Ticker')
        merged.rename(columns={'Close': 'CurrentPrice'}, inplace=True)
        merged['Date'] = latest_date
        merged['Action'] = 'keep'
        merged.loc[merged['CurrentPrice'] < merged['HistoricalAverage'] * 0.9, 'Action'] = 'buy'
        merged.loc[merged['CurrentPrice'] > merged['HistoricalAverage'] * 1.15, 'Action'] = 'sell'
        merged['GapPct'] = (merged['CurrentPrice'] - merged['HistoricalAverage']) / merged['HistoricalAverage']
        merged['HistoricalAverage'] = merged['HistoricalAverage'].round(0)
        merged['CurrentPrice'] = merged['CurrentPrice'].round(0)
        merged['GapPct'] = merged['GapPct'].round(2)

        def color_action(val):
            if val == 'buy':
                return 'background-color: lightgreen'
            elif val == 'sell':
                return 'background-color: salmon'
            else:
                return 'background-color: lightgray'

        # styled = merged[['Date', 'HistoricalAverage', 'CurrentPrice', 'GapPct', 'Action']].style.applymap(color_action, subset=['Action'])
        styled = merged[['Date', 'HistoricalAverage', 'CurrentPrice', 'GapPct', 'Action']].style \
            .applymap(color_action, subset=['Action']) \
            .format({"HistoricalAverage": "{:.0f}", "CurrentPrice": "{:.0f}", "GapPct": "{:.2%}"})

        return styled

    def _trendline_coverage(self, x_fit, y_fit, x, y, kind, tolerance=0):
        best_score = -1
        best_line = None
        n = len(x_fit)
        for i in range(n):
            for j in range(i + 1, n):
                xi, xj = x_fit[i], x_fit[j]
                yi, yj = y_fit[i], y_fit[j]
                if xi == xj:
                    continue
                slope = (yj - yi) / (xj - xi)
                intercept = yi - slope * xi
                line = slope * x + intercept
                if kind == 'support':
                    mask = y >= line - tolerance
                else:
                    mask = y <= line + tolerance
                score = np.sum(mask)
                span = abs(xj - xi)
                total_score = score + span / len(x)  # Favor lines covering more length
                if total_score > best_score:
                    best_score = total_score
                    best_line = (slope, intercept)
        return best_line

    def plot_trendlines(self, ticker, window=150, kind='support', max_angle_deg=45, graph_type='open-close',
                        method='regression'):
        """
        Plots trendlines for a given ticker with customizable price visualization.

        Parameters:
        - ticker (str): the stock symbol to plot
        - window (int): number of recent days to use for analysis
        - kind (str): 'support' or 'resistance'
        - max_angle_deg (float): maximum allowed angle (in degrees) for the trendline
        - graph_type (str): controls how the price is plotted:
            - 'open-close': bar chart from open to close (default)
            - 'close': line plot using closing prices
            - 'open': line plot using opening prices
            - 'mid': line plot using average of open and close

        - method:
            - 'regression': standard linear regression
            - 'strict': line under/above all prices from extrema
            - 'pairwise': test all pairs and retain valid
            - 'envelope': convex hull lower/upper line
            - 'ruler': heuristic top/bottom extremes
        """
        df = self.df[self.df['Ticker'] == ticker].tail(window).reset_index(drop=True)
        x = np.arange(len(df))
        y = df['Close'].values

        if kind == 'support':
            idx = y <= pd.Series(y).rolling(5, center=True).min()
        else:
            idx = y >= pd.Series(y).rolling(5, center=True).max()

        x_fit = x[idx]
        y_fit = y[idx]

        if len(x_fit) < 2:
            print("Fallback: not enough extrema. Using absolute values.")
            idx = np.argsort(y)[:2] if kind == 'support' else np.argsort(y)[-2:]
            x_fit, y_fit = x[idx], y[idx]

        trendline = None
        angle_deg = None

        if method == 'regression':
            model = LinearRegression().fit(x_fit.reshape(-1, 1), y_fit)
            slope = model.coef_[0]
            intercept = model.intercept_
            angle_deg = abs(math.degrees(math.atan(slope)))
            if angle_deg > max_angle_deg:
                print(f"Rejected: angle {angle_deg:.2f}° > {max_angle_deg}°")
                return
            trendline = model.predict(x.reshape(-1, 1))

        elif method == 'strict' or method == 'pairwise':
            best = None
            for i in range(len(x_fit)):
                for j in range(i + 1, len(x_fit)):
                    xi, xj = x_fit[i], x_fit[j]
                    yi, yj = y_fit[i], y_fit[j]
                    if xi == xj: continue
                    slope = (yj - yi) / (xj - xi)
                    intercept = yi - slope * xi
                    line = slope * x + intercept
                    if kind == 'support' and np.all(y >= line):
                        if not best or intercept > best[1]:
                            best = (slope, intercept)
                    elif kind == 'resistance' and np.all(y <= line):
                        if not best or intercept < best[1]:
                            best = (slope, intercept)
            if not best:
                print("No valid strict/pairwise trendline found.")
                return
            slope, intercept = best
            trendline = slope * x + intercept
            angle_deg = abs(math.degrees(math.atan(slope)))

        elif method == 'coverage':
            result = self._trendline_coverage(x_fit, y_fit, x, y, kind, tolerance=0.5)
            if result is None:
                print("No coverage trendline found")
                return
            slope, intercept = result
            trendline = slope * x + intercept
            angle_deg = abs(math.degrees(math.atan(slope)))

        elif method == 'envelope':
            points = np.column_stack((x, y))
            try:
                hull = ConvexHull(points)
                edges = []
                for i in range(len(hull.vertices)):
                    a = hull.vertices[i - 1]
                    b = hull.vertices[i]
                    xa, ya = x[a], y[a]
                    xb, yb = x[b], y[b]
                    if xb == xa: continue
                    slope = (yb - ya) / (xb - xa)
                    intercept = ya - slope * xa
                    edge_line = slope * x + intercept
                    if kind == 'support' and np.all(y >= edge_line):
                        edges.append((slope, intercept))
                    elif kind == 'resistance' and np.all(y <= edge_line):
                        edges.append((slope, intercept))
                if not edges:
                    print("No valid envelope edge found.")
                    return
                slope, intercept = max(edges, key=lambda e: e[1]) if kind == 'support' else min(edges,
                                                                                                key=lambda e: e[1])
                trendline = slope * x + intercept
                angle_deg = abs(math.degrees(math.atan(slope)))
            except:
                print("Convex hull failed")
                return

        elif method == 'ruler':
            idx_min = np.argmin(y)
            idx_max = np.argmax(y)
            if kind == 'support':
                slope = (y[idx_max] - y[idx_min]) / (idx_max - idx_min + 1e-5)
                intercept = y[idx_min] - slope * idx_min
            else:
                slope = (y[idx_min] - y[idx_max]) / (idx_min - idx_max + 1e-5)
                intercept = y[idx_max] - slope * idx_max
            trendline = slope * x + intercept
            angle_deg = abs(math.degrees(math.atan(slope)))

        else:
            print("Invalid method")
            return

        breaks = []
        self.last_breaks = []
        if kind == 'support':
            breaks = np.where(y < trendline)[0]
        else:
            breaks = np.where(y > trendline)[0]

        fig, ax = plt.subplots(figsize=(12, 6))
        if graph_type == 'open-close':
            ax.bar(df['Date'], df['Close'] - df['Open'], bottom=df['Open'], width=1,
                   color=(df['Close'] >= df['Open']).map({True: 'green', False: 'red'}))
        elif graph_type == 'close':
            ax.plot(df['Date'], df['Close'], label='Close', color='black')
        elif graph_type == 'open':
            ax.plot(df['Date'], df['Open'], label='Open', color='gray')
        elif graph_type == 'mid':
            ax.plot(df['Date'], (df['Open'] + df['Close']) / 2, label='Mid', color='purple')
        else:
            print("Invalid graph_type")
            return

        ax.plot(df['Date'], trendline, label=f"{kind.capitalize()} Trendline ({method})", color='blue')
        for b in breaks:
            ax.annotate('Break', xy=(df['Date'][b], y[b]), xytext=(0, -12), textcoords='offset points',
                        ha='center', color='red', fontsize=8,
                        arrowprops=dict(arrowstyle='->', color='red'))
            self.last_breaks.append({"date": df['Date'][b], "price": y[b]})

        ax.set_title(f"{ticker} {kind} Trendline ({method}, {angle_deg:.2f}°)")
        ax.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def generate_trade_signals(self, ticker, window=150, kind='support', method='coverage'):
        self.plot_trendlines(ticker, window=window, kind=kind, method=method)
        if not hasattr(self, 'last_breaks') or not self.last_breaks:
            return pd.DataFrame()
        df = self.df[self.df['Ticker'] == ticker].tail(window).reset_index(drop=True)
        df = df.sort_values('Date')

        signal_df = pd.DataFrame(self.last_breaks)
        signal_df['Ticker'] = ticker
        signal_df['Volume'] = signal_df['date'].map(
            lambda d: df[df['Date'] == d]['Volume'].values[0] if d in df['Date'].values else np.nan)
        signal_df['Volume_z'] = signal_df['date'].map(
            lambda d: df[df['Date'] == d]['Volume_zscore'].values[0] if d in df['Date'].values else np.nan)

        signal_df['ReturnIfExecuted'] = df.set_index('Date').reindex(signal_df['date'])['Close'].pct_change().shift(
            -1).values
        signal_df.rename(columns={"date": "Date", "price": "Close"}, inplace=True)
        signal_df['Action'] = 'Sell' if kind == 'support' else 'Buy'
        signal_df['Break_Confirmed'] = signal_df['Volume_z'] > 0
        signal_df['Time_Near_Level'] = signal_df['Date'].map(lambda d: self._compute_time_near_level(df, d, kind))
        return signal_df[
            ['Date', 'Ticker', 'Close', 'Volume', 'Volume_z', 'Time_Near_Level', 'Action', 'Break_Confirmed',
             'ReturnIfExecuted']]

    def _compute_time_near_level(self, df, date, kind, window=10, threshold=0.01):
        date_index = df[df['Date'] == date].index
        if date_index.empty or date_index[0] < window:
            return 0
        i = date_index[0]
        ref_price = df['Close'].iloc[i]
        recent = df.iloc[i - window:i]
        count = (abs(recent['Close'] - ref_price) / ref_price < threshold).sum()
        return int(count)

    def identify_zones(self, ticker, window=200, freq_thresh=3, tolerance=0.01):
        df = self.df[self.df['Ticker'] == ticker].tail(window)
        levels = df['Close'].round(1)  # bucketize levels
        zone_counts = levels.value_counts()
        zone_candidates = zone_counts[zone_counts >= freq_thresh].index

        zone_df = pd.DataFrame({'Zone': zone_candidates})
        current_price = df['Close'].iloc[-1]
        zone_df['CurrentPrice'] = current_price
        zone_df['GapPct'] = (current_price - zone_df['Zone']) / zone_df['Zone']

        def classify(gap):
            if abs(gap) <= tolerance:
                return 'ATTENTION'
            elif gap > 0:
                return 'ABOVE'
            else:
                return 'BELOW'

        zone_df['Zone_Status'] = zone_df['GapPct'].apply(classify)
        zone_df = zone_df.sort_values('GapPct')
        return zone_df.reset_index(drop=True)

    def anticipate_price_action(self, ticker, window=200, freq_thresh=3, tolerance=0.01):
        zones = self.identify_zones(ticker, window, freq_thresh, tolerance)
        if zones.empty:
            return "NO ZONE"

        zone = zones[zones['Zone_Status'] == 'ATTENTION']
        if zone.empty:
            return "WAIT"

        recent = self.df[self.df['Ticker'] == ticker].tail(window).copy()
        if 'Volume_zscore' not in recent.columns:
            recent['Volume_zscore'] = recent.groupby('Ticker')['Volume'].transform(lambda x: (x - x.mean()) / x.std())

        latest_volume_z = recent['Volume_zscore'].iloc[-1]
        latest_price = recent['Close'].iloc[-1]
        trend = recent['Trend'].iloc[-1]

        if latest_volume_z > 0 and trend > 0:
            return f"BUY: near zone {zone['Zone'].values[0]} with confirming volume"
        elif latest_volume_z > 0 and trend < 0:
            return f"SELL: near zone {zone['Zone'].values[0]} with rejecting volume"
        else:
            return f"OBSERVE: near zone {zone['Zone'].values[0]} with weak volume"

    def anticipate_all(self, window=200, freq_thresh=3, tolerance=0.01):
        tickers = self.df['Ticker'].unique()
        results = []
        for ticker in tickers:
            decision = self.anticipate_price_action(ticker, window, freq_thresh, tolerance)
            results.append({"Ticker": ticker, "Signal": decision})
        return pd.DataFrame(results)

# Usage:
# pds = PriceDrivenStrategy()
# forward_result = pds.forward_strategy()
# rolling_result = pds.rolling_forward_strategy()
# print(rolling_result.head())
