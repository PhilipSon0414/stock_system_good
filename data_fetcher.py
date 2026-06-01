import ssl
import certifi

# Python 3.14 macOS SSL 인증서 문제 해결
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta


def get_ohlcv(ticker: str, period_days: int = 400) -> pd.DataFrame:
    start = (datetime.now() - timedelta(days=period_days)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(ticker, start)
        if df is None or df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception:
        return pd.DataFrame()


def get_name(ticker: str) -> str:
    try:
        listing = fdr.StockListing('KRX')
        row = listing[listing['Code'] == ticker]
        if not row.empty:
            return row.iloc[0]['Name']
    except Exception:
        pass
    return ticker


def get_ticker_list(market: str = 'ALL') -> pd.DataFrame:
    """
    Returns DataFrame with columns: Code, Name, Market, Close, Volume, Marcap
    """
    frames = []
    markets = ['KOSPI', 'KOSDAQ'] if market == 'ALL' else [market]
    for m in markets:
        try:
            df = fdr.StockListing(m)
            df['Market'] = m
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return result
