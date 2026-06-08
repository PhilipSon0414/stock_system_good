import ssl
import certifi

# Python 3.14 macOS SSL 인증서 문제 해결
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta


def get_ohlcv(ticker: str, period_days: int = 400, end_date: str | None = None) -> pd.DataFrame:
    if end_date:
        end_dt = datetime.strptime(end_date, '%Y%m%d')
    else:
        end_dt = datetime.now()
    start = (end_dt - timedelta(days=period_days)).strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(ticker, start, end_str)
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


def get_ticker_list(market: str = 'ALL', date_str: str | None = None) -> pd.DataFrame:
    """
    Returns DataFrame with columns: Code, Name, Market, Close, Volume, Marcap
    date_str(YYYYMMDD): 해당 날짜 기준 종목 정보 (pykrx 사용). None이면 최신.
    """
    frames = []
    markets = ['KOSPI', 'KOSDAQ'] if market == 'ALL' else [market]

    # fdr 방식 시도
    fdr_ok = False
    if not date_str:
        for m in markets:
            try:
                df = fdr.StockListing(m)
                if df is not None and not df.empty:
                    df['Market'] = m
                    frames.append(df)
                    fdr_ok = True
            except Exception:
                pass

    # fdr 실패 또는 날짜 지정 시 pykrx 사용
    if not fdr_ok or date_str:
        frames = []
        try:
            from pykrx import stock as _stock
            target = date_str or datetime.now().strftime('%Y%m%d')
            for m in markets:
                try:
                    df_ok = _stock.get_market_ohlcv_by_ticker(target, market=m)
                    if df_ok is None or df_ok.empty:
                        continue
                    df_ok.index = df_ok.index.astype(str).str.zfill(6)
                    close_col = next((c for c in df_ok.columns if '종가' in c or c == 'Close'), None)
                    vol_col   = next((c for c in df_ok.columns if '거래량' in c or c == 'Volume'), None)
                    cap_col   = next((c for c in df_ok.columns if '시가총액' in c), None)
                    row = pd.DataFrame({
                        'Code':   df_ok.index,
                        'Name':   df_ok.index,
                        'Market': m,
                        'Close':  df_ok[close_col].values  if close_col else 0,
                        'Volume': df_ok[vol_col].values    if vol_col   else 0,
                        'Marcap': df_ok[cap_col].values    if cap_col   else 0,
                    })
                    frames.append(row)
                except Exception:
                    pass
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return result
