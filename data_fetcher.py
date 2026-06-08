import ssl
import certifi

# Python 3.14 macOS SSL 인증서 문제 해결
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta

# 종목명 캐시 — 프로세스 전체에서 한 번만 KRX 조회
_name_cache: dict[str, str] = {}
_listing_cache: pd.DataFrame | None = None


def _ensure_name_cache():
    """KRX 전체 종목명을 한 번만 조회해 _name_cache에 로드한다."""
    global _name_cache, _listing_cache
    if _name_cache:
        return
    try:
        listing = fdr.StockListing('KRX')
        if listing is not None and not listing.empty:
            _listing_cache = listing
            for _, row in listing.iterrows():
                code = str(row.get('Code', '')).zfill(6)
                name = str(row.get('Name', code))
                if code:
                    _name_cache[code] = name
    except Exception:
        pass


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
    """종목코드로 종목명 반환. 캐시 미스 시 KRX 전체 조회(1회)."""
    code = str(ticker).zfill(6)
    # 1) 캐시 확인
    if code in _name_cache:
        return _name_cache[code]
    # 2) 캐시 없으면 전체 로드 후 재조회
    _ensure_name_cache()
    return _name_cache.get(code, ticker)


def get_ticker_list(market: str = 'ALL') -> pd.DataFrame:
    """
    Returns DataFrame with columns: Code, Name, Market, Close, Volume, Marcap
    종목명 캐시도 함께 갱신한다.
    """
    frames = []
    markets = ['KOSPI', 'KOSDAQ'] if market == 'ALL' else [market]
    for m in markets:
        try:
            df = fdr.StockListing(m)
            df['Market'] = m
            frames.append(df)
            # 이름 캐시 갱신
            for _, row in df.iterrows():
                code = str(row.get('Code', '')).zfill(6)
                name = str(row.get('Name', code))
                if code:
                    _name_cache[code] = name
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return result
