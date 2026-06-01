"""
종목 스크리너

1단계: 당일 스냅샷(StockListing)으로 거래량/시가총액 1차 필터 (빠름)
2단계: 필터된 종목에 대해 상세 분석 (세력+OB+이평선+점수)
"""

import time
import pandas as pd
from data_fetcher import get_ohlcv, get_ticker_list
from indicators import add_all
from seoryeok import analyze
from order_block import get_order_blocks
from scorer import score
from config import (
    SCREEN_MIN_SCORE, SCREEN_MAX_RESULTS,
    SCREEN_MIN_VOLUME, SCREEN_MIN_PRICE
)


def _prefilter(market: str) -> list:
    """StockListing 기반 1차 필터: 고거래량 상위 종목만 선별"""
    listing = get_ticker_list(market)
    if listing.empty:
        return []

    vol_col = 'Volume' if 'Volume' in listing.columns else None
    price_col = 'Close' if 'Close' in listing.columns else None

    candidates = listing.copy()
    if price_col:
        candidates = candidates[candidates[price_col] >= SCREEN_MIN_PRICE]
    if vol_col:
        candidates = candidates[candidates[vol_col] >= SCREEN_MIN_VOLUME]
        candidates = candidates.sort_values(vol_col, ascending=False).head(200)

    return list(candidates['Code'].astype(str).str.zfill(6))


def run(market: str = 'ALL', verbose: bool = True) -> list:
    tickers = _prefilter(market)
    if not tickers:
        print("종목 목록을 가져오지 못했습니다. 네트워크 연결을 확인하세요.")
        return []

    if verbose:
        print(f"1차 필터: {len(tickers)}개 종목 정밀 분석 시작")

    results = []
    for i, ticker in enumerate(tickers):
        try:
            df = get_ohlcv(ticker, period_days=400)
            if len(df) < 60:
                continue

            close = df['Close'].iloc[-1]
            vol = df['Volume'].iloc[-1]
            if close < SCREEN_MIN_PRICE or vol < SCREEN_MIN_VOLUME:
                continue

            df = add_all(df)
            df = analyze(df)
            ob = get_order_blocks(df)
            pts, tags = score(df, ob)

            if pts >= SCREEN_MIN_SCORE:
                results.append({
                    'ticker': ticker,
                    'name': _name_from_listing(tickers, ticker),
                    'score': pts,
                    'price': close,
                    'vol_ratio': round(df['VolRatio'].iloc[-1], 2),
                    'tags': tags,
                    'df': df,
                    'ob': ob,
                })

        except Exception:
            pass

        if verbose and i % 25 == 0 and i > 0:
            print(f"  {i}/{len(tickers)} 분석 완료 (후보: {len(results)}개)")

        time.sleep(0.05)

    results.sort(key=lambda x: x['score'], reverse=True)
    top = results[:SCREEN_MAX_RESULTS]

    if verbose:
        _print_results(top)

    return top


# 종목 이름 캐시 (스크리닝 중 반복 호출 최소화)
_name_cache: dict = {}

def _name_from_listing(_, ticker: str) -> str:
    if ticker not in _name_cache:
        from data_fetcher import get_name
        _name_cache[ticker] = get_name(ticker)
    return _name_cache[ticker]


def _print_results(results: list):
    sep = '=' * 65
    print(f'\n{sep}')
    print(f'  스크리닝 결과 TOP {len(results)}')
    print(sep)
    for i, r in enumerate(results, 1):
        tags_str = ', '.join(r['tags'][:3])
        print(f"\n{i:2d}. [{r['ticker']}] {r['name']}")
        print(f"     점수: {r['score']}점  |  현재가: {r['price']:,.0f}원  |  거래량: {r['vol_ratio']}x")
        print(f"     신호: {tags_str}")
    print(f'\n{sep}')
