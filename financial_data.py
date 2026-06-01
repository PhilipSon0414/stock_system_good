"""
종목별 재무지표 (PER, PBR, 시가총액) 조회

스캔 첫 호출 시 KOSPI+KOSDAQ 전 종목을 bulk fetch → 메모리 캐시.
이후 get_financials()는 캐시 조회만 하므로 스캔 속도에 영향 없음.
"""

import pandas as pd
from datetime import datetime, timedelta

_fund_cache: dict[str, pd.DataFrame] = {}   # date_str → fundamental DataFrame
_cap_cache:  dict[str, pd.DataFrame] = {}   # date_str → market cap DataFrame


def _prev_trading_day(date_str: str) -> str:
    """주말이면 직전 금요일 날짜 반환"""
    dt = datetime.strptime(date_str, '%Y%m%d')
    while dt.weekday() >= 5:       # 토(5), 일(6)
        dt -= timedelta(days=1)
    return dt.strftime('%Y%m%d')


def _load(date_str: str):
    """bulk fetch — 이미 캐시에 있으면 skip"""
    if date_str in _fund_cache:
        return

    date_str = _prev_trading_day(date_str)

    try:
        from pykrx import stock as pkrx
        frames_f, frames_c = [], []
        for market in ('KOSPI', 'KOSDAQ'):
            try:
                df_f = pkrx.get_market_fundamental(date_str, market=market)
                if df_f is not None and not df_f.empty:
                    frames_f.append(df_f)
            except Exception:
                pass
            try:
                df_c = pkrx.get_market_cap(date_str, market=market)
                if df_c is not None and not df_c.empty:
                    frames_c.append(df_c)
            except Exception:
                pass

        _fund_cache[date_str] = pd.concat(frames_f) if frames_f else pd.DataFrame()
        _cap_cache[date_str]  = pd.concat(frames_c) if frames_c else pd.DataFrame()
        total = len(_fund_cache[date_str])
        if total:
            print(f'  [재무] {date_str} 전 종목 재무지표 로드 완료 ({total}개)')
    except Exception as e:
        print(f'  [재무] bulk fetch 실패: {e}')
        _fund_cache[date_str] = pd.DataFrame()
        _cap_cache[date_str]  = pd.DataFrame()


def get_financials(ticker: str, date_str: str | None = None) -> dict:
    """
    PER, PBR, 시가총액 반환.
    조회 실패 또는 적자 종목 → per/pbr = None.
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y%m%d')
    date_str = _prev_trading_day(date_str)

    _load(date_str)

    result: dict = {'per': None, 'pbr': None, 'market_cap': None}

    fund_df = _fund_cache.get(date_str, pd.DataFrame())
    if not fund_df.empty and ticker in fund_df.index:
        row = fund_df.loc[ticker]
        per = float(row.get('PER', 0) or 0)
        pbr = float(row.get('PBR', 0) or 0)
        result['per'] = per if per > 0 else None
        result['pbr'] = pbr if pbr > 0 else None

    cap_df = _cap_cache.get(date_str, pd.DataFrame())
    if not cap_df.empty and ticker in cap_df.index:
        cap = int(cap_df.loc[ticker].get('시가총액', 0) or 0)
        result['market_cap'] = cap if cap > 0 else None

    return result


def fmt_market_cap(cap: int | None) -> str:
    """시가총액 → 읽기 쉬운 문자열"""
    if cap is None:
        return '-'
    if cap >= 10_000_000_000_000:          # 10조+
        return f'{cap / 1_000_000_000_000:.0f}조'
    if cap >= 1_000_000_000_000:           # 1조+
        return f'{cap / 1_000_000_000_000:.1f}조'
    return f'{cap // 100_000_000:,}억'


def fmt_financials(fin: dict) -> str:
    """PER | PBR ★ | 시총 한 줄 포맷"""
    parts = []

    per = fin.get('per')
    parts.append(f'PER {per:.1f}배' if per else 'PER -')

    pbr = fin.get('pbr')
    if pbr:
        note = ' ★저평가' if pbr < 1.0 else ''
        parts.append(f'PBR {pbr:.2f}배{note}')
    else:
        parts.append('PBR -')

    parts.append(f'시총 {fmt_market_cap(fin.get("market_cap"))}')

    return ' | '.join(parts)
