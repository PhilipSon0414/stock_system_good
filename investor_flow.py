"""
외국인 · 기관 수급 분석

네이버 금융 (frgn.naver) 에서 기관/외국인 순매매량을 파싱.
로그인 없이 공개 페이지를 사용.

점수 기준:
  기관 3일+ 연속 순매수         → +30점
  외국인 3일+ 연속 순매수        → +30점
  외국인+기관 오늘 동시 매수     → +15점 보너스
  기관 역발상 매수 (주가 하락 중) → +10점
  세력 매집: 외국인↓ + 기관↑    → +20점 (핵심 신호)
  외국인 5일+ 연속 순매도 (단독) → -8점
"""

import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

_HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
_BASE_URL = 'https://finance.naver.com/item/frgn.naver'


def _fetch_page(ticker: str, page: int = 1) -> pd.DataFrame:
    try:
        r = requests.get(_BASE_URL, params={'code': ticker, 'page': page},
                         headers=_HEADERS, timeout=8)
        soup = BeautifulSoup(r.text, 'html.parser')
        tables = soup.find_all('table')
        # 날짜/종가/기관/외국인 테이블 탐색
        for table in tables:
            ths = [th.text.strip() for th in table.find_all('th')]
            if '기관' in ths and '외국인' in ths:
                rows = table.find_all('tr')
                data = []
                for row in rows[1:]:
                    cols = row.find_all('td')
                    if len(cols) < 7:
                        continue
                    date   = cols[0].text.strip()
                    close  = cols[1].text.strip().replace(',', '')
                    inst   = cols[5].text.strip().replace(',', '').replace('+', '')
                    frgn   = cols[6].text.strip().replace(',', '').replace('+', '')
                    if not date or not date[0].isdigit():
                        continue
                    try:
                        data.append({
                            'Date':    pd.to_datetime(date),
                            'Close':   int(close.replace(',', '')),
                            'Inst':    int(inst) if inst and inst != '-' else 0,
                            'Foreign': int(frgn) if frgn and frgn != '-' else 0,
                        })
                    except ValueError:
                        continue
                if data:
                    return pd.DataFrame(data).set_index('Date').sort_index()
    except Exception:
        pass
    return pd.DataFrame()


def _parse_end_date(end_date: str | None):
    """'YYYY-MM-DD' 또는 'YYYYMMDD' → datetime. None이면 None."""
    if not end_date:
        return None
    from datetime import datetime
    fmt = '%Y-%m-%d' if '-' in end_date else '%Y%m%d'
    return datetime.strptime(end_date, fmt)


def get_investor_df(ticker: str, days: int = 20,
                    end_date: str | None = None) -> pd.DataFrame:
    """기관/외국인 순매매 DataFrame (최근 days일).
    end_date 지정 시 해당일까지의 데이터만 사용 (백데이트 스캔 시 미래 누수 차단)."""
    page1 = _fetch_page(ticker, 1)
    if len(page1) < days:
        page2 = _fetch_page(ticker, 2)
        if not page2.empty:
            page1 = pd.concat([page2, page1]).sort_index()
    if page1.empty:
        return pd.DataFrame()
    end_dt = _parse_end_date(end_date)
    if end_dt is not None:
        page1 = page1[page1.index <= end_dt]
    return page1.tail(days) if not page1.empty else pd.DataFrame()


def _streak(series: pd.Series) -> int:
    """연속 양수(+) 일수 반환. 음수면 음의 연속 일수."""
    if series.empty:
        return 0
    vals = list(series.values)[::-1]  # 최근순
    sign = 1 if vals[0] > 0 else -1
    count = 0
    for v in vals:
        if (v > 0) == (sign > 0):
            count += 1
        else:
            break
    return sign * count


def get_pykrx_investor_df(ticker: str, days: int = 20,
                          end_date: str | None = None) -> pd.DataFrame:
    """pykrx 기반 기관/외국인 순매매 (네이버 대비 안정적, 더 많은 일수 지원).
    end_date 지정 시 해당일까지만 조회 (백데이트 스캔 시 미래 누수 차단)."""
    try:
        from pykrx import stock
        from datetime import datetime, timedelta
        end_dt = _parse_end_date(end_date) or datetime.now()
        end   = end_dt.strftime('%Y%m%d')
        start = (end_dt - timedelta(days=days * 2)).strftime('%Y%m%d')
        df = stock.get_market_trading_volume_by_date(start, end, ticker)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={'기관합계': 'Inst', '외국인합계': 'Foreign'})
        return df[['Inst', 'Foreign']].tail(days)
    except Exception:
        return pd.DataFrame()


def get_investor_flow_stats(ticker: str, end_date: str | None = None) -> dict:
    """
    기관/외국인 수급 통계 (pykrx 기반).
    end_date: 스캔 기준일 — 지정 시 해당일까지의 데이터만 사용.
    Returns:
        inst_streak     : 기관 연속 순매수 일수 (음수=매도)
        frgn_streak     : 외국인 연속 순매수 일수
        inst_transition : 기관 매도→매수 전환 여부
        frgn_transition : 외국인 매도→매수 전환 여부
        inst_5d_sum     : 기관 5일 누적 순매수
        frgn_5d_sum     : 외국인 5일 누적 순매수
        combo_buy       : 기관+외국인 동시 매수 일수 (최근 5일)
    """
    df = get_pykrx_investor_df(ticker, days=20, end_date=end_date)
    if df.empty or len(df) < 3:
        return {}

    inst = df['Inst']
    frgn = df['Foreign']

    def transition(series):
        """이전 3일 매도 → 최근 2일 매수 전환 여부"""
        if len(series) < 5:
            return False
        prev   = series.iloc[-5:-2]
        recent = series.iloc[-2:]
        return any(v < 0 for v in prev) and all(v > 0 for v in recent)

    combo_buy_days = sum(1 for i, f in zip(inst.tail(5), frgn.tail(5)) if i > 0 and f > 0)

    return {
        'inst_streak':     _streak(inst),
        'frgn_streak':     _streak(frgn),
        'inst_transition': transition(inst),
        'frgn_transition': transition(frgn),
        'inst_5d_sum':     int(inst.tail(5).sum()),
        'frgn_5d_sum':     int(frgn.tail(5).sum()),
        'combo_buy_days':  combo_buy_days,
    }


def score_investors(ticker: str, ohlcv_df: pd.DataFrame,
                    end_date: str | None = None) -> tuple[int, list[str]]:
    """외국인+기관 수급 점수 (0~100) — pykrx 우선, 네이버 폴백.
    end_date: 스캔 기준일('YYYY-MM-DD'|'YYYYMMDD') — 지정 시 해당일까지의
    수급만 사용한다. 백데이트 스캔/학습 시 급등 이후 수급이 섞이는 누수 차단."""
    # pykrx 데이터 우선 시도
    pykrx_df = get_pykrx_investor_df(ticker, days=20, end_date=end_date)
    if not pykrx_df.empty and len(pykrx_df) >= 3:
        df = pykrx_df
    else:
        df = get_investor_df(ticker, days=20, end_date=end_date)

    if df.empty or len(df) < 3:
        return 0, ['수급데이터 없음']

    pts = 0
    tags = []

    frgn = df['Foreign']
    inst = df['Inst']

    fstreak = _streak(frgn)
    istreak = _streak(inst)

    # ── 외국인 ────────────────────────────────────────────────────────────
    if fstreak >= 5:
        pts += 30; tags.append(f'외국인 {fstreak}일 연속 순매수')
    elif fstreak >= 3:
        pts += 22; tags.append(f'외국인 {fstreak}일 연속 순매수')
    elif fstreak >= 1:
        pts += 10; tags.append(f'외국인 순매수 중')
    elif fstreak <= -5:
        # 기관이 사고 있으면 세력 매집 패턴 — 패널티 없음 (아래 별도 처리)
        if istreak < 3:
            pts -= 8; tags.append(f'외국인 {abs(fstreak)}일 연속 순매도')
    elif fstreak <= -3:
        if istreak < 1:
            pts -= 3; tags.append(f'외국인 순매도 중')

    # 외국인 5일 누적
    f5 = frgn.tail(5).sum()
    if f5 > 0:
        tags.append(f'외국인 5일 누적 +{f5:,.0f}주')

    # ── 기관 ──────────────────────────────────────────────────────────────
    if istreak >= 5:
        pts += 30; tags.append(f'기관 {istreak}일 연속 순매수')
    elif istreak >= 3:
        pts += 22; tags.append(f'기관 {istreak}일 연속 순매수')
    elif istreak >= 1:
        pts += 10; tags.append(f'기관 순매수 중')
    elif istreak <= -5:
        pts -= 10; tags.append(f'기관 {abs(istreak)}일 연속 순매도')
    elif istreak <= -3:
        pts -= 5; tags.append(f'기관 순매도 중')

    # ── 세력 매집 신호: 외국인 매도 + 기관 매수 ─────────────────────────
    # 개인 물량 털기(외국인) + 기관 저가 매집 = 세력 매집 완료 직전
    if fstreak <= -3 and istreak >= 3:
        pts += 20; tags.append(f'세력 매집: 외국인 {abs(fstreak)}일 매도+기관 {istreak}일 매수')
    elif fstreak < 0 and istreak >= 2:
        pts += 10; tags.append('외국인↓ 기관↑ 매집 가능성')

    # ── 동시 매수 보너스 ──────────────────────────────────────────────────
    today_f = frgn.iloc[-1]
    today_i = inst.iloc[-1]
    if today_f > 0 and today_i > 0:
        pts += 15; tags.append('외국인+기관 동시 순매수')

    # ── 역발상 매수: 주가 하락 중 기관 순매수 ─────────────────────────────
    if not ohlcv_df.empty and len(ohlcv_df) >= 5:
        price_change_5d = (ohlcv_df['Close'].iloc[-1] - ohlcv_df['Close'].iloc[-5]) / ohlcv_df['Close'].iloc[-5]
        inst_5d = inst.tail(5).sum()
        if price_change_5d < -0.05 and inst_5d > 0:
            pts += 10; tags.append('역발상 기관 매수 (주가하락 중 기관 순매수)')

    # ── 전환 신호 (매도→매수) — 가장 강한 진입 신호 ─────────────────────
    def is_transition(series):
        if len(series) < 5:
            return False
        prev   = series.iloc[-5:-2]
        recent = series.iloc[-2:]
        return any(v < 0 for v in prev) and all(v > 0 for v in recent)

    if is_transition(frgn):
        pts += 15; tags.append('★ 외국인 순매도→순매수 전환 (핵심 신호)')
    elif len(frgn) >= 4 and frgn.iloc[-4:-1].mean() < 0 and frgn.iloc[-1] > 0:
        pts += 8;  tags.append('외국인 순매도→순매수 전환')

    if is_transition(inst):
        pts += 15; tags.append('★ 기관 순매도→순매수 전환 (핵심 신호)')

    # ── 기관+외국인 동시 매수 연속 ────────────────────────────────────────
    combo_days = sum(1 for i_v, f_v in zip(inst.tail(5), frgn.tail(5)) if i_v > 0 and f_v > 0)
    if combo_days >= 3:
        pts += 20; tags.append(f'기관+외국인 동시매수 {combo_days}일 연속')
    elif combo_days >= 2:
        pts += 10; tags.append(f'기관+외국인 동시매수 {combo_days}일')

    # ── 누적 순매수 볼륨 ─────────────────────────────────────────────────
    inst_10d = int(inst.tail(10).sum())
    frgn_10d = int(frgn.tail(10).sum())
    if inst_10d > 0 and frgn_10d > 0:
        tags.append(f'10일 누적: 기관+{inst_10d:,}주 / 외국인+{frgn_10d:,}주')
    elif inst_10d > 0:
        tags.append(f'10일 누적 기관 +{inst_10d:,}주')
    elif frgn_10d > 0:
        tags.append(f'10일 누적 외국인 +{frgn_10d:,}주')

    return max(0, min(100, pts)), tags
