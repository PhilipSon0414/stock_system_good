"""
세력 진입 차트 패턴 감지

감지 패턴:
  1. 이중 바닥 (W패턴)       — 하락 추세 반전, 세력 매집 완료
  2. 박스권 상단 돌파 임박    — 횡보 후 폭발 직전
  3. 상승 삼각형              — 수평 저항 + 상승하는 지지선
  4. 눌림목 반등              — 이평선 지지 후 재상승 시작
  5. 볼린저밴드 스퀴즈        — 밴드 수축 후 폭발 준비
  6. 장기 횡보 후 상방 이탈   — 세력 매집 완료 후 출발
"""

import numpy as np
import pandas as pd


# ── 보조 함수 ─────────────────────────────────────────────────────────────

def _local_minima(series: pd.Series, order: int = 5) -> list[int]:
    """로컬 저점 인덱스 목록"""
    mins = []
    vals = series.values
    for i in range(order, len(vals) - order):
        if vals[i] == min(vals[i - order: i + order + 1]):
            mins.append(i)
    return mins


def _local_maxima(series: pd.Series, order: int = 5) -> list[int]:
    """로컬 고점 인덱스 목록"""
    maxs = []
    vals = series.values
    for i in range(order, len(vals) - order):
        if vals[i] == max(vals[i - order: i + order + 1]):
            maxs.append(i)
    return maxs


def _bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0):
    mid  = df['Close'].rolling(period).mean()
    band = df['Close'].rolling(period).std() * std
    return mid, mid + band, mid - band


# ── 패턴 1: 이중 바닥 (W패턴) ────────────────────────────────────────────

def detect_double_bottom(df: pd.DataFrame, window: int = 80) -> tuple[int, str]:
    recent = df.iloc[-window:] if len(df) > window else df
    lows   = recent['Low']
    mins   = _local_minima(lows, order=4)

    if len(mins) < 2:
        return 0, ''

    # 최근 2개 저점
    b1, b2 = mins[-2], mins[-1]
    if b2 - b1 < 8 or b2 - b1 > 50:   # 8~50일 간격
        return 0, ''

    p1 = lows.iloc[b1]
    p2 = lows.iloc[b2]
    if abs(p1 - p2) / p1 > 0.04:       # 두 저점이 4% 이내
        return 0, ''

    # 두 저점 사이 고점 (넥라인)
    neckline = recent['High'].iloc[b1:b2].max()
    current  = recent['Close'].iloc[-1]
    dist     = (current - neckline) / neckline

    if -0.02 <= dist <= 0.05:           # 넥라인 돌파 직전/직후
        score = 25 if dist >= 0 else 18
        return score, f'이중바닥(W) 넥라인{"돌파" if dist >= 0 else "근접"}'
    elif -0.08 <= dist < -0.02:
        return 10, '이중바닥(W) 형성 중'

    return 0, ''


# ── 패턴 2: 박스권 상단 돌파 임박 ────────────────────────────────────────

def detect_box_breakout(df: pd.DataFrame, box_days: int = 20) -> tuple[int, str]:
    if len(df) < box_days + 5:
        return 0, ''

    recent = df.iloc[-box_days:]
    hi     = recent['High'].max()
    lo     = recent['Low'].min()
    rng    = (hi - lo) / lo if lo != 0 else 1.0
    cur    = df['Close'].iloc[-1]

    if rng > 0.12:                      # 박스권 폭이 12%+ 면 박스 아님
        return 0, ''

    dist_to_top = (hi - cur) / hi      # 현재가 → 박스 상단 거리

    if dist_to_top < 0.01:             # 1% 이내 = 돌파 직전
        return 22, f'박스권 상단 돌파 임박 (범위:{rng*100:.1f}%)'
    elif dist_to_top < 0.03:
        return 14, f'박스권 상단 근접 ({dist_to_top*100:.1f}% 이내)'
    elif dist_to_top < 0.06:
        return 7, f'박스권 중간 이상'

    return 0, ''


# ── 패턴 3: 상승 삼각형 ───────────────────────────────────────────────────

def detect_ascending_triangle(df: pd.DataFrame, window: int = 40) -> tuple[int, str]:
    if len(df) < window:
        return 0, ''

    recent = df.iloc[-window:]
    highs  = recent['High']
    lows   = recent['Low']

    # 고점이 수평 저항 (최고점 ±2% 내)
    max_hi = highs.max()
    hi_flat = (highs.tail(window // 2) > max_hi * 0.98).sum() >= 3

    if not hi_flat:
        return 0, ''

    # 저점이 상승 추세
    min_idxs = _local_minima(lows, order=3)
    if len(min_idxs) < 2:
        return 0, ''

    recent_mins = [lows.iloc[i] for i in min_idxs[-3:]]
    rising_lows = all(recent_mins[i] < recent_mins[i + 1]
                      for i in range(len(recent_mins) - 1))

    if not rising_lows:
        return 0, ''

    cur       = df['Close'].iloc[-1]
    dist_top  = (max_hi - cur) / max_hi

    if dist_top < 0.02:
        return 20, '상승삼각형 저항 돌파 직전'
    elif dist_top < 0.05:
        return 12, '상승삼각형 형성'

    return 0, ''


# ── 패턴 4: 눌림목 반등 완성 ─────────────────────────────────────────────

def detect_pullback_bounce(df: pd.DataFrame) -> tuple[int, str]:
    if len(df) < 30:
        return 0, ''

    latest  = df.iloc[-1]
    prev    = df.iloc[-2]
    ma5     = latest.get('MA5', np.nan)
    ma20    = latest.get('MA20', np.nan)
    ma60    = latest.get('MA60', np.nan)

    # 이평선 정배열 확인
    if any(pd.isna(v) for v in [ma5, ma20, ma60]):
        return 0, ''
    if not (ma5 > ma20 > ma60):
        return 0, ''

    cur = latest['Close']

    # 최근 2~7일 저점이 이평선 근처였나
    low_window = df.iloc[-7:-1]['Low']
    near_ma20  = any(abs(v - ma20) / ma20 < 0.025 for v in low_window)
    near_ma5   = any(abs(v - ma5)  / ma5  < 0.015 for v in low_window)

    # 오늘 반등 시작 (양봉)
    bouncing = latest['IsBull'] and latest['Close'] > prev['Close']

    if bouncing and (near_ma20 or near_ma5):
        ma_touched = 'MA5' if near_ma5 else 'MA20'
        # 거래량 감소 후 소폭 증가면 더 신뢰
        vol_ratio = latest.get('VolRatio', 1)
        if 0.5 < vol_ratio < 2.0:
            return 18, f'{ma_touched} 눌림목 반등 시작'
        return 12, f'{ma_touched} 눌림목 반등 시작'

    return 0, ''


# ── 패턴 5: 볼린저밴드 스퀴즈 + 상방 이탈 ───────────────────────────────

def detect_bb_squeeze(df: pd.DataFrame) -> tuple[int, str]:
    if len(df) < 25:
        return 0, ''

    _, bb_up, bb_dn = _bollinger(df)
    width     = ((bb_up - bb_dn) / df['Close']).dropna()

    if len(width) < 10:
        return 0, ''

    cur_width  = width.iloc[-1]
    avg_width  = width.iloc[-60:].mean() if len(width) >= 60 else width.mean()
    rel_width  = cur_width / avg_width if avg_width > 0 else 1.0

    cur_price  = df['Close'].iloc[-1]
    upper      = bb_up.iloc[-1]
    lower      = bb_dn.iloc[-1]

    if rel_width < 0.5:                # 밴드 폭이 평균의 50% 이하 = 강한 스퀴즈
        if cur_price >= upper * 0.99:  # 상단 밴드 터치 = 상방 이탈
            return 22, f'BB 스퀴즈 후 상방 이탈'
        return 15, f'BB 강한 스퀴즈 (폭:{rel_width*100:.0f}%)'
    elif rel_width < 0.7:
        return 8, f'BB 스퀴즈 진행 중'

    return 0, ''


# ── 패턴 6: 장기 횡보 후 상방 이탈 ──────────────────────────────────────

def detect_long_consolidation_breakout(df: pd.DataFrame) -> tuple[int, str]:
    """30일+ 횡보 후 첫 상방 돌파"""
    if len(df) < 35:
        return 0, ''

    window = df.iloc[-35:-3]           # 최근 3일 제외한 32일
    rng    = (window['High'].max() - window['Low'].min()) / window['Close'].mean()
    cur    = df['Close'].iloc[-1]
    res    = window['High'].max()

    if rng > 0.10:                     # 박스권 조건 아님
        return 0, ''

    recent3_max = df.iloc[-3:]['Close'].max()
    if recent3_max > res * 1.01:       # 3일 내 돌파 발생
        return 18, f'장기 횡보({rng*100:.1f}%) 후 상방 이탈'

    return 0, ''


# ── 패턴 7: 연속 저거래량 후 거래량 폭발 ─────────────────────────────────

def detect_vol_squeeze_breakout(df: pd.DataFrame) -> tuple[int, str]:
    """연속 저거래량 구간 후 갑작스러운 거래량 증가 = 세력 출동 직전/직후"""
    if 'VolRatio' not in df.columns or len(df) < 8:
        return 0, ''

    vol = df['VolRatio'].dropna()
    if len(vol) < 5:
        return 0, ''

    # 전일까지 연속 저거래량 일수 (당일 제외)
    quiet_days = 0
    for v in vol.iloc[:-1].values[::-1]:
        if v < 0.70:
            quiet_days += 1
        else:
            break

    today_vol = vol.iloc[-1]

    if quiet_days >= 5 and today_vol >= 1.5:
        return 28, f'저거래량 {quiet_days}일 후 거래량 폭발 (세력 출동)'
    elif quiet_days >= 3 and today_vol >= 1.2:
        return 18, f'저거래량 {quiet_days}일 후 거래량 증가'
    elif quiet_days >= 5:
        return 12, f'저거래량 {quiet_days}일 지속 (대기 중)'
    elif quiet_days >= 3:
        return 7, f'저거래량 {quiet_days}일 지속'

    return 0, ''


# ── 통합 패턴 점수 ────────────────────────────────────────────────────────

def score_patterns(df: pd.DataFrame) -> tuple[int, list[str]]:
    """모든 패턴 점수 합산 (0~100)"""
    pts  = 0
    tags = []

    for fn in [
        detect_double_bottom,
        detect_box_breakout,
        detect_ascending_triangle,
        detect_pullback_bounce,
        detect_bb_squeeze,
        detect_long_consolidation_breakout,
        detect_vol_squeeze_breakout,
    ]:
        try:
            s, label = fn(df)
            if s > 0 and label:
                pts += s
                tags.append(label)
        except Exception:
            pass

    return min(100, pts), tags
