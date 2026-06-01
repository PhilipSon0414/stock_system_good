"""
오더블록(Order Block) 탐지 — 개선판

강세 OB (Bullish OB):
  상승 직전 마지막 하락 캔들 → 세력이 매수한 구간.
  가격이 다시 돌아오면 재매수 기회.

약세 OB (Bearish OB):
  하락 직전 마지막 상승 캔들 → 세력이 매도한 구간.
  OB Flip: 가격이 위로 돌파해서 지금은 지지로 전환된 구간.

OB 분류 체계:
  ① 신선 강세OB  (visit=0, 가격 위에 존재) → 가장 강력한 지지
  ② 접촉 강세OB  (visit 1+)                 → 지지 약화 중
  ③ OB 플립      (약세OB를 위로 돌파)        → 전 저항이 지지로 전환 (핵심 신호)
  ④ 활성 약세OB  (아직 돌파 못한 저항)       → 저항 구간
"""

import pandas as pd
import numpy as np
from config import OB_MIN_MOVE, OB_LOOKBACK, OB_NEAR_TOLERANCE


def _find_obs(df: pd.DataFrame, bullish_type: bool) -> list:
    obs = []
    n = len(df)
    lookahead = 8

    for i in range(1, n - lookahead):
        candle = df.iloc[i]
        is_correct_type = not candle['IsBull'] if bullish_type else candle['IsBull']
        if not is_correct_type or candle['BodyRatio'] < 0.005:
            continue

        for j in range(i + 1, min(i + lookahead + 1, n)):
            future = df.iloc[j]
            if bullish_type:
                denom = candle['Low']
                move = (future['High'] - denom) / denom if denom != 0 else 0
            else:
                denom = candle['High']
                move = (denom - future['Low']) / denom if denom != 0 else 0

            if move < OB_MIN_MOVE:
                continue

            has_prior = False
            for k in range(i + 1, j):
                mid = df.iloc[k]
                wrong_dir = mid['IsBull'] if bullish_type else not mid['IsBull']
                if wrong_dir and mid['BodyRatio'] > 0.01:
                    has_prior = True
                    break

            if not has_prior:
                date_val = candle.get('Date', df.index[i]) if hasattr(candle, 'get') else df.index[i]
                obs.append({
                    'Date':       pd.Timestamp(date_val) if not isinstance(date_val, pd.Timestamp) else date_val,
                    'Type':       'Bullish' if bullish_type else 'Bearish',
                    'High':       candle['High'],
                    'Low':        candle['Low'],
                    'Mid':        (candle['High'] + candle['Low']) / 2,
                    'Idx':        i,
                    'vol_ratio':  candle.get('VolRatio', 1.0) or 1.0,
                    'body_ratio': candle.get('BodyRatio', 0.01) or 0.01,
                })
                break

    return obs


def _count_visits(df_slice: pd.DataFrame, ob_low: float, ob_high: float) -> int:
    """OB 형성 이후 가격이 해당 구간에 재진입한 횟수"""
    visits = 0
    inside = False
    for i in range(len(df_slice)):
        row = df_slice.iloc[i]
        in_zone = row['Low'] <= ob_high and row['High'] >= ob_low
        if in_zone and not inside:
            visits += 1
            inside = True
        elif not in_zone:
            inside = False
    return visits


def get_order_blocks(df: pd.DataFrame) -> dict:
    recent = df.iloc[-OB_LOOKBACK:] if len(df) > OB_LOOKBACK else df
    recent = recent.reset_index(drop=False)
    price  = df['Close'].iloc[-1]

    bull_obs = _find_obs(recent, bullish_type=True)
    bear_obs = _find_obs(recent, bullish_type=False)

    # ── 강세 OB: 가격이 Low 위에 있으면 유효 ─────────────────────────────
    active_bull = []
    for ob in bull_obs:
        if ob['Low'] > price:
            continue
        after = recent.iloc[ob['Idx'] + 1:]
        ob['visit_count'] = _count_visits(after, ob['Low'], ob['High'])
        ob['is_fresh']    = ob['visit_count'] == 0
        ob['is_strong']   = ob['vol_ratio'] >= 2.0
        active_bull.append(ob)

    # ── 약세 OB ───────────────────────────────────────────────────────────
    # 활성: 가격이 High 아래 (아직 저항)
    # 플립: 가격이 High 위 (저항 돌파 → 지지로 전환)
    active_bear  = []
    flipped_bear = []
    for ob in bear_obs:
        after = recent.iloc[ob['Idx'] + 1:]
        ob['visit_count'] = _count_visits(after, ob['Low'], ob['High'])
        if price > ob['High']:
            ob['is_flipped'] = True
            flipped_bear.append(ob)
        else:
            ob['is_flipped'] = False
            active_bear.append(ob)

    # ── 근접 판정 (거리 기반) ─────────────────────────────────────────────
    def _dist_to_bull(ob):
        if price < ob['Low']:
            return (ob['Low'] - price) / price
        if price > ob['High']:
            return (price - ob['High']) / price
        return 0.0  # 진입 중

    def _dist_to_bear(ob):
        # 아직 돌파 못한 활성 약세OB: 가격과 Low 사이 거리
        return (ob['Low'] - price) / price if ob['Low'] > price else 0.0

    def _dist_to_flipped(ob):
        # 플립 OB: 가격과 High 사이 거리 (High가 이제 지지선)
        return (price - ob['High']) / price if price > ob['High'] else 0.0

    # 가장 가까운 강세 OB까지 거리
    nearest_bull_dist = None
    nearest_bull      = None
    if active_bull:
        ranked = sorted(active_bull, key=_dist_to_bull)
        nearest_bull      = ranked[0]
        nearest_bull_dist = _dist_to_bull(nearest_bull)

    # 가장 가까운 활성 약세 OB까지 거리
    nearest_bear_dist = None
    if active_bear:
        nearest_bear_dist = min(_dist_to_bear(ob) for ob in active_bear)

    # 가장 가까운 플립 OB까지 거리
    nearest_flipped_dist = None
    if flipped_bear:
        nearest_flipped_dist = min(_dist_to_flipped(ob) for ob in flipped_bear)

    # 이진 플래그 (하위 호환)
    near_bull    = nearest_bull_dist is not None and nearest_bull_dist <= OB_NEAR_TOLERANCE
    near_bear    = nearest_bear_dist is not None and nearest_bear_dist <= OB_NEAR_TOLERANCE
    near_flipped = nearest_flipped_dist is not None and nearest_flipped_dist <= OB_NEAR_TOLERANCE * 2

    # 컨플루언스: 현재가 ±5% 안에 강세 OB 2개+
    confluence_bull = sum(1 for ob in active_bull if _dist_to_bull(ob) <= 0.05) >= 2

    return {
        # 목록
        'bull':          active_bull,
        'bear':          active_bear,
        'flipped_bear':  flipped_bear,
        # 이진 플래그 (기존 호환)
        'near_bull':     near_bull,
        'near_bear':     near_bear,
        'near_flipped':  near_flipped,
        # 거리 (세분화 점수에 사용)
        'nearest_bull_dist':    nearest_bull_dist,
        'nearest_bear_dist':    nearest_bear_dist,
        'nearest_flipped_dist': nearest_flipped_dist,
        # 품질 플래그
        'bull_fresh':     nearest_bull is not None and nearest_bull['is_fresh'],
        'bull_strong':    nearest_bull is not None and nearest_bull['is_strong'],
        'confluence_bull': confluence_bull,
    }
