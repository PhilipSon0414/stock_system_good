"""
종목 투자 점수 산출 (0~100)

점수 구성:
  이평선 정배열    20점
  세력 진입/매집   25점
  오더블록 근접    20점
  골든크로스       10점
  눌림목           10점
  상승 시작        15점
  세력 재발화      최대 +35점 보너스

페널티:
  세력 이탈 신호  -25점
  이평선 역배열   -10점
"""

import pandas as pd
import numpy as np


def get_accum_info(df: pd.DataFrame) -> dict:
    """SE_Accum 클러스터 분석 — 진짜 재발화(침묵 후 재출현) 감지

    클러스터 정의: SE_Accum 신호들 사이 간격이 30일+ 이상이면 새 클러스터로 구분.

    is_reactivating: 이전 클러스터 이력 3회+ AND 현재 클러스터 활성 (최근 10일 내 신호)
    is_fresh_refire: 현재 클러스터가 20일 이내에 시작 → 2~4주 내 급등 임박 후보
    days_since_refire: 현재 클러스터 첫 신호로부터 경과일
    """
    empty = {
        'days_since': None, 'is_reactivating': False, 'history_count': 0,
        'hist_count': 0, 'days_since_refire': None, 'is_fresh_refire': False,
    }
    if 'SE_Accum' not in df.columns or len(df) < 30:
        return empty

    accum_dates = df.index[df['SE_Accum'] == True]
    if len(accum_dates) == 0:
        return empty

    last_accum_date = accum_dates[-1]
    days_since      = (df.index[-1] - last_accum_date).days

    # 현재 신호가 10일 이상 없으면 매집 비활성
    if days_since > 10:
        return {**empty, 'days_since': days_since, 'history_count': len(accum_dates)}

    # 클러스터 분리: 신호 간 간격 30일+ → 새 클러스터
    CLUSTER_GAP = 30
    cluster_starts = [accum_dates[0]]
    for k in range(1, len(accum_dates)):
        if (accum_dates[k] - accum_dates[k - 1]).days > CLUSTER_GAP:
            cluster_starts.append(accum_dates[k])

    current_start     = cluster_starts[-1]
    days_since_refire = (df.index[-1] - current_start).days
    hist_count        = len(accum_dates[accum_dates < current_start])

    # 재발화 = 이전 클러스터 이력 3회+ & 현재 클러스터 존재
    is_reactivating = hist_count >= 3
    # 신선 재발화 = 현재 클러스터가 20일 이내에 시작
    is_fresh_refire = is_reactivating and days_since_refire <= 20

    return {
        'days_since':        days_since,
        'is_reactivating':   is_reactivating,
        'history_count':     len(accum_dates),
        'hist_count':        hist_count,
        'days_since_refire': days_since_refire,
        'is_fresh_refire':   is_fresh_refire,
    }


def score(df: pd.DataFrame, ob_info: dict) -> tuple[int, list[str]]:
    if len(df) < 60:
        return 0, []

    pts = 0
    tags = []
    latest = df.iloc[-1]
    w5 = df.iloc[-5:]

    # 이평선 정배열 — 현재가 MA20 위일 때만 유효 (버그 수정: 고점 이후 하락 중인 종목 오탐 방지)
    close_price = latest.get('Close', 0)
    ma20_val    = latest.get('MA20',  0)
    price_above_ma20 = close_price >= ma20_val * 0.98 if ma20_val > 0 else False  # -2% 허용

    if latest.get('MaBull', False):
        if price_above_ma20:
            pts += 20
            tags.append('이평선 정배열')
        else:
            pts += 5   # 이평 정배열이나 현재가 MA20 아래 → 약한 신호만
            tags.append('이평선 정배열(가격MA20아래)')
    elif latest.get('MA5', 0) > latest.get('MA20', 0):
        if price_above_ma20:
            pts += 10
            tags.append('단기 정배열(5>20)')
        else:
            pts += 3
            tags.append('단기 정배열(가격MA20아래)')

    # 골든크로스
    if w5['GoldenCross'].any() if 'GoldenCross' in w5 else False:
        pts += 10
        tags.append('골든크로스 발생')

    # 세력 진입
    if w5['SE_Entry'].any():
        pts += 25
        tags.append('세력 진입 신호')
    elif w5['SE_Accum'].any():
        pts += 15
        tags.append('세력 매집 구간')

    # 베어트랩 (하락 유도 후 회복)
    if w5['SE_BearTrap'].any():
        pts += 12
        tags.append('베어트랩 회복')

    # 상승 시작
    if w5['SE_Rally'].any():
        pts += 15
        tags.append('상승 시작 신호')

    # 눌림목
    if 'Pullback' in latest.index and latest['Pullback']:
        pts += 10
        tags.append('눌림목 매수 기회')

    # 오더블록 — 거리·품질 세분화 점수
    bull_dist = ob_info.get('nearest_bull_dist')
    if bull_dist is not None:
        if bull_dist == 0.0:
            pts += 25; tags.append('강세OB 진입 중')
        elif bull_dist < 0.01:
            pts += 22; tags.append(f'강세OB 근접 ({bull_dist*100:.1f}%)')
        elif bull_dist < 0.02:
            pts += 18; tags.append(f'강세OB 접근 ({bull_dist*100:.1f}%)')
        elif bull_dist < 0.04:
            pts += 12; tags.append(f'강세OB 4% 이내')
        elif bull_dist < 0.07:
            pts += 6;  tags.append(f'강세OB 7% 이내')
        elif ob_info.get('bull'):
            pts += 3;  tags.append('강세OB 존재')

        # 품질 보너스 (중복 가산)
        if ob_info.get('bull_fresh') and bull_dist < 0.04:
            pts += 6; tags.append('미접촉 강세OB (최초 방문)')
        if ob_info.get('bull_strong') and bull_dist < 0.04:
            pts += 4; tags.append('고거래량 강세OB')
        if ob_info.get('confluence_bull'):
            pts += 7; tags.append('강세OB 중첩 (컨플루언스)')

    # OB 플립: 전 저항을 위로 돌파 → 지지로 전환 (핵심 신호)
    flip_dist = ob_info.get('nearest_flipped_dist')
    if flip_dist is not None:
        if flip_dist < 0.02:
            pts += 18; tags.append(f'OB 플립 지지 ({flip_dist*100:.1f}% 위)')
        elif flip_dist < 0.05:
            pts += 12; tags.append(f'OB 플립 지지 ({flip_dist*100:.1f}% 위)')
        elif ob_info.get('near_flipped'):
            pts += 7;  tags.append('OB 플립 지지 근접')

    # 세력 재발화 (침묵 후 재출현 — 진짜 재발화만 인정)
    accum_info = get_accum_info(df)
    if accum_info['is_reactivating']:
        pts += 25
        dsr = accum_info['days_since_refire']
        if accum_info['is_fresh_refire']:
            pts += 15  # 20일 이내 신선 재발화 → 급등 임박 보너스
            tags.append(f'★ 급등 임박 재발화 ({dsr}일 전 재출현, 이력 {accum_info["hist_count"]}회)')
        else:
            tags.append(f'세력 재발화 ({dsr}일 전 재출현, 이력 {accum_info["hist_count"]}회)')
        if accum_info['hist_count'] >= 10:
            pts += 10
            tags.append('장기 매집 이력 (10회+)')

    # ── 페널티 ────────────────────────────────────────────────────────────
    if w5['SE_Exit'].any():
        pts -= 25
        tags.append('⚠ 세력 이탈 신호')

    if latest.get('MaBear', False):
        pts -= 10
        tags.append('⚠ 이평선 역배열')

    # 활성 약세OB 근접 — 아직 돌파 못한 저항 (플립된 것은 이미 위에서 가산)
    bear_dist = ob_info.get('nearest_bear_dist')
    if bear_dist is not None and bear_dist < 0.02:
        pts -= 5
        tags.append(f'⚠ 약세OB 저항 ({bear_dist*100:.1f}% 위)')

    return max(0, min(100, pts)), tags
