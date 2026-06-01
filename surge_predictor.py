"""
급상승 임박 예측 스코어러

세력 매집이 완료되고 곧 급등이 시작될 종목을 탐지.
높은 점수 = 단기간 내 급등 가능성 높음.

주요 판단 기준 (백테스트 최적화 기준):
  1. 이평선 배열/가격 위치 — MA(20,60,120) 정배열 + MA120 위 (1.54x lift)
  2. 거래량 압축   — 세력이 물량 모은 후 아무것도 안 하는 구간
  3. 횡보 기간     — 길게 눌릴수록 반등 폭이 큼
  4. 골든크로스 임박 — MA20/MA60 크로스 (2.15x lift, 기존 MA5/MA20: 0.62x)
  5. 지지 반복 테스트 — 같은 가격대를 여러 번 지지받은 경우
  6. 강세 오더블록 근접 — 세력 매수 구간에 가격이 돌아옴
"""

import pandas as pd
import numpy as np
from pathlib import Path as _Path


def score_surge(df: pd.DataFrame, ob_info: dict) -> tuple[int, list[str]]:
    if len(df) < 60:
        return 0, []

    pts = 0
    tags = []
    latest = df.iloc[-1]

    ma20  = latest.get('MA20',  np.nan)
    ma60  = latest.get('MA60',  np.nan)
    ma120 = latest.get('MA120', np.nan)
    close = latest['Close']

    # ── 1. 이평선 배열 및 가격 위치 (최대 25점) ────────────────────────────
    # 백테스트: MA(20,60,120) 정배열 1.54x lift / MA120 위 1.26x lift
    if not any(pd.isna(v) for v in [ma20, ma60, ma120]):
        if ma20 > ma60 > ma120:
            pts += 20; tags.append('MA 정배열 (20>60>120)')
            if close > ma120:
                pts += 5; tags.append('MA120 위 (장기 지지)')
        elif ma20 > ma60:
            pts += 12; tags.append('MA 단기 정배열 (20>60)')
            if close > ma120:
                pts += 5; tags.append('MA120 위 (장기 지지)')
    elif not any(pd.isna(v) for v in [ma20, ma60]):
        if ma20 > ma60:
            pts += 12; tags.append('MA 단기 정배열 (20>60)')

    # ── 2. 거래량 분석 (최대 20점) ──────────────────────────────────────
    # 백테스트: D-1 ≥2x → Lift 17.03x / ≥3x → 10.36x / 0.5~0.7x → lift<1.0x(역신호)
    vol_r = latest.get('VolRatio', np.nan)
    if not pd.isna(vol_r):
        if vol_r >= 5.0:
            pts += 20; tags.append('거래량 폭증 (5x+) — 세력 진입')
        elif vol_r >= 3.0:
            pts += 16; tags.append('거래량 폭증 (3x+)')
        elif vol_r >= 2.0:
            pts += 12; tags.append('거래량 상승 (2x+) — 돌파 신호')
        elif vol_r < 0.30:
            pts += 10; tags.append('거래량 극도 압축 (<30%) — 세력 잠복')
        elif vol_r < 0.50:
            pts += 4;  tags.append('거래량 강한 압축 (<50%)')
        # 0.50~0.70 구간: 백테스트 lift<1.0x → 점수 없음

    # ── 2b. 잠복 일수 + 압축→폭발 전환 감지 (최대 35점) ────────────────────
    # 전환 패턴 = 과거 연속 저거래량 + 현재 거래량 상승 시작
    if 'VolRatio' in df.columns and len(df) >= 3:
        vol_vals = df['VolRatio'].dropna().values
        # 오늘(latest) 제외하고 직전부터 거슬러 연속 저거래량 일수
        prev_vals = vol_vals[:-1] if len(vol_vals) > 1 else vol_vals
        quiet_streak = 0
        for v in reversed(prev_vals):
            if v < 0.70:
                quiet_streak += 1
            else:
                break

        if not pd.isna(vol_r) and vol_r >= 2.0 and quiet_streak >= 5:
            pts += 35; tags.append(f'★ 압축→폭발 전환 ({quiet_streak}일 잠복 후 2x+ 상승)')
        elif not pd.isna(vol_r) and vol_r >= 1.5 and quiet_streak >= 3:
            pts += 22; tags.append(f'압축→폭발 조짐 ({quiet_streak}일 잠복 후 1.5x+ 상승)')
        elif quiet_streak >= 7:
            pts += 18; tags.append(f'세력 잠복 {quiet_streak}일 연속 저거래량')
        elif quiet_streak >= 5:
            pts += 10; tags.append(f'연속 저거래량 {quiet_streak}일')
        elif quiet_streak >= 3:
            pts += 5;  tags.append(f'연속 저거래량 {quiet_streak}일')

    # ── 3. 횡보 기간 및 박스권 압축 (최대 20점) ───────────────────────────
    for days, label, score in [(20, '20일', 20), (15, '15일', 14), (10, '10일', 8)]:
        if len(df) >= days:
            w = df.iloc[-days:]
            rng = (w['High'].max() - w['Low'].min()) / close
            if rng < 0.03:
                pts += score; tags.append(f'{label} 초박스권 (<3%)'); break
            elif rng < 0.05:
                pts += int(score * 0.7); tags.append(f'{label} 박스권 (<5%)'); break
            elif rng < 0.08:
                pts += int(score * 0.4); tags.append(f'{label} 횡보 (<8%)'); break

    # ── 4. 골든크로스 임박 (최대 15점) — MA20/MA60 기준, 백테스트 2.15x lift ──
    if not any(pd.isna(v) for v in [ma20, ma60]):
        gap_ratio = (ma60 - ma20) / ma60  # 양수면 MA20 < MA60 (골든크로스 전)
        if 0 < gap_ratio < 0.003:
            pts += 15; tags.append('MA20/60 골든크로스 임박 (<0.3%)')
        elif 0 < gap_ratio < 0.008:
            pts += 10; tags.append('MA20/60 골든크로스 근접 (<0.8%)')
        elif 0 < gap_ratio < 0.015:
            pts += 5;  tags.append('MA20/60 골든크로스 접근 (<1.5%)')
        elif gap_ratio <= 0:
            # 이미 골든크로스 상태 — 최근 크로스인지 확인
            if len(df) >= 3:
                prev_gap = (df.iloc[-3].get('MA60', 0) - df.iloc[-3].get('MA20', 0)) / max(df.iloc[-3].get('MA60', 1), 1)
                if prev_gap > 0:
                    pts += 12; tags.append('MA20/60 골든크로스 최근 발생')

    # ── 5. 지지선 반복 테스트 (최대 10점) ────────────────────────────────
    if len(df) >= 30:
        w30 = df.iloc[-30:]
        support_zone_lo = close * 0.97
        support_zone_hi = close * 1.01
        touches = sum(
            1 for _, row in w30.iterrows()
            if support_zone_lo <= row['Low'] <= support_zone_hi
        )
        if touches >= 4:
            pts += 10; tags.append(f'지지선 {touches}회 반복 테스트')
        elif touches >= 2:
            pts += 5;  tags.append(f'지지선 {touches}회 테스트')

    # ── 6. 오더블록 근접 (최대 15점) ────────────────────────────────────
    bull_dist = ob_info.get('nearest_bull_dist')
    if bull_dist is not None:
        if bull_dist == 0.0:
            pts += 15; tags.append('강세OB 진입 중')
        elif bull_dist < 0.02:
            pts += 12; tags.append(f'강세OB 근접 ({bull_dist*100:.1f}%)')
        elif bull_dist < 0.05:
            pts += 7;  tags.append('강세OB 5% 이내')
        if ob_info.get('bull_fresh') and bull_dist < 0.05:
            pts += 5; tags.append('미접촉 강세OB')

    flip_dist = ob_info.get('nearest_flipped_dist')
    if flip_dist is not None and flip_dist < 0.04:
        pts += 10; tags.append(f'OB 플립 지지 ({flip_dist*100:.1f}%)')

    # ── 7. NR7 / VCP 변동성 압축 (최대 15점) ────────────────────────────
    if latest.get('VCP', False):
        pts += 15; tags.append('VCP 변동성 수축 — 폭발 직전')
    elif latest.get('NR7', False):
        pts += 10; tags.append('NR7 가격 압축 (7일 최소 범위)')

    # ── 8. OBV 다이버전스 (최대 15점) ────────────────────────────────────
    if latest.get('OBV_Diverge', False):
        pts += 15; tags.append('OBV 다이버전스 — 세력 매집 중')

    # ── 9. 52주 신고가 근접 (최대 15점) ──────────────────────────────────
    dist_52w = latest.get('High52W_Dist', np.nan)
    if not pd.isna(dist_52w):
        if dist_52w <= 0.02:
            pts += 15; tags.append(f'52주 신고가 근접 ({dist_52w*100:.1f}% 이내) — 저항 없는 구간')
        elif dist_52w <= 0.05:
            pts += 10; tags.append(f'52주 신고가 5% 이내')
        elif dist_52w <= 0.10:
            pts += 5;  tags.append(f'52주 신고가 10% 이내')

    # ── 페널티 ────────────────────────────────────────────────────────────
    w5 = df.iloc[-5:]
    if 'SE_Exit' in w5.columns and w5['SE_Exit'].any():
        pts -= 30; tags.append('⚠ 세력이탈 신호')
    if latest.get('MaBear', False):
        pts -= 20; tags.append('⚠ 이평선 역배열')
    if 'SE_Entry' in w5.columns and w5['SE_Entry'].any():
        pts -= 5   # 이미 진입 완료 = 저점 아님

    # 최근 급등 후면 점수 감점 (기준 완화: 25%+ 만 강하게 감점)
    if len(df) >= 5:
        gain5d = (close - df['Close'].iloc[-5]) / df['Close'].iloc[-5]
        if gain5d > 0.25:
            pts -= 10; tags.append('⚠ 최근 5일 25%+ 이미 상승')
        elif gain5d > 0.15:
            pts -= 4; tags.append('⚠ 최근 5일 15%+ 이미 상승')

    return max(0, min(100, pts)), tags


def score_surge_with_history(df: pd.DataFrame, ob_info: dict,
                              score_history: list[dict]) -> tuple[int, list[str]]:
    """score_history(최근 스캔 기록)를 반영한 급등 점수.

    score_history: [{'combined': int, 'seoryeok': int, 'surge': int}, ...] 최신순 정렬
    백테스트 결과 반영:
      - co_mean ≥ 54 & co_slope ≥ 3  → +20pt (lift 6.0x)
      - 세력 ≥ 75 & co_mean ≥ 54      → +15pt (lift 5.0x)
      - 세력 ≥ 70 & 전일합산+5 & co_mean ≥ 45 → +15pt (lift 5.0x)
      - co_slope < 0 (하락 추세)       → -10pt
      - 연속 등장 5일 이상             → -15pt (lift 0.00x 확인)
    """
    base_pts, base_tags = score_surge(df, ob_info)

    if not score_history or len(score_history) < 2:
        return base_pts, base_tags

    bonus = 0
    bonus_tags = []

    combined_series = [h['combined'] for h in score_history]
    seoryeok_series = [h['seoryeok'] for h in score_history]
    n = len(combined_series)

    # 5일 합산 평균 및 기울기
    co_mean = float(np.mean(combined_series))
    if n >= 3:
        x = np.arange(n, dtype=float)
        co_slope = float(np.polyfit(x, combined_series, 1)[0])
    else:
        co_slope = float(combined_series[-1] - combined_series[0]) if n >= 2 else 0.0

    se_last = seoryeok_series[-1]
    co_last = combined_series[-1]
    co_d1_chg = combined_series[-1] - combined_series[-2] if n >= 2 else 0

    consec = n  # 연속 등장 일수 근사

    # ── 백테스트 기반 보너스 규칙 ───────────────────────────────────────
    # R4: co_mean ≥ 54 & co_slope ≥ 3  (lift 6.0x, 적중 40%)
    if co_mean >= 54 and co_slope >= 3:
        bonus += 20
        bonus_tags.append(f'★ 합산평균{co_mean:.0f} 기울기+{co_slope:.1f} (R4 최강 패턴)')

    # R2: 세력 ≥ 75 & co_mean ≥ 54  (lift 5.0x, 적중 33%)
    elif se_last >= 75 and co_mean >= 54:
        bonus += 15
        bonus_tags.append(f'★ 세력{se_last}+합산평균{co_mean:.0f} (R2 패턴)')

    # R8: 세력 ≥ 70 & 전일합산+5 & co_mean ≥ 45  (lift 5.0x)
    elif se_last >= 70 and co_d1_chg >= 5 and co_mean >= 45:
        bonus += 15
        bonus_tags.append(f'★ 세력{se_last}+전일+{co_d1_chg}+평균{co_mean:.0f} (R8 패턴)')

    # 기울기 양수 보너스 (소폭)
    elif co_slope >= 2 and se_last >= 65:
        bonus += 8
        bonus_tags.append(f'기울기 상승세 (+{co_slope:.1f}/일)')

    # ── 페널티 규칙 ─────────────────────────────────────────────────────
    # 기울기 음수: 합산 점수 하락 추세
    if co_slope < -2:
        bonus -= 10
        bonus_tags.append(f'⚠ 합산점수 하락추세 (기울기{co_slope:.1f})')

    # 연속 5일+ 등장: 백테스트에서 lift 0.00x
    if consec >= 5:
        penalty = min((consec - 4) * 5, 20)
        bonus -= penalty
        bonus_tags.append(f'⚠ 연속{consec}일 등장 — 급등 시그널 약화 (-{penalty}pt)')

    # 현재가 MA20 대비 -10% 이상 이탈: 하락 추세 확인 페널티
    # (MA120 없어도 MA20만으로 단기 추세 파악 가능)
    ma20 = latest.get('MA20', np.nan)
    if not np.isnan(ma20) and ma20 > 0:
        price_vs_ma20 = (close - ma20) / ma20
        if price_vs_ma20 < -0.15:
            bonus -= 20
            bonus_tags.append(f'⚠ 현재가 MA20 -{abs(price_vs_ma20)*100:.0f}% 이탈 (하락 추세)')
        elif price_vs_ma20 < -0.08:
            bonus -= 10
            bonus_tags.append(f'⚠ 현재가 MA20 -{abs(price_vs_ma20)*100:.0f}% 이탈')

    final_pts = max(0, min(100, base_pts + bonus))
    final_tags = bonus_tags + base_tags
    return final_pts, final_tags


def combined_score(seoryeok_pts: int, surge_pts: int,
                   investor_pts: int = 0, pattern_pts: int = 0) -> int:
    """세력(25%) + 급등임박(30%) + 수급(25%) + 패턴(20%) 합산"""
    from config import W_SEORYEOK, W_SURGE, W_INVESTOR, W_PATTERN
    return round(
        seoryeok_pts * W_SEORYEOK +
        surge_pts    * W_SURGE    +
        investor_pts * W_INVESTOR +
        pattern_pts  * W_PATTERN
    )
