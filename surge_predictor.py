"""
급상승 임박 예측 스코어러

[2026-06-03 학습 갱신] 국내 2,468건 급등 역추적 분석 결과 반영:
  ① RangeRel d=+0.335, BoxRange5d d=+0.267 → D-1은 범위 확대 중 (압축→이완 전환)
  ② VolRatio d=+0.184, VolSlope5 d=+0.156 → 거래량 점진 증가 (1.29x) 신호
  ③ QuietStreak d=-0.134 → 잠복 기간 자체가 아닌 '잠복 후 회복 전환'이 핵심
  ④ NR7 d=-0.124 → 한/미 동일: NR7 단독 역방향. VolRecovery 조합 필수
  ⑤ RSI d=-0.086 → 급등 D-1 RSI=53.6 (중립) — 과매수 아님 신호
  ⑥ ATRCompress 1.09→1.13 확대 추세 — 변동성 이완 신호 추가

기존 '압축→폭발 전환' 로직 방향 유효 확인 (2x+/5일+ 잠복 후 폭발 = 핵심).
단, 순수 잠복 일수 점수 하향, 중간 거래량 회복(1.0~2.0x) 구간 신설.
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
    # [학습 갱신] D-1 급등군 VolRatio=1.29 vs 대조군 1.06 (d=+0.184)
    # 폭증뿐 아니라 1.0~2.0x 중간 활성 구간도 유의미한 신호로 확인
    vol_r     = latest.get('VolRatio',   np.nan)
    vol_slope = latest.get('VolSlope5',  np.nan)  # [신규] 5일 거래량 기울기
    vol_rec   = bool(latest.get('VolRecovery', False))  # [신규] 잠복→회복 전환

    if not pd.isna(vol_r):
        if vol_r >= 5.0:
            pts += 20; tags.append('거래량 폭증 (5x+) — 세력 진입')
        elif vol_r >= 3.0:
            pts += 16; tags.append('거래량 폭증 (3x+)')
        elif vol_r >= 2.0:
            pts += 12; tags.append('거래량 상승 (2x+) — 돌파 신호')
        elif vol_r >= 1.2:
            pts += 7;  tags.append(f'거래량 활성 {vol_r:.1f}x — 모멘텀 형성')  # [신규]
        elif vol_r < 0.30:
            # [학습 갱신] 순수 극압축 단독 점수 하향 (10→5점): 잠복 자체보다 전환이 핵심
            pts += 5;  tags.append('거래량 극압축 (<30%) — 회복 전환 대기')
        elif vol_r < 0.50:
            pts += 2;  tags.append('거래량 압축 (<50%)')
        # 0.50~0.70 구간: lift<1.0x → 점수 없음

    # [신규] VolSlope5: 5일 거래량 증가 추세 (학습 d=+0.156)
    if not pd.isna(vol_slope) and vol_slope > 0 and not pd.isna(vol_r) and vol_r < 2.0:
        pts += 5; tags.append('거래량 5일 증가 추세 (VolSlope↑)')

    # ── 2b. 잠복 일수 + 압축→폭발 전환 감지 (최대 35점) ────────────────────
    # [학습 확인] '압축→폭발 전환(VolRecovery)' 방향 유효. 순수 잠복 점수 하향.
    # QuietStreak d=-0.134: 잠복 중 상태 자체보다 잠복이 끝나는 시점이 핵심.
    if 'VolRatio' in df.columns and len(df) >= 3:
        vol_vals  = df['VolRatio'].dropna().values
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
        elif not pd.isna(vol_r) and vol_r >= 1.2 and quiet_streak >= 3:
            # [신규] 중간 회복형: 잠복 후 1.2x+ 상승도 유효 신호로 추가
            pts += 15; tags.append(f'잠복 후 거래량 회복 ({quiet_streak}일→{vol_r:.1f}x)')
        elif quiet_streak >= 7:
            # [학습 갱신] 순수 잠복 7일 → 18점에서 10점으로 하향
            pts += 10; tags.append(f'세력 잠복 {quiet_streak}일 (회복 대기)')
        elif quiet_streak >= 5:
            pts += 6;  tags.append(f'연속 저거래량 {quiet_streak}일')
        elif quiet_streak >= 3:
            pts += 3;  tags.append(f'연속 저거래량 {quiet_streak}일')

    # [신규] VolRecovery 복합 신호 (indicators.py 계산값 활용)
    if vol_rec:
        pts += 8; tags.append('★ 거래량 잠복→회복 전환 (VolRecovery)')

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

    # ── 7. NR7 / VCP 변동성 압축 ──────────────────────────────────────────
    # [학습 갱신] NR7 d=-0.124 (역방향): 단독 점수 하향, VolRecovery 조합 시 완전 점수
    if latest.get('VCP', False):
        pts += 15; tags.append('VCP 변동성 수축 — 폭발 직전')
    elif latest.get('NR7', False):
        if vol_rec or (not pd.isna(vol_slope) and vol_slope > 0):
            pts += 12; tags.append('NR7 + 거래량 회복 복합 — 폭발 임박')
        else:
            pts += 5;  tags.append('NR7 가격 압축 (거래량 회복 대기)')

    # ── 8. OBV 다이버전스 (최대 15점) ────────────────────────────────────
    if latest.get('OBV_Diverge', False):
        pts += 15; tags.append('OBV 다이버전스 — 세력 매집 중')

    # ── 8b. ATR 모멘텀 [신규] ─────────────────────────────────────────────
    # 학습: ATRCompress D-5→D-1 = 1.091→1.125 (점진 확대 = 폭발 준비)
    atr_c = float(latest.get('ATRCompress', np.nan))
    if not pd.isna(atr_c):
        if atr_c >= 1.15:
            pts += 7; tags.append(f'ATR 모멘텀 확대 ({atr_c:.2f}x) — 변동성 이완')
        elif atr_c >= 1.05:
            pts += 3; tags.append(f'ATR 소폭 확대 ({atr_c:.2f}x)')

    # ── 8c. RSI 회복 구간 [신규] ─────────────────────────────────────────
    # 학습: 급등 D-1 RSI=53.6 vs 대조군 55.1 — 과매수 아닌 중립 구간
    rsi14 = float(latest.get('RSI14', np.nan))
    if not pd.isna(rsi14):
        if 30 <= rsi14 <= 57:
            pts += 5; tags.append(f'RSI {rsi14:.0f} 회복 구간 (30~57) — 과매수 아님')
        elif rsi14 > 75:
            pts -= 5; tags.append(f'RSI {rsi14:.0f} 과매수 — 급등 확률 저하')

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

    10% 기준 재검증 결과 반영 (2026-06):
      - 세력≥80 & 합산≥45  → lift 1.92x (핵심)
      - 세력 60-69          → lift 0.62x (노이즈 — 패널티)
      - 합산 70+            → lift 0.52x (연속보너스 허수 — 패널티)
      - co_mean ≥ 54 & co_slope ≥ 3  → +20pt (이전 기준 유지)
      - 세력 ≥ 75 & co_mean ≥ 54      → +15pt
      - 세력 ≥ 70 & 전일합산+5 & co_mean ≥ 45 → +15pt
      - co_slope < 0 (하락 추세)       → -10pt
      - 연속 등장 5일 이상             → 페널티
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

    # 기울기 양수 보너스 (소폭, 세력 70+만)
    elif co_slope >= 2 and se_last >= 70:
        bonus += 8
        bonus_tags.append(f'기울기 상승세 (+{co_slope:.1f}/일)')

    # ── 10% 검증 기반 노이즈 패널티 ──────────────────────────────────
    # 세력 60-69: lift 0.62x → 페널티
    if 60 <= se_last < 70:
        bonus -= 5
        bonus_tags.append(f'⚠ 세력{se_last} 노이즈 구간 (60-69 lift 0.62x)')
    # 합산 70+: lift 0.52x → 연속보너스 허수 패널티
    if co_last >= 70:
        bonus -= 8
        bonus_tags.append(f'⚠ 합산{co_last} 고점 (연속보너스 허수, lift 0.52x)')

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
    latest = df.iloc[-1]
    close  = latest['Close']
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
    """세력(40%) + 급등임박(20%) + 수급(25%) + 패턴(15%) 합산
    데이터 분석 기반 수정 (2026-06):
      세력점수 = 유일한 양의 신호 (+5.9점 차이) → 비중 대폭 증가
      급등점수 = 오히려 역효과 (-2.2점) → 비중 축소
    """
    from config import W_SEORYEOK, W_SURGE, W_INVESTOR, W_PATTERN
    return round(
        seoryeok_pts * W_SEORYEOK +
        surge_pts    * W_SURGE    +
        investor_pts * W_INVESTOR +
        pattern_pts  * W_PATTERN
    )


def predict_surge_timing(seoryeok: int, vol_ratio: float,
                          ma_bull: bool, consec_days: int) -> dict:
    """
    급등 '발생 시' 예상 소요일 통계 (조건부 — 생존자 편향 주의).

    ⚠ 주의: 아래 수치는 '스캔 후 실제로 급등한 57건'만 조건화한 통계다.
      급등하지 않은 스캔은 분모에 없으므로 confidence_pct는 '급등 확률'이
      아니라 '급등한 케이스 중 3일 내였던 비율'이다. 급등 여부 자체의
      확률은 surge_prob(GBM)·티어 적중률을 봐야 한다.

    분석 결과 (57건 스캔→급등 케이스):
      거래량 2x+ & 세력80+  → 평균 2.0일 / 세력 70-79 → 평균 2.4일
      MA정배열 → 평균 1.4일 / 세력80+ → 평균 3.5일

    Returns:
        expected_days    : 급등 시 예상 소요 거래일
        confidence_pct   : 급등 케이스 중 3거래일 내 비율 (%) — 급등 확률 아님
        timing_label     : 설명 문자열
        holding_advice   : 보유 기간 권고
    """
    # 조건별 소요일 + 급등 케이스 중 3일내 비율 (생존자 통계 — 급등 확률 아님)
    if vol_ratio >= 2.0 and seoryeok >= 80:
        exp_days, conf3d = 2.0, 100
        label   = '★ 거래량폭발+세력 — 급등 시 즉시~2일내 (과거 급등례 기준)'
        holding = '진입 후 1~3거래일 집중 모니터링'
    elif ma_bull and seoryeok >= 65:
        exp_days, conf3d = 1.4, 100
        label   = '★ MA정배열+세력 — 급등 시 1~2일내 (과거 급등례 기준)'
        holding = '진입 후 1~2거래일 단기 홀딩'
    elif 70 <= seoryeok < 80:
        exp_days, conf3d = 2.4, 78
        label   = '세력70대 — 급등 시 2~3일내 (급등례 중 78%)'
        holding = '진입 후 3거래일 보유 권고'
    elif seoryeok >= 80:
        exp_days, conf3d = 3.5, 59
        label   = '세력80+ — 급등 시 3~5일내 (급등례 중 59%)'
        holding = '진입 후 3~5거래일 보유'
    elif vol_ratio < 0.3 and seoryeok >= 65:
        exp_days, conf3d = 4.1, 42
        label   = '거래량 압축 — 폭발 대기 중 (4~5일 소요 예상)'
        holding = '진입 후 5거래일 보유 필요'
    else:
        exp_days, conf3d = 5.0, 35
        label   = '일반 패턴 — 4~7일 이후 급등 가능'
        holding = '최소 5거래일 보유'

    return {
        'expected_days':  exp_days,
        'confidence_pct': conf3d,
        'timing_label':   label,
        'holding_advice': holding,
    }
