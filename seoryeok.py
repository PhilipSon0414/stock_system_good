"""
세력 행동 패턴 감지 모듈

패턴 정의:
  ENTRY      - 세력 진입: 거래량 폭증 + 양봉 + 이평선 돌파
  ACCUM      - 세력 매집: 저거래량 횡보 후 서서히 물량 모음
  BEAR_TRAP  - 세력 하락 유도: 음봉+거래량 폭증으로 개인 물량 털기 후 회복
  RALLY      - 상승 시작: 거래량 폭증 + 장대양봉 + 저항 돌파
  EXIT       - 세력 이탈: 고가권 음봉 + 거래량 폭증 (물량 던지기)
  PULLBACK   - 눌림목 매수 기회: 정배열 상태에서 이평선까지 되돌림
"""

import pandas as pd
import numpy as np
from config import (
    VOL_ENTRY_THRESHOLD, VOL_STRONG_THRESHOLD, VOL_QUIET_THRESHOLD,
    CONSOLIDATION_DAYS, CONSOLIDATION_RANGE
)


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    entry = pd.Series(False, index=df.index)
    accum = pd.Series(False, index=df.index)
    bear_trap = pd.Series(False, index=df.index)
    rally = pd.Series(False, index=df.index)
    exit_ = pd.Series(False, index=df.index)

    for i in range(CONSOLIDATION_DAYS + 20, n):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        vol_r = row.get('VolRatio', np.nan)
        if pd.isna(vol_r):
            continue

        ma20  = row.get('MA20',  np.nan)
        ma60  = row.get('MA60',  np.nan)
        ma120 = row.get('MA120', np.nan)

        # ── 세력 진입 ──────────────────────────────────────────────────────
        if vol_r >= VOL_ENTRY_THRESHOLD and row['IsBull'] and row['BodyRatio'] > 0.02:
            breaks_ma20  = not pd.isna(ma20)  and prev['Close'] < ma20  and row['Close'] >= ma20
            breaks_ma60  = not pd.isna(ma60)  and prev['Close'] < ma60  and row['Close'] >= ma60
            breaks_ma120 = not pd.isna(ma120) and prev['Close'] < ma120 and row['Close'] >= ma120
            if breaks_ma20 or breaks_ma60 or breaks_ma120:
                entry.iloc[i] = True

        # ── 세력 매집 ──────────────────────────────────────────────────────
        window = df.iloc[i - CONSOLIDATION_DAYS: i]
        avg_vol_r = window['VolRatio'].mean() if 'VolRatio' in window else np.nan
        if not pd.isna(avg_vol_r) and avg_vol_r < VOL_QUIET_THRESHOLD:
            price_range = (window['High'].max() - window['Low'].min()) / window['Close'].mean()
            if price_range < CONSOLIDATION_RANGE:
                near_ma = (
                    (not pd.isna(ma20)  and abs(row['Close'] - ma20)  / ma20  < 0.05) or
                    (not pd.isna(ma60)  and abs(row['Close'] - ma60)  / ma60  < 0.05) or
                    (not pd.isna(ma120) and abs(row['Close'] - ma120) / ma120 < 0.05)
                )
                if near_ma:
                    accum.iloc[i] = True

        # ── 세력 하락 유도 (베어트랩) ──────────────────────────────────────
        if i >= 2:
            prev2 = df.iloc[i - 2]
            trap_day = (
                prev['VolRatio'] >= VOL_ENTRY_THRESHOLD and
                not prev['IsBull'] and
                not pd.isna(ma20) and prev['Low'] < ma20
            )
            recovery = row['IsBull'] and not pd.isna(ma20) and row['Close'] > ma20
            if trap_day and recovery:
                bear_trap.iloc[i] = True

        # ── 상승 시작 ──────────────────────────────────────────────────────
        if vol_r >= VOL_STRONG_THRESHOLD and row['IsBull'] and row['BodyRatio'] > 0.03:
            look = df.iloc[max(0, i - CONSOLIDATION_DAYS): i]
            resistance = look['High'].max()
            if row['Close'] > resistance:
                rally.iloc[i] = True

        # ── 세력 이탈 ──────────────────────────────────────────────────────
        recent20 = df.iloc[max(0, i - 20): i]
        at_high = row['Close'] >= recent20['Close'].quantile(0.80)
        if at_high and vol_r >= VOL_ENTRY_THRESHOLD and not row['IsBull'] and row['BodyRatio'] > 0.02:
            exit_.iloc[i] = True

    df['SE_Entry'] = entry
    df['SE_Accum'] = accum
    df['SE_BearTrap'] = bear_trap
    df['SE_Rally'] = rally
    df['SE_Exit'] = exit_
    return df
