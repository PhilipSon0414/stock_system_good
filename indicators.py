import pandas as pd
import numpy as np
from config import MA_PERIODS, VOLUME_AVG_PERIOD


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _moving_averages(df)
    df = _volume_stats(df)
    df = _candle_props(df)
    df = _ma_alignment(df)
    df = _pullback(df)
    df = _nr7_vcp(df)
    df = _obv(df)
    df = _high_52w(df)
    return df


def _moving_averages(df):
    for p in MA_PERIODS:
        df[f'MA{p}'] = df['Close'].rolling(p).mean()
    return df


def _volume_stats(df):
    df['VolMA'] = df['Volume'].rolling(VOLUME_AVG_PERIOD).mean()
    df['VolRatio'] = df['Volume'] / df['VolMA'].replace(0, np.nan)
    df['VolRatio'] = df['VolRatio'].replace([np.inf, -np.inf], np.nan)
    return df


def _candle_props(df):
    df['IsBull'] = df['Close'] >= df['Open']
    df['Body'] = (df['Close'] - df['Open']).abs()
    df['BodyRatio'] = df['Body'] / df['Close'].replace(0, np.nan)
    df['UpperWick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['LowerWick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Change'] = df['Close'].pct_change()
    return df


def _ma_alignment(df):
    has = lambda col: col in df.columns and not df[col].isna().all()

    # 정배열/역배열: MA(20,60,120) — 백테스트 결과 1.54x lift (vs 기존 5/20/60: 1.35x)
    if has('MA20') and has('MA60') and has('MA120'):
        df['MaBull'] = (df['MA20'] > df['MA60']) & (df['MA60'] > df['MA120'])
        df['MaBear'] = (df['MA20'] < df['MA60']) & (df['MA60'] < df['MA120'])
    elif has('MA20') and has('MA60'):
        df['MaBull'] = df['MA20'] > df['MA60']
        df['MaBear'] = df['MA20'] < df['MA60']

    # 골든크로스: MA20/MA60 — 백테스트 결과 2.15x lift (vs 기존 MA5/MA20: 0.62x)
    if has('MA20') and has('MA60'):
        prev_cross = (df['MA20'].shift(1) <= df['MA60'].shift(1)) & (df['MA20'] > df['MA60'])
        df['GoldenCross'] = prev_cross

    return df


def _pullback(df):
    """눌림목: 상승 추세 중 이평선까지 되돌림"""
    from config import PULLBACK_MA_TOLERANCE
    if 'MA20' not in df.columns:
        return df
    near_ma20 = (df['Low'] - df['MA20']).abs() / df['MA20'].replace(0, np.nan) < PULLBACK_MA_TOLERANCE
    near_ma60 = (df['Low'] - df['MA60']).abs() / df['MA60'].replace(0, np.nan) < PULLBACK_MA_TOLERANCE if 'MA60' in df.columns else False
    df['Pullback'] = df.get('MaBull', False) & (near_ma20 | near_ma60) & (df['VolRatio'] < 1.0)
    return df


def _nr7_vcp(df):
    """1순위: NR7(7일 최소 가격범위) + VCP(변동성 수축 패턴)"""
    df['Range'] = df['High'] - df['Low']

    # NR7: 오늘 가격 범위가 최근 7일 중 가장 좁음 → 에너지 압축
    rolling_min = df['Range'].rolling(7).min()
    df['NR7'] = (df['Range'] == rolling_min) & rolling_min.notna()

    # VCP: 최근 20일을 4구간(각 5일)으로 나눠 변동폭이 연속 감소
    df['VCP'] = False
    if len(df) >= 20:
        close = df['Close'].iloc[-1]
        if close > 0:
            segs = [df.iloc[-20:-15], df.iloc[-15:-10], df.iloc[-10:-5], df.iloc[-5:]]
            ranges = [(s['High'].max() - s['Low'].min()) / close
                      for s in segs if len(s) > 0]
            if len(ranges) == 4 and all(ranges[k] > ranges[k + 1] for k in range(3)):
                df.loc[df.index[-1], 'VCP'] = True

    return df


def _obv(df):
    """2순위: OBV(On-Balance Volume) + 다이버전스 감지
    가격 횡보/하락 중 OBV 상승 = 세력 매집 신호"""
    close = df['Close'].values
    vol   = df['Volume'].values
    obv   = np.zeros(len(df))
    for i in range(1, len(df)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + vol[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - vol[i]
        else:
            obv[i] = obv[i - 1]
    df['OBV'] = obv

    # 다이버전스: 최근 20일 가격 변화 < +3% & OBV 증가
    df['OBV_Diverge'] = False
    if len(df) >= 20:
        base_close = df['Close'].iloc[-20]
        if base_close > 0:
            price_chg = (df['Close'].iloc[-1] - base_close) / base_close
            obv_chg   = df['OBV'].iloc[-1] - df['OBV'].iloc[-20]
            if price_chg < 0.03 and obv_chg > 0:
                df.loc[df.index[-1], 'OBV_Diverge'] = True

    return df


def _high_52w(df):
    """3순위: 52주(250거래일) 신고가 대비 현재가 거리
    신고가에 가까울수록 저항 없는 구간 → 세력 작업 유리"""
    lookback = min(250, len(df))
    df['High52W']      = df['High'].rolling(lookback, min_periods=lookback).max()
    # 0이면 신고가 돌파, 양수면 신고가까지 남은 거리 비율
    df['High52W_Dist'] = (df['High52W'] - df['Close']) / df['High52W'].replace(0, np.nan)
    df['High52W_Dist'] = df['High52W_Dist'].clip(lower=0)
    return df
