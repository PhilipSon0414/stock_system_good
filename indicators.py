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
    df = _rsi(df)
    df = _atr_bb_compress(df)
    df = _vol_momentum(df)
    df = _box_range(df)
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


def _rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI(14) + 회복 구간 플래그.
    학습 결과(KR): 급등 D-1 RSI=53.6 vs 대조군 55.1 — 과매수 아닌 중립~회복 구간.
    학습 결과(US): 급등 D-1 RSI=51.1 vs 대조군 52.3 — 동일 패턴."""
    delta  = df['Close'].diff()
    gain   = delta.clip(lower=0).rolling(period).mean()
    loss   = (-delta.clip(upper=0)).rolling(period).mean()
    rs     = gain / loss.replace(0, np.nan)
    df['RSI14']        = 100 - (100 / (1 + rs))
    df['RSI_Recovery'] = df['RSI14'].between(30, 57)   # 과매도 회복 ~ 중립
    return df


def _atr_bb_compress(df: pd.DataFrame) -> pd.DataFrame:
    """ATR 모멘텀 + 볼린저밴드 폭.
    학습 결과(KR): D-5→D-1 ATRCompress 1.091→1.125, BBCompress 1.066→1.120 (점진 확대).
    학습 결과(US): 동일 방향 — 급등 전날은 변동성이 확대 중인 상태.
    핵심: '압축 중'이 아닌 '압축 후 확대 시작' 단계가 매수 타이밍.

    NRStreak: 연속 NR 일수 — 학습에서 급등 전 감소(잠복→회복 전환 증거).
    VolDecDays: 연속 거래량 감소일 — 학습에서 급등 전 낮음 (회복 시작)."""
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df['ATR14']       = tr.rolling(14).mean()
    df['ATRRel']      = df['ATR14'] / c.replace(0, np.nan)
    df['ATRCompress'] = df['ATR14'] / df['ATR14'].rolling(20).mean().replace(0, np.nan)

    bb_std = c.rolling(20).std()
    df['BBWidth']    = (2 * bb_std) / df['MA20'].replace(0, np.nan)
    df['BBCompress'] = df['BBWidth'] / df['BBWidth'].rolling(20).mean().replace(0, np.nan)

    # NR 연속일
    if 'NR7' in df.columns:
        nr = df['NR7'].astype(int)
        ns = nr.groupby((nr != nr.shift()).cumsum()).cumcount() + 1
        df['NRStreak'] = (ns * nr).astype(int)

    # 연속 거래량 감소일
    v = df['Volume']
    vd = (v < v.shift(1)).astype(int)
    vs = vd.groupby((vd != vd.shift()).cumsum()).cumcount() + 1
    df['VolDecDays'] = (vs * vd).astype(int)

    return df


def _vol_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """거래량 모멘텀 — 압축 후 회복(증가) 신호.
    학습 결과(KR): VolRatio D-5→D-1 = 1.198→1.286 (점진 증가).
                  QuietStreak D-5→D-1 = 1.16→0.99 (잠복 끝나는 중).
    핵심 패턴: 세력 잠복(저거래량) 이후 거래량이 회복되기 시작하는 시점.
    VolRecovery = 직전 3일+ 저거래량 이후 거래량 증가 = 가장 신뢰도 높은 신호."""
    v    = df['Volume']
    v_ma = df.get('VolMA', v.rolling(VOLUME_AVG_PERIOD).mean())
    ratio = v / v_ma.replace(0, np.nan)

    # 연속 저거래량 일수 (0.7x 미만)
    quiet = (ratio < 0.7).astype(int)
    qs = quiet.groupby((quiet != quiet.shift()).cumsum()).cumcount() + 1
    df['QuietStreak'] = (qs * quiet).astype(int)

    # 5일 거래량 기울기 부호
    def slope_sign(w):
        w = w.dropna()
        if len(w) < 3:
            return 0.0
        x = np.arange(len(w), dtype=float)
        cov = np.cov(x, w.values)[0, 1]
        return float(np.sign(cov))

    df['VolSlope5'] = ratio.rolling(5).apply(slope_sign, raw=False)

    # 핵심 복합 신호: 잠복(QuietStreak ≥ 3) 이후 거래량 증가
    had_quiet = df['QuietStreak'].shift(1) >= 3
    now_up    = ratio >= ratio.shift(1)
    df['VolRecovery'] = had_quiet & now_up

    return df


def _box_range(df: pd.DataFrame) -> pd.DataFrame:
    """박스권 범위 지표 — 한국 시장 핵심.
    학습 결과(KR): BoxRange5d d=+0.267, BoxRange10d d=+0.167
    D-1의 박스 범위는 대조군보다 넓음 → 폭발 직전 변동성 확대 중.
    단, D-5~D-10 시점에 박스가 좁았다가 확대되는 '수축→이완' 패턴이 핵심."""
    c, h, l = df['Close'], df['High'], df['Low']
    for days in [5, 10, 20]:
        bh = h.rolling(days).max()
        bl = l.rolling(days).min()
        df[f'BoxRange{days}d'] = (bh - bl) / c.replace(0, np.nan)
    return df
