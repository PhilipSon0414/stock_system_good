# -*- coding: utf-8 -*-
"""
반등형(reversal/oversold) 신호 회수 가능성 진단.

가설: 현재 시스템(세력 매집·정배열형)이 놓치는 급등의 다수가
'역배열·과매도 낙폭과대 후 반등' 부류다(6/29 학습: 역배열 76.7%).
라벨 유니버스(score_history, surged_by_5d)에서 각 종목의 스캔일 시점
OHLCV를 재구성해 반등 시그니처의 lift/recall/precision을 정량화한다.

- base rate, 현재 시스템(combined>=40) 포착 대비 '반등 신호 증분 회수'
- 검증 철칙: lift>1.3x 유지 못하면 편입 금지(DART/OB 전례)

결과: /tmp/diag_reversal.json
"""
import json
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import get_ohlcv
from indicators import add_all

recs = [r for r in json.load(open('score_history.json'))['records']
        if r.get('surged_by_5d') is not None]
print(f'대상 라벨 레코드: {len(recs)}건', flush=True)

out = []
for i, r in enumerate(recs):
    t, d = r['ticker'], r['scan_date']
    row = {
        'y': 1 if r.get('surged_by_5d') else 0,
        'y3': 1 if r.get('surged_by_3d') else 0,
        'se': r.get('seoryeok') or 0,
        'inv': r.get('investor') or 0,
        'comb': r.get('combined') or 0,
        'vr_log': r.get('surge') or 0,  # 참고용
    }
    try:
        df = get_ohlcv(t, end_date=d)
        if df is None or len(df) < 60:
            continue
        df = add_all(df)
        last = df.iloc[-1]
        close = float(last['Close'])
        def fnum(v):
            return float(v) if v == v and v is not None else None
        ma20 = fnum(last.get('MA20'))
        ma60 = fnum(last.get('MA60'))
        rsi = fnum(last.get('RSI14'))
        vr = fnum(last.get('VolRatio'))
        # 낙폭(최근 5·20봉 수익률)
        c6 = float(df['Close'].iloc[-6]) if len(df) >= 6 else close
        c21 = float(df['Close'].iloc[-21]) if len(df) >= 21 else close
        ret5 = close / c6 - 1 if c6 else 0
        ret20 = close / c21 - 1 if c21 else 0
        # 최근 20봉 저점 대비 위치
        low20 = float(df['Low'].iloc[-20:].min())
        high20 = float(df['High'].iloc[-20:].max())
        pos_in_range = (close - low20) / (high20 - low20) if high20 > low20 else 0.5
        # 캔들: 아래꼬리(매도세 소진)·양봉
        lw = float(last.get('LowerWick') or 0)
        body = float(last.get('Body') or 0)
        rng = float(last['High'] - last['Low']) or 1e-9
        lowerwick_ratio = lw / rng
        isbull = bool(last.get('IsBull'))
        row.update({
            'rsi': rsi,
            'pma20': (close / ma20 - 1) if ma20 else None,   # MA20 대비 이격(음수=아래)
            'below_ma20': (ma20 is not None and close < ma20),
            'mabear': (ma20 is not None and ma60 is not None and ma20 < ma60),
            'vr': vr,
            'ret5': ret5, 'ret20': ret20,
            'pos20': pos_in_range,
            'lwick': lowerwick_ratio,
            'isbull': isbull,
            # 이미 계산된 반등·압축 신호(보너스)
            'rsi_recov': bool(last.get('RSI_Recovery')),
            'vol_recov': bool(last.get('VolRecovery')),
            'vcp': bool(last.get('VCP')),
            'bbcompress': bool(last.get('BBCompress')),
            'nr7': bool(last.get('NR7')),
        })
        out.append(row)
    except Exception:
        continue
    if i % 100 == 0:
        print(f'  진행 {i}/{len(recs)}  (수집 {len(out)})', flush=True)

json.dump(out, open('/tmp/diag_reversal.json', 'w'))
print(f'완료: {len(out)}건 저장', flush=True)
