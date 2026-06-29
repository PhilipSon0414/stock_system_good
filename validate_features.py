# -*- coding: utf-8 -*-
"""
피처 증강 검증 — 기존 4점수 GBM vs +낙폭(ret5)+거래량(vr)+ret20 증강 GBM.
surge_model.py와 동일 방법론(TimeSeriesSplit, 시간순, 다중시드 AUC).

1) score_history 라벨 레코드별 ret5/ret20/vr/pma20 재구성 → /tmp/feat_aug.json 캐시
2) 베이스(4점수) vs 증강(4점수+ret5+vr+ret20) AUC를 8시드 평균으로 비교
"""
import json
import sys
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).parent
CACHE = Path('/tmp/feat_aug.json')
COMP = ['seoryeok', 'surge', 'investor', 'pattern']
NEW = ['ret5', 'vr', 'ret20']


def build_cache():
    from data_fetcher import get_ohlcv
    from indicators import add_all
    recs = [r for r in json.load(open(SCRIPT_DIR / 'score_history.json'))['records']
            if all(k in r for k in COMP) and r.get('surged_by_3d') is not None]
    recs.sort(key=lambda r: r['scan_date'])
    print(f'재구성 대상 {len(recs)}건', flush=True)
    out = []
    for i, r in enumerate(recs):
        row = {'date': r['scan_date'],
               'y3': 1 if r.get('surged_by_3d') else 0,
               'y5': 1 if r.get('surged_by_5d') else 0}
        for k in COMP:
            v = r.get(k, 0); row[k] = float(v) if v is not None else 0.0
        try:
            df = get_ohlcv(r['ticker'], end_date=r['scan_date'])
            if df is None or len(df) < 30:
                continue
            df = add_all(df)
            last = df.iloc[-1]
            close = float(last['Close'])
            c6 = float(df['Close'].iloc[-6]) if len(df) >= 6 else close
            c21 = float(df['Close'].iloc[-21]) if len(df) >= 21 else close
            vr = last.get('VolRatio')
            ma20 = last.get('MA20')
            row['ret5'] = close / c6 - 1 if c6 else 0.0
            row['ret20'] = close / c21 - 1 if c21 else 0.0
            row['vr'] = float(vr) if vr == vr and vr is not None else 1.0
            row['pma20'] = (close / float(ma20) - 1) if ma20 == ma20 and ma20 else 0.0
            out.append(row)
        except Exception:
            continue
        if i % 200 == 0:
            print(f'  {i}/{len(recs)} (수집 {len(out)})', flush=True)
    json.dump(out, open(CACHE, 'w'))
    print(f'캐시 저장 {len(out)}건 → {CACHE}', flush=True)
    return out


def auc_multiseed(X, y, seeds=8):
    accs = []
    for s in range(seeds):
        t = TimeSeriesSplit(n_splits=4)
        fold = []
        for tr, te in t.split(X):
            if len(np.unique(y[te])) < 2:
                continue
            m = GradientBoostingClassifier(n_estimators=80, max_depth=2,
                                           learning_rate=0.05, subsample=0.8,
                                           random_state=s)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            fold.append(roc_auc_score(y[te], p))
        if fold:
            accs.append(np.mean(fold))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    data = json.load(open(CACHE)) if CACHE.exists() else build_cache()
    data = [r for r in data if all(k in r for k in NEW)]
    data.sort(key=lambda r: r['date'])
    print(f'\n검증 데이터 {len(data)}건')
    for horizon in ('y3', 'y5'):
        y = np.array([r[horizon] for r in data])
        Xb = np.array([[r[k] for k in COMP] for r in data])
        Xa = np.array([[r[k] for k in COMP + NEW] for r in data])
        mb, sb = auc_multiseed(Xb, y)
        ma, sa = auc_multiseed(Xa, y)
        hz = '3일' if horizon == 'y3' else '5일'
        print(f'\n■ {hz} 급등  (n={len(data)}, 기저 {y.mean()*100:.1f}%)')
        print(f'  베이스 GBM(4점수)          AUC {mb:.3f} ± {sb:.3f}')
        print(f'  증강 GBM(+낙폭+거래량+ret20) AUC {ma:.3f} ± {sa:.3f}')
        print(f'  → 변화 {(ma-mb)*100:+.1f}%p')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'rebuild':
        CACHE.unlink(missing_ok=True)
    main()
