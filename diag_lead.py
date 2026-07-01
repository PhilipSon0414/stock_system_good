# -*- coding: utf-8 -*-
"""
Task 4 — 리드타임(cold-start) 예측력 측정.

핵심 질문: 이 시스템이 '아직 안 움직인(미발동) 종목의 급등을 사전에' 예측할 수 있나?
아니면 '이미 움직인 종목의 모멘텀 연속'만 맞추나?(헛점 5)

not_extended(미발동/cold) = vol_ratio<1.5 & ret5<0.08 & high52w_dist>0.05
  → cold 부분집합에서 GBM이 surged_by_5d를 예측하는 AUC·precision@k를
    hot(발동)과 비교. cold AUC~0.5면 사전포착 능력 없음(모멘텀만).
"""
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

FEATS = ['seoryeok', 'surge', 'investor', 'pattern', 'ret5', 'ret20']

recs = [r for r in json.load(open('score_history.json'))['records']
        if r.get('surged_by_5d') is not None and 'ret5' in r
        and r.get('vol_ratio') is not None and r.get('high52w_dist') is not None]
recs.sort(key=lambda r: r['scan_date'])


def not_extended(r):
    return (r['vol_ratio'] < 1.5) and (r['ret5'] < 0.08) and (r['high52w_dist'] > 0.05)


cold = [r for r in recs if not_extended(r)]
hot = [r for r in recs if not not_extended(r)]


def rate(rs):
    p = sum(1 for r in rs if r['surged_by_5d'])
    return p, len(rs), (p / len(rs) if rs else 0)


pc, nc, rc = rate(cold)
ph, nh, rh = rate(hot)
pa, na, ra = rate(recs)
print(f'전체 {na}건 급등률 {ra*100:.1f}%')
print(f'  cold(미발동) {nc}건 급등률 {rc*100:.1f}% (급등 {pc})')
print(f'  hot (발동)   {nh}건 급등률 {rh*100:.1f}% (급등 {ph})')
print()


def wf_auc(rs, seeds=8):
    if len({r['surged_by_5d'] for r in rs}) < 2:
        return None
    X = np.array([[float(r.get(k, 0) or 0) for k in FEATS] for r in rs])
    y = np.array([1 if r['surged_by_5d'] else 0 for r in rs])
    accs = []
    for s in range(seeds):
        fold = []
        for tr, te in TimeSeriesSplit(n_splits=4).split(X):
            if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
                continue
            m = GradientBoostingClassifier(n_estimators=80, max_depth=2,
                                           learning_rate=0.05, subsample=0.8,
                                           min_samples_leaf=30, random_state=s)
            m.fit(X[tr], y[tr])
            fold.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        if fold:
            accs.append(np.mean(fold))
    return (np.mean(accs), np.std(accs)) if accs else None


print('■ 워크포워드 AUC (surged_by_5d 예측, 6피처 GBM):')
for lab, rs in (('전체', recs), ('cold(미발동)', cold), ('hot(발동)', hot)):
    a = wf_auc(rs)
    if a:
        print(f'  {lab:14s} AUC {a[0]:.3f} ± {a[1]:.3f}  (n={len(rs)})')
    else:
        print(f'  {lab:14s} 계산불가(표본/클래스 부족, n={len(rs)})')

# cold 부분집합 precision@k (in-sample GBM 채점) — 사전포착 실전 지표
print('\n■ cold 부분집합 precision@k (미발동 중 상위 골랐을 때 급등률):')
if len({r['surged_by_5d'] for r in cold}) >= 2:
    X = np.array([[float(r.get(k, 0) or 0) for k in FEATS] for r in cold])
    y = np.array([1 if r['surged_by_5d'] else 0 for r in cold])
    m = GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                                   subsample=0.8, min_samples_leaf=30, random_state=0)
    m.fit(X, y)
    p = m.predict_proba(X)[:, 1]
    order = np.argsort(-p)
    for k in (10, 20, 30, 50):
        if k <= len(cold):
            hit = int(y[order[:k]].sum())
            print(f'  상위 {k:>2}개: 적중 {hit}/{k} = {hit/k*100:.0f}%  lift {hit/k/rc:.2f}x (cold기저 {rc*100:.1f}%)')
else:
    print('  cold 양성 부족 — 측정 불가')
