# -*- coding: utf-8 -*-
"""
급등 예측 ML 모델 (풍부 피처) — 4개 압축점수의 한계를 넘기 위해
원시 서브신호 + 시장 레짐까지 모델에 직접 공급한다.

배경(2026-06): 선형 게이지(4점수)는 3일 급등 AUC ~0.48~0.57로 약함.
원시 피처를 더한 GBM/로지스틱은 3일 AUC ~0.56~0.60으로 개선(n=390 검증).
데이터가 쌓일수록 신뢰도↑. 충분히 검증되면 daily_scan 랭킹에 편입.

데이터: score_history.json 의 'se_entry' 보유 + 라벨 확정 레코드.

사용법:
    python3 surge_model.py eval               # 워크포워드 AUC (선형 vs 모델)
    python3 surge_model.py train surged_by_3d # 학습·저장
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import joblib

SCRIPT_DIR = Path(__file__).parent
HISTORY = SCRIPT_DIR / 'score_history.json'

COMP = ['seoryeok', 'surge', 'investor', 'pattern']
RAW = ['vol_ratio', 'se_entry', 'se_accum', 'se_exit', 'ma_bull', 'golden_cross',
       'nr7', 'vcp', 'obv_diverge', 'high52w_dist', 'consec_days', 'sector_rs']
REGIME = ['market_regime']
FEATURES = COMP + RAW + REGIME

# 현재 선형 게이지 가중치(비교 기준)
LIN_W = {'seoryeok': 0.428, 'surge': 0.143, 'investor': 0.333, 'pattern': 0.095}


def _ready_records():
    recs = json.load(open(HISTORY, encoding='utf-8'))['records']
    recs = [r for r in recs if 'se_entry' in r]   # 원시피처 보유
    recs.sort(key=lambda r: r['scan_date'])
    return recs


def _matrix(recs):
    def f(r, k):
        v = r.get(k, 0)
        return float(v) if v is not None else 0.0   # 결측(레짐 등)→0
    X = np.array([[f(r, k) for k in FEATURES] for r in recs])
    lin = np.array([sum(LIN_W[k] * f(r, k) for k in COMP) for r in recs])
    return X, lin


def _mk(kind):
    if kind == 'gbm':
        return GradientBoostingClassifier(n_estimators=80, max_depth=2,
                                          learning_rate=0.05, subsample=0.8)
    return LogisticRegression(max_iter=2000, class_weight='balanced')


def evaluate(horizon='surged_by_3d'):
    recs = [r for r in _ready_records() if r.get(horizon) is not None]
    X, lin = _matrix(recs)
    y = np.array([1 if r.get(horizon) else 0 for r in recs])
    print(f"■ 급등 예측 워크포워드 검증  ({horizon}, n={len(recs)}, "
          f"기저 {y.mean()*100:.1f}%)")

    def cv(scores_or_model, is_model, X_=None):
        t = TimeSeriesSplit(n_splits=4)
        a = []
        for tr, te in t.split(X if X_ is None else X_):
            if len(np.unique(y[te])) < 2:
                continue
            if is_model:
                m = scores_or_model()
                m.fit(X[tr], y[tr])
                p = m.predict_proba(X[te])[:, 1]
            else:
                p = scores_or_model[te]
            a.append(roc_auc_score(y[te], p))
        return float(np.mean(a)) if a else float('nan')

    print(f"  선형 게이지(4점수)         {cv(lin, False):.3f}")
    print(f"  로지스틱(풍부피처)          {cv(lambda: _mk('lr'), True):.3f}")
    print(f"  GBM(풍부피처)              {cv(lambda: _mk('gbm'), True):.3f}")
    print(f"  피처: {len(FEATURES)}개 (압축점수4 + 원시{len(RAW)} + 레짐1)")


def train_and_save(horizon='surged_by_3d', kind='gbm'):
    recs = [r for r in _ready_records() if r.get(horizon) is not None]
    X, _ = _matrix(recs)
    y = np.array([1 if r.get(horizon) else 0 for r in recs])
    base = _mk(kind)
    clf = CalibratedClassifierCV(base, method='isotonic',
                                 cv=TimeSeriesSplit(n_splits=4))
    clf.fit(X, y)
    path = SCRIPT_DIR / f'surge_model_{horizon}.pkl'
    joblib.dump({'model': clf, 'features': FEATURES, 'n': len(recs),
                 'horizon': horizon}, path)
    print(f"  [학습] {horizon}: n={len(recs)} (양성 {int(y.sum())}) → {path}")
    return path


def predict(record: dict, horizon='surged_by_3d'):
    """단일 종목 record로 급등확률 예측. 모델 없으면 None."""
    path = SCRIPT_DIR / f'surge_model_{horizon}.pkl'
    if not path.exists():
        return None
    bundle = joblib.load(path)
    def f(k):
        v = record.get(k, 0)
        return float(v) if v is not None else 0.0
    x = np.array([[f(k) for k in bundle['features']]])
    return float(bundle['model'].predict_proba(x)[0, 1])


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'eval'
    horizon = sys.argv[2] if len(sys.argv) > 2 else 'surged_by_3d'
    if cmd == 'train':
        print("■ 급등 예측 모델 학습")
        for h in ('surged_by_3d', 'surged_by_5d'):
            train_and_save(h)
    else:
        for h in ('surged_by_3d', 'surged_by_5d'):
            evaluate(h)
            print()
