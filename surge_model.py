# -*- coding: utf-8 -*-
"""
급등 예측 ML 모델 — 4개 컴포넌트 점수(세력·급등·수급·패턴)를 GBM으로
비선형 결합해 선형 가중합 게이지의 한계를 넘는다.

검증(2026-06-16, 라벨 1033건 워크포워드, 2달 백필 데이터):
  - 선형 게이지(4점수): AUC 0.584
  - GBM(4점수):        AUC 0.623±0.007 (8시드, +3.9%p) ← 채택
  - 원시피처(거래량/OB/DART) 추가는 기여 없음(4점수에 이미 반영, 과적합) → 제외
  핵심: 데이터가 늘자 GBM의 비선형 결합이 안정적으로 선형을 능가.

운영: shadow 모드(평가/예측 제공). daily_scan 랭킹 편입은 별도 결정.

사용법:
    python3 surge_model.py eval               # 워크포워드 AUC (선형 vs GBM)
    python3 surge_model.py train              # 학습·저장
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
# [2026-06-16 검증] 라벨 1033건: GBM(4점수) 0.623 > 선형 0.584. 거래량/OB/DART 무기여.
# [2026-06-30 검증] 라벨 1712건 워크포워드(8시드, TimeSeriesSplit)로 낙폭 피처 편입:
#   +ret5(5일수익률)+ret20(20일) → 5일급등 AUC 0.541→0.680(+13.9%p),
#   3일급등 0.610→0.647(+3.7%p). ret5가 핵심('강세 급눌림' 선행). vr은 노이즈라 제외.
#   반등/과매도(RSI<30 lift 0.43x)는 반박돼 미편입. ret5/ret20은 daily_scan이 산출.
EXTRA = ['ret5', 'ret20']
FEATURES = COMP + EXTRA

# 현재 선형 게이지 가중치(비교 기준, 교정 후)
LIN_W = {'seoryeok': 0.428, 'surge': 0.143, 'investor': 0.398, 'pattern': 0.03}


def _ready_records():
    recs = json.load(open(HISTORY, encoding='utf-8'))['records']
    recs = [r for r in recs if all(k in r for k in COMP)]   # 4점수 보유 전체
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

    print(f"  선형 게이지(4점수)          {cv(lin, False):.3f}")
    print(f"  로지스틱({len(FEATURES)}피처)           {cv(lambda: _mk('lr'), True):.3f}")
    print(f"  GBM({len(FEATURES)}피처)                {cv(lambda: _mk('gbm'), True):.3f}")
    print(f"  피처: {len(FEATURES)}개 ({', '.join(FEATURES)})")


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
