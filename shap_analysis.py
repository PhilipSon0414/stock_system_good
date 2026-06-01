"""
SHAP 피처 중요도 분석 + 예측 오류 분류

어떤 피처가 실제로 급등 예측에 기여하는지 SHAP으로 측정.
예측 오류 종목을 세 유형으로 분류:
  ① 세력 없음  — 세력 점수가 원래 낮았던 종목
  ② 타이밍 오차 — 세력은 있었으나 급등 시점 다름
  ③ 외부 이벤트 — 공시/테마/정책 등 외부 변수
"""

import json
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR   = Path(__file__).parent
HISTORY_FILE = SCRIPT_DIR / 'score_history.json'

FEATURES = [
    'seoryeok', 'surge', 'investor', 'pattern', 'combined',
    'vol_ratio', 'se_entry', 'se_accum', 'se_exit',
    'nr7', 'vcp', 'obv_diverge', 'high52w_dist',
    'ma_bull', 'golden_cross', 'consec_days',
]


def load_labeled_records() -> list[dict]:
    """surged_by_5d 확정 레코드 로드"""
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding='utf-8'))
        return [r for r in data.get('records', []) if r.get('surged_by_5d') is not None]
    except Exception:
        return []


def run_shap_analysis() -> dict:
    """SHAP 피처 중요도 분석. shap 미설치 시 permutation importance 대체."""
    records = load_labeled_records()
    if len(records) < 30:
        return {'error': f'데이터 부족 ({len(records)}건)'}

    import numpy as np
    X = np.array([[r.get(f, 0) or 0 for f in FEATURES] for r in records], dtype=float)
    y = np.array([int(r['surged_by_5d']) for r in records])

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)
        model  = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                             random_state=42, learning_rate=0.05)
        model.fit(X_s, y)

        imp = model.feature_importances_
        feat_imp = sorted(zip(FEATURES, imp), key=lambda x: x[1], reverse=True)

        # SHAP 시도
        shap_vals = None
        try:
            import shap
            explainer  = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_s)
            shap_mean  = np.abs(shap_values).mean(axis=0)
            shap_imp   = sorted(zip(FEATURES, shap_mean), key=lambda x: x[1], reverse=True)
            shap_vals  = shap_imp
        except ImportError:
            pass

        return {
            'n_samples':   len(records),
            'n_positive':  int(y.sum()),
            'feat_imp':    feat_imp,        # (feature, importance) 리스트
            'shap_imp':    shap_vals,       # SHAP 값 (없으면 None)
            'model':       model,
            'scaler':      scaler,
        }

    except ImportError:
        return {'error': 'sklearn 필요'}
    except Exception as e:
        return {'error': str(e)}


def classify_false_positives() -> dict:
    """
    False Positive 오류 분류 (예측 O → 급등 X).
    세 유형으로 분류:
      ① 세력 없음   — seoryeok < 50
      ② 타이밍 오차 — seoryeok ≥ 50, 연속 등장 많음 (5일+)
      ③ 외부 이벤트  — 판단 불가 (기타)
    """
    records = load_labeled_records()
    fp = [r for r in records if r.get('surged_by_5d') is False
          and r.get('combined', 0) >= 40]   # 모델이 추천했던 종목

    if not fp:
        return {'total_fp': 0}

    type1 = [r for r in fp if r.get('seoryeok', 0) < 50]
    type2 = [r for r in fp if r.get('seoryeok', 0) >= 50
             and r.get('consec_days', 0) >= 5]
    type3 = [r for r in fp if r not in type1 and r not in type2]

    # 평균 점수
    def avg(lst, key):
        vals = [r.get(key, 0) for r in lst if r.get(key) is not None]
        return round(sum(vals) / len(vals), 1) if vals else 0

    return {
        'total_fp': len(fp),
        'type1_no_force': {
            'count': len(type1),
            'pct':   round(len(type1) / len(fp) * 100, 1),
            'avg_seoryeok': avg(type1, 'seoryeok'),
            'desc': '세력 없음 (세력<50점)',
        },
        'type2_timing': {
            'count': len(type2),
            'pct':   round(len(type2) / len(fp) * 100, 1),
            'avg_consec': avg(type2, 'consec_days'),
            'desc': '타이밍 오차 (연속5일+ 이미 알려진 종목)',
        },
        'type3_external': {
            'count': len(type3),
            'pct':   round(len(type3) / len(fp) * 100, 1),
            'desc': '외부 이벤트 또는 미분류',
        },
    }


def print_shap_report():
    """콘솔 출력용 SHAP 리포트"""
    print('=' * 60)
    print('  SHAP 피처 중요도 분석')
    print('=' * 60)
    result = run_shap_analysis()
    if 'error' in result:
        print(f'  오류: {result["error"]}')
        return

    n, np_ = result['n_samples'], result['n_positive']
    print(f'  학습 데이터: {n}건 (양성 {np_}건 / {np_/n*100:.1f}%)')
    print()

    imp_data = result.get('shap_imp') or result.get('feat_imp', [])
    label = 'SHAP' if result.get('shap_imp') else 'GBM 피처 중요도'
    print(f'  [{label}] 상위 피처:')
    for feat, imp in imp_data[:12]:
        bar = '█' * int(imp * 300)
        print(f'    {feat:<18} {imp:.4f}  {bar}')

    print()
    print('=' * 60)
    print('  False Positive 오류 분류')
    print('=' * 60)
    fp = classify_false_positives()
    total = fp.get('total_fp', 0)
    if total == 0:
        print('  FP 데이터 없음')
        return
    print(f'  총 FP: {total}건 (합산40+ 추천 → 급등 미발생)')
    for key in ['type1_no_force', 'type2_timing', 'type3_external']:
        t = fp.get(key, {})
        print(f'  {t.get("desc","")}: {t.get("count",0)}건 ({t.get("pct",0):.1f}%)')


if __name__ == '__main__':
    print_shap_report()
