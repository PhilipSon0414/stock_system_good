"""
ML 기반 급등 예측 신뢰도 분류기

백테스트 캘리브레이션 (2026-05, score_history 447건 / 31 positives, 15%+ 기준):
  Tier S: 세력≥80 + 합산≥45 → 18.6% hit rate (2.68x lift)
  Tier A: 세력≥70           → 14.2% hit rate (2.04x lift)
  Tier B: 합산≥40 + 세력≥50 →  9.6% hit rate (1.39x lift)
  Tier C: 합산≥35           →  8.0% hit rate (1.16x lift)
  Tier D: 그 외             →  6.9% hit rate (기본)

sklearn 사용 가능 + 확정 기록 50개 이상이면 로지스틱 회귀 자동 학습 후 확률 보정.
"""

import json
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
HISTORY_FILE = SCRIPT_DIR / 'score_history.json'

# ── 캘리브레이션 테이블 (min_seoryeok, min_combined, tier, hit_rate, lift) ──
# 검증 결과 (2026-06, score_history 544건 / 32 positives, 15%+ 기준):
#   세력80+: 12.7% hit (2.15x lift)  세력70~79: 8.6% (1.46x)
#   세력60~69: 2.4% (0.41x) ← 기준선 이하 (노이즈 구간)
#   세력~59: 4.6% (0.78x)
# 성공 패턴: 이평선정배열+눌림목 복합 = 7건 중 5건 (71%)
_TIERS = [
    (80, 40, 'S', 0.127, 2.15),
    (70,  0, 'A', 0.086, 1.46),
    (60,  0, 'B', 0.059, 1.00),  # 노이즈 구간 — 기준선 수준
    ( 0, 30, 'C', 0.059, 1.00),
    ( 0,  0, 'D', 0.059, 1.00),
]

_TIER_LABELS = {
    'S': '★★★★ Tier-S (세력80+합산40+ / 예상적중 ~13%)',
    'A': '★★★  Tier-A (세력70+ / 예상적중 ~9%)',
    'B': '★★   Tier-B (세력60+ / 주의: 노이즈 구간)',
    'C': '★    Tier-C (합산30+ / 기준선 수준)',
    'D': '      Tier-D (기본)',
}

_STRATEGIES = {
    'S': '즉시 진입 검토 (세력 최강 신호 — 거래량 확인 후 진입)',
    'A': '거래량 2x+ 확인 후 진입 권고',
    'B': '추가 확인 신호 대기 (SE_Entry 또는 거래량 2x+)',
    'C': '모니터링 (추가 신호 없으면 보류)',
    'D': '관망',
}


# ── 캘리브레이션 기반 분류 ────────────────────────────────────────────────────

def get_confidence_tier(seoryeok: int, surge: int,
                        investor: int, pattern: int,
                        combined: int) -> dict:
    """
    Returns:
        tier      : 'S'|'A'|'B'|'C'|'D'
        hit_rate  : 예상 5일내 20%+ 급등 확률 (0~1)
        lift      : 기본 대비 배수
        label     : 표시용 문자열
        strategy  : 매매 전략 권고
        ml_prob   : sklearn ML 확률 (없으면 None)
    """
    tier_val = 'D'
    hit_rate = 0.048
    lift     = 1.00

    for min_se, min_co, t, rate, lft in _TIERS:
        if seoryeok >= min_se and combined >= min_co:
            tier_val = t
            hit_rate = rate
            lift     = lft
            break

    ml_prob = _get_ml_prob(seoryeok, surge, investor, pattern, combined)

    # ML 확률이 있으면 캘리브레이션 값과 가중 평균 (ML 40% + 캘리브레이션 60%)
    final_rate = hit_rate
    if ml_prob is not None:
        final_rate = ml_prob * 0.4 + hit_rate * 0.6

    return {
        'tier':     tier_val,
        'hit_rate': round(final_rate, 3),
        'lift':     lift,
        'label':    _TIER_LABELS[tier_val],
        'strategy': _STRATEGIES[tier_val],
        'ml_prob':  round(ml_prob, 3) if ml_prob is not None else None,
    }


# ── sklearn ML 모델 (선택적) ─────────────────────────────────────────────────

_ml_cache = None   # None = 아직 학습 안함, False = 학습 실패/불가


def _load_training_data():
    try:
        data    = json.loads(HISTORY_FILE.read_text(encoding='utf-8'))
        records = data.get('records', [])
        return [r for r in records if r.get('surged_by_5d') is not None]
    except Exception:
        return []


def _build_ml_model():
    records = _load_training_data()
    if len(records) < 50:
        return False

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = np.array([
            [r['seoryeok'], r['surge'], r['investor'], r['pattern'], r['combined']]
            for r in records
        ], dtype=float)
        y = np.array([int(r['surged_by_5d']) for r in records])

        if y.sum() < 5:   # 양성 샘플 5개 미만이면 무의미
            return False

        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)
        # class_weight 미사용: 실제 양성률에 캘리브레이션된 확률 유지
        model  = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_s, y)

        return {'model': model, 'scaler': scaler,
                'n_train': len(records), 'n_pos': int(y.sum())}
    except Exception:
        return False


def _get_ml_prob(seoryeok, surge, investor, pattern, combined) -> float | None:
    global _ml_cache
    if _ml_cache is None:
        _ml_cache = _build_ml_model()

    if not _ml_cache:
        return None

    try:
        import numpy as np
        X = np.array([[seoryeok, surge, investor, pattern, combined]], dtype=float)
        X_s  = _ml_cache['scaler'].transform(X)
        prob = _ml_cache['model'].predict_proba(X_s)[0][1]
        return float(prob)
    except Exception:
        return None


# ── 진입 타이밍 권고 ──────────────────────────────────────────────────────────

def get_entry_timing(tier: str, vol_ratio: float,
                     rs_vs_market: float | None,
                     has_se_entry: bool,
                     consecutive_days: int) -> str:
    """
    ML 티어 + 현재 시장 컨디션으로 진입 타이밍 권고문 생성.

    tier           : get_confidence_tier() 결과의 'tier'
    vol_ratio      : 현재 거래량 / 30일 평균 거래량
    rs_vs_market   : 종목 20일 수익률 - 코스피 20일 수익률 (%)
    has_se_entry   : SE_Entry 신호 여부
    consecutive_days: 연속 등장 일수
    """
    signals = []

    # 강점 신호 집계
    if has_se_entry:
        signals.append('SE진입신호')
    if vol_ratio >= 3.0:
        signals.append(f'거래량{vol_ratio:.1f}x폭증')
    elif vol_ratio >= 2.0:
        signals.append(f'거래량{vol_ratio:.1f}x확인')
    if rs_vs_market is not None and rs_vs_market >= 5.0:
        signals.append(f'RS+{rs_vs_market:.1f}%강세')
    if consecutive_days >= 3:
        signals.append(f'{consecutive_days}일연속')

    strength = len(signals)

    if tier == 'S':
        if strength >= 2:
            return f'진입 적극 권고 ({", ".join(signals[:2])} 복합 확인)'
        elif strength == 1:
            return f'진입 권고 ({signals[0]} 확인)'
        else:
            return '진입 검토 (추가 거래량 확인 권장)'

    elif tier == 'A':
        if strength >= 2:
            return f'진입 권고 ({", ".join(signals[:2])})'
        elif vol_ratio >= 2.0 or has_se_entry:
            return f'진입 검토 ({signals[0] if signals else "신호 미약"})'
        else:
            return '대기 — 거래량 2x+ 또는 SE_Entry 확인 후 진입'

    elif tier == 'B':
        if has_se_entry and vol_ratio >= 2.0:
            return f'조건 충족 진입 검토 (SE진입+거래량{vol_ratio:.1f}x)'
        else:
            return '대기 — SE_Entry + 거래량 2x+ 동시 확인 필요'

    else:   # C, D
        return '관망 — 세력 신호 강화 대기'


# ── 모델 통계 출력 ────────────────────────────────────────────────────────────

def auto_recalibrate() -> dict:
    """score_history 최신 데이터로 _TIERS 캘리브레이션을 자동 갱신.

    기존 하드코딩 값 대신 실측 데이터 기반 hit_rate 를 재계산한다.
    반환: {'updated': bool, 'tiers': list, 'n': int, 'n_pos': int}
    """
    global _TIERS, _TIER_LABELS, _ml_cache

    records = _load_training_data()
    n   = len(records)
    np_ = sum(1 for r in records if r.get('surged_by_5d'))
    if n < 30 or np_ < 5:
        return {'updated': False, 'reason': f'데이터 부족 ({n}건/{np_}양성)'}

    base = np_ / n

    def calc_tier(fn):
        g = [r for r in records if fn(r)]
        if len(g) < 5:
            return None, None
        pos = sum(1 for r in g if r['surged_by_5d'])
        rate = pos / len(g)
        lift = rate / base
        return rate, lift

    new_tiers = []
    tier_defs = [
        ('S', lambda r: r.get('seoryeok', 0) >= 80 and r.get('combined', 0) >= 45),
        ('A', lambda r: r.get('seoryeok', 0) >= 70),
        ('B', lambda r: r.get('seoryeok', 0) >= 50 and r.get('combined', 0) >= 40),
        ('C', lambda r: r.get('combined', 0) >= 35),
        ('D', lambda r: True),
    ]

    prev_min_se  = [80, 70, 50,  0, 0]
    prev_min_co  = [45,  0, 40, 35, 0]

    for (tier, fn), min_se, min_co in zip(tier_defs, prev_min_se, prev_min_co):
        rate, lift = calc_tier(fn)
        if rate is None:
            # 데이터 부족 시 기존 값 유지
            for t in _TIERS:
                if t[2] == tier:
                    new_tiers.append(t)
                    break
        else:
            new_tiers.append((min_se, min_co, tier, round(rate, 4), round(lift, 3)))

    if new_tiers:
        _TIERS = new_tiers
        _ml_cache = None   # ML 모델도 재학습
        return {'updated': True, 'tiers': new_tiers, 'n': n, 'n_pos': np_,
                'base_rate': round(base, 4)}
    return {'updated': False, 'reason': '계산 실패'}


def model_stats() -> str:
    records = _load_training_data()
    n   = len(records)
    np_ = sum(1 for r in records if r.get('surged_by_5d'))
    if n == 0:
        return '학습 데이터 없음'

    # trigger lazy build if needed
    _get_ml_prob(0, 0, 0, 0, 0)

    ml_info = ''
    if _ml_cache and isinstance(_ml_cache, dict):
        ml_info = f' | sklearn LR 활성 ({_ml_cache["n_train"]}건/{_ml_cache["n_pos"]}양성)'
    else:
        ml_info = ' | sklearn: 데이터 부족 (50건+ 필요)'

    return (
        f'{n}건 (양성 {np_}건 / {np_/n*100:.1f}%)'
        f'{ml_info}'
    )
