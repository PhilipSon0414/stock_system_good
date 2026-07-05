"""
학습 데이터 기반 모델 가중치 자동 조정 (lift 기반)

학습 원리 (2026-07 재설계):
  기존 방식은 급등 종목(양성)만 보고 컴포넌트 평균을 상수 50과 비교했다.
  → 모든 종목에 후한 점수를 주는 컴포넌트(변별력 0)가 계속 가중치를 얻는
    통계적 결함. 비교 기준은 상수가 아니라 '급등하지 않은 종목'이어야 한다.

  새 방식: score_history.json의 라벨 확정 레코드(양성+음성, 탈락 유니버스
  샘플 포함)로 컴포넌트별 표준화 평균차(Cohen's d)를 계산해,
  d가 큰(급등/비급등을 잘 가르는) 컴포넌트에 가중치를 배분한다.
  EMA(α=0.15)로 점진 반영 — 급격한 변동 방지.

  평가 지표도 교체: 기존 hit_rate(급등 종목 중 합산 40+ 비율)는 가중치를
  키우면 예측력 변화 없이도 올라가는 순환 지표였다. → forward_tracker의
  precision@K(스캔일별 상위 K픽 중 실제 급등 비율)로 대체.

파일:
  learning_log.json    — 급등 종목 관측 이력 + 가중치 변경 이력
  learned_weights.json — 현재 학습된 가중치 (config.py가 자동 로드)
  score_history.json   — 라벨 확정 레코드 (lift 계산의 데이터 소스)
"""

import json
from pathlib import Path
from datetime import datetime

SCRIPT_DIR      = Path(__file__).parent
LEARNING_LOG    = SCRIPT_DIR / 'learning_log.json'
LEARNED_WEIGHTS = SCRIPT_DIR / 'learned_weights.json'

ALPHA           = 0.15   # EMA 학습률 (새 데이터 15%, 기존 85%)
LEARNING_WINDOW = 30     # 이메일 표시용 최근 관측치 수 (피처 출현률 계산)
LIFT_WINDOW     = 600    # lift 계산에 사용할 최근 라벨 확정 레코드 수
LIFT_HORIZON    = 'surged_by_5d'   # 가중치 학습 기준 호라이즌
MIN_LABELED     = 60     # lift 계산 최소 레코드 수
MIN_POSITIVES   = 8      # lift 계산 최소 양성 수
MIN_WEIGHT      = 0.10   # 컴포넌트 최소 가중치(기본)
MAX_WEIGHT      = 0.50   # 컴포넌트 최대 가중치
# [2026-06-16] 컴포넌트별 하한 — 차트패턴은 925건 실증상 급등 역상관(lift 0.24x)이라
#   기존 0.10 플로어가 매 학습마다 패턴을 ~0.095로 끌어올리는 부작용이 있었다.
#   패턴 하한을 0.02로 완화해 자동학습이 패턴을 낮게 유지하도록 한다.
MIN_WEIGHT_BY   = {'W_PATTERN': 0.02}

DEFAULT_WEIGHTS = {
    'W_SEORYEOK': 0.30,
    'W_SURGE':    0.30,
    'W_INVESTOR': 0.37,
    'W_PATTERN':  0.03,
}

COMPONENT_MAP = {
    'seoryeok': 'W_SEORYEOK',
    'surge':    'W_SURGE',
    'investor': 'W_INVESTOR',
    'pattern':  'W_PATTERN',
}


def load_learning_log() -> dict:
    if LEARNING_LOG.exists():
        try:
            data = json.loads(LEARNING_LOG.read_text(encoding='utf-8'))
            data.setdefault('observations', [])
            data.setdefault('weight_history', [])
            return data
        except Exception:
            pass
    return {'observations': [], 'weight_history': []}


def save_learning_log(data: dict):
    LEARNING_LOG.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def load_current_weights() -> dict:
    if LEARNED_WEIGHTS.exists():
        try:
            return json.loads(LEARNED_WEIGHTS.read_text(encoding='utf-8'))
        except Exception:
            pass
    return DEFAULT_WEIGHTS.copy()


def save_learned_weights(weights: dict):
    LEARNED_WEIGHTS.write_text(
        json.dumps(weights, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def _normalize(weights: dict) -> dict:
    total = sum(weights.values())
    if total == 0:
        return DEFAULT_WEIGHTS.copy()
    normalized = {k: v / total for k, v in weights.items()}
    # 클램핑 후 재정규화
    clamped = {k: max(MIN_WEIGHT_BY.get(k, MIN_WEIGHT), min(MAX_WEIGHT, v))
               for k, v in normalized.items()}
    total2 = sum(clamped.values())
    return {k: round(v / total2, 6) for k, v in clamped.items()}


def compute_feature_importance(recent: list) -> list[dict]:
    """
    최근 급등 종목들의 전일 피처 출현 빈도 계산.
    출현률이 높을수록 급등 전 공통 패턴으로 볼 수 있음.
    """
    bool_features = [
        'SE_Entry', 'SE_Accum', 'SE_BearTrap', 'SE_Rally', 'SE_Exit',
        'MaBull', 'MaBear', 'MaConverge', 'GoldenCross',
        'near_bull_ob', 'near_bear_ob',
    ]
    label_map = {
        'SE_Entry':    '세력 진입 신호',
        'SE_Accum':    '세력 매집 구간',
        'SE_BearTrap': '베어트랩 회복',
        'SE_Rally':    '상승 시작 신호',
        'SE_Exit':     '⚠ 세력 이탈',
        'MaBull':      '이평선 정배열',
        'MaBear':      '이평선 역배열',
        'MaConverge':  '이평선 수렴',
        'GoldenCross': '골든크로스',
        'near_bull_ob':'강세 오더블록 근접',
        'near_bear_ob':'약세 오더블록 근접',
    }
    n = len(recent)
    if n == 0:
        return []

    counts = {f: 0 for f in bool_features}
    for obs in recent:
        feats = obs.get('features', {})
        for f in bool_features:
            if feats.get(f, False):
                counts[f] += 1

    result = []
    for f in bool_features:
        rate = counts[f] / n
        if rate > 0:
            result.append({
                'feature': f,
                'label':   label_map.get(f, f),
                'count':   counts[f],
                'rate':    round(rate, 4),
            })
    result.sort(key=lambda x: x['rate'], reverse=True)
    return result


def _load_labeled_records(horizon: str = LIFT_HORIZON) -> list[dict]:
    """score_history.json에서 라벨 확정 레코드(양성+음성) 로드.
    탈락 유니버스 샘플(below_gate)도 포함 — 음성 대조군의 핵심."""
    try:
        from forward_tracker import HISTORY_FILE
        data = json.loads(HISTORY_FILE.read_text(encoding='utf-8'))
        recs = [r for r in data.get('records', [])
                if r.get(horizon) is not None
                and all(k in r for k in COMPONENT_MAP)]
        recs.sort(key=lambda r: r.get('scan_date', ''))
        return recs[-LIFT_WINDOW:]
    except Exception:
        return []


def _cohens_d(pos_vals: list, neg_vals: list) -> float:
    """표준화 평균차 — 컴포넌트가 급등/비급등을 얼마나 가르는지.
    d>0: 급등 종목에서 점수가 더 높음 (예측력 있음)."""
    n1, n2 = len(pos_vals), len(neg_vals)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1 = sum(pos_vals) / n1
    m2 = sum(neg_vals) / n2
    v1 = sum((x - m1) ** 2 for x in pos_vals) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in neg_vals) / (n2 - 1)
    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    if pooled <= 0:
        return 0.0
    return (m1 - m2) / pooled ** 0.5


def update_model(log: dict) -> dict:
    """
    lift 기반 가중치 EMA 업데이트.

    score_history의 라벨 확정 레코드에서 컴포넌트별 Cohen's d
    (급등 vs 비급등 표준화 평균차)를 구하고, d에 비례해 목표 가중치를
    배분한 뒤 EMA로 점진 반영한다. 데이터 부족 시 가중치를 건드리지 않는다.
    """
    observations = log.get('observations', [])
    recent = observations[-LEARNING_WINDOW:]

    recs = _load_labeled_records(LIFT_HORIZON)
    pos  = [r for r in recs if r.get(LIFT_HORIZON)]
    neg  = [r for r in recs if not r.get(LIFT_HORIZON)]

    if len(recs) < MIN_LABELED or len(pos) < MIN_POSITIVES:
        print(f'    가중치 학습 보류: 라벨 확정 {len(recs)}건/양성 {len(pos)}건 '
              f'(최소 {MIN_LABELED}건/{MIN_POSITIVES}양성 필요)')
        return {}

    old_weights = load_current_weights()

    # ── 컴포넌트별 변별력 (Cohen's d) ────────────────────────────────────
    new_raw = {}
    component_stats = {}
    for comp, wkey in COMPONENT_MAP.items():
        pv = [float(r.get(comp, 0)) for r in pos]
        nv = [float(r.get(comp, 0)) for r in neg]
        mu_pos = sum(pv) / len(pv)
        mu_neg = sum(nv) / len(nv)
        d      = _cohens_d(pv, nv)

        # 목표 가중치 ∝ max(d, 0) + ε — 역상관(d<0) 컴포넌트는 바닥으로
        new_raw[wkey] = max(d, 0.0) + 0.01

        component_stats[comp] = {
            'avg_score':      round(mu_pos, 1),   # 급등 종목 평균 (이메일 표시용)
            'mu_neg':         round(mu_neg, 1),   # 비급등 종목 평균 (대조군)
            'cohens_d':       round(d, 3),
            'default_weight': round(DEFAULT_WEIGHTS[wkey], 4),
            'old_weight':     round(old_weights.get(wkey, DEFAULT_WEIGHTS[wkey]), 4),
            'new_weight':     0.0,   # 정규화 후 채움
            'weight_change':  0.0,
        }

    # 목표 정규화 → EMA 점진 반영 → 클램핑 재정규화
    total_raw = sum(new_raw.values())
    targets = {k: v / total_raw for k, v in new_raw.items()}
    ema = {
        wkey: (1.0 - ALPHA) * old_weights.get(wkey, DEFAULT_WEIGHTS[wkey])
              + ALPHA * targets[wkey]
        for wkey in new_raw
    }
    new_weights = _normalize(ema)

    for comp, wkey in COMPONENT_MAP.items():
        nw = new_weights[wkey]
        component_stats[comp]['new_weight']    = round(nw, 4)
        component_stats[comp]['weight_change'] = round(
            nw - component_stats[comp]['old_weight'], 4
        )

    save_learned_weights(new_weights)

    # ── 평가 지표: precision@5 (순환 지표였던 hit_rate 대체) ──────────────
    precision5 = None
    try:
        from forward_tracker import precision_at_k
        p = precision_at_k(k=5, horizon=LIFT_HORIZON)
        if p:
            precision5 = p['precision']
    except Exception:
        pass

    avg_combined = (sum(float(r.get('combined', 0)) for r in pos) / len(pos))

    # 피처 중요도 (급등 관측치 기반 — 표시용)
    feature_importance = compute_feature_importance(recent)

    # 히스토리 기록
    log['weight_history'].append({
        'date':         datetime.now().strftime('%Y-%m-%d %H:%M'),
        'weights':      {k: round(v, 4) for k, v in new_weights.items()},
        'n_total':      len(observations),
        'n_labeled':    len(recs),
        'n_pos':        len(pos),
        'avg_combined': round(avg_combined, 2),
        'precision_at_5': round(precision5, 4) if precision5 is not None else None,
        'cohens_d':     {c: component_stats[c]['cohens_d'] for c in COMPONENT_MAP},
    })

    for comp, stat in component_stats.items():
        arrow = '↑' if stat['weight_change'] > 0 else ('↓' if stat['weight_change'] < 0 else '→')
        print(
            f'    {comp:<12}: 급등 {stat["avg_score"]:5.1f} vs 비급등 {stat["mu_neg"]:5.1f} '
            f'(d={stat["cohens_d"]:+.3f})  '
            f'{stat["old_weight"]:.4f} → {stat["new_weight"]:.4f} {arrow}'
        )
    if precision5 is not None:
        print(f'    precision@5 ({LIFT_HORIZON}): {precision5*100:.1f}%')

    return {
        'window':             len(recs),
        'n_pos':              len(pos),
        'total_obs':          len(observations),
        'avg_combined':       round(avg_combined, 2),
        'precision_at_5':     round(precision5, 4) if precision5 is not None else None,
        # 하위호환: 예전 이메일 코드가 참조하던 키 — precision@5로 대체
        'hit_rate':           round(precision5, 4) if precision5 is not None else 0.0,
        'component_stats':    component_stats,
        'old_weights':        old_weights,
        'new_weights':        new_weights,
        'default_weights':    DEFAULT_WEIGHTS,
        'feature_importance': feature_importance,
    }
