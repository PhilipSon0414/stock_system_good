"""
학습 데이터 기반 모델 가중치 자동 조정

학습 원리:
  실제로 급등한 종목의 전일 점수를 누적 수집하여,
  각 컴포넌트(세력/급등/수급/패턴)가 급등을 얼마나 잘 예측했는지 분석.
  예측력이 높은 컴포넌트는 가중치 증가, 낮은 컴포넌트는 감소.
  EMA(α=0.15)로 점진적 업데이트 — 급격한 변동 방지.

파일:
  learning_log.json    — 급등 종목 관측 이력 + 가중치 변경 이력
  learned_weights.json — 현재 학습된 가중치 (config.py가 자동 로드)
"""

import json
from pathlib import Path
from datetime import datetime

SCRIPT_DIR      = Path(__file__).parent
LEARNING_LOG    = SCRIPT_DIR / 'learning_log.json'
LEARNED_WEIGHTS = SCRIPT_DIR / 'learned_weights.json'

ALPHA           = 0.15   # EMA 학습률 (새 데이터 15%, 기존 85%)
LEARNING_WINDOW = 30     # 가중치 계산에 사용할 최근 관측치 수
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


def update_model(log: dict) -> dict:
    """
    학습 로그 기반 가중치 EMA 업데이트.

    급등 종목들의 전일 점수를 분석:
    - 각 컴포넌트 평균이 50점 초과 → 예측력 있음 → 가중치 증가
    - 50점 미만 → 예측력 부족 → 가중치 감소
    """
    observations = log.get('observations', [])
    recent = observations[-LEARNING_WINDOW:]
    if not recent:
        return {}

    n = len(recent)
    old_weights = load_current_weights()

    # 컴포넌트별 평균 점수
    comp_avgs = {k: 0.0 for k in COMPONENT_MAP}
    for obs in recent:
        sc = obs.get('scores', {})
        for comp in COMPONENT_MAP:
            comp_avgs[comp] += sc.get(comp, 0)
    for comp in comp_avgs:
        comp_avgs[comp] /= n

    avg_combined = sum(
        obs.get('scores', {}).get('combined', 0) for obs in recent
    ) / n

    # 합산 40점+ = 모델이 사전에 포착했을 것으로 간주 (예측 성공)
    hit_rate = sum(
        1 for obs in recent
        if obs.get('scores', {}).get('combined', 0) >= 40
    ) / n

    # EMA 기반 가중치 조정
    new_raw = {}
    component_stats = {}

    for comp, wkey in COMPONENT_MAP.items():
        avg_score = comp_avgs[comp]
        old_w     = old_weights.get(wkey, DEFAULT_WEIGHTS[wkey])
        default_w = DEFAULT_WEIGHTS[wkey]

        # 예측 신호: 평균 점수가 50점 기준 대비 얼마나 높은가
        # signal 범위: -0.5 (최저) ~ +0.5 (최고)
        signal   = (avg_score - 50.0) / 100.0
        target_w = old_w * (1.0 + signal * 0.3)
        new_w    = (1.0 - ALPHA) * old_w + ALPHA * target_w

        new_raw[wkey] = new_w
        component_stats[comp] = {
            'avg_score':     round(avg_score, 1),
            'default_weight': round(default_w, 4),
            'old_weight':    round(old_w, 4),
            'new_weight':    0.0,   # 정규화 후 채움
            'weight_change': 0.0,
        }

    new_weights = _normalize(new_raw)

    for comp, wkey in COMPONENT_MAP.items():
        nw = new_weights[wkey]
        component_stats[comp]['new_weight']    = round(nw, 4)
        component_stats[comp]['weight_change'] = round(
            nw - component_stats[comp]['old_weight'], 4
        )

    save_learned_weights(new_weights)

    # 피처 중요도
    feature_importance = compute_feature_importance(recent)

    # 히스토리 기록
    log['weight_history'].append({
        'date':         datetime.now().strftime('%Y-%m-%d %H:%M'),
        'weights':      {k: round(v, 4) for k, v in new_weights.items()},
        'n_total':      len(observations),
        'n_window':     n,
        'avg_combined': round(avg_combined, 2),
        'hit_rate':     round(hit_rate, 4),
    })

    for comp, stat in component_stats.items():
        arrow = '↑' if stat['weight_change'] > 0 else ('↓' if stat['weight_change'] < 0 else '→')
        print(
            f'    {comp:<12}: 평균 {stat["avg_score"]:5.1f}점  '
            f'{stat["old_weight"]:.4f} → {stat["new_weight"]:.4f} {arrow}'
        )

    return {
        'window':             n,
        'total_obs':          len(observations),
        'avg_combined':       round(avg_combined, 2),
        'hit_rate':           round(hit_rate, 4),
        'component_stats':    component_stats,
        'old_weights':        old_weights,
        'new_weights':        new_weights,
        'default_weights':    DEFAULT_WEIGHTS,
        'feature_importance': feature_importance,
    }
