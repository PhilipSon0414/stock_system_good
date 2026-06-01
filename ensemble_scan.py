"""
3단계 앙상블 필터: 단일 모델 한계 극복

[1단계] 고재현율(High Recall) 필터 — 급등 가능 후보 50~100개
  기준: 세력≥20점 OR 거래량≥1.5x OR 트리플필터 1개 이상
  목적: 진짜 급등 종목을 놓치지 않음

[2단계] 고정밀도(High Precision) 스코어링 — Top 20 선별
  백테스트 최적 규칙 적용:
    - 세력 70+ (lift 2.04x)
    - co_mean ≥ 54 & slope ≥ 3 (lift 6.0x)
    - 트리플필터 통과 (예상 적중 15~25%)
    - 기관/외국인 전환 신호

[3단계] 시장 국면 게이트 — 최종 결정
  상승장: 전략 유지
  하락장: 포지션 축소, Tier-S만 진입
  박스권: 트리플필터 통과 종목만
"""

from typing import Optional


# ── 1단계: 고재현율 필터 ────────────────────────────────────────────────────

def stage1_recall_filter(results: list) -> list:
    """
    세력≥20 OR 거래량≥1.5x OR 트리플필터 1개 이상 통과 종목.
    기본 스캔 필터보다 넓게 잡아 급등 종목 포착률 극대화.
    """
    passed = []
    for r in results:
        se  = r.get('seoryeok', 0)
        vr  = r.get('vol_ratio', 0) or 0
        tf  = r.get('triple_filter', {})
        tf_score = tf.get('score', 0) if tf else 0

        if se >= 20 or vr >= 1.5 or tf_score >= 1:
            passed.append(r)

    return passed


# ── 2단계: 고정밀도 스코어링 ────────────────────────────────────────────────

def stage2_precision_score(r: dict, score_history_context: dict) -> int:
    """
    백테스트 검증 규칙 기반 정밀 점수 (0~100).
    score_history_context: {co_mean, co_slope, consec} 사전 계산값
    """
    score = 0

    se   = r.get('seoryeok', 0)
    vr   = r.get('vol_ratio', 0) or 0
    tf   = r.get('triple_filter', {})
    inv  = r.get('investor', 0)
    pvm  = r.get('price_vs_ma20')

    co_mean  = score_history_context.get('co_mean', 0)
    co_slope = score_history_context.get('co_slope', 0)
    consec   = score_history_context.get('consec', 0)

    # ① 세력 점수 기반 (최대 30점)
    if se >= 80:   score += 30
    elif se >= 70: score += 22
    elif se >= 60: score += 14
    elif se >= 50: score += 8

    # ② 백테스트 R4 규칙 (최대 20점)
    if co_mean >= 54 and co_slope >= 3:
        score += 20   # lift 6.0x
    elif se >= 75 and co_mean >= 54:
        score += 15   # lift 5.0x
    elif se >= 70 and co_slope >= 2:
        score += 10

    # ③ 트리플 필터 (최대 20점)
    tf_score = tf.get('score', 0) if tf else 0
    if tf_score == 3:  score += 20   # 세 조건 모두
    elif tf_score == 2: score += 12  # 두 조건
    elif tf_score == 1: score += 5

    # ④ 기관/외국인 전환 신호 (최대 15점)
    inv_data = r.get('_investor_stats', {})
    if inv_data.get('inst_transition') or inv_data.get('frgn_transition'):
        score += 15
    elif inv_data.get('combo_buy_days', 0) >= 3:
        score += 10
    elif inv >= 50:
        score += 5

    # ⑤ MA20 위치 보정 (최대 10점)
    if pvm is not None:
        if pvm >= 5:    score += 10
        elif pvm >= 0:  score += 6
        elif pvm >= -3: score += 2
        else:           score -= 10   # MA20 아래는 감점

    # ⑥ 거래량 (최대 5점)
    if vr >= 3.0:   score += 5
    elif vr >= 2.0: score += 3

    # 연속 5일+ 페널티 (데이터 기반)
    if consec >= 5:
        score -= min((consec - 4) * 3, 15)

    return max(0, min(100, score))


# ── 3단계: 시장 국면 게이트 ─────────────────────────────────────────────────

def stage3_market_gate(results: list, mkt_ctx: dict) -> tuple[list, str]:
    """
    시장 국면에 따라 최종 종목 수와 기준 조정.

    Returns:
        (최종 후보 리스트, 국면 설명)
    """
    k5 = mkt_ctx.get('kospi_5d', 0)
    q5 = mkt_ctx.get('kosdaq_5d', 0)

    if k5 > 3.0 and q5 > 3.0:
        # 강세장: 모든 Tier 허용, 상위 15개
        regime   = '강세장 — 전체 추천 유지'
        filtered = results
        top_n    = 15

    elif k5 < -3.0 and q5 < -3.0:
        # 하락장: Tier-S만 (세력80+), 트리플필터 통과 필수
        regime   = '하락장 — Tier-S + 트리플필터 통과 종목만'
        filtered = [
            r for r in results
            if r.get('ml_tier', {}).get('tier') == 'S'
            or r.get('triple_filter', {}).get('passed')
        ]
        top_n = 5

    elif abs(k5) < 1.5 and abs(q5) < 1.5:
        # 박스권: 트리플필터 통과 또는 세력80+ 종목만
        regime   = '박스권 — 트리플필터 또는 세력80+ 종목만'
        filtered = [
            r for r in results
            if r.get('triple_filter', {}).get('score', 0) >= 2
            or r.get('seoryeok', 0) >= 80
        ]
        top_n = 10

    else:
        # 중립장: 일반 기준
        regime   = '중립장 — 일반 기준 적용'
        filtered = results
        top_n    = 12

    return filtered[:top_n], regime


# ── 앙상블 메인 ──────────────────────────────────────────────────────────────

def run_ensemble(results: list, mkt_ctx: dict,
                 score_history_map: Optional[dict] = None) -> dict:
    """
    3단계 앙상블 실행.

    Args:
        results           : daily_scan analyze_one() 결과 리스트
        mkt_ctx           : get_market_context() 결과
        score_history_map : {ticker: {co_mean, co_slope, consec}} 사전 계산값

    Returns:
        stage1   : 1단계 통과 종목 수
        stage2   : 2단계 정밀 점수 계산 완료 종목
        stage3   : 최종 추천 종목 (시장 국면 반영)
        regime   : 시장 국면 설명
        triple   : 트리플필터 통과 종목
    """
    if score_history_map is None:
        score_history_map = {}

    # 1단계
    s1 = stage1_recall_filter(results)

    # 2단계: 정밀 점수 계산
    for r in s1:
        ctx = score_history_map.get(r['ticker'], {})
        r['ensemble_score'] = stage2_precision_score(r, ctx)

    s1.sort(key=lambda x: x['ensemble_score'], reverse=True)
    s2 = s1[:30]   # 상위 30개로 압축

    # 트리플 필터 통과 별도 추출
    triple_passed = [r for r in s2 if r.get('triple_filter', {}).get('passed')]

    # 3단계
    s3, regime = stage3_market_gate(s2, mkt_ctx)

    return {
        'stage1':  len(s1),
        'stage2':  s2,
        'stage3':  s3,
        'regime':  regime,
        'triple':  triple_passed,
    }


def build_ensemble_report_section(ensemble: dict) -> list[str]:
    """앙상블 결과 리포트 섹션 생성"""
    sep  = '═' * 70
    sep2 = '─' * 70
    L    = []

    L.append('')
    L.append(sep)
    L.append('  ★★★★★ 앙상블 스코어링 — 3단계 정밀 필터 결과')
    L.append(f'  시장 국면: {ensemble["regime"]}')
    L.append(f'  1단계 후보: {ensemble["stage1"]}개 → 2단계: {len(ensemble["stage2"])}개 → 최종: {len(ensemble["stage3"])}개')
    L.append(sep2)

    # 트리플 필터 별도 강조
    triple = ensemble.get('triple', [])
    if triple:
        L.append(f'\n  ◆ 트리플 필터 통과 (최고 신뢰): {len(triple)}종목')
        for r in triple:
            tf  = r.get('triple_filter', {})
            pvm = r.get('price_vs_ma20')
            ma_str = f'MA20 위 {pvm:+.1f}%' if pvm and pvm >= 0 else (f'MA20 {pvm:+.1f}%' if pvm else '')
            L.append(
                f"    ★ {r['name']} ({r['ticker']})  앙상블{r.get('ensemble_score',0)}점  "
                f"세력{r['seoryeok']}  거래량{r.get('vol_ratio',0):.1f}x  {ma_str}"
            )
            L.append(f"       {tf.get('signal', '')}")

    # 최종 추천 상위 10
    L.append(f'\n  최종 추천 종목 (앙상블 점수 기준)')
    L.append(f'  {"종목명":<12} {"앙상블":>5} {"세력":>5} {"합산":>5} {"거래량":>6} {"vsMA20":>8}  신호')
    L.append(sep2)
    for r in ensemble['stage3'][:10]:
        pvm = r.get('price_vs_ma20')
        ma_flag = '✅' if pvm and pvm >= 0 else ('⚡' if pvm and pvm >= -5 else '❌')
        ma_str  = f'{ma_flag}{pvm:+.1f}%' if pvm is not None else '   ?'
        tf_sig  = r.get('triple_filter', {}).get('signal', '')[:20]
        L.append(
            f"  {r['name']:<12} {r.get('ensemble_score',0):>5} {r['seoryeok']:>5} "
            f"{r['combined']:>5} {r.get('vol_ratio',0):>5.1f}x {ma_str:>8}  {tf_sig}"
        )
    L.append(sep)
    return L
