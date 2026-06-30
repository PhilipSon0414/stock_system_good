#!/usr/bin/env python3
"""
일일 세력 분석 스캔

매일 저녁 8시 자동 실행:
  1. 코스피 + 코스닥 전체 종목 순차 스캔
  2. 세력 흔적 종목 필터링
  3. 급상승 임박도 점수 산출 및 랭킹
  4. 리포트 파일 저장 (reports/ 폴더)
  5. 이메일 발송 (email_config.json 설정 시)

직접 실행: python3 daily_scan.py
"""

import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from data_fetcher import get_ohlcv, get_name, get_ticker_list
from indicators import add_all
from seoryeok import analyze
from order_block import get_order_blocks
from scorer import score as seoryeok_score, get_accum_info
from surge_predictor import score_surge, score_surge_with_history, combined_score, predict_surge_timing
from investor_flow import score_investors
from chart_patterns import score_patterns
from email_sender import send_report
from config import SCREEN_MIN_PRICE, SCREEN_MIN_VOLUME
from forward_tracker import record_scores, build_tracker_section
from financial_data import get_financials, fmt_financials, fmt_market_cap
from short_interest import get_short_interest
from surge_ml import get_confidence_tier, get_entry_timing, model_stats
from triple_filter import scan_triple_filter
from ensemble_scan import run_ensemble, build_ensemble_report_section
from investor_flow import get_investor_flow_stats
from sector_momentum import get_sector_rs
from macro_context import get_full_macro_context, build_macro_report_lines, get_dart_flags

REPORTS_DIR = Path(__file__).parent / 'reports'


def get_market_context() -> dict:
    """코스피/코스닥 추세 + 매크로 외부 지표 통합 컨텍스트"""
    ctx = {'kospi_5d': 0.0, 'kosdaq_5d': 0.0, 'kospi_20d': 0.0,
           'warning': False, 'desc': '', 'macro': {}}
    try:
        idx = get_ohlcv('KS11', period_days=30)
        if len(idx) >= 5:
            ctx['kospi_5d'] = round(
                (idx['Close'].iloc[-1] - idx['Close'].iloc[-5]) / idx['Close'].iloc[-5] * 100, 2)
        if len(idx) >= 20:
            ctx['kospi_20d'] = round(
                (idx['Close'].iloc[-1] - idx['Close'].iloc[-20]) / idx['Close'].iloc[-20] * 100, 2)
        idx2 = get_ohlcv('KQ11', period_days=10)
        if len(idx2) >= 5:
            ctx['kosdaq_5d'] = round(
                (idx2['Close'].iloc[-1] - idx2['Close'].iloc[-5]) / idx2['Close'].iloc[-5] * 100, 2)
    except Exception:
        pass

    k5, q5 = ctx['kospi_5d'], ctx['kosdaq_5d']

    # ── 매크로 외부 지표 수집 ─────────────────────────────────────────────
    print('  외부 매크로 지표 수집 중...')
    try:
        macro = get_full_macro_context()
        ctx['macro'] = macro
        # 매크로 gate가 stop이면 경고 강화
        if macro.get('gate') == 'stop':
            ctx['warning'] = True
        # 매크로 점수를 포함한 종합 설명
        macro_score = macro.get('score', 0)
        macro_regime = {'bull':'강세','neutral':'중립','bear':'약세','panic':'극도공포'}.get(
            macro.get('regime','neutral'), '?')
        vix = macro.get('details', {}).get('VIX', 0)
        krw = macro.get('details', {}).get('USD_KRW', 0)
        macro_summary = f'VIX:{vix:.0f} 원달러:{krw:.0f} 매크로{macro_score:+d}pt[{macro_regime}]'
    except Exception as e:
        macro_summary = ''
        print(f'  매크로 수집 오류: {e}')

    if k5 < -3.0 and q5 < -3.0:
        ctx['warning'] = True
        ctx['desc'] = (f'⚠ 시장 하락장 경고 (코스피 {k5:+.1f}% / 코스닥 {q5:+.1f}%) | {macro_summary}')
    elif k5 > 2.0 and q5 > 2.0:
        ctx['desc'] = (f'✅ 시장 강세 (코스피 {k5:+.1f}% / 코스닥 {q5:+.1f}%) | {macro_summary}')
    else:
        ctx['desc'] = (f'시장: 코스피 {k5:+.1f}% / 코스닥 {q5:+.1f}% | {macro_summary}')
    return ctx
REPORTS_DIR.mkdir(exist_ok=True)

# 스캔 사전 필터 설정
# 검증 결과 (2026-06): 급등종목 97.9%가 거래량 사전필터에서 탈락
# → 종목수 1500으로 확대, 최소거래량 20k로 완화
PRE_FILTER_MIN_PRICE     = 2_000      # 2천원 이상
PRE_FILTER_MIN_VOLUME    = 20_000     # 2만주 이상 (50k→20k: 소형주 포착)
PRE_FILTER_MAX_STOCKS    = 1_500      # 최대 종목 수 (1000→1500: 포착률 향상)
SEORYEOK_MIN_GATE        = 15         # 세력 흔적 최소 점수 (20→15: 완화)
FINAL_MIN_COMBINED       = 25         # 최종 리포트 포함 최소 합산점수 (35→25: 완화)
SEORYEOK_OVERRIDE_GATE   = 60         # 세력 60+ 이면 합산 점수 무시하고 포함
TOP_N_REPORT             = 50         # 전체 순위 50개
UNIVERSE_SAMPLE_RATE     = 0.15       # 탈락 종목 학습용 샘플링 비율(선택편향 해소·ML 음성)
TOP_N_DETAIL             = 20         # 상세 분석 상위 20개

SCORE_HISTORY_PATH       = Path(__file__).parent / 'score_history.json'

# 날짜 기준 스캔 (None이면 오늘, 'YYYYMMDD'이면 해당 종가 기준)
_SCAN_END_DATE: str | None = None
# 데이터검증(2026-06): 연속2일=0%, 연속3일+=2.8% (기저율14.9% 대비 역효과)
# 연속 출현 = 이미 급등이 반영된 종목 → 신규 진입 시점 지남
# → 연속 보너스 완전 제거, 신규 신호(1일)만 소폭 유지
PERSISTENCE_BONUS_TABLE  = {1: 5}  # 첫 출현만 +5 (데이터: 1일째가 최고 신선도)

# 엘리트 픽 기준 — 10% 검증: 세력 80+ 핵심 (1.92x lift)
# 세력 60-69는 노이즈, 합산 점수보다 세력 점수 우선
ELITE_SE_TIER1  = 90    # 세력 90+ + 급등 60+
ELITE_SG_TIER1  = 60    # 65→60 (10% 기준 완화)
ELITE_SE_TIER2  = 80    # 세력 80+ + 거래량 2x+
ELITE_VOL_TIER2 = 2.0   # 3x→2x (더 많은 포착)
ELITE_SE_TIER3  = 75    # 세력 75+ + 연속 3일+
ELITE_CONSEC_TIER3 = 3
ELITE_SG_TIER4  = 75    # 급등 75+ + 연속 2일+ (80→75)
ELITE_CONSEC_TIER4 = 2


def prefilter_tickers(market: str = 'ALL') -> list:
    """StockListing 기반 사전 필터 — 분석 대상 축소"""
    print('  종목 목록 다운로드 중...')
    listing = get_ticker_list(market, date_str=_SCAN_END_DATE)
    if listing.empty:
        print('  ⚠ 종목 목록 없음. 네트워크 확인 요망.')
        return []

    df = listing.copy()

    # 코드 정규화
    df['Code'] = df['Code'].astype(str).str.zfill(6)

    # 우선주 제외 (코드 끝자리 5 또는 7)
    df = df[~df['Code'].str.endswith(('5', '7'))]

    # 가격·거래량 필터
    if 'Close' in df.columns:
        df = df[df['Close'] >= PRE_FILTER_MIN_PRICE]
    if 'Volume' in df.columns:
        df = df[df['Volume'] >= PRE_FILTER_MIN_VOLUME]
        df = df.sort_values('Volume', ascending=False)

    # 거래량 상위 N개만
    df = df.head(PRE_FILTER_MAX_STOCKS)
    return list(df['Code'])


def _get_ticker_score_history(ticker: str, limit: int = 5) -> list[dict]:
    """score_history에서 특정 종목의 최근 N일 스캔 기록 반환 (최신순 → 역순).
    반환 형식: [{'combined': int, 'seoryeok': int, 'surge': int}, ...] 오래된 것 먼저
    """
    if not SCORE_HISTORY_PATH.exists():
        return []
    try:
        records = json.loads(SCORE_HISTORY_PATH.read_text(encoding='utf-8')).get('records', [])
        today   = datetime.now().strftime('%Y-%m-%d')
        hist    = sorted(
            [r for r in records if r['ticker'] == ticker and r['scan_date'] < today],
            key=lambda r: r['scan_date'], reverse=True
        )[:limit]
        return [{'combined': r['combined'], 'seoryeok': r['seoryeok'], 'surge': r['surge']}
                for r in reversed(hist)]  # 오래된 것 먼저
    except Exception:
        return []


def _build_score_history_context(results: list) -> dict:
    """각 종목의 score_history 기반 co_mean/co_slope/consec 계산."""
    import numpy as np
    sh_map = {}
    for r in results:
        ticker = r['ticker']
        hist   = _get_ticker_score_history(ticker, limit=7)
        if not hist:
            sh_map[ticker] = {}
            continue
        co_vals = [h['combined'] for h in hist]
        co_mean = float(np.mean(co_vals))
        if len(co_vals) >= 3:
            x = np.arange(len(co_vals), dtype=float)
            co_slope = float(np.polyfit(x, co_vals, 1)[0])
        else:
            co_slope = float(co_vals[-1] - co_vals[0]) if len(co_vals) >= 2 else 0.0
        sh_map[ticker] = {
            'co_mean':  round(co_mean, 1),
            'co_slope': round(co_slope, 1),
            'consec':   len(hist),
        }
    return sh_map


def _get_consecutive_days() -> dict:
    """score_history.json에서 종목별 연속 등장 일수 반환 {ticker: consecutive_days}
    오늘 이전 스캔 날짜들 기준으로 연속성 체크 (최대 7일).
    """
    if not SCORE_HISTORY_PATH.exists():
        return {}
    try:
        records = json.loads(SCORE_HISTORY_PATH.read_text(encoding='utf-8')).get('records', [])
        today = datetime.now().strftime('%Y-%m-%d')
        dates = sorted(
            {r['scan_date'] for r in records if r['scan_date'] < today},
            reverse=True
        )[:7]
        if not dates:
            return {}

        # 날짜별 ticker 집합 빌드
        date_tickers: dict[str, set] = {}
        for d in dates:
            date_tickers[d] = {r['ticker'] for r in records if r['scan_date'] == d}

        # 각 ticker별 최근부터 연속 등장 일수
        all_tickers = {r['ticker'] for r in records}
        result = {}
        for ticker in all_tickers:
            streak = 0
            for d in dates:
                if ticker in date_tickers[d]:
                    streak += 1
                else:
                    break
            if streak > 0:
                result[ticker] = streak
        return result
    except Exception:
        return {}


def _get_elite_picks(results: list, consec_map: dict) -> list:
    """백테스트 기반 엘리트 픽 필터 (2026-06-12 925건 재학습).

    조건 (복수 해당 가능):
      Tier1: 세력≥90 + 급등≥65                  (극강 세력 + 급등 임박)
      Tier2: 세력≥80 + 거래량>2x                 (세력 포착 + 폭발형 진입)
      Tier3: 세력≥70 + 수급≥60 + 급등30~70       (3중 복합 47~60% 적중, lift 3.2~4.0x)
      Tier4: 세력≥70 + 수급≥80                   (최강 복합 75%+ 적중, lift 5.0x)
      Tier5: 세력≥70 + 수급≥50                   (복합 33.8% 적중, lift 2.26x)
      Tier6: 쇼트스퀴즈+세력≥70
      Tier7: 거래량2x+ + 수급≥50                 (폭발 거래량 + 기관/외인 동반)
      Tier8: 거래량3x+ 단독                       (수급 무관, 실증 lift 2.06x — 저세력 급등 회수)
    """
    elite = []
    seen  = set()
    for r in results:
        ticker = r['ticker']
        se     = r.get('seoryeok',  0)
        sg     = r.get('surge',     0)
        vr     = r.get('vol_ratio', 0)
        inv    = r.get('investor',  0)
        consec = consec_map.get(ticker, 0)
        reasons = []

        if se >= ELITE_SE_TIER1 and sg >= ELITE_SG_TIER1:
            reasons.append(f'Tier1 세력{se}+급등{sg}')
        if se >= ELITE_SE_TIER2 and vr > ELITE_VOL_TIER2:
            reasons.append(f'Tier2 세력{se}+거래량{vr:.1f}x')
        # Tier3: 세력70+수급60+급등30~70 (3중 복합: 47~60% 적중, lift 3.2~4.0x — 최강 실증 조합)
        if se >= 70 and inv >= 60 and 30 <= sg < 70:
            reasons.append(f'Tier3 3중복합(SE{se}+수급{inv}+급등{sg}) 47~60%적중')
        # Tier4: 세력70+수급80+ (75%+ 적중)
        if se >= 70 and inv >= 80:
            reasons.append(f'Tier4 세력{se}+수급{inv}(75%+ 적중)')
        # Tier5: 세력70+수급50+ (33.8% 적중)
        elif se >= 70 and inv >= 50:
            reasons.append(f'Tier5 세력{se}+수급{inv}(33.8% 적중)')
        # Tier6: 쇼트 스퀴즈 가능성 + 세력 진입
        short = r.get('short', {})
        if short.get('squeeze') and se >= 70:
            reasons.append(f'Tier6 쇼트스퀴즈+세력{se}')
        # Tier7: 거래량 폭발 + 수급 동반
        if vr >= 2.0 and inv >= 50:
            reasons.append(f'Tier7 거래량{vr:.1f}x+수급{inv}')
        # Tier8: 거래량 폭발 단독 (수급 무관) — 2026-06-16 실증: 거래량3x+ 단독 lift 2.06x
        #   (급등률 27.5% vs 기저 13.3%). 세력 미포착(저세력) 급등을 거래량으로 회수.
        if vr >= 3.0:
            reasons.append(f'Tier8 거래량폭발{vr:.1f}x단독(lift2.1x)')

        if reasons and ticker not in seen:
            seen.add(ticker)
            elite.append({**r, 'elite_reasons': reasons, 'consec': consec})

    # ① GBM 급등확률 우선 정렬(검증 +3.9%p AUC), 동률 시 티어수·합산점수
    return sorted(elite, key=lambda x: (round(x.get('surge_prob') or 0, 3),
                                        len(x['elite_reasons']), x['combined']),
                  reverse=True)


def _ret_feats(df):
    """급락+거래량 검증 피처: 5·20일 수익률(ret5/ret20). surge_model GBM 입력.
    검증(라벨 1712건 워크포워드, 2026-06-30): 4점수 대비 5일급등 AUC +13.9%p,
    3일 +3.7%p (ret5가 핵심 동력). '강세 급눌림'이 급등 선행."""
    try:
        close = float(df['Close'].iloc[-1])
        c6 = float(df['Close'].iloc[-6]) if len(df) >= 6 else close
        c21 = float(df['Close'].iloc[-21]) if len(df) >= 21 else close
        ret5 = round(close / c6 - 1, 4) if c6 else 0.0
        ret20 = round(close / c21 - 1, 4) if c21 else 0.0
        return ret5, ret20
    except Exception:
        return 0.0, 0.0


def analyze_one(ticker: str) -> dict | None:
    """단일 종목 전체 분석. 실패 시 None 반환."""
    try:
        df = get_ohlcv(ticker, period_days=400, end_date=_SCAN_END_DATE)
        if len(df) < 60:
            return None

        close = df['Close'].iloc[-1]
        vol   = df['Volume'].iloc[-1]
        if close < PRE_FILTER_MIN_PRICE or vol < PRE_FILTER_MIN_VOLUME:
            return None

        df = add_all(df)
        df = analyze(df)
        ob = get_order_blocks(df)

        s_pts, s_tags = seoryeok_score(df, ob)
        accum_info = get_accum_info(df)
        if s_pts < SEORYEOK_MIN_GATE:
            return None  # 세력 흔적 부족 → 제외

        # score_history에서 직전 5일 합산/세력 이력 로드 (통계 피처용)
        _sh_hist = _get_ticker_score_history(ticker, limit=5)
        surge_pts, surge_tags = score_surge_with_history(df, ob, _sh_hist)
        inv_pts,   inv_tags   = score_investors(ticker, df)
        pat_pts,   pat_tags   = score_patterns(df)
        combo = combined_score(s_pts, surge_pts, inv_pts, pat_pts)

        # ── 데이터 기반 복합 보너스/패널티 (2026-06-12 925건 분석) ──────────────
        # 3중 최강 조합 (표본 10+, 기저율 14.9% 대비):
        #   SE70+ 수급60+ 급등30~70 = 47~60% (lift 3.2~4.0x) → +25pt ★★★
        #   SE80+ 수급60+ 급등30~70 = 50% (lift 3.35x)       → +25pt ★★★
        #   SE70+수급80+ = 75%+ 적중률                        → +20pt ★★
        #   SE70+수급60+ = 33.8%+                             → +10pt ★
        # 패널티:
        #   급등90+ 수급<40 = 3.0% (lift 0.20x) — 거의 0에 가까운 역효과  → -25pt
        #   급등80~90 수급<40 = 8.6% (lift 0.57x) — 기저율 이하           → -15pt
        #   급등80+ 수급<60 = 6.8% (lift 0.46x) — 과열 확실               → -15pt
        #   패턴40+ = 3.6% 극역상관 (더 높을수록 나쁨)                      → -10pt
        #   패턴20~40 = 11% (기저율 이하)                                   → -5pt
        #   SE70+ 수급<20 = 9.8% (lift 0.66x, 기저율 이하)                 → -10pt

        # 3중 최강 조합 우선 (보너스 중복 허용)
        if s_pts >= 70 and inv_pts >= 60 and 30 <= surge_pts < 70:
            combo = min(100, combo + 25)
            surge_tags = list(surge_tags) + ['★★★ 세력+수급+급등 3중(47~60% 적중)']
        elif s_pts >= 70 and inv_pts >= 80:
            combo = min(100, combo + 20)
            surge_tags = list(surge_tags) + ['★★ 세력+수급 최강(75% 적중)']
        elif s_pts >= 70 and inv_pts >= 50:
            combo = min(100, combo + 10)
            surge_tags = list(surge_tags) + ['★ 세력+수급 복합(33.8% 적중)']

        # 패턴 역상관 패널티 (패턴 높을수록 급등률 하락)
        if pat_pts >= 40:
            combo = max(0, combo - 10)   # 패턴40+: 3.6% (극역상관)
        elif pat_pts >= 20:
            combo = max(0, combo - 5)    # 패턴20~40: 11% (기저율 이하)

        # 급등 과열 패널티 (수급 없는 급등 과열은 기저율보다 낮음)
        if surge_pts >= 90 and inv_pts < 40:
            combo = max(0, combo - 25)   # 급등90+수급<40: 3.0% — 거의 폭탄
            surge_tags = list(surge_tags) + ['⚠ 급등과열+수급없음(3% 역효과)']
        elif surge_pts >= 80 and inv_pts < 40:
            combo = max(0, combo - 15)   # 급등80~90수급<40: 8.6%
            surge_tags = list(surge_tags) + ['⚠ 급등과열(8.6% 역효과)']
        elif surge_pts >= 80 and inv_pts < 60:
            combo = max(0, combo - 15)   # 급등80+수급<60: 6.8%
            surge_tags = list(surge_tags) + ['⚠ 급등과열(6.8% 역효과)']
        elif s_pts >= 70 and inv_pts < 20:
            combo = max(0, combo - 10)   # 세력70+수급<20: 9.8% 기저율 이하

        # 세력 60+ 이면 합산 무관 포함 (검증: 세력고점 종목은 저합산도 급등 가능)
        if s_pts < SEORYEOK_OVERRIDE_GATE and combo < FINAL_MIN_COMBINED:
            # 광범위 유니버스 로깅: 탈락 종목도 일부만 경량 기록(ML 학습 음성·선택편향 해소).
            # 비싼 후속 분석(재무·공매도·DART·ML)은 생략하고 4점수+df만 담는다.
            if random.random() < UNIVERSE_SAMPLE_RATE:
                _r5, _r20 = _ret_feats(df)
                return {
                    'ticker': ticker, 'name': get_name(ticker), 'price': close,
                    'seoryeok': s_pts, 'surge': surge_pts, 'investor': inv_pts,
                    'pattern': pat_pts, 'combined': combo, 'raw_combined': combo,
                    'vol_ratio': round(df['VolRatio'].iloc[-1], 2) if 'VolRatio' in df.columns else 0,
                    'ret5': _r5, 'ret20': _r20,
                    'df': df, 'ob': ob, 'below_gate': True,
                }
            return None

        vol_ratio = round(df['VolRatio'].iloc[-1], 2) if 'VolRatio' in df.columns else 0
        fin   = get_financials(ticker)
        short = get_short_interest(ticker)

        # ── 트리플 필터 ─────────────────────────────────────────────────
        from triple_filter import check_triple
        triple = check_triple(ticker, df)

        # ── 기관/외국인 흐름 통계 (pykrx) ──────────────────────────────
        inv_stats = get_investor_flow_stats(ticker)

        # ── DART 부정 공시 필터 ──────────────────────────────────────
        dart_info = get_dart_flags(ticker, days=3)

        # ── 현재가 vs MA20 위치 계산 ────────────────────────────────────
        import math as _math
        _ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns else float('nan')
        if not _math.isnan(float(_ma20)) and _ma20 > 0:
            price_vs_ma20 = round((close - _ma20) / _ma20 * 100, 1)
        else:
            price_vs_ma20 = None

        # ── RS vs KOSPI (20일 상대 강도) ──────────────────────────────────
        rs_vs_market = None
        stock_20d_ret = 0.0
        if len(df) >= 20:
            mkt_ctx = getattr(run_scan, '_market_ctx', {})
            kospi_20d = mkt_ctx.get('kospi_20d', 0.0)
            stock_20d_ret = (close - df['Close'].iloc[-20]) / df['Close'].iloc[-20] * 100
            rs_vs_market = round(stock_20d_ret - kospi_20d, 1)

        # ── 업종 상대강도 ─────────────────────────────────────────────
        sector_rs = get_sector_rs(ticker, stock_20d_ret)

        # ── ML 신뢰도 티어 ────────────────────────────────────────────────
        latest_row = df.iloc[-1]
        ml_tier = get_confidence_tier(s_pts, surge_pts, inv_pts, pat_pts, combo)
        has_se_entry = bool(latest_row.get('SE_Entry', False))
        consec_days  = 0  # 연속일은 run_scan 이후 보너스 적용 시점에서 업데이트됨
        entry_timing = get_entry_timing(
            ml_tier['tier'], vol_ratio, rs_vs_market, has_se_entry, consec_days
        )

        # ── 급등 사이클 스테이지 예측 (D-5 ~ D+1) ──────────────────────────
        stage_result = None
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent.parent))
            from surge_stage_model import predict_from_df, stage_tag
            stage_result = predict_from_df(df)
        except Exception:
            pass

        _r5, _r20 = _ret_feats(df)
        return {
            'ticker':        ticker,
            'name':          get_name(ticker),
            'price':         close,
            'vol_ratio':     vol_ratio,
            'ret5':          _r5,
            'ret20':         _r20,
            'seoryeok':      s_pts,
            'surge':         surge_pts,
            'investor':      inv_pts,
            'pattern':       pat_pts,
            'combined':      combo,
            's_tags':        s_tags,
            'surge_tags':    surge_tags,
            'inv_tags':      inv_tags,
            'pat_tags':      pat_tags,
            'df':            df,
            'ob':            ob,
            'accum_info':    accum_info,
            'fin':           fin,
            'short':         short,
            'rs_vs_market':    rs_vs_market,
            'ml_tier':         ml_tier,
            'entry_timing':    entry_timing,
            'price_vs_ma20':   price_vs_ma20,
            'triple_filter':   triple,
            '_investor_stats': inv_stats,
            'sector_rs':       sector_rs,
            'dart_info':       dart_info,
            'stage':           stage_result,   # 급등 사이클 스테이지
        }
    except Exception:
        return None


def run_scan(market: str = 'ALL') -> list:
    now = datetime.now()
    print(f'\n{"="*65}')
    print(f'  일일 세력 분석 스캔 시작')
    print(f'  실행 시각: {now.strftime("%Y-%m-%d %H:%M")}')
    print(f'  대상 시장: {market}')
    print(f'{"="*65}')

    # 시장 컨텍스트 조회 (4순위)
    mkt_ctx = get_market_context()
    print(f'\n  [{mkt_ctx["desc"]}]')
    run_scan._market_ctx = mkt_ctx

    tickers = prefilter_tickers(market)
    total = len(tickers)
    if not tickers:
        return []

    print(f'\n  사전 필터 통과: {total}개 종목 분석 시작')
    print(f'  예상 소요 시간: 약 {total // 10}분\n')

    results = []
    for i, ticker in enumerate(tickers, 1):
        result = analyze_one(ticker)
        if result:
            results.append(result)

        # 진행 상황 출력 (30종목마다)
        if i % 30 == 0 or i == total:
            pct = i / total * 100
            bar = '█' * (i * 20 // total) + '░' * (20 - i * 20 // total)
            print(f'  [{bar}] {pct:.0f}%  {i}/{total}  후보: {len(results)}개',
                  end='\r')

        time.sleep(0.1)

    print()

    # ② 광범위 유니버스 샘플 분리 (탈락 종목 — 학습 로깅용, 픽/리포트에서 제외)
    universe_sample = [r for r in results if r.get('below_gate')]
    results = [r for r in results if not r.get('below_gate')]
    run_scan._universe_sample = universe_sample
    if universe_sample:
        print(f'  유니버스 샘플(탈락) 로깅 대상: {len(universe_sample)}개')

    # raw_combined 저장 (연속 보너스 적용 전 순수 점수 — ML 학습 및 역설 분석용)
    for r in results:
        r['raw_combined'] = r.get('combined', 0)

    # 연속 등장 보너스 적용 (스케일: 1일+10, 2일+10, 3일+20, 4일++30)
    # ※ 이평선 하향 + 현재가 MA20 -10% 이상 이탈 시 보너스 제한 (모델 버그 수정)
    consec_map = _get_consecutive_days()
    if consec_map:
        bonus_count = 0
        for r in results:
            days = consec_map.get(r['ticker'], 0)
            if days >= 1:
                bonus = PERSISTENCE_BONUS_TABLE.get(min(days, 4), 30)

                # 차트 하향 시 보너스 상한 제한
                df_r = r.get('df')
                if df_r is not None and len(df_r) > 0:
                    latest_r  = df_r.iloc[-1]
                    ma20_r    = latest_r.get('MA20', float('nan'))
                    close_r   = latest_r.get('Close', 0)
                    import math
                    if not math.isnan(float(ma20_r)) and ma20_r > 0:
                        gap = (close_r - ma20_r) / ma20_r
                        if gap < -0.10:
                            # 현재가가 MA20 아래 10%+ → 보너스 절반
                            bonus = bonus // 2
                            r['s_tags'] = [f'⚠ MA20 이탈({gap*100:.0f}%)로 연속보너스 제한'] + r['s_tags']

                r['combined']  = min(100, r['combined'] + bonus)
                r['consec_days'] = days
                label = f'★ {days}일 연속등장 보너스 (+{bonus}점)'
                r['s_tags'] = [label] + r['s_tags']
                bonus_count += 1
        if bonus_count:
            print(f'  연속등장 보너스: {bonus_count}개 종목 적용')

    # 결과에 consec_map 저장 + entry_timing 재계산 (consec_days 반영)
    for r in results:
        r.setdefault('consec_days', consec_map.get(r['ticker'], 0))
        # entry_timing에 consec_days 반영하여 재계산
        ml = r.get('ml_tier', {})
        if ml:
            r['entry_timing'] = get_entry_timing(
                ml['tier'],
                r.get('vol_ratio', 0),
                r.get('rs_vs_market'),
                bool(r.get('df', None) is not None and r['df'].iloc[-1].get('SE_Entry', False)),
                r.get('consec_days', 0),
            )
    run_scan._consec_map = consec_map   # 전달용 임시 저장

    # ① GBM 급등확률(검증된 4점수 모델, +3.9%p AUC) — 랭킹 보강용
    try:
        from surge_model import predict as _surge_predict
        for r in results:
            p = _surge_predict(r, horizon='surged_by_3d')
            if p is not None:
                r['surge_prob'] = round(p, 3)
        if any('surge_prob' in r for r in results):
            print('  GBM 급등확률 계산 완료(랭킹 보강)')
    except Exception as e:
        print(f'  [GBM] 급등확률 스킵: {e}')

    results.sort(key=lambda x: x['combined'], reverse=True)
    results = results[:TOP_N_REPORT]

    # 트리플 필터 + 앙상블 스코어링 실행
    triple_hits, double_hits = scan_triple_filter(results)
    if triple_hits:
        print(f'  트리플 필터 통과: {len(triple_hits)}종목 | 더블: {len(double_hits)}종목')

    sh_map = _build_score_history_context(results)
    ensemble = run_ensemble(results, mkt_ctx, sh_map)
    run_scan._ensemble   = ensemble
    run_scan._triple_hits = triple_hits

    return results


def _calc_ob_trade_params(ob: dict, price: float, r: dict) -> dict:
    """OB 기반 진입가·목표가·손절가·기대수익률 계산.

    진입전략:
      - 강세OB 진입 중 or OB플립 지지 → 현재가 진입
      - 강세OB 5% 이내 접근 → OB 상단 근처 눌림목 대기
      - 세력80+ & MA20 위 → 즉시 진입

    목표가 산출 우선순위:
      1. 가장 가까운 약세OB (저항) 하단
      2. 52주 신고가
      3. 기본 +10% (데이터 없을 때)

    손절가: 강세OB 하단 또는 현재가 -5%
    """
    result = {
        'entry':    price,
        'target':   None,
        'stop':     None,
        'rr_ratio': None,
        'expected_return': None,
        'entry_desc': '',
        'target_desc': '',
        'stop_desc': '',
    }

    seoryeok = r.get('seoryeok', 0)
    pvm      = r.get('price_vs_ma20')
    ml_tier  = r.get('ml_tier', {}).get('tier', 'D')
    combined = r.get('combined', 0)

    # ── 진입가 결정 ──────────────────────────────────────────────
    bull_dist = ob.get('nearest_bull_dist')
    flip_dist = ob.get('nearest_flipped_dist')
    bull_obs  = sorted(ob.get('bull', []), key=lambda x: abs(price - x['Mid']))
    flip_obs  = sorted(ob.get('flipped_bear', []), key=lambda x: abs(price - x['High']))

    if bull_dist is not None and bull_dist == 0.0:
        # 강세OB 안에 있음 → 현재가 진입
        result['entry'] = price
        result['entry_desc'] = f'강세OB 진입 중 → 현재가({price:,.0f}원) 즉시 진입'
    elif flip_dist is not None and flip_dist < 0.03:
        # OB 플립 지지 위 → 현재가 진입
        result['entry'] = price
        result['entry_desc'] = f'OB플립 지지({flip_dist*100:.1f}% 위) → 현재가 진입'
    elif bull_obs and bull_dist is not None and bull_dist < 0.05:
        # 강세OB 5% 이내 접근 → OB 상단(High)까지 눌림목 대기
        ob_high = bull_obs[0]['High']
        if ob_high < price:  # 이미 OB 위에 있음
            result['entry'] = price
            result['entry_desc'] = f'강세OB 상단({ob_high:,.0f}원) 돌파 확인 → 현재가 진입'
        else:
            result['entry'] = ob_high
            result['entry_desc'] = f'강세OB 상단({ob_high:,.0f}원) 눌림목 대기 (현재가 {price:,.0f}원)'
    elif pvm is not None and pvm >= 0 and seoryeok >= 80:
        result['entry'] = price
        result['entry_desc'] = f'세력{seoryeok}+ MA20 위 → 현재가 즉시 진입'
    else:
        result['entry'] = price
        result['entry_desc'] = f'현재가 진입 (추가 확인 권장)'

    entry = result['entry']

    # ── 목표가 결정 ──────────────────────────────────────────────
    # 1순위: 가장 가까운 위쪽 약세OB. 단 저항이 현재가 +3% 안쪽이면 목표로 부적합
    # (×0.99 할인 시 목표가 현재가 아래로 떨어져 R:R 음수가 되는 버그 방지).
    TARGET_FLOOR = 1.03
    bear_obs = [b for b in ob.get('bear', []) if b['Low'] > entry * TARGET_FLOOR]
    bear_obs_sorted = sorted(bear_obs, key=lambda x: x['Low'])

    df = r.get('df')
    high52w = None
    if df is not None and 'High52W' in df.columns:
        high52w = float(df['High52W'].iloc[-1])

    if bear_obs_sorted:
        target = bear_obs_sorted[0]['Low'] * 0.99  # 저항 직전 99%
        result['target']      = int(target)
        result['target_desc'] = f'약세OB 저항({bear_obs_sorted[0]["Low"]:,.0f}원) 직전 목표'
    elif high52w and high52w > entry * 1.03:
        result['target']      = int(high52w * 0.99)
        result['target_desc'] = f'52주 신고가({high52w:,.0f}원) 근접 목표'
    else:
        # 기본: Tier에 따라 목표 수익률 다르게
        target_pct = {'S': 0.15, 'A': 0.12, 'B': 0.10, 'C': 0.08, 'D': 0.08}.get(ml_tier, 0.10)
        result['target']      = int(entry * (1 + target_pct))
        result['target_desc'] = f'기본 목표 +{target_pct*100:.0f}% (OB 데이터 없음)'

    # ── 손절가 결정 ──────────────────────────────────────────────
    if bull_obs:
        ob_low = bull_obs[0]['Low']
        if ob_low < entry:
            result['stop']      = int(ob_low * 0.99)
            result['stop_desc'] = f'강세OB 하단({ob_low:,.0f}원) -1% 손절'
        else:
            result['stop']      = int(entry * 0.95)
            result['stop_desc'] = f'기본 -5% 손절'
    elif flip_obs:
        flip_low = flip_obs[0]['Low']
        result['stop']      = int(flip_low * 0.99)
        result['stop_desc'] = f'OB플립 하단({flip_low:,.0f}원) 이탈 시 손절'
    else:
        result['stop']      = int(entry * 0.95)
        result['stop_desc'] = f'기본 -5% 손절'

    # ── 기대수익률 · R:R 계산 ────────────────────────────────────
    if result['target'] and result['stop']:
        profit   = result['target'] - entry
        risk     = entry - result['stop']
        if risk > 0:
            result['rr_ratio']        = round(profit / risk, 2)
            result['expected_return'] = round(profit / entry * 100, 1)

    # ── R:R 불리 시 진입 등급 강등 ───────────────────────────────
    # 목표≤현재가 또는 R:R<1.0 이면 기대값 비우호 → '즉시 진입' 라벨을 보류로 교정.
    # (신호 레이어가 '즉시진입'이라도 매매 수학이 음수면 신규진입 부적합)
    rr = result.get('rr_ratio')
    unfavorable = (result['target'] is not None and result['target'] <= entry) or \
                  (rr is not None and rr < 1.0)
    result['rr_unfavorable'] = bool(unfavorable)
    if unfavorable:
        rr_txt = f"R:R {rr:.1f}" if rr is not None else "목표<현재가"
        result['entry_desc'] = (f"⚠ 진입보류({rr_txt} 불리·저항 근접) — "
                                f"눌림목 대기 권장 (현재가 {price:,.0f}원)")

    return result


def _build_recommendation_rationale(r: dict, ob_params: dict) -> list[str]:
    """추천 근거 상세 텍스트 생성."""
    lines = []
    se    = r.get('seoryeok', 0)
    sg    = r.get('surge', 0)
    inv   = r.get('investor', 0)
    vr    = r.get('vol_ratio', 0) or 0
    pvm   = r.get('price_vs_ma20')
    ml    = r.get('ml_tier', {})
    tf    = r.get('triple_filter', {})
    dart  = r.get('dart_info', {})
    consec= r.get('consec_days', 0)

    # 핵심 근거 (양성)
    strengths = []
    if se >= 80:
        strengths.append(f'세력{se}점 (10%급등 예상적중 {ml.get("hit_rate",0)*100:.0f}%)')
    elif se >= 70:
        strengths.append(f'세력{se}점 (유의미 신호)')

    # 이평선 위치
    if pvm is not None:
        if pvm >= 5:
            strengths.append(f'MA20 위 {pvm:+.1f}% (상승 추세 확인)')
        elif pvm >= 0:
            strengths.append(f'MA20 위 {pvm:+.1f}% (추세 돌파)')
        elif pvm >= -5:
            strengths.append(f'MA20 근접 {pvm:+.1f}% (돌파 시도)')

    # 거래량
    if vr >= 3.0:
        strengths.append(f'거래량 {vr:.1f}x 폭발 (세력 진입 확인)')
    elif vr >= 2.0:
        strengths.append(f'거래량 {vr:.1f}x 상승')
    elif vr < 0.4:
        strengths.append(f'거래량 {vr:.1f}x 극압축 (급등 직전 잠복 패턴)')

    # 연속 등장
    if 2 <= consec <= 4:
        strengths.append(f'{consec}일 연속 등장 (지속 세력 확인)')
    elif consec >= 5:
        strengths.append(f'{consec}일 연속 (시그널 약화 주의)')

    # 트리플 필터
    tf_score = tf.get('score', 0)
    if tf_score == 3:
        strengths.append('★ 트리플필터 통과 (공매도↓+기관매수+거래량)')
    elif tf_score == 2:
        strengths.append('더블필터 통과 (2/3 조건)')

    # 기관 전환
    inv_stats = r.get('_investor_stats', {})
    if inv_stats.get('inst_transition'):
        strengths.append('★ 기관 순매도→매수 전환')
    if inv_stats.get('frgn_transition'):
        strengths.append('★ 외국인 순매도→매수 전환')
    if inv_stats.get('combo_buy_days', 0) >= 3:
        strengths.append(f'기관+외국인 동시매수 {inv_stats["combo_buy_days"]}일')

    # OB 상황
    ob = r.get('ob', {})
    if ob.get('nearest_bull_dist') == 0.0:
        strengths.append('강세OB 내 진입 (최적 지지)')
    elif ob.get('nearest_bull_dist') is not None and ob['nearest_bull_dist'] < 0.03:
        strengths.append(f'강세OB 근접 ({ob["nearest_bull_dist"]*100:.1f}% 내)')

    # DART 공시
    if dart.get('has_positive'):
        strengths.append(f"✅ 긍정공시: {dart.get('positive_title','')[:15]}")
    if dart.get('has_negative'):
        strengths.append(f"⚠ 부정공시 주의: {dart.get('negative_title','')[:15]}")

    # 약점
    warnings = []
    if pvm is not None and pvm < -10:
        warnings.append(f'현재가 MA20 -{abs(pvm):.0f}% 이탈 (하락 추세)')
    if consec >= 5:
        warnings.append(f'연속{consec}일 → 급등 시그널 약화')
    if dart.get('has_negative'):
        warnings.append('부정공시 → 희석/차입 위험')

    # 포매팅
    lines.append(f'       ┌─ 추천 근거 {"─"*45}')
    for s in strengths[:5]:
        lines.append(f'       │  ✓ {s}')
    for w in warnings[:2]:
        lines.append(f'       │  ⚠ {w}')

    # OB 진입 파라미터
    if ob_params and ob_params.get('rr_ratio'):
        lines.append(f'       ├─ OB 기반 매매 전략 {"─"*40}')
        lines.append(f'       │  진입가: {ob_params["entry"]:>10,.0f}원  ← {ob_params["entry_desc"][:30]}')
        lines.append(f'       │  목표가: {ob_params["target"]:>10,.0f}원  ({ob_params["expected_return"]:+.1f}%)  ← {ob_params["target_desc"][:28]}')
        lines.append(f'       │  손절가: {ob_params["stop"]:>10,.0f}원  ({(ob_params["stop"]-ob_params["entry"])/ob_params["entry"]*100:+.1f}%)  ← {ob_params["stop_desc"][:28]}')
        lines.append(f'       │  리스크·리워드: {ob_params["rr_ratio"]:.1f}:1  {"✅ 유리" if ob_params["rr_ratio"] >= 2 else ("⚠ 보통" if ob_params["rr_ratio"] >= 1 else "❌ 불리")}')
    lines.append(f'       └{"─"*56}')

    return lines


def _format_ob_lines(ob: dict, price: float) -> list:
    """오더블록 정보를 리포트 라인으로 변환"""
    lines = []
    has_ob = ob.get('bull') or ob.get('bear') or ob.get('flipped_bear')
    if not has_ob:
        return lines

    lines.append('       오더블록:')

    # ① OB 플립 (전 저항 → 현재 지지) — 가장 먼저 표시
    flipped = sorted(ob.get('flipped_bear', []),
                     key=lambda x: abs(price - x['High']))
    for b in flipped[:2]:
        dist_pct = (price - b['High']) / b['High'] * 100
        date_str = b['Date'].strftime('%Y-%m-%d') if hasattr(b['Date'], 'strftime') else str(b['Date'])[:10]
        visit_str = f'접촉{b["visit_count"]}회' if b.get('visit_count', 0) > 0 else '미접촉'
        if dist_pct <= 2:
            proximity = f'  ← 플립지지 진입! (+{dist_pct:.1f}%)'
        elif dist_pct <= 5:
            proximity = f'  ← 플립지지 근접 (+{dist_pct:.1f}%)'
        else:
            proximity = f'  (돌파 +{dist_pct:.1f}%)'
        lines.append(
            f'         🔄 OB플립: {b["Low"]:>10,.0f} ~ {b["High"]:,.0f}원'
            f'  ({date_str}, {visit_str}){proximity}'
        )

    # ② 강세 오더블록 (현재가에 가까운 순 3개)
    bull_obs = sorted(ob.get('bull', []),
                      key=lambda x: abs(price - x['Mid']))
    for b in bull_obs[:3]:
        dist_pct = (price - b['High']) / b['High'] * 100
        date_str = b['Date'].strftime('%Y-%m-%d') if hasattr(b['Date'], 'strftime') else str(b['Date'])[:10]
        fresh_str = '미접촉' if b.get('is_fresh') else f'접촉{b.get("visit_count",0)}회'
        strong_str = ' ★고거래량' if b.get('is_strong') else ''
        if dist_pct == 0 or abs(dist_pct) <= 1:
            proximity = '  ← 진입 중!'
        elif dist_pct <= 4:
            proximity = f'  ← {dist_pct:+.1f}% (근접)'
        else:
            proximity = f'  ({dist_pct:+.1f}%)'
        lines.append(
            f'         ✅ 강세OB: {b["Low"]:>10,.0f} ~ {b["High"]:,.0f}원'
            f'  ({date_str}, {fresh_str}{strong_str}){proximity}'
        )

    # ③ 활성 약세 오더블록 (아직 저항인 것, 2개)
    bear_obs = sorted(ob.get('bear', []),
                      key=lambda x: abs(price - x['Mid']))
    for b in bear_obs[:2]:
        dist_pct = (b['Low'] - price) / price * 100
        date_str = b['Date'].strftime('%Y-%m-%d') if hasattr(b['Date'], 'strftime') else str(b['Date'])[:10]
        if 0 < dist_pct <= 2:
            proximity = f'  ← 저항 근접! (+{dist_pct:.1f}%)'
        elif 0 < dist_pct <= 5:
            proximity = f'  ← {dist_pct:+.1f}% 위 (저항)'
        else:
            proximity = f'  ({dist_pct:+.1f}%)'
        lines.append(
            f'         ⚠  약세OB: {b["Low"]:>10,.0f} ~ {b["High"]:,.0f}원'
            f'  ({date_str}){proximity}'
        )

    return lines


def _build_focus_picks(results: list, top_n: int = 5) -> list:
    """오늘의 집중 픽: GBM(surge_prob) 상위 + R:R 유리 + 미발동(D+0/D+1 제외) top_n.

    데이터 검증(2026-06-30, 라벨 1661건 GBM 채점): 매일 상위 N개 매수 시 적중률
    N=1 45.8% → N=5 37.5% → N=10 30.4%(기저 11.4%). 정밀도는 소수 픽에서 포화하고
    N≥6부터 하락 → 분산까지 감안한 실전 최적 N=4~5. R:R불리·이미발동은 기대값↓라 제외.
    """
    cand = []
    for r in results:
        if r.get('surge_prob') is None:
            continue
        ob = _calc_ob_trade_params(r.get('ob', {}), r['price'], r)
        if ob.get('rr_unfavorable'):          # 목표≤현재가 or R:R<1 → 제외
            continue
        stage = (r.get('stage') or {}).get('predicted', 'normal')
        if stage in ('D+0', 'D+1'):           # 이미 발동/후속 → 추격 제외
            continue
        r['_focus_ob'] = ob
        r['_focus_stage'] = stage
        cand.append(r)
    cand.sort(key=lambda r: -(r.get('surge_prob') or 0))
    return cand[:top_n]


def _focus_reason(r) -> str:
    """집중 픽 한 줄 근거(지배적 신호 기준)."""
    se = r.get('seoryeok', 0); inv = r.get('investor', 0); vr = r.get('vol_ratio', 0) or 0
    bits = []
    if se >= 70 and inv >= 80:
        bits.append('세력+수급80 (75%+ 실증조합)')
    elif se >= 70 and inv >= 60:
        bits.append('세력+수급 복합 (47~60%)')
    elif se >= 70 and inv >= 50:
        bits.append('세력+수급 복합 (33.8%)')
    if vr >= 3:
        bits.append(f'거래량{vr:.1f}x폭발(lift2.1x)')
    elif vr >= 2:
        bits.append(f'거래량{vr:.1f}x동반')
    if se >= 90:
        bits.append(f'세력{se} 최상위')
    if not bits:
        bits.append(f'세력{se}/수급{inv}')
    return ' · '.join(bits[:2])


def _render_focus_picks(picks: list, top_n: int = 5) -> list:
    sep = '═' * 70
    L = [sep,
         f'  🎯 오늘의 집중 픽 TOP {top_n}  (GBM 상위 · R:R 유리 · 미발동만)',
         f'  ※ 실증: 매일 상위 {top_n}개 ≈ 적중률 ~38%(기저 11%의 3.3x). R:R불리·이미발동 제외, 분산 위해 {top_n}종목.',
         '─' * 70]
    if not picks:
        L.append('  (조건 충족 종목 없음 — R:R 유리 & 미발동 픽이 오늘은 부재)')
        L.append(sep)
        return L
    for i, r in enumerate(picks, 1):
        ob = r.get('_focus_ob', {})
        sp = (r.get('surge_prob') or 0) * 100
        tgt = ob.get('expected_return')
        rr = ob.get('rr_ratio')
        stage = r.get('_focus_stage', 'normal')
        stage_s = {'D-1': 'D-1(임박)', 'D-2': 'D-2(1~2일)', 'D-3': 'D-3',
                   'D-4': 'D-4', 'D-5': 'D-5(이른편)'}.get(stage, '대기')
        # 비현실적 원거리 목표(>30%·R:R 2자리)는 신뢰도 표기 — 먼 저항/신고가 기반
        far = (tgt is not None and tgt > 30) or (rr is not None and rr >= 6)
        tgt_s = (f'목표 {tgt:+.1f}%' + ('⚠원거리' if far else '')) if tgt is not None else '목표 -'
        rr_s = f'R:R {rr:.1f}' if rr is not None else 'R:R -'
        L.append(f'  {i}. {r["name"]}({r["ticker"]})  '
                 f'GBM {sp:.0f}%  세력{r.get("seoryeok",0)}/수급{r.get("investor",0)}  '
                 f'거래량{(r.get("vol_ratio",0) or 0):.1f}x  {rr_s}  {tgt_s}  [{stage_s}]')
        L.append(f'      └ {_focus_reason(r)}')
    L.append('─' * 70)
    L.append('  ※ GBM=종목별 검증 급등확률(1659건 학습). 절대치는 다소 낙관 — 순위·선별 효과가 핵심.')
    L.append('  ※ 참고용이며 매매 권유 아님. 고용지표 등 이벤트 임박 시 분할·보수적 진입.')
    L.append(sep)
    return L


def build_report(results: list, scan_market: str) -> str:
    now = datetime.now()
    lines = []
    sep  = '═' * 70
    sep2 = '─' * 70

    mkt_ctx    = getattr(run_scan, '_market_ctx', {})
    consec_map = getattr(run_scan, '_consec_map', {})

    # ── 헤더 ────────────────────────────────────────────────────────────────
    lines.append(sep)
    lines.append(f'  일일 세력 분석 리포트')
    lines.append(f'  {now.strftime("%Y년 %m월 %d일 %H:%M")}  |  시장: {scan_market}')
    lines.append(sep)
    lines.append(f'  발굴 종목: {len(results)}개')
    if mkt_ctx.get('desc'):
        lines.append(f'  {mkt_ctx["desc"]}')

    # 데이터 기반 핵심 가이드
    lines.append('')
    lines.append('  ┌─ 의사결정 핵심 원칙 (925건 실증 데이터, 기저율 14.9%) ─────────────')
    lines.append('  │  ★★★ SE70+수급60+급등30~70 = 47~60% (lift 3.2~4.0x) ← 최강 3중')
    lines.append('  │  ★★  SE70+수급80+ = 75%+  (lift 5.0x)  ← 세력+수급 최고조합')
    lines.append('  │  ★   SE70+수급50+ = 33.8% (lift 2.26x) ← 중간 복합 신호')
    lines.append('  │  ⚠  급등90+수급<40 = 3.0%  (lift 0.20x) ← 과열=역효과 (폭탄!)')
    lines.append('  │  ⚠  급등80+수급<60 = 6.8%  (lift 0.46x) ← 과열 패널티 적용')
    lines.append('  │  ⚠  패턴40+ = 3.6%  (lift 0.24x) ← 패턴 고점=역효과 (제외 대상)')
    lines.append('  │  ⚠  합산70+ = 6.1%  (lift 0.41x) ← 합산 맹신 금지 (패턴 인플레)')
    lines.append('  │  수급80+단독 = 38.5% (lift 2.58x) | 세력80+ = 31.7% (lift 2.13x)')
    lines.append('  │  연속2일=0% / 연속3일+=2.8% → 연속 출현은 급등 소진 신호')
    lines.append(f'  │  ML모델: {model_stats()}')
    lines.append('  └──────────────────────────────────────────────────────────────────')

    # 매크로 컨텍스트
    macro = mkt_ctx.get('macro', {})
    if macro:
        for mline in build_macro_report_lines(macro):
            lines.append(mline)
    lines.append('')

    # ── 🎯 오늘의 집중 픽 TOP 5 (선택 폭 압축 → 적중률↑) ──────────────────────
    focus = _build_focus_picks(results, top_n=5)
    lines += _render_focus_picks(focus, top_n=5)
    lines.append('')

    # ── 엘리트 픽 (세력점수 기준 재편) ────────────────────────────────────────
    elite_picks = _get_elite_picks(results, consec_map)

    lines.append(sep)
    if elite_picks:
        lines.append('  ★★★★ 엘리트픽 — 세력×수급×급등 복합 신호 (925건 실증)')
        lines.append('  ┌ Tier1: 세력≥90+급등 ┬ Tier2: 세력≥80+거래량2x+')
        lines.append('  ├ Tier3: SE70+수급60+급등30~70(47~60%!) ┼ Tier4: SE70+수급80+(75%+)')
        lines.append('  ├ Tier5: SE70+수급50+(33.8%) ┼ Tier6: 쇼트스퀴즈 ┼ Tier7: 거래량2x++수급50+ ┘')
        lines.append(sep2)
        lines.append(
            f'  {"종목명":<14} {"현재가":>9} {"세력":>4} {"수급":>4} {"거래량":>6}'
            f' {"vsMA20":>7} {"GBM":>5} {"ML티어":>8} {"타이밍예측":>12}  조건'
        )
        lines.append(sep2)
        for r in elite_picks:
            vr    = r.get('vol_ratio', 0) or 0
            pvm   = r.get('price_vs_ma20')
            ml    = r.get('ml_tier', {})
            inv   = r.get('investor', 0)
            cond  = ' + '.join(r['elite_reasons'])
            ma_s  = f'{pvm:+.1f}%' if pvm is not None else '  nan'
            ma_f  = '✅' if pvm and pvm>=0 else ('⚡' if pvm and pvm>=-5 else '❌')
            tier  = ml.get('tier','?')
            rate  = ml.get('hit_rate',0)
            sp    = r.get('surge_prob')
            sp_s  = f'{sp*100:.0f}%' if sp is not None else '  -'

            # 급등 타이밍 예측
            df_r   = r.get('df')
            ma_bull = bool(df_r.iloc[-1].get('MaBull',False)) if df_r is not None and len(df_r)>0 else False
            timing = predict_surge_timing(r['seoryeok'], vr, ma_bull, r.get('consec_days',0))
            timing_s = f'{timing["expected_days"]:.0f}일내({timing["confidence_pct"]}%)'

            lines.append(
                f'  {r["name"]:<14} {r["price"]:>9,.0f}원'
                f' {r["seoryeok"]:>4}점 {inv:>4}점 {vr:>5.1f}x'
                f' {ma_s:>6}{ma_f}'
                f' {sp_s:>5}'
                f' Tier-{tier}({rate*100:.0f}%)'
                f' {timing_s:>12}  {cond}'
            )
        lines.append(sep2)
        lines.append(f'  ※ 엘리트픽 {len(elite_picks)}종목 | 즉시 모니터링')
        lines.append('  ※ 타이밍 N일내(X%)는 거래량·세력 유형별 평균(57건 소표본)이라 종목별 확률 아님')
        lines.append('     → 종목별 검증 확률은 GBM 컬럼(1659건 학습) 참고')
    else:
        lines.append('  ★★★★ 엘리트픽: 해당 없음')
        lines.append(sep2)
    lines.append('')

    # ── 폭발형 / 잠복형 (거래량 기반) ─────────────────────────────────────────
    explosive = [r for r in results if (r.get('vol_ratio',0) or 0) >= 2.0 and r.get('seoryeok',0) >= 65]
    latent    = [r for r in results if (r.get('vol_ratio',0) or 0) < 0.3 and r.get('seoryeok',0) >= 65]

    if explosive or latent:
        lines.append(sep)
        lines.append('  ★★★ 거래량 기반 즉시 진입 후보')
        lines.append(f'  폭발형(거래량2x+&세력65+): {len(explosive)}개  '
                     f'잠복형(거래량<0.3x&세력65+): {len(latent)}개')
        lines.append(sep2)
        all_vol = sorted(explosive + latent, key=lambda x: x.get('vol_ratio',0), reverse=True)
        lines.append(
            f'  {"종목명":<14} {"현재가":>9} {"세력":>4} {"거래량":>6}'
            f' {"vsMA20":>7} {"타이밍":>10}  유형 | 주요 신호'
        )
        lines.append(sep2)
        seen_v = set()
        for r in all_vol[:8]:
            if r['ticker'] in seen_v: continue
            seen_v.add(r['ticker'])
            vr   = r.get('vol_ratio',0) or 0
            pvm  = r.get('price_vs_ma20')
            ma_s = f'{pvm:+.1f}%' if pvm is not None else ''
            df_r = r.get('df')
            ma_bull = bool(df_r.iloc[-1].get('MaBull',False)) if df_r is not None and len(df_r)>0 else False
            t    = predict_surge_timing(r['seoryeok'], vr, ma_bull, r.get('consec_days',0))
            vtype = '폭발형🔥' if vr >= 2.0 else '잠복형💤'
            sig  = next((x for x in r.get('s_tags',[]) if '연속' not in x and '보너스' not in x), '')
            lines.append(
                f'  {r["name"]:<14} {r["price"]:>9,.0f}원'
                f' {r["seoryeok"]:>4}점 {vr:>5.1f}x'
                f' {ma_s:>7}  {t["expected_days"]:.0f}일({t["confidence_pct"]}%)'
                f'  {vtype} | {sig[:25]}'
            )
        lines.append(sep2)
        lines.append('')

    # ── 급등 임박 후보 (재발화) ─────────────────────────────────────────────
    imminent = [r for r in results if r.get('accum_info', {}).get('is_fresh_refire')]
    if imminent:
        lines.append(sep)
        lines.append('  ★★ 급등 임박 후보 — 매집 재발화 (이력 3회+ 후 재출현)')
        lines.append(sep2)
        lines.append(
            f'  {"종목명":<14} {"현재가":>9} {"세력":>4} {"수급":>4}'
            f' {"재발화":>7} {"이력":>4} {"시총":>8}  OB 타이밍'
        )
        lines.append(sep2)
        for r in imminent:
            ai  = r.get('accum_info', {})
            fin = r.get('fin', {})
            dsr = ai.get('days_since_refire', 0)
            hc  = ai.get('hist_count', 0)
            cap = fmt_market_cap(fin.get('market_cap'))
            vr  = r.get('vol_ratio',0) or 0
            df_r = r.get('df')
            ma_bull = bool(df_r.iloc[-1].get('MaBull',False)) if df_r is not None and len(df_r)>0 else False
            t   = predict_surge_timing(r['seoryeok'], vr, ma_bull, r.get('consec_days',0))
            lines.append(
                f'  {r["name"]:<14} {r["price"]:>9,.0f}원'
                f' {r["seoryeok"]:>4}점 {r.get("investor",0):>4}점'
                f' {dsr:>4}일전 {hc:>4}회 {cap:>8}'
                f'  ▶ {t["timing_label"][:25]}'
            )
        lines.append(sep2)
        lines.append('')

    # ── 전체 순위표 (세력점수 기준, 합산점수 제거) ─────────────────────────────
    # 세력점수 기준 정렬
    results_sorted = sorted(results, key=lambda x: (x['seoryeok'], x.get('investor',0)), reverse=True)

    lines.append(sep2)
    lines.append(
        f'  {"순위":<4} {"종목명":<14} {"코드":<8}'
        f' {"현재가":>9} {"세력":>4} {"수급":>4} {"거래량":>6}'
        f' {"vsMA20":>7} {"스테이지":>8} {"ML티어":>8} {"급등예상":>9}'
    )
    lines.append(sep2)

    for rank, r in enumerate(results_sorted, 1):
        inv  = r.get('investor', 0)
        fin  = r.get('fin', {})
        pvm  = r.get('price_vs_ma20')
        ml   = r.get('ml_tier', {})
        vr   = r.get('vol_ratio',0) or 0
        df_r = r.get('df')
        ma_bull = bool(df_r.iloc[-1].get('MaBull',False)) if df_r is not None and len(df_r)>0 else False
        t    = predict_surge_timing(r['seoryeok'], vr, ma_bull, r.get('consec_days',0))
        ai   = r.get('accum_info',{})
        # 수급80+ 종목은 최강 신호(5.03x lift) → 별도 마킹
        if inv >= 80 and r['seoryeok'] >= 70:  rflag = '💎'
        elif ai.get('is_fresh_refire'):          rflag = '★★'
        elif ai.get('is_reactivating'):          rflag = '★ '
        else:                                    rflag = '  '
        tier  = ml.get('tier','?')
        rate  = ml.get('hit_rate',0)
        ma_s  = f'{pvm:+.1f}%' if pvm is not None else '  nan'
        ma_f  = '✅' if pvm and pvm>=0 else ('⚡' if pvm and pvm>=-5 else '❌')
        # 스테이지 태그
        stage_r = r.get('stage')
        if stage_r:
            STAGE_ICONS = {'D-1':'🔥','D-2':'⚡','D-3':'📈','D-4':'👀','D-5':'💤','D+0':'🚀','D+1':'📉'}
            st = stage_r.get('predicted','normal')
            sp = stage_r.get('confidence', 0)
            st_s = f'{STAGE_ICONS.get(st,"")}{"" if st=="normal" else st}({sp:.0%})' if st != 'normal' else '  ─  '
        else:
            st_s = '  ?  '
        lines.append(
            f'  {rflag}{rank:<3} {r["name"]:<14} {r["ticker"]:<8}'
            f' {r["price"]:>9,.0f}원'
            f' {r["seoryeok"]:>4}점 {inv:>4}점 {vr:>5.1f}x'
            f' {ma_s:>6}{ma_f}'
            f' {st_s:>8}'
            f' T{tier}({rate*100:.0f}%)'
            f' {t["expected_days"]:.0f}일({t["confidence_pct"]}%)'
        )

    lines.append(sep2)
    lines.append('')

    # ── 상세 분석 (의사결정 필수 정보만) ───────────────────────────────────────
    lines.append(sep)
    lines.append(f'  [ 상위 {TOP_N_DETAIL}종목 상세 분석 — OB 매매전략 포함 ]')
    lines.append(sep)

    for rank, r in enumerate(results_sorted[:TOP_N_DETAIL], 1):
        se  = r['seoryeok']
        inv = r.get('investor', 0)
        vr  = r.get('vol_ratio',0) or 0
        pvm = r.get('price_vs_ma20')
        ml  = r.get('ml_tier', {})
        df_r = r.get('df')
        ma_bull = bool(df_r.iloc[-1].get('MaBull',False)) if df_r is not None and len(df_r)>0 else False
        timing = predict_surge_timing(se, vr, ma_bull, r.get('consec_days',0))
        ob_params = _calc_ob_trade_params(r.get('ob', {}), r['price'], r)

        # 투자 판단 (세력점수 + ML 티어 기준)
        tier = ml.get('tier','D')
        if se >= 80 and tier in ('S','A'):
            verdict = '★★★ 즉시 진입 검토'
        elif se >= 70 or tier == 'S':
            verdict = '★★  진입 검토'
        elif se >= 60:
            verdict = '★   모니터링'
        else:
            verdict = '    관망'
        # R:R 비우호(목표≤현재가 or R:R<1.0)면 '즉시 진입' 강등 — 매매 수학 우선
        if ob_params.get('rr_unfavorable') and '즉시 진입' in verdict:
            verdict = '★★  진입보류(R:R 불리·저항근접)'

        # 스테이지 태그
        stage_r = r.get('stage')
        STAGE_ICONS = {'D-1':'🔥','D-2':'⚡','D-3':'📈','D-4':'👀','D-5':'💤','D+0':'🚀','D+1':'📉','normal':''}
        if stage_r:
            st = stage_r.get('predicted','normal')
            sp = stage_r.get('confidence', 0)
            pre_p = stage_r.get('pre_surge_prob', 0)
            if st in ('D-1','D-2','D-3'):
                stage_line = (f'  ┌─ 🎯 급등 사이클 스테이지: {STAGE_ICONS.get(st,"")} {st}'
                              f'  (신뢰도 {sp:.0%}  사전급등확률 {pre_p:.0%}) ─────────────')
            elif st in ('D+0','D+1'):
                stage_line = (f'  ┌─ ⚠  스테이지: {STAGE_ICONS.get(st,"")} {st}'
                              f'  (이미 발동/후속 {sp:.0%}) — 신규 진입 주의 ──────────────')
            elif st == 'normal':
                stage_line = None
            else:
                stage_line = (f'  ┌─ 스테이지: {STAGE_ICONS.get(st,"")} {st}'
                              f'  (신뢰도 {sp:.0%}  사전급등확률 {pre_p:.0%}) ─────────────')
        else:
            stage_line = None

        lines.append(f'\n  {rank:>2}위. {r["name"]} ({r["ticker"]})  [{verdict}]')
        lines.append(
            f'       세력:{se:>3}점  수급:{inv:>3}점  거래량:{vr:.1f}x'
            f'  ML:{ml.get("label","?")[:20]}'
        )
        if stage_line:
            lines.append(stage_line)
        ai = r.get('accum_info', {})
        ds = ai.get('days_since')
        accum_str = f'{ds}일 전' if ds is not None else '-'
        if ai.get('is_fresh_refire'):
            refire_str = f'  ★★ 급등임박 재발화! ({ai["days_since_refire"]}일 전 재출현)'
        elif ai.get('is_reactivating'):
            refire_str = f'  ★ 재발화 ({ai["days_since_refire"]}일 전 재출현)'
        else:
            refire_str = ''
        pvm = r.get('price_vs_ma20')
        if pvm is not None:
            ma20_label = f'✅ MA20 위 {pvm:+.1f}%' if pvm >= 0 else \
                         (f'⚡ MA20 근접 {pvm:+.1f}%' if pvm >= -5 else
                          f'❌ MA20 아래 {pvm:+.1f}% (진입 주의)')
        else:
            ma20_label = 'MA20 데이터 없음'
        lines.append(
            f'       현재가: {r["price"]:,.0f}원  |  거래량: {r["vol_ratio"]}x'
            f'  |  {ma20_label}  |  매집경과: {accum_str}{refire_str}'
        )

        # 가격 위치
        ma_s = f'MA20 위 {pvm:+.1f}%' if pvm and pvm>=0 else (f'MA20 {pvm:+.1f}%' if pvm else '')
        rs   = r.get('rs_vs_market')
        rs_s = f'  |  RS(코스피) {rs:+.1f}%{"★" if rs and rs>=5 else ""}' if rs else ''
        lines.append(f'       현재가: {r["price"]:,.0f}원  |  {ma_s}{rs_s}')

        # 급등 타이밍 예측 (핵심)
        lines.append(f'       ▶ 급등타이밍: {timing["timing_label"]}')
        lines.append(f'         예상 {timing["expected_days"]:.0f}거래일 후  |  '
                     f'3일내 {timing["confidence_pct"]}%  |  {timing["holding_advice"]}')

        # OB 매매전략 (항상 표시) — ob_params는 verdict 계산 시 이미 산출됨(재사용)
        if ob_params.get('rr_ratio'):
            rr_icon = '✅ 유리' if ob_params['rr_ratio']>=2 else ('⚠ 보통' if ob_params['rr_ratio']>=1 else '❌ 불리')
            lines.append(f'       ┌─ OB 매매전략 {"─"*45}')
            lines.append(f'       │  진입: {r["price"]:>10,.0f}원  ← {ob_params["entry_desc"][:35]}')
            lines.append(f'       │  목표: {ob_params["target"]:>10,.0f}원  ({ob_params["expected_return"]:+.1f}%)')
            lines.append(f'       │  손절: {ob_params["stop"]:>10,.0f}원  ({(ob_params["stop"]-r["price"])/r["price"]*100:+.1f}%)  |  R:R {ob_params["rr_ratio"]:.1f}:1  {rr_icon}')
            lines.append(f'       └{"─"*57}')

        # 세력 신호 (핵심만)
        s_tags_clean = [t for t in r['s_tags'] if '연속등장' not in t and '보너스' not in t][:2]
        if s_tags_clean:
            lines.append(f'       세력신호: {" | ".join(s_tags_clean)}')

        # 수급 신호
        inv_tags = r.get('inv_tags', [])
        inv_clean = [t for t in inv_tags if not t.startswith('수급데이터') and '★' in t][:2]
        if inv_clean:
            lines.append(f'       수급신호: {" | ".join(inv_clean)}')

        # 공매도
        short = r.get('short', {})
        if short.get('data_ok') and short.get('ratio', 0) >= 1.0:
            sq = ' ★쇼트스퀴즈!' if short.get('squeeze') or short.get('rapid_cover') else ''
            lines.append(f'       공매도: {short["signal"]}{sq}')

        # 트리플 필터
        tf = r.get('triple_filter', {})
        if tf.get('score', 0) >= 2:
            lines.append(f'       트리플필터: {tf["signal"]}')

        # DART 공시 (경고만)
        dart = r.get('dart_info', {})
        if dart.get('has_negative'):
            lines.append(f'       ⚠ DART: {dart["signal"]}')
        elif dart.get('has_positive'):
            lines.append(f'       ✅ DART: {dart["signal"]}')

        # 재무 (시총·PER만)
        fin = r.get('fin', {})
        cap = fmt_market_cap(fin.get('market_cap'))
        per = f'PER {fin["per"]:.0f}' if fin.get('per') else ''
        pbr = f'PBR {fin["pbr"]:.2f}' if fin.get('pbr') else ''
        if cap or per:
            lines.append(f'       재무: 시총 {cap}  {per}  {pbr}')

        # 오더블록 상세
        lines.extend(_format_ob_lines(r['ob'], r['price']))

        # ML 신뢰도 티어 + 진입 타이밍
        lines.append(f'       {sep2}')

    # 앙상블 스코어링 섹션
    ensemble = getattr(run_scan, '_ensemble', None)
    if ensemble:
        lines.extend(build_ensemble_report_section(ensemble))

    # 점수 적중률 추적 섹션
    lines.extend(build_tracker_section())

    lines.append('')
    lines.append(sep)
    lines.append('  ※ 이 리포트는 참고용입니다. 투자 판단은 본인 책임입니다.')
    lines.append(sep)

    return '\n'.join(lines)


def _build_email_body(now_str: str, results: list, elite_list: list,
                      imminent: list, mkt_ctx: dict) -> str:
    sep  = '═' * 60
    sep2 = '─' * 60
    L = []

    # ── 헤더 ────────────────────────────────────────────────────
    L.append(sep)
    L.append(f'  세력 분석 리포트  |  {now_str}')
    L.append(f'  총 후보: {len(results)}종목  |  {mkt_ctx.get("desc", "")}')
    L.append(sep)

    # ── 엘리트 픽 ───────────────────────────────────────────────
    L.append('')
    L.append('★★★★ 엘리트 픽 (백테스트 최고 신뢰 — 즉시 모니터링)')
    L.append(sep2)
    if elite_list:
        for r in elite_list:
            ml   = r.get('ml_tier', {})
            tier = ml.get('tier', '?')
            rate = ml.get('hit_rate', 0)
            cond = ' / '.join(r['elite_reasons'])
            entry = r.get('entry_timing', '')
            rs   = r.get('rs_vs_market')
            rs_str = f'  RS코스피대비 {rs:+.1f}%' if rs is not None else ''
            pvm = r.get('price_vs_ma20')
            if pvm is not None:
                ma20_str = f'  ✅MA20위{pvm:+.1f}%' if pvm >= 0 else \
                           (f'  ⚡MA20근접{pvm:+.1f}%' if pvm >= -5 else
                            f'  ❌MA20아래{pvm:+.1f}%(주의)')
            else:
                ma20_str = ''
            L.append(
                f"  {r['name']} ({r['ticker']})  {r['price']:,.0f}원"
                f"  합산{r['combined']}점 세력{r['seoryeok']}점 급등{r['surge']}점"
                f"  거래량{r.get('vol_ratio',0):.1f}x  연속{r.get('consec_days',0)}일{ma20_str}"
            )
            L.append(f"    조건: {cond}")
            L.append(f"    ML: Tier-{tier} / 예상적중 {rate*100:.1f}%{rs_str}")
            L.append(f"    진입전략: {entry}")
            L.append('')
    else:
        L.append('  해당 없음')
    L.append(sep2)

    # ── 급등 임박 후보 ───────────────────────────────────────────
    L.append('')
    L.append('★★ 급등 임박 후보 (매집 재발화 — 2~4주 내 급등 가능성)')
    L.append(sep2)
    if imminent:
        for r in imminent:
            ai  = r.get('accum_info', {})
            dsr = ai.get('days_since_refire', 0)
            hc  = ai.get('hist_count', 0)
            ml  = r.get('ml_tier', {})
            tier = ml.get('tier', '?')
            rate = ml.get('hit_rate', 0)
            rs   = r.get('rs_vs_market')
            rs_str = f'  RS {rs:+.1f}%' if rs is not None else ''
            entry = r.get('entry_timing', '')
            L.append(
                f"  {r['name']} ({r['ticker']})  {r['price']:,.0f}원"
                f"  합산{r['combined']}점 세력{r['seoryeok']}점"
                f"  재발화{dsr}일전 / 이력{hc}회{rs_str}"
            )
            L.append(f"    ML: Tier-{tier} / 예상적중 {rate*100:.1f}%")
            L.append(f"    진입전략: {entry}")
            L.append('')
    else:
        L.append('  해당 없음')
    L.append(sep2)

    # ── 전체 순위 TOP 10 ─────────────────────────────────────────
    L.append('')
    L.append('전체 순위 TOP 10')
    L.append(sep2)
    for i, r in enumerate(results[:10], 1):
        ml   = r.get('ml_tier', {})
        tier = ml.get('tier', '?')
        rate = ml.get('hit_rate', 0)
        rs   = r.get('rs_vs_market')
        rs_str = f'  RS {rs:+.1f}%' if rs is not None else ''
        L.append(
            f"  {i:>2}. {r['name']} ({r['ticker']})"
            f"  {r['price']:,.0f}원  합산{r['combined']}점"
            f"  세력{r['seoryeok']} 급등{r['surge']} 거래량{r.get('vol_ratio',0):.1f}x"
            f"  Tier-{tier}({rate*100:.0f}%){rs_str}"
        )
    L.append(sep2)

    # ── 수익화 전략 ──────────────────────────────────────────────
    k5 = mkt_ctx.get('kospi_5d', 0)
    q5 = mkt_ctx.get('kosdaq_5d', 0)
    market_bull = k5 > 2.0 and q5 > 2.0
    market_bear = k5 < -3.0 and q5 < -3.0

    L.append('')
    L.append(sep)
    L.append('  ★ 오늘의 수익화 전략')
    L.append(sep)

    # 시장 환경 판단
    if market_bull:
        mkt_stance = f'강세장 (코스피 {k5:+.1f}%) — 공격적 진입 가능, 개별 신호 신뢰도 ↑'
    elif market_bear:
        mkt_stance = f'하락장 (코스피 {k5:+.1f}%) — 보수적 대응, 포지션 축소 권고'
    else:
        mkt_stance = f'중립장 (코스피 {k5:+.1f}%) — 신호 강도 높은 종목 선별적 진입'

    L += [
        f'  시장 환경: {mkt_stance}',
        '',
        '  ┌─ 진입 원칙 (ML 티어별) ─────────────────────────────────────',
        '  │',
        '  │  Tier-S (세력80+)  예상적중 ~18~25%',
        '  │    → 당일 또는 익일 거래량 1.5x+ 확인 후 진입',
        '  │    → 포지션 비중: 2~3%  목표: +10~15%  손절: -5%',
        '  │',
        '  │  Tier-A (세력70+)  예상적중 ~12%',
        '  │    → 거래량 2x+ 또는 SE_Entry 신호 발생 시 진입',
        '  │    → 포지션 비중: 1~2%  목표: +10%  손절: -5%',
        '  │',
        '  │  Tier-B/C          예상적중 ~6~8%',
        '  │    → 추가 신호(SE_Entry + 거래량 2x+) 동시 확인 필수',
        '  │    → 포지션 비중: 0.5~1%  관망 우선',
        '  │',
        '  └──────────────────────────────────────────────────────────────',
        '',
        '  ┌─ 매매 체크리스트 ────────────────────────────────────────────',
        '  │  진입 전 확인사항:',
        '  │  □ 세력 점수 70점 이상인가?',
        '  │  □ 당일 거래량이 30일 평균의 1.5배 이상인가?',
        '  │  □ 코스피 대비 RS 양수(상대 강세)인가?',
        '  │  □ MA 정배열(20>60>120) 상태인가?',
        '  │  □ 연속 등장 2일 이상인가?',
        '  │',
        '  │  리스크 관리:',
        '  │  □ 동시 보유 최대 3종목 (총 노출 6% 이내)',
        '  │  □ 손절선 매수가 -5% 엄수',
        '  │  □ 급등 후 고점 대비 -7% 하락 시 익절 고려',
        '  │  □ 하락장(코스피 -3%↓) 신규 진입 자제',
        '  └──────────────────────────────────────────────────────────────',
        '',
        '  ┌─ 오늘 최우선 관심 종목 ─────────────────────────────────────',
    ]

    # Tier-S 또는 Tier-A 중 최상위 3개 추천
    top_picks = sorted(
        [r for r in results if r.get('ml_tier', {}).get('tier') in ('S', 'A')],
        key=lambda x: (x.get('ml_tier', {}).get('hit_rate', 0), x['combined']),
        reverse=True
    )[:3]

    if not top_picks:
        top_picks = results[:3]

    for i, r in enumerate(top_picks, 1):
        ml    = r.get('ml_tier', {})
        tier  = ml.get('tier', '?')
        rate  = ml.get('hit_rate', 0)
        entry = r.get('entry_timing', '')
        rs    = r.get('rs_vs_market')
        rs_str = f'  RS {rs:+.1f}%' if rs is not None else ''
        L.append(
            f'  │  {i}. {r["name"]} ({r["ticker"]})  {r["price"]:,.0f}원'
            f'  Tier-{tier} 예상{rate*100:.0f}%  세력{r["seoryeok"]}점{rs_str}'
        )
        L.append(f'  │     → {entry}')

    L += [
        '  └──────────────────────────────────────────────────────────────',
        '',
        sep,
        '  ※ 이 분석은 참고용입니다. 투자 판단은 본인 책임입니다.',
        '  ※ 세부 내용은 첨부 리포트를 확인하세요.',
        sep,
    ]

    return '\n'.join(L)


def save_report(report_text: str) -> str:
    fname = datetime.now().strftime('%Y-%m-%d_%Hh%M_report.txt')
    fpath = REPORTS_DIR / fname
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f'\n  리포트 저장: {fpath}')
    return str(fpath)


def mac_notify(title: str, message: str):
    """macOS 알림 팝업"""
    try:
        os.system(
            f"osascript -e 'display notification \"{message}\" with title \"{title}\"'"
        )
    except Exception:
        pass


def _git_auto_push():
    """코드 변경 사항을 GitHub에 자동 커밋 & 푸시."""
    import subprocess
    from pathlib import Path

    repo_dir = Path(__file__).parent

    try:
        # 변경된 .py 파일 확인
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=repo_dir, capture_output=True, text=True, timeout=30
        )
        changed = [l for l in result.stdout.splitlines()
                   if l.strip() and not l.endswith('.json') and not l.endswith('.png')]
        if not changed:
            print('  [GitHub] 변경된 코드 파일 없음 — push 생략')
            return

        # 변경된 .py 파일만 stage
        subprocess.run(['git', 'add', '*.py', '.gitignore'],
                       cwd=repo_dir, capture_output=True, timeout=30)

        date_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        msg = f'auto: daily scan update {date_str}'
        commit = subprocess.run(
            ['git', 'commit', '-m', msg],
            cwd=repo_dir, capture_output=True, text=True, timeout=30
        )
        if commit.returncode != 0 and 'nothing to commit' in commit.stdout:
            print('  [GitHub] 커밋할 변경 없음')
            return

        push = subprocess.run(
            ['git', 'push', 'origin', 'main'],
            cwd=repo_dir, capture_output=True, text=True, timeout=60
        )
        if push.returncode == 0:
            print(f'  [GitHub] ✅ push 완료 → PhilipSon0414/stock_system_good')
        else:
            print(f'  [GitHub] ⚠ push 실패: {push.stderr.strip()[:80]}')
    except Exception as e:
        print(f'  [GitHub] 오류 (스캔 결과에는 영향 없음): {e}')


def main(market: str = 'ALL', scan_date: str | None = None):
    global _SCAN_END_DATE
    _SCAN_END_DATE = scan_date
    results = run_scan(market)

    if not results:
        print('\n  세력 흔적 종목 없음.')
        return

    # 오늘 점수 추적 기록 (+ ② 탈락 유니버스 샘플 = 학습 음성)
    record_scores(results + getattr(run_scan, '_universe_sample', []))

    report_text = build_report(results, market)
    print('\n' + report_text)

    # 파일 저장
    report_path = save_report(report_text)

    # 이메일 발송
    now_str    = datetime.now().strftime('%Y-%m-%d %H:%M')
    consec_map = getattr(run_scan, '_consec_map', {})
    elite_list = _get_elite_picks(results, consec_map)
    imminent   = [r for r in results if r.get('accum_info', {}).get('is_fresh_refire')]
    mkt_ctx    = getattr(run_scan, '_market_ctx', {})
    elite_str  = f' 엘리트픽 {len(elite_list)}종목!' if elite_list else ''
    subject    = f'[세력 분석] {now_str} —{elite_str} 급등 후보 {len(results)}종목'

    body = _build_email_body(now_str, results, elite_list, imminent, mkt_ctx)
    send_report(subject=subject, body=body, attachment_path=report_path)

    # macOS 알림
    if elite_list:
        top = elite_list[0]
        cond = top['elite_reasons'][0]
        mac_notify(
            f'★★★★ 엘리트픽 {len(elite_list)}종목 발견!',
            f'{top["name"]} ({cond}) | 세력{top["seoryeok"]} 급등{top["surge"]}'
        )
    else:
        top1 = results[0]
        mac_notify(
            '세력 분석 완료',
            f'1위: {top1["name"]} {top1["combined"]}점 | 총 {len(results)}종목'
        )

    # ── 스테이지 예측 저장 & 학습 루틴 ──────────────────────────────────────
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from stage_tracker import save_predictions, daily_routine

        preds = []
        for r in results:
            sr = r.get('stage') or {}
            if not isinstance(sr, dict):
                sr = {}
            latest = r['df'].iloc[-1] if r.get('df') is not None else {}
            preds.append({
                'ticker':         r['ticker'],
                'stage':          sr.get('predicted', 'normal'),
                'confidence':     float(sr.get('confidence', 0) or 0),
                'pre_surge_prob': float(sr.get('pre_surge_prob', 0) or 0),
                'price':          float(r.get('price', 0) or 0),
                'vol_ratio':      float(r.get('vol_ratio', 0) or 0),
                'rsi14':          float(latest.get('RSI14', 0) or 0),
                'atr_compress':   float(latest.get('ATRCompress', 1) or 1),
                'rs_vs_market':   float(r.get('rs_vs_market', 0) or 0),
                'price_vs_ma20':  float(r.get('price_vs_ma20', 0) or 0),
                'vol_recovery':   bool(latest.get('VolRecovery', False)),
                'short_ratio':    float((r.get('short') or {}).get('ratio', 0) or 0),
                'sector':         '',
                'sm_score':       0,
            })

        n_saved = save_predictions(preds, market='KR')
        print(f'  [학습] 예측 저장: {n_saved}건 (D-1/D-2/D-3)')

        filt = daily_routine(market='KR', verbose=True)
        if filt and filt.get('n_samples', 0) >= 5:
            hr = filt.get('overall_hit_rate', 0)
            n  = filt.get('n_samples', 0)
            n_pos = len(filt.get('positive_filters', []))
            print(f'  [학습] 필터 갱신 — 적중률: {hr:.0%} ({n}샘플) 긍정조건: {n_pos}개')
    except Exception as e:
        print(f'  [학습] 오류 (무시): {e}')

    print(f'\n  완료. 총 {len(results)}종목 발견.')

    # ── GitHub 자동 동기화 ────────────────────────────────────────────────
    _git_auto_push()


if __name__ == '__main__':
    # 날짜 단독 전달 허용: python3 daily_scan.py 2026-06-11
    if len(sys.argv) == 2 and sys.argv[1][:2] == '20' and '-' in sys.argv[1]:
        market_arg = 'ALL'
        date_arg   = sys.argv[1]
    else:
        market_arg = sys.argv[1] if len(sys.argv) > 1 else 'ALL'
        date_arg   = sys.argv[2] if len(sys.argv) > 2 else None
    main(market_arg, scan_date=date_arg)
