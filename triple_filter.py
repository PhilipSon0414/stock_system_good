"""
트리플 필터: 세력 개입 고정밀 포착

세 조건의 교집합만 통과:
  ① 공매도 잔고 감소 추세 (숏 세력이 청산 중)
  ② 기관 순매수 전환   (기관이 매도→매수로 바뀌는 시점)
  ③ 거래량 20일 평균 대비 200%+ (실제 자금 유입 확인)

백테스트 근거:
  단독 조건 적중률 ~5~8% → 세 조건 교집합: 예상 15~25%
  특히 ②+③ 동시 = 기관 저점 매집 패턴의 핵심 신호
"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd


def check_triple(ticker: str, df: pd.DataFrame) -> dict:
    """
    세 조건 체크 후 결과 반환.

    Returns:
        passed        : 세 조건 모두 통과 여부
        short_ok      : 공매도 잔고 감소 여부
        inst_ok       : 기관 순매수 전환 여부
        vol_ok        : 거래량 200%+ 여부
        score         : 통과 조건 수 (0~3)
        signal        : 리포트용 태그
        inst_days     : 기관 순매수 연속 일수
        short_change  : 공매도 잔고 5일 변화율 (%)
        vol_ratio     : 현재 거래량 / 20일 평균
    """
    result = {
        'passed': False, 'short_ok': False, 'inst_ok': False, 'vol_ok': False,
        'score': 0, 'signal': '', 'inst_days': 0, 'short_change': None, 'vol_ratio': 0.0,
    }

    # ── 조건 ③: 거래량 200%+ ─────────────────────────────────────────────
    if 'VolRatio' in df.columns:
        vol_ratio = float(df['VolRatio'].iloc[-1]) if not pd.isna(df['VolRatio'].iloc[-1]) else 0.0
        result['vol_ratio'] = round(vol_ratio, 2)
        if vol_ratio >= 2.0:
            result['vol_ok'] = True
            result['score'] += 1

    # ── 조건 ① : 공매도 잔고 감소 ────────────────────────────────────────
    try:
        from short_interest import get_short_interest
        si = get_short_interest(ticker)
        if si.get('data_ok'):
            trend = si.get('trend', 'stable')
            change = si.get('change_rate_5d')
            result['short_change'] = change
            if trend == 'decreasing':
                result['short_ok'] = True
                result['score'] += 1
    except Exception:
        pass

    # ── 조건 ② : 기관 순매수 전환 ────────────────────────────────────────
    try:
        inst_days, inst_transition = _check_institutional_flow(ticker)
        result['inst_days'] = inst_days
        if inst_days >= 1 and inst_transition:
            result['inst_ok'] = True
            result['score'] += 1
        elif inst_days >= 3:   # 전환 없어도 3일+ 연속 매수면 통과
            result['inst_ok'] = True
            result['score'] += 1
    except Exception:
        pass

    result['passed'] = result['score'] == 3

    # 신호 태그 생성
    tags = []
    if result['short_ok']:
        sc = f'{result["short_change"]:+.1f}%' if result['short_change'] else ''
        tags.append(f'공매도잔고감소{sc}')
    if result['inst_ok']:
        tags.append(f'기관순매수{result["inst_days"]}일')
    if result['vol_ok']:
        tags.append(f'거래량{result["vol_ratio"]:.1f}x')

    if result['passed']:
        result['signal'] = f'★★ 트리플필터 통과: {" + ".join(tags)}'
    elif result['score'] == 2:
        result['signal'] = f'▲ 더블필터({result["score"]}/3): {" + ".join(tags)}'
    elif result['score'] == 1:
        result['signal'] = f'단일조건({" + ".join(tags)})'

    return result


def _check_institutional_flow(ticker: str) -> tuple[int, bool]:
    """
    pykrx로 최근 20일 기관 순매수 데이터 조회.
    Returns: (연속_순매수_일수, 매도→매수_전환_여부)
    """
    from pykrx import stock
    end   = datetime.now().strftime('%Y%m%d')
    start = (datetime.now() - timedelta(days=35)).strftime('%Y%m%d')

    df = stock.get_market_trading_volume_by_date(start, end, ticker)
    if df is None or df.empty or '기관합계' not in df.columns:
        return 0, False

    df = df[df.index.to_series().diff().dt.days.fillna(1) > 0].tail(20)
    if len(df) < 3:
        return 0, False

    inst = df['기관합계'].values

    # 연속 순매수 일수 (최근부터 역산)
    consec = 0
    for v in reversed(inst):
        if v > 0:
            consec += 1
        else:
            break

    # 전환 감지: 이전 3일 중 매도 → 최근 2일 매수
    prev_window = inst[-5:-2] if len(inst) >= 5 else inst[:-2]
    recent      = inst[-2:]
    transition  = (
        len(prev_window) > 0 and
        any(v < 0 for v in prev_window) and   # 이전에 매도 있었음
        all(v > 0 for v in recent)              # 최근 2일 매수 전환
    )

    return consec, transition


def scan_triple_filter(results: list) -> list:
    """
    daily_scan 결과 리스트에서 트리플 필터 조건 체크 후 score 추가.
    results: analyze_one() 결과 딕셔너리 리스트
    """
    triple_hits  = []
    double_hits  = []

    for r in results:
        ticker = r['ticker']
        df     = r.get('df')
        if df is None:
            continue

        tf = check_triple(ticker, df)
        r['triple_filter'] = tf

        if tf['passed']:
            triple_hits.append(r)
        elif tf['score'] == 2:
            double_hits.append(r)

    return triple_hits, double_hits
