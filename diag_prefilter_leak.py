# -*- coding: utf-8 -*-
"""
Task 0 (교정판) — 거래량 사전필터 누수 + 코일 신호 회수 가능성 정량화.

지시서 원안은 below_gate 레코드를 쓰나, 거래량 프리필터(daily_scan:150)는
스코어링/로깅 '이전'에 작동 → 프리필터 탈락 종목은 below_gate에도 없음.
따라서 실제 급등 이벤트를 재구성해 '전일 거래량<필터' 여부와 코일 신호를 직접 측정.

측정:
  1) 급등(단일일 10%+) 이벤트의 전일 거래량이 PRE_FILTER_MIN_VOLUME 미만 비율
     = 거래량 게이트가 버린 recall
  2) 그 탈락 급등 중 코일 신호(NR7/VCP/OBV_Diverge/BBCompress<0.9/VolRecovery)
     하나라도 보유 비율 = 코일 게이트로 건질 수 있는 비율 (지시서 수용기준 ②)
  3) 대조군(비급등 무작위)의 동일 코일 신호율 → lift(정밀도 맥락, recall-only 함정 방지)

결과: /tmp/diag_prefilter.json
"""
import json
import sys
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import FinanceDataReader as fdr

from data_fetcher import get_ohlcv
from indicators import add_all
from daily_learner import get_surge_tickers
from daily_scan import PRE_FILTER_MIN_VOLUME, PRE_FILTER_MIN_PRICE

START = '2026-05-15'
END   = '2026-06-26'          # 5일 라벨 성숙 위해 최근 며칠 제외
MAX_PER_DAY = 25              # 하루 급등 표본 상한(런타임 관리)


def _trading_days(start, end):
    """KOSPI 지수로 실제 거래일 추출."""
    idx = fdr.DataReader('KS11', start, end)
    return [d.strftime('%Y%m%d') for d in idx.index]


def _coil_flags(df):
    """전일(마지막 봉) 코일 신호 dict. 데이터부족→None."""
    if df is None or len(df) < 30:
        return None
    df = add_all(df)
    last = df.iloc[-1]
    def b(k):
        return bool(last.get(k)) if last.get(k) == last.get(k) else False
    bbc = last.get('BBCompress')
    return {
        'nr7': b('NR7'), 'vcp': b('VCP'), 'obv_diverge': b('OBV_Diverge'),
        'vol_recovery': b('VolRecovery'),
        'bbcompress_lt': (float(bbc) < 0.90) if bbc == bbc and bbc is not None else False,
        'volume': float(last['Volume']),
        'close': float(last['Close']),
    }


def _any_coil(c):
    return c['nr7'] or c['vcp'] or c['obv_diverge'] or c['vol_recovery'] or c['bbcompress_lt']


def main():
    days = _trading_days(START, END)
    print(f'거래일 {len(days)}일 ({days[0]}~{days[-1]})', flush=True)

    surges = []   # (ticker, surge_date)
    seen = set()
    for i, d in enumerate(days):
        try:
            lst = get_surge_tickers(10.0, date_str=d) or []
        except Exception:
            continue
        for s in lst[:MAX_PER_DAY]:
            key = (s['ticker'], d)
            if key not in seen:
                seen.add(key); surges.append((s['ticker'], d))
        if i % 5 == 0:
            print(f'  급등수집 {i}/{len(days)} (누적 {len(surges)})', flush=True)
    print(f'급등 이벤트 {len(surges)}건', flush=True)

    rejected = recov = total = 0
    coil_surge = 0
    for j, (t, d) in enumerate(surges):
        # 전일까지 데이터로 코일/거래량 판정
        prev = (datetime.strptime(d, '%Y%m%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        try:
            df = get_ohlcv(t, end_date=prev)
        except Exception:
            continue
        c = _coil_flags(df)
        if c is None:
            continue
        total += 1
        if c['close'] < PRE_FILTER_MIN_PRICE:
            continue
        is_rejected = c['volume'] < PRE_FILTER_MIN_VOLUME   # 거래량 게이트 탈락
        if is_rejected:
            rejected += 1
            if _any_coil(c):
                recov += 1
        if _any_coil(c):
            coil_surge += 1
        if j % 50 == 0:
            print(f'  판정 {j}/{len(surges)}', flush=True)

    # 대조군: 비급등 무작위(급등셋 제외 랜덤 종목×랜덤일)의 코일율
    import FinanceDataReader as fdr2
    listing = fdr2.StockListing('KRX')
    codes = [str(c).zfill(6) for c in listing['Code'].tolist()
             if not str(c).endswith(('5', '7'))]
    surge_set = {t for t, _ in surges}
    ctrl_codes = [c for c in codes if c not in surge_set]
    # 결정적 샘플(무작위 금지 환경) — 코드 해시로 200개
    ctrl_codes = sorted(ctrl_codes, key=lambda x: x[::-1])[:200]
    ctrl_total = ctrl_coil = 0
    mid = days[len(days) // 2]
    prev_mid = (datetime.strptime(mid, '%Y%m%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    for c in ctrl_codes:
        try:
            df = get_ohlcv(c, end_date=prev_mid)
        except Exception:
            continue
        cf = _coil_flags(df)
        if cf is None or cf['close'] < PRE_FILTER_MIN_PRICE:
            continue
        ctrl_total += 1
        if _any_coil(cf):
            ctrl_coil += 1

    out = {
        'surge_events': total, 'rejected': rejected, 'recoverable': recov,
        'coil_surge': coil_surge,
        'ctrl_total': ctrl_total, 'ctrl_coil': ctrl_coil,
    }
    json.dump(out, open('/tmp/diag_prefilter.json', 'w'))
    print('\n' + '=' * 60)
    print(f'급등 이벤트(판정) {total}건')
    if total:
        print(f'  ① 거래량 게이트 탈락(전일 vol<{PRE_FILTER_MIN_VOLUME:,}): '
              f'{rejected}건 = {rejected/total*100:.1f}% (버린 recall)')
    if rejected:
        print(f'  ② 그중 코일신호 보유(회수가능): {recov}건 = {recov/rejected*100:.1f}% '
              f'← 수용기준(40%+면 Task1 최우선)')
    surge_coil_rate = coil_surge / total if total else 0
    ctrl_coil_rate = ctrl_coil / ctrl_total if ctrl_total else 0
    print(f'  ③ 코일신호율: 급등 {surge_coil_rate*100:.1f}% vs 대조 {ctrl_coil_rate*100:.1f}% '
          f'→ lift {surge_coil_rate/ctrl_coil_rate if ctrl_coil_rate else 0:.2f}x '
          f'(1.3x+ 이어야 정밀도 유효)')
    print('=' * 60)


if __name__ == '__main__':
    main()
