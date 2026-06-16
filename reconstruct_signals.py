# -*- coding: utf-8 -*-
"""
라벨 유니버스(score_history) 재구성 — 각 (종목, 스캔일) 시점 기준으로
near_bull_ob(OB 근접)와 DART 공시 신호를 과거 시점대로 재계산해
급등(surged_by_5d) 대비 lift를 검증한다. (선행 피처 편입 판단용)

결과: /tmp/recon_signals.json
"""
import json
import sys
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import get_ohlcv
from indicators import add_all
from order_block import get_order_blocks
from dart_utils import get_dart_disclosures, classify_dart_impact

recs = [r for r in json.load(open('score_history.json'))['records']
        if r.get('surged_by_5d') is not None]
print(f'대상 라벨 레코드: {len(recs)}건', flush=True)

out = []
for i, r in enumerate(recs):
    t, d = r['ticker'], r['scan_date']
    rec = {'y': 1 if r.get('surged_by_5d') else 0,
           'se': r.get('seoryeok', 0), 'inv': r.get('investor', 0),
           'vr': r.get('vol_ratio', 0)}
    # OB 재구성
    try:
        df = get_ohlcv(t, end_date=d)
        if df is not None and len(df) >= 60:
            df = add_all(df)
            ob = get_order_blocks(df)
            rec['near_bull'] = bool(ob.get('near_bull'))
            rec['near_bear'] = bool(ob.get('near_bear'))
    except Exception:
        pass
    # DART 재구성 (스캔일 기준 최근 3일 공시)
    try:
        info = get_dart_disclosures(t, d, lookback_days=3)
        rec['has_pos'] = bool(info.get('has_positive'))
        rec['has_neg'] = bool(info.get('has_negative'))
        rec['surge_type'] = classify_dart_impact(info).get('surge_type', 'unknown')
    except Exception:
        pass
    out.append(rec)
    if i % 50 == 0:
        print(f'  진행 {i}/{len(recs)}', flush=True)

json.dump(out, open('/tmp/recon_signals.json', 'w'))
print(f'완료: {len(out)}건 저장', flush=True)
