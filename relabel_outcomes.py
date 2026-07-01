# -*- coding: utf-8 -*-
"""
Task 6 — record-pull 재라벨링.

문제: forward_tracker.update_outcomes()는 '그날 급등 리스트' 의존 → 실행 공백(예:
6/23→6/30 7일 갭)에 급등한 종목이 리스트에 없으면 미포착·잘못된 False 라벨.

해결: 각 레코드의 스캔일 이후 N거래일 OHLCV를 직접 조회해 단일일 10%+(종가/전일종가)
여부를 판정(이벤트-푸시 → 레코드-풀). 급등 정의는 _surge_via_fdr와 동일.

사용:
  python3 relabel_outcomes.py          # dry-run(변경 규모만 보고)
  python3 relabel_outcomes.py --apply  # 백업 후 실제 반영
"""
import json
import sys
import shutil
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from data_fetcher import get_ohlcv

SH = Path(__file__).parent / 'score_history.json'
SURGE_PCT = 0.10
HORIZONS = {'surged_by_3d': 3, 'surged_by_5d': 5,
            'surged_by_10d': 10, 'surged_by_20d': 20}


def _relabel(df, scan_date, n):
    """scan_date 이후 n거래일 내 단일일 10%+ → True / 만료 & 없음 → False / 미성숙 → None."""
    idx = [i for i, d in enumerate(df.index) if str(d.date()) <= scan_date]
    if not idx:
        return None
    pos = idx[-1]
    fut = df.iloc[pos + 1: pos + 1 + n]
    if len(fut) < n:
        return None   # 윈도우 미성숙
    prev = float(df['Close'].iloc[pos])
    for j in range(len(fut)):
        c = float(fut['Close'].iloc[j])
        if prev > 0 and c / prev - 1 >= SURGE_PCT:
            return True
        prev = c
    return False


def main():
    apply = '--apply' in sys.argv
    data = json.load(open(SH, encoding='utf-8'))
    recs = data['records']
    print(f'레코드 {len(recs)}건, 재라벨링 {"적용" if apply else "DRY-RUN"}', flush=True)

    cache = {}
    stat = {h: {'filled': 0, 'recovered': 0, 'corrected': 0, 'same': 0}
            for h in HORIZONS}
    for i, r in enumerate(recs):
        t, d = r.get('ticker'), r.get('scan_date')
        if not t or not d:
            continue
        if t not in cache:
            try:
                cache[t] = get_ohlcv(t, period_days=400)
            except Exception:
                cache[t] = None
        df = cache[t]
        if df is None or len(df) < 5:
            continue
        for h, n in HORIZONS.items():
            new = _relabel(df, d, n)
            old = r.get(h)
            if new is None:
                continue
            if old is None:
                stat[h]['filled'] += 1
            elif old == new:
                stat[h]['same'] += 1
            elif not old and new:
                stat[h]['recovered'] += 1     # False→True (놓친 급등 복원)
            elif old and not new:
                stat[h]['corrected'] += 1      # True→False (드묾)
            if apply:
                r[h] = new
        if i % 200 == 0:
            print(f'  {i}/{len(recs)}', flush=True)

    print('\n' + '=' * 64)
    print(f'  {"호라이즌":<16}{"채움(None→)":>12}{"복원(F→T)":>12}{"교정(T→F)":>12}{"동일":>8}')
    for h in HORIZONS:
        s = stat[h]
        print(f'  {h:<16}{s["filled"]:>12}{s["recovered"]:>12}{s["corrected"]:>12}{s["same"]:>8}')
    print('=' * 64)

    if apply:
        shutil.copy(SH, SH.with_suffix('.json.bak_relabel'))
        json.dump(data, open(SH, 'w', encoding='utf-8'), ensure_ascii=False)
        print('  ✅ 적용 완료 (백업: score_history.json.bak_relabel)')
    else:
        print('  (DRY-RUN — 실제 반영하려면 --apply)')


if __name__ == '__main__':
    main()
