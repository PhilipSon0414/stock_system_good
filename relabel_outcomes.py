# -*- coding: utf-8 -*-
"""
전체 이력 record-pull 재라벨링 (일회성 복구 도구).

forward_tracker.update_outcomes()가 이제 동일한 pull 방식으로 미확정(None)
라벨을 채우므로, 이 스크립트는 과거 push 방식이 남긴 **잘못 확정된 라벨**
(실행 공백기의 False 오라벨 등)까지 전수 재검증·교정할 때 사용한다.
급등 판정 로직은 forward_tracker.label_from_ohlcv 단일 정의를 공유한다.

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
from forward_tracker import label_from_ohlcv as _relabel, HORIZONS

SH = Path(__file__).parent / 'score_history.json'


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
