# -*- coding: utf-8 -*-
"""
score_history 백필 — 각 라벨 레코드에 ret5/ret20(낙폭 피처)을 과거시점 재구성해 추가.
surge_model GBM이 학습에 쓰도록(검증 +13.9%p AUC). 1회성. 백업 생성 후 in-place.
"""
import json
import shutil
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from data_fetcher import get_ohlcv

SH = Path(__file__).parent / 'score_history.json'
data = json.load(open(SH, encoding='utf-8'))
recs = data['records']
shutil.copy(SH, SH.with_suffix('.json.bak_retfill'))
print(f'전체 레코드 {len(recs)}건, 백업 생성', flush=True)

cache = {}   # (ticker, date) 중복 fetch 방지
done = 0
for i, r in enumerate(recs):
    if 'ret5' in r and 'ret20' in r:
        done += 1
        continue
    t, d = r.get('ticker'), r.get('scan_date')
    if not t or not d:
        continue
    key = (t, d)
    if key not in cache:
        try:
            df = get_ohlcv(t, end_date=d)
            if df is None or len(df) < 6:
                cache[key] = (0.0, 0.0)
            else:
                close = float(df['Close'].iloc[-1])
                c6 = float(df['Close'].iloc[-6]) if len(df) >= 6 else close
                c21 = float(df['Close'].iloc[-21]) if len(df) >= 21 else close
                cache[key] = (round(close / c6 - 1, 4) if c6 else 0.0,
                              round(close / c21 - 1, 4) if c21 else 0.0)
        except Exception:
            cache[key] = (0.0, 0.0)
    r['ret5'], r['ret20'] = cache[key]
    done += 1
    if i % 200 == 0:
        print(f'  {i}/{len(recs)} (완료 {done})', flush=True)

json.dump(data, open(SH, 'w', encoding='utf-8'), ensure_ascii=False)
print(f'백필 완료: {done}건에 ret5/ret20 기록', flush=True)
