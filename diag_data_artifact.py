# -*- coding: utf-8 -*-
"""
Task 0 (2차) — 수정주가 미적용 데이터 아티팩트 오염 측정.

get_ohlcv는 fdr 원주가(무수정) → 액면분할·권리락이 인접일 |종가변동|>30%로 나타남.
KRX 일간 상하한 ±30% → 30% 초과 = 물리적 불가 = 아티팩트.
_relabel은 단일일 10%+를 급등으로 보므로, 아티팩트(>30%)가 가짜 양성을 만들 수 있음.

측정:
  ① 인접일 |변동|>30% 아티팩트 발생 건/종목
  ② surged_by_5d==True 인데 그 급등이 '오직 >30% 아티팩트로만' 성립한 가짜 양성
  ③ 해당 종목 리스트
결과: /tmp/diag_artifact.json
"""
import json
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from data_fetcher import get_ohlcv

PRICE_LIMIT = 0.30
SURGE_PCT = 0.10
SH = Path(__file__).parent / 'score_history.json'

recs = [r for r in json.load(open(SH, encoding='utf-8'))['records']
        if r.get('scan_date') and r.get('ticker')]
print(f'레코드 {len(recs)}건', flush=True)

cache = {}
artifact_events = 0
artifact_tickers = set()
fake_pos = []          # surged_by_5d=True인데 아티팩트로만 성립
true_pos_checked = 0

for i, r in enumerate(recs):
    t, d = r['ticker'], r['scan_date']
    if t not in cache:
        try:
            cache[t] = get_ohlcv(t, period_days=400)
        except Exception:
            cache[t] = None
    df = cache[t]
    if df is None or len(df) < 5:
        continue
    # 스캔일 이후 5거래일 윈도우
    idx = [k for k, dt in enumerate(df.index) if str(dt.date()) <= d]
    if not idx:
        continue
    pos = idx[-1]
    fut = df.iloc[pos + 1: pos + 1 + 5]
    if len(fut) == 0:
        continue
    prev = float(df['Close'].iloc[pos])
    real_surge = artifact_surge = False
    for j in range(len(fut)):
        c = float(fut['Close'].iloc[j])
        if prev <= 0:
            prev = c; continue
        chg = c / prev - 1
        if abs(chg) > PRICE_LIMIT:
            artifact_events += 1; artifact_tickers.add(t)
            if chg >= SURGE_PCT:
                artifact_surge = True       # 아티팩트가 급등조건 충족
        elif chg >= SURGE_PCT:
            real_surge = True               # 정상 10~30% 급등
        prev = c
    if r.get('surged_by_5d'):
        true_pos_checked += 1
        # 가짜 양성: 급등이 아티팩트로만 성립(정상 급등일 없음)
        if artifact_surge and not real_surge:
            fake_pos.append({'ticker': t, 'name': r.get('name', t), 'scan_date': d})
    if i % 300 == 0:
        print(f'  {i}/{len(recs)}', flush=True)

out = {'artifact_events': artifact_events,
       'artifact_tickers': sorted(artifact_tickers),
       'true_pos_checked': true_pos_checked,
       'fake_positives': fake_pos}
json.dump(out, open('/tmp/diag_artifact.json', 'w'), ensure_ascii=False)

print('\n' + '=' * 60)
print(f'① 인접일 |변동|>30% 아티팩트: {artifact_events}건 / {len(artifact_tickers)}종목')
print(f'② surged_by_5d=True 점검 {true_pos_checked}건 중 '
      f'아티팩트로만 성립한 가짜 양성: {len(fake_pos)}건')
if fake_pos:
    print('③ 가짜 양성 종목:')
    for f in fake_pos[:20]:
        print(f'   {f["name"]}({f["ticker"]}) @ {f["scan_date"]}')
else:
    print('③ 가짜 양성 없음 — 라벨 오염 0, Task A는 예방 가드만')
print('=' * 60)
