"""급등 종목별 세력 흔적 최초 발생 시점 분석"""
import sys, json
sys.path.insert(0, '.')
from pathlib import Path
from datetime import datetime
import pandas as pd

from data_fetcher import get_ohlcv
from indicators import add_all
from seoryeok import analyze

log  = json.loads(Path('learning_log.json').read_text(encoding='utf-8'))
obs  = log['observations']

# 종목당 최고 급등률 기록만 사용
best = {}
for o in obs:
    t = o['ticker']
    if t not in best or o['surge_pct'] > best[t]['surge_pct']:
        best[t] = o
stocks = sorted(best.values(), key=lambda x: x['surge_pct'], reverse=True)

SE_COLS  = ['SE_Entry', 'SE_Accum', 'SE_BearTrap', 'SE_Rally']
SE_LABEL = {'SE_Entry':'진입', 'SE_Accum':'매집', 'SE_BearTrap':'베어트랩', 'SE_Rally':'상승시작'}

results  = []
print(f'분석 중: {len(stocks)}종목\n', flush=True)

for i, o in enumerate(stocks, 1):
    ticker   = o['ticker']
    name     = o['name']
    surge_dt = datetime.strptime(o['date'], '%Y-%m-%d').date()

    df_full = get_ohlcv(ticker, period_days=400)
    if df_full.empty or len(df_full) < 62:
        continue

    mask = [d.date() < surge_dt for d in df_full.index]
    df   = df_full.loc[mask].copy()
    if len(df) < 62:
        continue

    df = add_all(df)
    df = analyze(df)

    signal_info = {}
    for col in SE_COLS:
        if col not in df.columns:
            continue
        sig_dates = df.index[df[col] == True]
        if len(sig_dates) == 0:
            continue
        first = sig_dates[0].date()
        last  = sig_dates[-1].date()
        signal_info[col] = {
            'first':       first,
            'last':        last,
            'count':       len(sig_dates),
            'days_before': (surge_dt - last).days,
            'first_days':  (surge_dt - first).days,
        }

    results.append({
        'ticker':    ticker,
        'name':      name,
        'surge_dt':  surge_dt,
        'surge_pct': o['surge_pct'],
        'signals':   signal_info,
    })
    sigs_found = [SE_LABEL[c] for c in SE_COLS if c in signal_info]
    print(f'  [{i:2d}/{len(stocks)}] {name}({ticker})  신호: {sigs_found}', flush=True)

# ── 결과 테이블 ───────────────────────────────────────────────────────────────
SEP  = '=' * 80
SEP2 = '-' * 80
print('\n' + SEP)
print('  급등 종목 세력 흔적 선행 분석 (급등일 기준 N일 전 마지막 신호)')
print(SEP)
hdr = '{:<13} {:<7} {:>6}  {:>6} {:>6} {:>8} {:>8}  {}'.format(
    '종목명', '코드', '급등률', '진입', '매집', '베어트랩', '상승시작', '최초신호(N일전)')
print('  ' + hdr)
print(SEP2)

all_first_days = []
no_signal_list = []

for r in results:
    sigs = r['signals']

    def fmt(col):
        if col not in sigs:
            return '  -   '
        d = sigs[col]['days_before']
        return f'{d:>3}일전'

    first_dates = [sigs[c]['first'] for c in SE_COLS if c in sigs]
    if first_dates:
        earliest      = min(first_dates)
        days_earliest = (r['surge_dt'] - earliest).days
        all_first_days.append(days_earliest)
        earliest_str  = f'{earliest} ({days_earliest}일전)'
    else:
        earliest_str = '없음'
        no_signal_list.append(r['name'])

    flag = '★' if sigs else '  '
    row = '{}{:<12} {:<7} +{:>4.0f}%  {:>6} {:>6} {:>8} {:>8}  {}'.format(
        flag,
        r['name'], r['ticker'], r['surge_pct'],
        fmt('SE_Entry'), fmt('SE_Accum'), fmt('SE_BearTrap'), fmt('SE_Rally'),
        earliest_str,
    )
    print('  ' + row)

print(SEP)

# ── 통계 요약 ─────────────────────────────────────────────────────────────────
has_sig = [r for r in results if r['signals']]
print(f'\n세력 신호 있음: {len(has_sig)}/{len(results)}종목 ({len(has_sig)/len(results)*100:.0f}%)')
if no_signal_list:
    print(f'세력 신호 없음: {", ".join(no_signal_list)}')

if all_first_days:
    med = sorted(all_first_days)[len(all_first_days)//2]
    print(f'\n최초 세력 신호 → 급등까지 선행 기간:')
    print(f'  평균   : {sum(all_first_days)/len(all_first_days):.0f}일')
    print(f'  중앙값 : {med}일')
    print(f'  최단   : {min(all_first_days)}일  /  최장: {max(all_first_days)}일')

    print('\n기간 분포:')
    buckets = [(0,7,'1주 이내'), (7,14,'1~2주'), (14,30,'2~4주'),
               (30,60,'1~2개월'), (60,120,'2~4개월'), (120,999,'4개월+')]
    for lo, hi, label in buckets:
        cnt = sum(1 for d in all_first_days if lo <= d < hi)
        if cnt:
            bar = '█' * cnt
            print(f'  {label:<10}: {cnt:3}건  {bar}')

# ── SE_Accum 매집 지속 기간 ───────────────────────────────────────────────────
print('\n' + SEP)
print('  SE_Accum(매집 구간) 지속 기간 상세')
print(SEP2)
accum_spans = []
for r in sorted(results, key=lambda x: x['surge_pct'], reverse=True):
    if 'SE_Accum' not in r['signals']:
        continue
    s    = r['signals']['SE_Accum']
    span = (s['last'] - s['first']).days
    accum_spans.append(span)
    print(f'  {r["name"]:<13} {s["first"]} ~ {s["last"]} '
          f'({span:>3}일간 {s["count"]:>2}회)  급등 {s["days_before"]}일 전 마지막 신호')

if accum_spans:
    print(SEP2)
    print(f'  매집 평균 지속: {sum(accum_spans)/len(accum_spans):.0f}일  '
          f'중앙값: {sorted(accum_spans)[len(accum_spans)//2]}일  '
          f'최장: {max(accum_spans)}일')
print(SEP)
