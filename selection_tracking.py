"""
선정 종목 급등 소요 기간 분석

각 날짜의 리포트에서 선정된 종목(40점+)이
이후 몇 거래일 만에 20%+ 급등했는지 추적.
"""

import re, json, sys, os, warnings
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/Volumes/iMac/Python Programming/Python_Invest/stock_system')

from pykrx import stock as pkrx

REPORTS_DIR = '/Volumes/iMac/Python Programming/Python_Invest/stock_system/reports'
SCORE_THRESHOLD = 30   # 이 점수 이상 선정된 종목 추적 (30/40/50 비교)
SURGE_THRESHOLD = 0.20  # 20%+ 단일일 급등 기준


# ─── 1. 리포트 파싱 ────────────────────────────────────────────────────────────
def parse_report(filepath):
    """리포트 파일에서 (코드, 종목명, 점수) 추출"""
    stocks = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            # "  1    포스코DX          022100      35,250원    44점 ..." 형식
            m = re.match(r'\s+(\d+)\s+([\w\(\)&·가-힣A-Za-z0-9]+)\s+(\d{6})\s+[\d,]+원\s+(\d+)점', line)
            if m:
                rank  = int(m.group(1))
                name  = m.group(2).strip()
                code  = m.group(3)
                score = int(m.group(4))
                stocks.append({'rank': rank, 'name': name, 'code': code, 'score': score})
    return stocks

# 날짜별 최신 리포트 선택
report_files = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith('.txt')])
latest_per_day = {}
for fname in report_files:
    m = re.match(r'(\d{4}-\d{2}-\d{2})_', fname)
    if m:
        day = m.group(1)
        latest_per_day[day] = fname  # 정렬된 순서라 마지막이 최신

print(f"분석 대상 리포트: {len(latest_per_day)}일")
for d, f in sorted(latest_per_day.items()):
    print(f"  {d}: {f}")

# 날짜별 선정 종목 추출
selections = {}  # {date: [{code, name, score, rank}, ...]}
for day, fname in sorted(latest_per_day.items()):
    fpath = os.path.join(REPORTS_DIR, fname)
    stocks = parse_report(fpath)
    filtered = [s for s in stocks if s['score'] >= SCORE_THRESHOLD]
    selections[day] = filtered
    print(f"  {day}: {len(filtered)}개 선정 (30점+) / 전체 {len(stocks)}개")


# ─── 2. pykrx 데이터 수집 ──────────────────────────────────────────────────────
def fetch_ohlcv(ticker, start, end, timeout=20):
    import threading
    result = [pd.DataFrame()]
    def _run():
        try:
            df = pkrx.get_market_ohlcv(start, end, ticker)
            if df is None or df.empty:
                return
            rename = {'시가':'Open','고가':'High','저가':'Low','종가':'Close','거래량':'Volume'}
            df = df.rename(columns=rename)
            df.index = pd.to_datetime(df.index)
            result[0] = df[['Open','High','Low','Close','Volume']].copy()
        except Exception:
            pass
    t = threading.Thread(target=_run, daemon=True)
    t.start(); t.join(timeout=timeout)
    return result[0]

TODAY = datetime.now().strftime('%Y%m%d')
START = '20260507'

# 분석 대상 종목 수집 (중복 제거, 최초 선정일 기록)
all_picks = {}  # {code: {name, first_date, first_score, all_dates}}
for day, stocks in sorted(selections.items()):
    for s in stocks:
        code = s['code']
        if code not in all_picks:
            all_picks[code] = {
                'name':       s['name'],
                'first_date': day,
                'first_score': s['score'],
                'all_dates':  [],
            }
        all_picks[code]['all_dates'].append((day, s['score']))

print(f"\n전체 고유 선정 종목: {len(all_picks)}개")
print("가격 데이터 수집 중...", flush=True)

# 캐시
PICK_CACHE = '/tmp/pick_tracking_cache.pkl'
import pickle
if os.path.exists(PICK_CACHE):
    with open(PICK_CACHE, 'rb') as f:
        price_data = pickle.load(f)
    print(f"  캐시 로드: {len(price_data)}개")
else:
    price_data = {}

new_fetch = 0
for i, (code, info) in enumerate(all_picks.items()):
    if code in price_data:
        continue
    df = fetch_ohlcv(code, START, TODAY)
    if not df.empty:
        price_data[code] = df
        new_fetch += 1
        if new_fetch % 20 == 0:
            print(f"  {i+1}/{len(all_picks)} 수집 중...", flush=True)

if new_fetch > 0:
    with open(PICK_CACHE, 'wb') as f:
        pickle.dump(price_data, f)
    print(f"  신규 수집: {new_fetch}개 저장")


# ─── 3. 급등 소요 기간 계산 ────────────────────────────────────────────────────
def days_to_surge(df, from_date_str, surge_thr=SURGE_THRESHOLD, max_days=30):
    """from_date 이후 첫 20%+ 급등 발생까지 거래일 수 반환 (없으면 None)"""
    from_dt = datetime.strptime(from_date_str, '%Y-%m-%d')
    after = df[df.index > from_dt]
    if after.empty:
        return None, None
    for i, (dt, row) in enumerate(after.iterrows()):
        chg = (row['Close'] - row['Open']) / row['Open']
        # 시가 대비 종가 급등 OR 전일 종가 대비 당일 종가 급등
        if i > 0:
            prev_close = after.iloc[i-1]['Close']
        else:
            before = df[df.index <= from_dt]
            prev_close = before.iloc[-1]['Close'] if not before.empty else row['Open']
        daily_chg = (row['Close'] - prev_close) / prev_close
        if daily_chg >= surge_thr or chg >= surge_thr:
            return i + 1, dt.strftime('%Y-%m-%d')  # i+1 = 거래일 수
        if i + 1 >= max_days:
            break
    return None, None  # max_days 내 급등 없음


# ─── 4. 날짜별 분석 ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("날짜별 선정 종목 급등 소요 기간 분석")
print("="*70)

all_surge_days = []       # 전체 급등 소요 거래일
date_results   = {}

for score_thr_label, score_thr in [('30점+', 30), ('40점+', 40)]:
    print(f"\n{'─'*70}")
    print(f"  기준: {score_thr_label} 선정 종목")
    print(f"{'─'*70}")
    print(f"  {'날짜':12s} {'선정':>5s} {'급등':>5s} {'평균일':>7s} {'최단':>5s} {'최장':>5s}  급등 종목")
    print(f"  {'─'*65}")

    all_days_thr = []
    date_detail  = {}

    for day, stocks in sorted(selections.items()):
        filtered = [s for s in stocks if s['score'] >= score_thr]
        surged = []
        not_surged = []

        for s in filtered:
            code = s['code']
            df = price_data.get(code)
            if df is None or df.empty:
                continue
            n_days, surge_date = days_to_surge(df, day)
            if n_days is not None:
                surged.append({**s, 'days': n_days, 'surge_date': surge_date})
                all_days_thr.append(n_days)
            else:
                not_surged.append(s)

        n_sel  = len(filtered)
        n_surg = len(surged)
        avg_d  = np.mean([x['days'] for x in surged]) if surged else None
        min_d  = min([x['days'] for x in surged]) if surged else None
        max_d  = max([x['days'] for x in surged]) if surged else None

        surged_names = ', '.join(f"{x['name']}({x['days']}일)" for x in sorted(surged, key=lambda x: x['days'])[:4])

        date_detail[day] = {
            'n_selected':   n_sel,
            'n_surged':     n_surg,
            'avg_days':     round(avg_d, 1) if avg_d else None,
            'min_days':     min_d,
            'max_days':     max_d,
            'surged_stocks': surged,
            'not_surged':   not_surged,
        }

        avg_str = f"{avg_d:.1f}일" if avg_d else "  — "
        min_str = f"{min_d}일"     if min_d else "—"
        max_str = f"{max_d}일"     if max_d else "—"
        ratio   = f"{n_surg}/{n_sel}"

        print(f"  {day}  {ratio:>5s}  {n_surg:4d}개  {avg_str:>6s}  {min_str:>4s}  {max_str:>4s}  {surged_names}")

    # 전체 요약
    if all_days_thr:
        print(f"\n  {'─'*65}")
        print(f"  전체 평균: {np.mean(all_days_thr):.1f}거래일  "
              f"중앙값: {np.median(all_days_thr):.1f}일  "
              f"최단: {min(all_days_thr)}일  "
              f"최장: {max(all_days_thr)}일")

        # 분포
        buckets = [(1,1),(2,3),(4,5),(6,7),(8,10),(11,15),(16,30)]
        print(f"\n  소요 기간 분포:")
        for lo, hi in buckets:
            cnt = sum(1 for d in all_days_thr if lo <= d <= hi)
            bar = '█' * cnt
            lbl = f"당일" if lo==1 and hi==1 else f"{lo}~{hi}일"
            print(f"    {lbl:8s}: {cnt:3d}건  {bar}")
    else:
        print(f"  급등 발생 없음")

# ─── 5. 전체 통합 요약 ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("전체 통합 요약 (30점+ 기준)")
print("="*70)

all_selected_30 = []
all_surged_30   = []
for day, stocks in sorted(selections.items()):
    for s in [s for s in stocks if s['score'] >= 30]:
        code = s['code']
        df   = price_data.get(code)
        if df is None: continue
        n, sd = days_to_surge(df, day)
        all_selected_30.append({'date': day, **s, 'days_to_surge': n, 'surge_date': sd})
        if n:
            all_surged_30.append(n)

total_sel   = len(all_selected_30)
total_surg  = len([x for x in all_selected_30 if x['days_to_surge']])
hit_rate    = total_surg / total_sel if total_sel > 0 else 0

print(f"\n  선정 종목 총계  : {total_sel}개")
print(f"  30일내 급등 발생: {total_surg}개 ({hit_rate*100:.1f}%)")
if all_surged_30:
    print(f"  평균 소요 기간  : {np.mean(all_surged_30):.1f}거래일")
    print(f"  중앙값          : {np.median(all_surged_30):.1f}거래일")
    print(f"  최단            : {min(all_surged_30)}거래일")
    print(f"  최장            : {max(all_surged_30)}거래일")

# 점수대별 급등률 + 평균 소요일
print(f"\n  점수대별 성과:")
print(f"  {'점수대':10s} {'선정':>6s} {'급등':>6s} {'급등률':>7s} {'평균일':>8s}")
print(f"  {'─'*45}")
for lo, hi in [(30,39),(40,49),(50,59),(60,101)]:
    grp = [x for x in all_selected_30 if lo <= x['score'] <= hi]
    surg= [x for x in grp if x['days_to_surge']]
    avg = np.mean([x['days_to_surge'] for x in surg]) if surg else None
    lbl = f"{lo}~{hi}점" if hi < 101 else f"{lo}점+"
    avg_str = f"{avg:.1f}일" if avg else "—"
    print(f"  {lbl:10s} {len(grp):5d}개  {len(surg):5d}개  "
          f"{len(surg)/len(grp)*100 if grp else 0:6.1f}%  {avg_str:>7s}")

# 결과 저장
result_out = {
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'total_selected': total_sel,
    'total_surged_30d': total_surg,
    'hit_rate': round(hit_rate, 3),
    'avg_days_to_surge': round(float(np.mean(all_surged_30)), 1) if all_surged_30 else None,
    'median_days': round(float(np.median(all_surged_30)), 1) if all_surged_30 else None,
}
with open('selection_tracking_results.json', 'w', encoding='utf-8') as f:
    json.dump(result_out, f, ensure_ascii=False, indent=2)

print("\n결과 저장: selection_tracking_results.json")
print("완료.")
