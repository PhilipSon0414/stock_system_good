"""
모델 역추적 백테스트

Part 1: 선정 종목 리드타임 분석
  - 리포트에서 선정된 종목이 실제 급등까지 평균 며칠 걸렸는지
  - 선정 당일 점수와 급등 여부의 관계

Part 2: 급등 전 모델 점수 역추적
  - 역사적 급등 이벤트(20%+)에서 D-1~D-10 시점의 세력/급등 점수
  - 점수 기준별 급등 탐지율 (Recall) 및 정밀도 (Precision)
  - 리드타임: 점수가 기준을 처음 초과한 시점~급등일 간격
"""

import sys, os, re, json, pickle, warnings, threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/Volumes/iMac/Python Programming/Python_Invest/stock_system')

from pykrx import stock as pkrx
from indicators import add_all
from seoryeok import analyze
from order_block import get_order_blocks
from scorer import score as seoryeok_score
from surge_predictor import score_surge
from config import W_SEORYEOK, W_SURGE

REPORTS_DIR  = Path('/Volumes/iMac/Python Programming/Python_Invest/stock_system/reports')
SURGE_THR    = 0.20   # 20%+ 급등 기준
BACKTEST_START = '20250901'   # 역추적 시작 (MA120 워밍업 포함)
BACKTEST_END   = '20260519'
BACKTEST_CACHE = '/tmp/backtest_full_cache.pkl'
LOOKBACK_DAYS  = [1, 3, 5, 10]
SCORE_THRESHOLDS = [30, 40, 50, 60]


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: 리포트 기반 리드타임 분석 (기존 데이터)
# ══════════════════════════════════════════════════════════════════════════════

def parse_report(filepath):
    stocks = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            m = re.match(r'\s+(\d+)\s+([\w\(\)&·가-힣A-Za-z0-9]+)\s+(\d{6})\s+[\d,]+원\s+(\d+)점', line)
            if m:
                stocks.append({
                    'rank': int(m.group(1)),
                    'name': m.group(2).strip(),
                    'code': m.group(3),
                    'score': int(m.group(4))
                })
    return stocks


def load_reports():
    """모든 리포트 파일에서 날짜별 최신 선정 종목 추출"""
    report_files = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith('.txt')])
    latest_per_day = {}
    for fname in report_files:
        m = re.match(r'(\d{4}-\d{2}-\d{2})_', fname)
        if m:
            latest_per_day[m.group(1)] = fname

    selections = {}
    all_tickers = set()
    for day, fname in sorted(latest_per_day.items()):
        stocks = parse_report(REPORTS_DIR / fname)
        selections[day] = stocks
        all_tickers.update(s['code'] for s in stocks)

    return selections, list(all_tickers)


def fetch_ohlcv_safe(ticker, start, end, timeout=20):
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


def find_surge_events(df, surge_thr=SURGE_THR):
    """DataFrame에서 20%+ 급등 이벤트 날짜 리스트 반환"""
    events = []
    if df is None or len(df) < 5:
        return events
    prev_close = df['Close'].shift(1)
    daily_ret   = (df['Close'] - prev_close) / prev_close
    if 'Open' in df.columns:
        otc = (df['Close'] - df['Open']) / df['Open']
    else:
        otc = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        if daily_ret.iloc[i] >= surge_thr or otc.iloc[i] >= surge_thr:
            events.append({
                'date':       df.index[i],
                'daily_ret':  round(daily_ret.iloc[i] * 100, 1),
                'otc_ret':    round(otc.iloc[i] * 100, 1),
            })
    return events


# ─── Part 1 실행 ─────────────────────────────────────────────────────────────
print("=" * 70)
print("  Part 1: 선정 종목 리드타임 분석 (리포트 파일 기반)")
print("=" * 70)

selections, all_tickers = load_reports()
print(f"  리포트: {len(selections)}일  |  고유 선정 종목: {len(all_tickers)}개\n")

# 기존 pick_tracking 캐시 로드
if os.path.exists('/tmp/pick_tracking_cache.pkl'):
    with open('/tmp/pick_tracking_cache.pkl', 'rb') as f:
        price_data = pickle.load(f)
    print(f"  기존 캐시 로드: {len(price_data)}개 종목 (기간 짧음 — 역추적용 별도 다운로드 필요)")
else:
    price_data = {}

# 각 선정 종목의 최초 선정일, 점수, 이후 급등 여부 정리
stock_first_seen = {}  # {code: {name, date, score}}
for day, stocks in sorted(selections.items()):
    for s in stocks:
        if s['code'] not in stock_first_seen:
            stock_first_seen[s['code']] = {'name': s['name'], 'date': day, 'score': s['score']}

# 기존 캐시로 리드타임 분석 (May 7-19 데이터 내에서)
lead_times = []
no_surge   = []
for code, info in stock_first_seen.items():
    df = price_data.get(code)
    if df is None or df.empty:
        continue
    sel_date = pd.Timestamp(info['date'])
    after = df[df.index > sel_date]
    surge_found = False
    for i in range(len(after)):
        if i == 0:
            prev_close = df[df.index <= sel_date]['Close'].iloc[-1] if not df[df.index <= sel_date].empty else after.iloc[0]['Open']
        else:
            prev_close = after.iloc[i-1]['Close']
        row = after.iloc[i]
        daily_ret  = (row['Close'] - prev_close) / prev_close
        otc        = (row['Close'] - row['Open']) / row['Open']
        if daily_ret >= SURGE_THR or otc >= SURGE_THR:
            days_elapsed = i + 1
            lead_times.append({
                'code': code, 'name': info['name'],
                'score': info['score'], 'days': days_elapsed,
                'surge_date': after.index[i].strftime('%Y-%m-%d'),
                'sel_date': info['date'],
            })
            surge_found = True
            break
    if not surge_found:
        no_surge.append({'code': code, 'name': info['name'], 'score': info['score']})

print(f"  선정 종목 중 급등 확인 (캐시 기간 내):")
print(f"    급등 발생: {len(lead_times)}건  |  미발생: {len(no_surge)}건")
if lead_times:
    days_list = [x['days'] for x in lead_times]
    print(f"    평균 리드타임: {np.mean(days_list):.1f}거래일  "
          f"중앙값: {np.median(days_list):.1f}일  "
          f"최단: {min(days_list)}일  최장: {max(days_list)}일")
    print(f"\n  성공 사례 상세:")
    print(f"  {'선정일':12s} {'종목':14s} {'점수':>4s} {'급등일':12s} {'리드타임':>6s}")
    print(f"  {'─'*55}")
    for x in sorted(lead_times, key=lambda x: x['days']):
        print(f"  {x['sel_date']:12s} {x['name']:<14s} {x['score']:4d}점  {x['surge_date']:12s}  {x['days']:4d}거래일")

    # 점수대별 분석
    print(f"\n  점수대별 성과 (캐시 내 모든 선정 종목):")
    all_30plus = [s for day, stocks in selections.items() for s in stocks if s['score'] >= 30]
    surge_codes = {x['code'] for x in lead_times}
    for lo, hi in [(30,39),(40,49),(50,59),(60,101)]:
        grp = [s for s in all_30plus if lo <= s['score'] <= hi]
        hit = [s for s in grp if s['code'] in surge_codes]
        lbl = f"{lo}~{hi}점" if hi < 101 else f"{lo}점+"
        print(f"    {lbl:8s}: 선정 {len(grp):4d}개  급등 {len(hit):3d}개  "
              f"적중률 {len(hit)/len(grp)*100 if grp else 0:5.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: 역추적 백테스트 (Sep 2025 ~ May 2026 데이터)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  Part 2: 모델 역추적 백테스트 (2025-09 ~ 2026-05)")
print("=" * 70)

# ─── 2-1. 역추적용 전체 데이터 다운로드 ─────────────────────────────────────
if os.path.exists(BACKTEST_CACHE):
    with open(BACKTEST_CACHE, 'rb') as f:
        full_data = pickle.load(f)
    print(f"  역추적 캐시 로드: {len(full_data)}개 종목")
else:
    full_data = {}

# 새로 다운로드 필요한 종목
need_download = [t for t in all_tickers if t not in full_data]
print(f"  신규 다운로드 필요: {len(need_download)}개 종목 (약 {len(need_download)//10}~{len(need_download)//5}분 소요)")

new_fetch = 0
for i, ticker in enumerate(need_download):
    df = fetch_ohlcv_safe(ticker, BACKTEST_START, BACKTEST_END)
    if not df.empty and len(df) >= 60:
        full_data[ticker] = df
        new_fetch += 1
    if (i+1) % 50 == 0:
        print(f"    다운로드 진행: {i+1}/{len(need_download)}  (성공: {new_fetch}개)", flush=True)

if new_fetch > 0:
    with open(BACKTEST_CACHE, 'wb') as f:
        pickle.dump(full_data, f)
    print(f"  신규 다운로드: {new_fetch}개 저장 → {BACKTEST_CACHE}")

print(f"  역추적 데이터: {len(full_data)}개 종목\n")


# ─── 2-2. 급등 이벤트 탐지 ────────────────────────────────────────────────────
print("  급등 이벤트 탐지 중...")
all_surge_events = []  # [{ticker, name, surge_date, daily_ret, df}, ...]

for ticker, df in full_data.items():
    name = stock_first_seen.get(ticker, {}).get('name', ticker)
    events = find_surge_events(df)
    for ev in events:
        # 분석 기간: 2026-01-01 이후만 포함
        if ev['date'] < pd.Timestamp('2026-01-01'):
            continue
        all_surge_events.append({
            'ticker':    ticker,
            'name':      name,
            'surge_date': ev['date'],
            'daily_ret': ev['daily_ret'],
            'df':        df,
        })

print(f"  탐지된 급등 이벤트: {len(all_surge_events)}건 "
      f"(종목: {len({e['ticker'] for e in all_surge_events})}개)")


# ─── 2-3. 점수 계산 함수 ──────────────────────────────────────────────────────
def score_at(df, ref_date):
    """ref_date 기준 (그 날 제외) seoryeok + surge 점수 계산"""
    try:
        sl = df[df.index < ref_date].copy()
        if len(sl) < 65:
            return None
        sl = add_all(sl)
        sl = analyze(sl)
        ob = get_order_blocks(sl)
        s_pts,  _ = seoryeok_score(sl, ob)
        sg_pts, _ = score_surge(sl, ob)
        # 조정 합산: 세력(25%) + 급등(30%) = 55% → 100점 스케일로 환산
        adj = round((s_pts * W_SEORYEOK + sg_pts * W_SURGE) / (W_SEORYEOK + W_SURGE))
        return {'seoryeok': s_pts, 'surge': sg_pts, 'adj_combo': adj}
    except Exception:
        return None


# ─── 2-4. 급등 이벤트별 D-1~D-10 점수 계산 ──────────────────────────────────
print(f"\n  D-1~D-10 점수 계산 중 (최대 {len(all_surge_events)}개 이벤트)...")
print(f"  {'─'*65}")

surge_scores  = defaultdict(list)   # {lookback_day: [score_dict, ...]}
event_details = []

for idx, ev in enumerate(all_surge_events):
    df          = ev['df']
    surge_date  = ev['surge_date']
    dates_before = df.index[df.index < surge_date]
    if len(dates_before) < 65:
        continue

    row_result = {'ticker': ev['ticker'], 'name': ev['name'],
                  'surge_date': surge_date.strftime('%Y-%m-%d'),
                  'daily_ret': ev['daily_ret']}

    for n in LOOKBACK_DAYS:
        if len(dates_before) < n:
            row_result[f'D-{n}_adj'] = None
            continue
        ref_date = dates_before[-n]   # 급등일 기준 N거래일 전
        sc = score_at(df, ref_date)
        if sc:
            row_result[f'D-{n}_seoryeok'] = sc['seoryeok']
            row_result[f'D-{n}_surge']    = sc['surge']
            row_result[f'D-{n}_adj']      = sc['adj_combo']
            surge_scores[n].append(sc)
        else:
            row_result[f'D-{n}_adj'] = None

    event_details.append(row_result)
    if (idx+1) % 20 == 0 or idx+1 == len(all_surge_events):
        print(f"    진행: {idx+1}/{len(all_surge_events)}  완료", flush=True)


# ─── 2-5. 통제군 (Control) 점수 계산 ─────────────────────────────────────────
print("\n  통제군 점수 계산 중...")

control_scores = []
surge_dates_by_ticker = defaultdict(set)
for ev in all_surge_events:
    surge_dates_by_ticker[ev['ticker']].add(ev['surge_date'].strftime('%Y-%m-%d'))

rng = np.random.default_rng(42)
for ticker, df in list(full_data.items())[:80]:   # 80개 종목 샘플
    valid_dates = df.index[
        (df.index >= pd.Timestamp('2026-01-01')) &
        (df.index < pd.Timestamp('2026-05-01'))
    ]
    if len(valid_dates) < 10:
        continue
    # 급등일 아닌 날 10개씩 샘플
    non_surge = [d for d in valid_dates if d.strftime('%Y-%m-%d') not in surge_dates_by_ticker[ticker]]
    sample_dates = rng.choice(non_surge, min(10, len(non_surge)), replace=False) if non_surge else []
    for sd in sample_dates:
        sc = score_at(df, pd.Timestamp(sd))
        if sc:
            control_scores.append(sc)

print(f"  통제군 점수: {len(control_scores)}건")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: 결과 출력
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  결과: 급등 전 모델 점수 분포 (조정 합산 기준)")
print("  ※ 조정 합산 = 세력(25%) + 급등임박(30%) 성분만, 100점 스케일 환산")
print("=" * 70)

print(f"\n  {'시점':8s} {'N건':>5s} {'평균':>6s} {'중앙값':>6s} {'40+비율':>8s} {'50+비율':>8s} {'60+비율':>8s}")
print(f"  {'─'*55}")
for n in LOOKBACK_DAYS:
    scores = [s['adj_combo'] for s in surge_scores[n] if s]
    if not scores:
        print(f"  D-{n:>2s}      0건  데이터 없음")
        continue
    arr = np.array(scores)
    print(f"  D-{n:<2d}   {len(arr):5d}건  {np.mean(arr):5.1f}점  {np.median(arr):5.1f}점"
          f"  {(arr>=40).mean()*100:7.1f}%  {(arr>=50).mean()*100:7.1f}%  {(arr>=60).mean()*100:7.1f}%")

# 통제군 비교
if control_scores:
    c_arr = np.array([s['adj_combo'] for s in control_scores])
    print(f"  {'─'*55}")
    print(f"  통제군   {len(c_arr):5d}건  {np.mean(c_arr):5.1f}점  {np.median(c_arr):5.1f}점"
          f"  {(c_arr>=40).mean()*100:7.1f}%  {(c_arr>=50).mean()*100:7.1f}%  {(c_arr>=60).mean()*100:7.1f}%")

# 세력 / 급등 개별 점수
print("\n\n  세력 흔적 점수 vs 급등 임박 점수 (D-1 기준):")
print(f"  {'':12s} {'세력 평균':>8s} {'급등 평균':>8s} {'세력 50+':>9s} {'급등 50+':>9s}")
print(f"  {'─'*55}")
if 1 in surge_scores and surge_scores[1]:
    se_d1 = [s['seoryeok'] for s in surge_scores[1]]
    sg_d1 = [s['surge']    for s in surge_scores[1]]
    print(f"  급등 전 D-1   {np.mean(se_d1):7.1f}점  {np.mean(sg_d1):7.1f}점  "
          f"{(np.array(se_d1)>=50).mean()*100:8.1f}%  {(np.array(sg_d1)>=50).mean()*100:8.1f}%")
if control_scores:
    c_se = [s['seoryeok'] for s in control_scores]
    c_sg = [s['surge']    for s in control_scores]
    print(f"  통제군        {np.mean(c_se):7.1f}점  {np.mean(c_sg):7.1f}점  "
          f"{(np.array(c_se)>=50).mean()*100:8.1f}%  {(np.array(c_sg)>=50).mean()*100:8.1f}%")


# ─── 점수 기준별 탐지율 (Recall) ─────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  탐지율 분석: 점수 기준별 급등 포착률 (D-1 기준)")
print("=" * 70)
print(f"\n  점수 기준  탐지 건수  전체 급등  탐지율(Recall)  통제군 비율  Lift")
print(f"  {'─'*65}")
d1_scores = surge_scores.get(1, [])
for thr in SCORE_THRESHOLDS:
    detected = [s for s in d1_scores if s and s['adj_combo'] >= thr]
    total    = len(d1_scores)
    recall   = len(detected) / total if total else 0
    c_ratio  = (np.array([s['adj_combo'] for s in control_scores]) >= thr).mean() if control_scores else 0
    lift     = recall / c_ratio if c_ratio > 0 else 0
    print(f"  조정 {thr:>2d}+점   {len(detected):6d}건  {total:6d}건   {recall*100:8.1f}%"
          f"      {c_ratio*100:7.1f}%   {lift:5.2f}x")


# ─── 리드타임 분포 (언제 처음 점수 초과?) ─────────────────────────────────────
print("\n\n" + "=" * 70)
print("  리드타임 분석: 급등 전 처음으로 점수 기준을 초과한 시점")
print("  (각 급등 이벤트에 대해 D-10~D-1 중 처음으로 기준 초과한 날)")
print("=" * 70)

LEAD_THR = 40  # 리드타임 기준 점수 (조정합산)

print(f"\n  기준 점수: 조정합산 {LEAD_THR}점+\n")
print(f"  {'시점':8s} {'건수':>6s} {'비율':>7s}  (누적)")
print(f"  {'─'*40}")

# 각 이벤트별로 D-10~D-1 스코어 계산하여 처음 초과한 날 찾기
lead_time_counts = defaultdict(int)
total_events_with_data = 0

for ev in all_surge_events[:200]:   # 너무 많으면 상위 200개만
    df        = ev['df']
    surge_date = ev['surge_date']
    dates_before = df.index[df.index < surge_date]
    if len(dates_before) < 65:
        continue
    total_events_with_data += 1
    first_cross = None
    for n in range(10, 0, -1):   # D-10 → D-1 순으로 확인
        if len(dates_before) < n:
            continue
        ref_date = dates_before[-n]
        sc = score_at(df, ref_date)
        if sc and sc['adj_combo'] >= LEAD_THR:
            first_cross = n
    if first_cross:
        lead_time_counts[first_cross] += 1
    else:
        lead_time_counts[0] += 1   # 기준 미달

cumulative = 0
for n in sorted(lead_time_counts.keys(), reverse=True):
    if n == 0:
        continue
    cnt = lead_time_counts[n]
    cumulative += cnt
    pct = cnt / total_events_with_data * 100 if total_events_with_data else 0
    cum_pct = cumulative / total_events_with_data * 100 if total_events_with_data else 0
    print(f"  D-{n:<2d}     {cnt:5d}건  {pct:6.1f}%  (누적 {cum_pct:.1f}%)")

missed = lead_time_counts.get(0, 0)
print(f"  미탐지   {missed:5d}건  {missed/total_events_with_data*100 if total_events_with_data else 0:6.1f}%")
if total_events_with_data:
    detected_total = sum(v for k, v in lead_time_counts.items() if k > 0)
    avg_lead = sum(k * v for k, v in lead_time_counts.items() if k > 0) / max(detected_total, 1)
    print(f"\n  전체 탐지율: {detected_total}/{total_events_with_data} "
          f"({detected_total/total_events_with_data*100:.1f}%)  "
          f"탐지된 경우 평균 리드타임: {avg_lead:.1f}거래일")


# ─── 상위 급등 이벤트 상세 ─────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  상위 급등 이벤트 D-1 점수 상세 (세력+급등 60+ 이벤트)")
print("=" * 70)
print(f"  {'종목':14s} {'급등일':12s} {'급등율':>7s}  {'D-1 세력':>8s}  {'D-1 급등':>8s}  {'D-1 조정':>8s}")
print(f"  {'─'*65}")

high_score_events = [
    ev for ev in event_details
    if ev.get('D-1_adj') is not None and ev.get('D-1_adj', 0) >= 50
]
high_score_events.sort(key=lambda x: x.get('D-1_adj', 0), reverse=True)

for ev in high_score_events[:30]:
    print(f"  {ev['name']:<14s} {ev['surge_date']:12s} {ev['daily_ret']:+6.1f}%"
          f"  {ev.get('D-1_seoryeok', '-'):>8}점  {ev.get('D-1_surge', '-'):>8}점"
          f"  {ev.get('D-1_adj', '-'):>8}점")


# ─── 결과 저장 ────────────────────────────────────────────────────────────────
results_summary = {
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'total_surge_events': len(all_surge_events),
    'events_analyzed': total_events_with_data,
    'control_samples': len(control_scores),
    'D1_avg_adj': round(float(np.mean([s['adj_combo'] for s in surge_scores.get(1, []) if s])), 1) if surge_scores.get(1) else None,
    'control_avg_adj': round(float(np.mean([s['adj_combo'] for s in control_scores])), 1) if control_scores else None,
    'lead_time_avg': avg_lead if total_events_with_data else None,
}
with open('model_backtest_results.json', 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

print("\n\n  결과 저장: model_backtest_results.json")
print("  완료.")
