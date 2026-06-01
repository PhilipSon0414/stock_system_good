"""
거래량 패턴 심층 분석

분석 항목:
  1. 급등 전 D-10~D0 거래량 타임라인 — 언제 거래량이 살아나나?
  2. VOL_ENTRY_THRESHOLD 최적화 (현재 3.0x)
  3. VOL_QUIET_THRESHOLD 최적화 (현재 0.7)
  4. 거래량 패턴 유형 분류 (4가지)
  5. 압축→폭발 비율 (세력 잠복 후 폭발 패턴)
  6. 연속 저거래량 일수와 급등 규모 상관관계
  7. D-1 거래량 분포 분석
"""

import json, pickle, sys, warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/Volumes/iMac/Python Programming/Python_Invest/stock_system')

from pykrx import stock as pkrx

# ─── 데이터 로드 ────────────────────────────────────────────────────────────────
with open('learning_log.json') as f:
    log = json.load(f)
observations = log['observations']
print(f"학습 관측치: {len(observations)}개", flush=True)

CACHE_FILE = '/tmp/tf_analysis_cache.pkl'
with open(CACHE_FILE, 'rb') as f:
    price_cache = pickle.load(f)
print(f"가격 캐시: {len(price_cache)}개 종목", flush=True)

VOL_AVG_PERIOD = 30  # 업데이트된 설정

def add_volume(df, period=VOL_AVG_PERIOD):
    df = df.copy()
    df['VolMA'] = df['Volume'].rolling(period).mean()
    df['VolRatio'] = df['Volume'] / df['VolMA'].replace(0, np.nan)
    df['VolRatio'] = df['VolRatio'].replace([np.inf, -np.inf], np.nan)
    return df

def get_window(df, surge_date_str, days_before, days_after=1):
    surge_dt = datetime.strptime(surge_date_str, '%Y-%m-%d')
    before = df[df.index < surge_dt]
    if len(before) < days_before:
        return None, None
    window = before.iloc[-days_before:]
    surge_rows = df[df.index >= surge_dt].iloc[:days_after]
    return window, surge_rows


def _fetch_ohlcv(ticker, start, end):
    try:
        df = pkrx.get_market_ohlcv(start, end, ticker)
        if df is None or df.empty:
            return pd.DataFrame()
        rename = {'시가': 'Open', '고가': 'High', '저가': 'Low', '종가': 'Close', '거래량': 'Volume'}
        df = df.rename(columns=rename)
        df.index = pd.to_datetime(df.index)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    except Exception:
        return pd.DataFrame()

def _fetch_safe(ticker, start, end, timeout=25):
    import threading
    result = [pd.DataFrame()]
    def _run(): result[0] = _fetch_ohlcv(ticker, start, end)
    t = threading.Thread(target=_run, daemon=True)
    t.start(); t.join(timeout=timeout)
    return result[0]


# ─── 컨트롤 그룹 준비 ───────────────────────────────────────────────────────────
CTL_CACHE = '/tmp/vol_control_cache.pkl'
import os
if os.path.exists(CTL_CACHE):
    with open(CTL_CACHE, 'rb') as f:
        control_data = pickle.load(f)
    print(f"컨트롤 캐시 로드: {len(control_data)}개", flush=True)
else:
    print("컨트롤 그룹 다운로드 중...", flush=True)
    try:
        kospi  = list(pkrx.get_market_ticker_list('20260401', market='KOSPI'))[:50]
        kosdaq = list(pkrx.get_market_ticker_list('20260401', market='KOSDAQ'))[:50]
        ctl_tickers = [t for t in (kospi + kosdaq) if t not in {o['ticker'] for o in observations}][:60]
    except Exception:
        ctl_tickers = []

    control_data = {}
    for i, ticker in enumerate(ctl_tickers):
        print(f"  {i+1}/{len(ctl_tickers)}: {ticker}", flush=True)
        df = _fetch_safe(ticker, '20250101', '20260501')
        if df.empty or 'Volume' not in df.columns:
            continue
        df = add_volume(df)
        if len(df) >= 80:
            control_data[ticker] = df
    with open(CTL_CACHE, 'wb') as f:
        pickle.dump(control_data, f)
    print(f"컨트롤 수집 완료: {len(control_data)}개", flush=True)


# ─── 급등 종목 거래량 데이터 준비 ────────────────────────────────────────────────
surge_data = []
for obs in observations:
    key = obs['ticker']
    if key not in price_cache:
        continue
    cache = price_cache[key]
    df = add_volume(cache['df'])
    surge_dt = datetime.strptime(cache['surge_date'], '%Y-%m-%d')
    df_before = df[df.index < surge_dt]
    df_surge  = df[df.index >= surge_dt].iloc[:1]
    if len(df_before) < 15 or df_surge.empty:
        continue
    surge_data.append({
        'ticker':     key,
        'name':       obs['name'],
        'surge_date': cache['surge_date'],
        'surge_pct':  obs.get('surge_pct', 0),
        'df_before':  df_before,
        'df_surge':   df_surge,
    })

N = len(surge_data)
print(f"\n분석 가능 관측치: {N}개\n", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# [1] D-10 ~ D0 거래량 타임라인
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("[1] 급등 전 거래량 타임라인 (D-10 ~ 급등당일 D0)")
print("=" * 62, flush=True)

OFFSETS = list(range(10, 0, -1)) + [0]  # D-10, D-9, ... D-1, D-0
timeline = {d: [] for d in OFFSETS}

for rec in surge_data:
    df_b = rec['df_before']
    df_s = rec['df_surge']
    for d in range(1, 11):
        if len(df_b) >= d:
            vr = df_b.iloc[-d].get('VolRatio', np.nan)
            if not pd.isna(vr):
                timeline[d].append(vr)
    # D-0 (급등 당일)
    if not df_s.empty:
        vr0 = df_s.iloc[0].get('VolRatio', np.nan)
        if not pd.isna(vr0):
            timeline[0].append(vr0)

print(f"\n  {'일':6s} {'평균':>7s} {'중앙':>7s} {'25%':>7s} {'75%':>7s} {'3x+비율':>8s}")
print("  " + "-" * 50)
for d in sorted(OFFSETS, reverse=True):
    vals = [v for v in timeline[d] if not np.isnan(v)]
    if not vals:
        continue
    arr = np.array(vals)
    label = f"D-{d}" if d > 0 else "D0(급등)"
    above3 = (arr >= 3.0).mean()
    print(f"  {label:6s}  {arr.mean():6.2f}x  {np.median(arr):6.2f}x  "
          f"{np.percentile(arr,25):6.2f}x  {np.percentile(arr,75):6.2f}x  {above3*100:6.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# [2] VOL_ENTRY_THRESHOLD 최적화
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("[2] 세력 진입 임계값 최적화 — D-1 기준")
print("=" * 62, flush=True)

ENTRY_THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]

surge_entry = {}
for thr in ENTRY_THRESHOLDS:
    hit = sum(1 for rec in surge_data
              if not pd.isna(rec['df_before'].iloc[-1].get('VolRatio', np.nan))
              and rec['df_before'].iloc[-1]['VolRatio'] >= thr)
    surge_entry[thr] = hit / N if N > 0 else 0

# D-5 이내 언제라도 기준
surge_entry5 = {}
for thr in ENTRY_THRESHOLDS:
    hit5 = 0
    for rec in surge_data:
        w5 = rec['df_before'].iloc[-5:]
        vr = w5['VolRatio'].dropna()
        if (vr >= thr).any():
            hit5 += 1
    surge_entry5[thr] = hit5 / N if N > 0 else 0

# 컨트롤 기저율 (D-1)
ctl_entry = {thr: [] for thr in ENTRY_THRESHOLDS}
for df_c in control_data.values():
    if len(df_c) < 10:
        continue
    vr = df_c.iloc[-1].get('VolRatio', np.nan)
    for thr in ENTRY_THRESHOLDS:
        if not pd.isna(vr):
            ctl_entry[thr].append(1 if vr >= thr else 0)

print(f"\n  {'임계값':>8s} {'D-1 감지':>9s} {'D-5내':>8s} {'기저율':>8s} {'D-1 Lift':>9s}")
print("  " + "-" * 50)
best_entry_thr = 3.0
best_entry_lift = 0
for thr in ENTRY_THRESHOLDS:
    base_vals = ctl_entry[thr]
    base_rate = np.mean(base_vals) if base_vals else 0
    s_rate = surge_entry[thr]
    lift = (s_rate / base_rate) if base_rate > 0 else 0
    label = ' ← 현재' if thr == 3.0 else (' ★' if lift == max(
        (surge_entry[t] / np.mean(ctl_entry[t]) if np.mean(ctl_entry[t]) > 0 else 0)
        for t in ENTRY_THRESHOLDS) else '')
    print(f"  {thr:>7.1f}x  {s_rate*100:8.1f}%  {surge_entry5[thr]*100:7.1f}%  "
          f"{base_rate*100:7.1f}%  {lift:8.2f}x{label}")
    if lift > best_entry_lift:
        best_entry_lift = lift
        best_entry_thr = thr


# ═══════════════════════════════════════════════════════════════════════════════
# [3] VOL_QUIET_THRESHOLD 최적화
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("[3] 세력 매집 기준 (VOL_QUIET) 최적화")
print("=" * 62, flush=True)

QUIET_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# D-5 이내 최소 VolRatio (가장 조용한 날)
quiet_res = {}
for qt in QUIET_THRESHOLDS:
    # 급등 전 D-3~D-10 사이에 한 번이라도 해당 임계 이하인 비율
    hit = 0
    for rec in surge_data:
        w = rec['df_before'].iloc[-10:-1]  # D-10 ~ D-2
        vr = w['VolRatio'].dropna()
        if (vr < qt).any():
            hit += 1
    quiet_res[qt] = hit / N if N > 0 else 0

# 컨트롤
ctl_quiet = {qt: [] for qt in QUIET_THRESHOLDS}
for df_c in control_data.values():
    if len(df_c) < 12:
        continue
    w = df_c.iloc[-10:-1]['VolRatio'].dropna()
    for qt in QUIET_THRESHOLDS:
        ctl_quiet[qt].append(1 if (w < qt).any() else 0)

print(f"\n  {'임계값':>8s} {'급등전 출현':>11s} {'기저율':>8s} {'Lift':>7s}")
print("  " + "-" * 42)
best_quiet_thr = 0.7
best_quiet_lift = 0
for qt in QUIET_THRESHOLDS:
    base_rate = np.mean(ctl_quiet[qt]) if ctl_quiet[qt] else 0
    s_rate = quiet_res[qt]
    lift = s_rate / base_rate if base_rate > 0 else 0
    label = ' ← 현재' if qt == 0.70 else ''
    print(f"  {qt:.2f}      {s_rate*100:10.1f}%  {base_rate*100:7.1f}%  {lift:6.2f}x{label}")
    if lift > best_quiet_lift:
        best_quiet_lift = lift
        best_quiet_thr = qt


# ═══════════════════════════════════════════════════════════════════════════════
# [4] 거래량 패턴 유형 분류
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("[4] 급등 전 거래량 패턴 유형 분류")
print("=" * 62, flush=True)

"""
패턴 정의 (D-10 ~ D-1 기준):
  A: 잠복형  — D-10~D-2 내내 조용(< 0.7), D-1에 폭발(>= 2x)
  B: 축적형  — D-10~D-2 중 저거래량 구간 있음, D-1 보통(0.5~2x)
  C: 점화형  — D-5~D-2 기간 거래량 꾸준히 증가 (slope > 0)
  D: 기습형  — D-1 또는 D-2에 갑자기 급등 (나머지는 평범)
  E: 기타
"""

pattern_counts = {'A_잠복폭발': 0, 'B_축적대기': 0, 'C_점진증가': 0, 'D_기습형': 0, 'E_기타': 0}
pattern_surge_pct = {'A_잠복폭발': [], 'B_축적대기': [], 'C_점진증가': [], 'D_기습형': [], 'E_기타': []}

for rec in surge_data:
    df_b = rec['df_before']
    if len(df_b) < 10:
        continue
    vr = df_b['VolRatio'].iloc[-10:].values
    vr_d1 = vr[-1] if not np.isnan(vr[-1]) else 1.0
    vr_d2 = vr[-2] if len(vr) >= 2 and not np.isnan(vr[-2]) else 1.0
    vr_early = [v for v in vr[:-2] if not np.isnan(v)]  # D-10 ~ D-3

    if not vr_early:
        ptype = 'E_기타'
    else:
        early_min = min(vr_early)
        early_mean = np.mean(vr_early)
        # slope of last 5 days before D-1
        last5 = [v for v in vr[-6:-1] if not np.isnan(v)]
        slope = np.polyfit(range(len(last5)), last5, 1)[0] if len(last5) >= 3 else 0

        is_quiet_early = early_min < 0.70  # 이전에 조용한 구간 있음
        is_burst_d1 = vr_d1 >= 2.5
        is_burst_d2 = vr_d2 >= 2.5
        is_increasing = slope > 0.05  # 꾸준히 증가

        if is_quiet_early and is_burst_d1:
            ptype = 'A_잠복폭발'
        elif is_burst_d1 and not is_quiet_early:
            ptype = 'D_기습형'
        elif is_burst_d2 and not is_quiet_early:
            ptype = 'D_기습형'
        elif is_increasing and early_mean < 1.0:
            ptype = 'C_점진증가'
        elif is_quiet_early:
            ptype = 'B_축적대기'
        else:
            ptype = 'E_기타'

    pattern_counts[ptype] += 1
    pattern_surge_pct[ptype].append(rec['surge_pct'])

print(f"\n  {'패턴':15s} {'건수':>6s} {'비율':>7s} {'평균급등':>9s} {'설명'}")
print("  " + "-" * 62)
descriptions = {
    'A_잠복폭발': '조용히 매집 후 D-1 폭발 — 세력 전형 패턴',
    'B_축적대기': '저거래량 구간 있으나 아직 폭발 전',
    'C_점진증가': '거래량 서서히 증가 — 추세 추종 유입',
    'D_기습형':   'D-1~D-2 갑작스러운 폭발 (이전 평범)',
    'E_기타':     '패턴 불명확',
}
for ptype, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1]):
    pct = cnt / N
    avg_surge = np.mean(pattern_surge_pct[ptype]) if pattern_surge_pct[ptype] else 0
    print(f"  {ptype:15s} {cnt:5d}개  {pct*100:6.1f}%  {avg_surge:8.1f}%  {descriptions.get(ptype,'')}")


# ═══════════════════════════════════════════════════════════════════════════════
# [5] 압축→폭발 비율 분석 (Compression Ratio)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("[5] 거래량 압축→폭발 비율 분석")
print("=" * 62, flush=True)
print("  압축비 = 급등당일 거래량 / D-10~D-2 중 최저 거래량")

compression_ratios = []
for rec in surge_data:
    df_b = rec['df_before']
    df_s = rec['df_surge']
    if len(df_b) < 10 or df_s.empty:
        continue
    min_quiet = df_b['Volume'].iloc[-10:-1].min()
    surge_vol = df_s.iloc[0]['Volume']
    if min_quiet > 0:
        ratio = surge_vol / min_quiet
        compression_ratios.append(ratio)

if compression_ratios:
    arr = np.array(compression_ratios)
    buckets = [(0, 3), (3, 5), (5, 10), (10, 20), (20, 50), (50, 999)]
    print(f"\n  {'압축비':12s} {'건수':>6s} {'비율':>7s}")
    print("  " + "-" * 30)
    for lo, hi in buckets:
        cnt = ((arr >= lo) & (arr < hi)).sum()
        label = f"{lo}x~{hi}x" if hi < 999 else f"{lo}x 이상"
        bar = '█' * int(cnt / len(arr) * 25)
        print(f"  {label:12s} {cnt:5d}개  {cnt/len(arr)*100:6.1f}%  {bar}")
    print(f"\n  평균 압축비: {arr.mean():.1f}x  중앙값: {np.median(arr):.1f}x  "
          f"75퍼: {np.percentile(arr,75):.1f}x")


# ═══════════════════════════════════════════════════════════════════════════════
# [6] 연속 저거래량 일수와 급등 규모 상관관계
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("[6] 연속 저거래량 일수 vs 급등 규모")
print("=" * 62, flush=True)
print("  (D-2 기준으로 뒤에서부터 세어 VolRatio < 0.70 연속일 수)")

quiet_vs_surge = {}
for rec in surge_data:
    df_b = rec['df_before']
    if len(df_b) < 5:
        continue
    vr_vals = df_b['VolRatio'].iloc[:-1].dropna().values  # D-2 이전까지
    quiet_days = 0
    for v in reversed(vr_vals):
        if v < 0.70:
            quiet_days += 1
        else:
            break
    bucket = min(quiet_days // 2, 6) * 2  # 0~1, 2~3, 4~5, 6~7, 8~9, 10~11, 12+
    if bucket not in quiet_vs_surge:
        quiet_vs_surge[bucket] = []
    quiet_vs_surge[bucket].append(rec['surge_pct'])

print(f"\n  {'연속 저거래량':14s} {'건수':>6s} {'평균 급등':>10s} {'최대 급등':>10s}")
print("  " + "-" * 46)
for bucket in sorted(quiet_vs_surge.keys()):
    vals = quiet_vs_surge[bucket]
    label = f"{bucket}~{bucket+1}일" if bucket < 12 else "12일+"
    avg_s = np.mean(vals)
    max_s = max(vals)
    print(f"  {label:14s} {len(vals):5d}개  {avg_s:9.1f}%  {max_s:9.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# [7] D-1 거래량 분포 — 급등을 예측할 수 있나?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("[7] D-1 거래량 비율 분포 분석")
print("=" * 62, flush=True)

d1_vols = []
for rec in surge_data:
    vr = rec['df_before'].iloc[-1].get('VolRatio', np.nan)
    if not pd.isna(vr):
        d1_vols.append(vr)

arr_d1 = np.array(d1_vols)
print(f"\n  급등 전날(D-1) 거래량 비율 분포 (N={len(arr_d1)})")
buckets_d1 = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0),
              (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 999)]
labels_d1 = ['< 0.3x (극도 조용)', '0.3~0.5x (매우 조용)', '0.5~0.7x (조용)',
             '0.7~1.0x (보통 이하)', '1.0~1.5x (보통)', '1.5~2.0x (약간 활발)',
             '2.0~3.0x (활발)', '3.0~5.0x (세력 진입)', '5.0x+ (폭발)']

print(f"\n  {'D-1 거래량':20s} {'건수':>6s} {'비율':>7s} 막대")
print("  " + "-" * 55)
for (lo, hi), lbl in zip(buckets_d1, labels_d1):
    cnt = ((arr_d1 >= lo) & (arr_d1 < hi)).sum()
    pct = cnt / len(arr_d1)
    bar = '█' * int(pct * 30)
    print(f"  {lbl:20s} {cnt:5d}개  {pct*100:6.1f}%  {bar}")

# 컨트롤 그룹과 비교
ctl_d1_vols = []
for df_c in control_data.values():
    if len(df_c) < 2:
        continue
    vr = df_c.iloc[-1].get('VolRatio', np.nan)
    if not pd.isna(vr):
        ctl_d1_vols.append(vr)

arr_ctl = np.array(ctl_d1_vols) if ctl_d1_vols else np.array([1.0])
print(f"\n  비교 통계:")
print(f"  급등 종목  D-1 평균: {arr_d1.mean():.2f}x  중앙값: {np.median(arr_d1):.2f}x")
print(f"  컨트롤    D-1 평균: {arr_ctl.mean():.2f}x  중앙값: {np.median(arr_ctl):.2f}x")
print(f"  D-1 3x 이상: 급등 {(arr_d1>=3).mean()*100:.1f}% vs 컨트롤 {(arr_ctl>=3).mean()*100:.1f}%  "
      f"Lift: {((arr_d1>=3).mean()/(arr_ctl>=3).mean() if (arr_ctl>=3).mean()>0 else 0):.2f}x")
print(f"  D-1 < 0.7: 급등 {(arr_d1<0.7).mean()*100:.1f}% vs 컨트롤 {(arr_ctl<0.7).mean()*100:.1f}%  "
      f"(조용한 D-1 비율)")


# ═══════════════════════════════════════════════════════════════════════════════
# [8] 결과 요약 및 최적 파라미터
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("[8] 결과 요약 및 권장 파라미터")
print("=" * 62, flush=True)

# 최적 Entry threshold 재계산 with proper lift
best_thr = max(ENTRY_THRESHOLDS,
    key=lambda t: (surge_entry[t] / np.mean(ctl_entry[t])) if np.mean(ctl_entry[t]) > 0 else 0)

results = {
    'analysis_date':        datetime.now().strftime('%Y-%m-%d %H:%M'),
    'n_surge':              N,
    'n_control':            len(control_data),
    'best_entry_threshold': best_thr,
    'best_entry_lift':      round(surge_entry[best_thr] / np.mean(ctl_entry[best_thr]), 3)
                            if np.mean(ctl_entry[best_thr]) > 0 else 0,
    'best_quiet_threshold': best_quiet_thr,
    'pattern_counts':       pattern_counts,
    'pattern_surge_avg':    {k: round(float(np.mean(v)), 2) if v else 0
                             for k, v in pattern_surge_pct.items()},
    'd1_vol_mean':          round(float(arr_d1.mean()), 3),
    'd1_vol_median':        round(float(np.median(arr_d1)), 3),
    'compression_ratio_median': round(float(np.median(compression_ratios)), 1) if compression_ratios else 0,
}

with open('volume_analysis_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  거래량 패턴 분석 결과 요약 ({N}개 관측치)               │
  ├──────────────────────────────────────────────────────────┤
  │  [세력 진입 임계값]                                       │
  │    최적: {best_thr:.1f}x  (현재: 3.0x)                        │
  │                                                          │
  │  [세력 매집 임계값]                                       │
  │    최적: {best_quiet_thr:.2f}  (현재: 0.70)                     │
  │                                                          │
  │  [급등 압축비]                                            │
  │    중앙값: {np.median(compression_ratios) if compression_ratios else 0:.1f}x  (최저→급등일 거래량 배율)   │
  └──────────────────────────────────────────────────────────┘
""")

print("결과 저장: volume_analysis_results.json")
print("\n완료.")
