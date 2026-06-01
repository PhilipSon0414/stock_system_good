"""
타임프레임 종합 분석 — 기존 급등 데이터 기반

분석 항목:
  1. MA 기간별 정배열 적중률 (5/10/20/60/120/240)
  2. 거래량 이평 기간 (10/20/30일)
  3. OB 탐색 범위 (30/60/90일)
  4. 매집 횡보 기간 (5/10/15/20일)
  5. 세력 탐지 이평선 조합 비교
  6. 전일 신호 선행력 (D-1, D-3, D-5, D-10)

결과 → 코드 자동 반영
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from itertools import combinations

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Volumes/iMac/Python Programming/Python_Invest/stock_system')
sys.stdout.reconfigure(line_buffering=True)

# ─── 데이터 로드 ────────────────────────────────────────────────────────────────
with open('learning_log.json') as f:
    log = json.load(f)

observations = log['observations']
print(f"학습 관측치: {len(observations)}개")

from pykrx import stock as pkrx

def _fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """pykrx로 OHLCV 수집 후 영문 컬럼으로 변환"""
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


def _fetch_ohlcv_safe(ticker: str, start: str, end: str, timeout: int = 25) -> pd.DataFrame:
    """타임아웃 포함 fetch — 25초 초과 시 빈 DataFrame 반환"""
    import threading
    result = [pd.DataFrame()]
    def _run():
        result[0] = _fetch_ohlcv(ticker, start, end)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return result[0]

# ─── 급등 종목 이력 수집 ────────────────────────────────────────────────────────
surge_records = []
for obs in observations:
    surge_records.append({
        'ticker': obs['ticker'],
        'name': obs['name'],
        'surge_date': obs['date'],
        'surge_pct': obs['surge_pct'],
    })

print(f"분석 대상: {len(surge_records)}개 급등 이벤트")

# ─── 주가 데이터 다운로드 (캐시) ────────────────────────────────────────────────
import os, pickle

CACHE_FILE = '/tmp/tf_analysis_cache.pkl'

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as f:
        price_cache = pickle.load(f)
    print(f"캐시 로드: {len(price_cache)}개 종목")
else:
    price_cache = {}

print("\n주가 데이터 수집 중...")
failed = []
for rec in surge_records:
    key = rec['ticker']
    if key in price_cache:
        continue
    try:
        surge_dt = datetime.strptime(rec['surge_date'], '%Y-%m-%d')
        start = (surge_dt - timedelta(days=400)).strftime('%Y%m%d')
        end   = (surge_dt + timedelta(days=5)).strftime('%Y%m%d')
        df = _fetch_ohlcv(rec['ticker'], start, end)
        if len(df) >= 60:
            price_cache[key] = {
                'df': df,
                'surge_date': rec['surge_date'],
                'name': rec['name'],
            }
            sys.stdout.write(f"  ✓ {rec['ticker']} {rec['name']}\n")
        else:
            failed.append(rec['ticker'])
    except Exception as e:
        failed.append(rec['ticker'])
        sys.stdout.write(f"  ✗ {rec['ticker']}: {e}\n")
    sys.stdout.flush()

with open(CACHE_FILE, 'wb') as f:
    pickle.dump(price_cache, f)

print(f"\n수집 완료: {len(price_cache)}개 / 실패: {len(failed)}개")


# ─── 분석 헬퍼 ──────────────────────────────────────────────────────────────────
def add_mas(df, periods):
    df = df.copy()
    for p in periods:
        df[f'MA{p}'] = df['Close'].rolling(p).mean()
    return df

def add_volume(df, vol_period):
    df = df.copy()
    df['VolMA'] = df['Volume'].rolling(vol_period).mean()
    df['VolRatio'] = df['Volume'] / df['VolMA'].replace(0, np.nan)
    return df

def get_pre_surge_row(df, surge_date_str, offset=1):
    """급등일 기준 D-offset 행 반환"""
    try:
        surge_dt = datetime.strptime(surge_date_str, '%Y-%m-%d')
        df_before = df[df.index < surge_dt]
        if len(df_before) < offset:
            return None
        return df_before.iloc[-offset]
    except Exception:
        return None

def lift_ratio(feature_rate, base_rate):
    """급등 종목 피처 출현율 / 전체 출현율 → lift"""
    if base_rate == 0:
        return float('inf')
    return round(feature_rate / base_rate, 3)


# ─── [1] MA 기간별 정배열 분석 ─────────────────────────────────────────────────
print("\n" + "="*60)
print("[1] MA 기간 조합별 정배열 적중률 분석 (D-1 기준)")
print("="*60)

ALL_MA_PERIODS = [5, 10, 20, 30, 60, 120, 240]
triplets = [
    (5, 20, 60),   # 기존
    (10, 30, 60),
    (10, 30, 120),
    (20, 60, 120), # 이전 백테스트 우승
    (20, 60, 240),
    (30, 60, 120),
    (30, 120, 240),
    (60, 120, 240),
]

ma_results = {}
for triplet in triplets:
    short, mid, long = triplet
    hit = 0
    total = 0
    for rec in surge_records:
        key = rec['ticker']
        if key not in price_cache:
            continue
        cache = price_cache[key]
        df = add_mas(cache['df'], [short, mid, long])
        row = get_pre_surge_row(df, cache['surge_date'], offset=1)
        if row is None:
            continue
        v_s  = row.get(f'MA{short}',  np.nan)
        v_m  = row.get(f'MA{mid}',    np.nan)
        v_l  = row.get(f'MA{long}',   np.nan)
        if any(pd.isna(x) for x in [v_s, v_m, v_l]):
            continue
        total += 1
        if v_s > v_m > v_l:
            hit += 1
    rate = hit / total if total > 0 else 0
    ma_results[triplet] = {'hit': hit, 'total': total, 'rate': round(rate, 3)}

# 기저율 (랜덤 KOSPI 종목 200개 샘플)
print("\n  기저율 계산 중 (랜덤 200 종목)...", flush=True)
try:
    ref_date = '20260401'
    kospi_tickers  = list(pkrx.get_market_ticker_list(ref_date, market='KOSPI'))[:80]
    kosdaq_tickers = list(pkrx.get_market_ticker_list(ref_date, market='KOSDAQ'))[:80]
    control_tickers = kospi_tickers + kosdaq_tickers
except Exception:
    control_tickers = []

control_ma_hit = {t: 0 for t in triplets}
control_total  = {t: 0 for t in triplets}
control_df_cache = {}  # 섹션[2] 이후 재사용

if control_tickers:
    for i_ctl, ticker in enumerate(control_tickers[:60]):
        print(f"  컨트롤 {i_ctl+1}/60: {ticker}", flush=True)
        try:
            df_c = _fetch_ohlcv_safe(ticker, '20250801', '20260501')
            if len(df_c) < 130:
                continue
            control_df_cache[ticker] = df_c
            for triplet in triplets:
                short, mid, long = triplet
                df_t = add_mas(df_c, [short, mid, long])
                row  = df_t.iloc[-1]
                v_s  = row.get(f'MA{short}',  np.nan)
                v_m  = row.get(f'MA{mid}',    np.nan)
                v_l  = row.get(f'MA{long}',   np.nan)
                if any(pd.isna(x) for x in [v_s, v_m, v_l]):
                    continue
                control_total[triplet] += 1
                if v_s > v_m > v_l:
                    control_ma_hit[triplet] += 1
        except Exception:
            continue

print(f"\n  {'조합':20s} {'급등D-1':>8s} {'기저율':>8s} {'Lift':>7s} {'판정':>10s}")
print("  " + "-"*55)
best_ma_triplet = None
best_lift = 0
for triplet in sorted(triplets, key=lambda t: -ma_results[t]['rate']):
    r = ma_results[triplet]
    base_n = control_ma_hit.get(triplet, 0)
    base_d = control_total.get(triplet, 1)
    base_rate = base_n / base_d if base_d > 0 else 0
    lft = lift_ratio(r['rate'], base_rate) if base_rate > 0 else 0
    label_cur = '← 기존' if triplet == (5, 20, 60) else ('← 현재' if triplet == (20, 60, 120) else '')
    print(f"  MA{triplet[0]}/{triplet[1]}/{triplet[2]}{'':10s} {r['rate']*100:7.1f}%  {base_rate*100:7.1f}%  {lft:6.2f}x  {label_cur}")
    if lft > best_lift:
        best_lift = lft
        best_ma_triplet = triplet


# ─── [2] 골든크로스 페어 분석 ───────────────────────────────────────────────────
print("\n" + "="*60)
print("[2] 골든크로스 페어 분석 (D-1 기준)")
print("="*60)

gc_pairs = [(5, 20), (5, 60), (10, 30), (20, 60), (20, 120), (60, 120)]
gc_results = {}

for pair in gc_pairs:
    short, long = pair
    hit = 0; total = 0
    for rec in surge_records:
        key = rec['ticker']
        if key not in price_cache:
            continue
        cache = price_cache[key]
        df = add_mas(cache['df'], [short, long])
        surge_dt = datetime.strptime(cache['surge_date'], '%Y-%m-%d')
        df_before = df[df.index < surge_dt]
        if len(df_before) < 5:
            continue
        total += 1
        # 최근 5일 내 골든크로스 발생 여부
        for i in range(max(1, len(df_before)-5), len(df_before)):
            curr = df_before.iloc[i]
            prev = df_before.iloc[i-1]
            vs_c, vl_c = curr.get(f'MA{short}', np.nan), curr.get(f'MA{long}', np.nan)
            vs_p, vl_p = prev.get(f'MA{short}', np.nan), prev.get(f'MA{long}', np.nan)
            if any(pd.isna(v) for v in [vs_c, vl_c, vs_p, vl_p]):
                continue
            if vs_p <= vl_p and vs_c > vl_c:
                hit += 1
                break
    gc_results[pair] = {'hit': hit, 'total': total, 'rate': round(hit/total, 3) if total > 0 else 0}

# 기저율 — 섹션[1]에서 캐시된 데이터 재사용
control_gc_hit = {p: 0 for p in gc_pairs}
control_gc_tot = {p: 0 for p in gc_pairs}
for ticker in control_tickers[:60]:
    try:
        df_c = control_df_cache.get(ticker, pd.DataFrame())
        if len(df_c) < 130:
            continue
        for pair in gc_pairs:
            short, long = pair
            df_t = add_mas(df_c, [short, long])
            control_gc_tot[pair] += 1
            for i in range(max(1, len(df_t)-5), len(df_t)):
                curr = df_t.iloc[i]; prev = df_t.iloc[i-1]
                vs_c = curr.get(f'MA{short}', np.nan); vl_c = curr.get(f'MA{long}', np.nan)
                vs_p = prev.get(f'MA{short}', np.nan); vl_p = prev.get(f'MA{long}', np.nan)
                if any(pd.isna(v) for v in [vs_c, vl_c, vs_p, vl_p]):
                    continue
                if vs_p <= vl_p and vs_c > vl_c:
                    control_gc_hit[pair] += 1
                    break
    except Exception:
        continue

print(f"\n  {'페어':15s} {'급등D-5내':>9s} {'기저율':>8s} {'Lift':>7s} {'판정':>12s}")
print("  " + "-"*55)
best_gc_pair = None
best_gc_lift = 0
for pair in sorted(gc_pairs, key=lambda p: -gc_results[p]['rate']):
    r = gc_results[pair]
    base_n = control_gc_hit.get(pair, 0)
    base_d = control_gc_tot.get(pair, 1)
    base_rate = base_n / base_d if base_d > 0 else 0
    lft = lift_ratio(r['rate'], base_rate) if base_rate > 0 else 0
    label_cur = '← 기존' if pair == (5, 20) else ('← 현재' if pair == (20, 60) else '')
    print(f"  MA{pair[0]}/MA{pair[1]}{'':9s} {r['rate']*100:8.1f}%  {base_rate*100:7.1f}%  {lft:6.2f}x  {label_cur}")
    if lft > best_gc_lift:
        best_gc_lift = lft
        best_gc_pair = pair


# ─── [3] 거래량 이평 기간 분석 ─────────────────────────────────────────────────
print("\n" + "="*60)
print("[3] 거래량 이평 기간별 세력 탐지 분석 (D-1 기준)")
print("="*60)

VOL_PERIODS = [5, 10, 15, 20, 30, 60]
VOL_THRESHOLD = 3.0  # 세력 진입 기준

vol_results = {}
for vp in VOL_PERIODS:
    high_vol = 0; total = 0
    for rec in surge_records:
        key = rec['ticker']
        if key not in price_cache:
            continue
        cache = price_cache[key]
        df = add_volume(cache['df'], vp)
        row = get_pre_surge_row(df, cache['surge_date'], offset=1)
        if row is None:
            continue
        vol_r = row.get('VolRatio', np.nan)
        if pd.isna(vol_r):
            continue
        total += 1
        if vol_r >= VOL_THRESHOLD:
            high_vol += 1
    # 추가: 5일 이내 언제라도 폭증 탐지
    hit5 = 0; total5 = 0
    for rec in surge_records:
        key = rec['ticker']
        if key not in price_cache:
            continue
        cache = price_cache[key]
        df = add_volume(cache['df'], vp)
        surge_dt = datetime.strptime(cache['surge_date'], '%Y-%m-%d')
        df_w = df[df.index < surge_dt].iloc[-5:]
        if len(df_w) == 0:
            continue
        total5 += 1
        if (df_w['VolRatio'].dropna() >= VOL_THRESHOLD).any():
            hit5 += 1
    vol_results[vp] = {
        'd1_rate': round(high_vol/total, 3) if total > 0 else 0,
        'd5_rate': round(hit5/total5, 3) if total5 > 0 else 0,
    }

print(f"\n  {'기간':8s} {'D-1 폭증':>9s} {'D-5내 폭증':>11s}")
print("  " + "-"*35)
for vp in VOL_PERIODS:
    r = vol_results[vp]
    label = ' ← 현재' if vp == 20 else ''
    print(f"  {vp}일{'':5s} {r['d1_rate']*100:8.1f}%  {r['d5_rate']*100:10.1f}%{label}")


# ─── [4] OB 탐색 범위 분석 ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("[4] OB(오더블록) 탐색 범위별 감지율 분석")
print("="*60)

OB_LOOKBACKS = [30, 45, 60, 90, 120]
OB_MIN_MOVE  = 0.04
OB_NEAR_TOL  = 0.03

ob_results = {}
for lb in OB_LOOKBACKS:
    near_count = 0; total = 0
    for rec in surge_records:
        key = rec['ticker']
        if key not in price_cache:
            continue
        cache = price_cache[key]
        df = cache['df'].copy()
        surge_dt = datetime.strptime(cache['surge_date'], '%Y-%m-%d')
        df_b = df[df.index < surge_dt]
        if len(df_b) < lb + 5:
            continue
        total += 1
        close = df_b.iloc[-1]['Close']
        window = df_b.iloc[-lb:]
        # 단순 OB 탐색: 강한 양봉 중 이후 OB_MIN_MOVE+ 이동한 것
        found_near = False
        for i in range(len(window) - 5):
            row = window.iloc[i]
            body = (row['Close'] - row['Open']) / row['Open'] if row['Open'] > 0 else 0
            if body < 0.02:
                continue
            future = window.iloc[i+1:i+6]
            if future.empty:
                continue
            future_high = future['High'].max()
            if (future_high - row['Close']) / row['Close'] >= OB_MIN_MOVE:
                ob_low  = min(row['Open'], row['Close'])
                ob_high = max(row['Open'], row['Close'])
                dist = max(0, (close - ob_high) / ob_high) if close > ob_high else max(0, (ob_low - close) / close)
                if dist < OB_NEAR_TOL:
                    found_near = True
                    break
        if found_near:
            near_count += 1
    ob_results[lb] = {'near': near_count, 'total': total, 'rate': round(near_count/total, 3) if total > 0 else 0}

print(f"\n  {'탐색범위':10s} {'OB근접율':>10s}")
print("  " + "-"*25)
best_ob = max(ob_results, key=lambda k: ob_results[k]['rate'])
for lb in OB_LOOKBACKS:
    r = ob_results[lb]
    label = ' ← 현재' if lb == 60 else (' ★최적' if lb == best_ob else '')
    print(f"  {lb}일{'':6s} {r['rate']*100:9.1f}%{label}")


# ─── [5] 세력 탐지 이평선 거리 기준 분석 ──────────────────────────────────────
print("\n" + "="*60)
print("[5] 세력 매집 이평선 근접 거리 기준 (near_ma 허용범위)")
print("="*60)

NEAR_MA_THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.08]
MA_FOR_ACCUM = [20, 60, 120]

accum_results = {}
for thr in NEAR_MA_THRESHOLDS:
    hit = 0; total = 0
    for rec in surge_records:
        key = rec['ticker']
        if key not in price_cache:
            continue
        cache = price_cache[key]
        df = add_mas(cache['df'], MA_FOR_ACCUM)
        row = get_pre_surge_row(df, cache['surge_date'], offset=1)
        if row is None:
            continue
        close = row.get('Close', np.nan)
        if pd.isna(close) or close == 0:
            continue
        total += 1
        for ma in MA_FOR_ACCUM:
            ma_val = row.get(f'MA{ma}', np.nan)
            if not pd.isna(ma_val) and abs(close - ma_val) / ma_val < thr:
                hit += 1
                break
    accum_results[thr] = {'hit': hit, 'total': total, 'rate': round(hit/total, 3) if total > 0 else 0}

print(f"\n  {'허용범위':10s} {'이평근접율':>10s}  해석")
print("  " + "-"*45)
for thr in NEAR_MA_THRESHOLDS:
    r = accum_results[thr]
    label = ' ← 현재' if thr == 0.03 else ''
    print(f"  {thr*100:.0f}%{'':7s} {r['rate']*100:9.1f}%{label}")


# ─── [6] 선행 신호 기간 분석 (D-N 기준) ─────────────────────────────────────
print("\n" + "="*60)
print("[6] 신호 선행력 분석 — 급등 며칠 전이 가장 잘 잡히나?")
print("="*60)

OFFSETS = [1, 2, 3, 5, 7, 10]

offset_results = {}
for offset in OFFSETS:
    ma_bull = 0; gc = 0; se_entry = 0; total = 0

    for rec in surge_records:
        key = rec['ticker']
        if key not in price_cache:
            continue
        cache = price_cache[key]
        df_raw = cache['df'].copy()
        df_raw = add_mas(df_raw, [20, 60, 120])
        df_raw = add_volume(df_raw, 20)

        from config import VOL_ENTRY_THRESHOLD, VOL_QUIET_THRESHOLD, CONSOLIDATION_DAYS, CONSOLIDATION_RANGE
        surge_dt = datetime.strptime(cache['surge_date'], '%Y-%m-%d')
        df_b = df_raw[df_raw.index < surge_dt]
        if len(df_b) < offset + 5:
            continue
        row = df_b.iloc[-offset]
        total += 1

        # 정배열 (20,60,120)
        v20 = row.get('MA20', np.nan); v60 = row.get('MA60', np.nan); v120 = row.get('MA120', np.nan)
        if not any(pd.isna(x) for x in [v20, v60, v120]) and v20 > v60 > v120:
            ma_bull += 1

        # 골든크로스 MA20/MA60 (최근 3일 내)
        for i in range(max(1, len(df_b)-offset-3), len(df_b)-offset+1):
            if i < 1 or i >= len(df_b):
                continue
            curr = df_b.iloc[i]; prev = df_b.iloc[i-1]
            vs_c = curr.get('MA20', np.nan); vl_c = curr.get('MA60', np.nan)
            vs_p = prev.get('MA20', np.nan); vl_p = prev.get('MA60', np.nan)
            if all(not pd.isna(v) for v in [vs_c, vl_c, vs_p, vl_p]):
                if vs_p <= vl_p and vs_c > vl_c:
                    gc += 1
                    break

        # 세력 진입 신호 (volume 폭증)
        vol_r = row.get('VolRatio', np.nan)
        if not pd.isna(vol_r) and vol_r >= VOL_ENTRY_THRESHOLD:
            se_entry += 1

    offset_results[offset] = {
        'total': total,
        'ma_bull': round(ma_bull/total, 3) if total > 0 else 0,
        'gc':      round(gc/total, 3) if total > 0 else 0,
        'entry':   round(se_entry/total, 3) if total > 0 else 0,
    }

print(f"\n  {'D-N':6s} {'정배열(20/60/120)':>18s} {'골든크로스(20/60)':>17s} {'세력진입폭증':>13s}")
print("  " + "-"*60)
for offset in OFFSETS:
    r = offset_results[offset]
    print(f"  D-{offset}{'':4s} {r['ma_bull']*100:17.1f}%  {r['gc']*100:16.1f}%  {r['entry']*100:12.1f}%")


# ─── [7] 거래량 압축 기간 분석 ─────────────────────────────────────────────────
print("\n" + "="*60)
print("[7] 거래량 압축 지속 기간 vs 급등 발생 분포")
print("="*60)

quiet_dist = {}
for rec in surge_records:
    key = rec['ticker']
    if key not in price_cache:
        continue
    cache = price_cache[key]
    df = add_volume(cache['df'], 20)
    surge_dt = datetime.strptime(cache['surge_date'], '%Y-%m-%d')
    df_b = df[df.index < surge_dt]
    if len(df_b) < 10:
        continue
    vol_vals = df_b['VolRatio'].dropna().values
    quiet_days = 0
    for v in reversed(vol_vals):
        if v < 0.70:
            quiet_days += 1
        else:
            break
    bucket = min(quiet_days // 3, 6) * 3  # 0~3, 3~6, 6~9, ... 18+
    quiet_dist[bucket] = quiet_dist.get(bucket, 0) + 1

total_obs = sum(quiet_dist.values())
print(f"\n  {'연속 저거래량':15s} {'비율':>8s} {'막대'}")
print("  " + "-"*45)
for bucket in sorted(quiet_dist.keys()):
    cnt = quiet_dist[bucket]
    pct = cnt / total_obs
    bar = '█' * int(pct * 30)
    label = f'{bucket}~{bucket+2}일' if bucket < 18 else '18일+'
    print(f"  {label:13s}  {pct*100:6.1f}%  {bar}")


# ─── [8] 결과 요약 및 최적 설정 도출 ──────────────────────────────────────────
print("\n" + "="*60)
print("[8] 최적 파라미터 요약")
print("="*60)

print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  분석 결과 요약 (77개 급등 관측치 기반)               │
  ├─────────────────────────────────────────────────────┤
  │                                                     │
  │  [MA 정배열]                                        │
  │    최적 조합 : MA{best_ma_triplet[0]}/{best_ma_triplet[1]}/{best_ma_triplet[2]}   (Lift {best_lift:.2f}x)              │
  │                                                     │
  │  [골든크로스]                                        │
  │    최적 페어 : MA{best_gc_pair[0]}/MA{best_gc_pair[1]}   (Lift {best_gc_lift:.2f}x)              │
  │                                                     │
  │  [OB 탐색 범위]                                      │
  │    최적 범위 : {best_ob}일                                │
  └─────────────────────────────────────────────────────┘
""")

# 결과 저장
results_summary = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'n_observations': len(surge_records),
    'best_ma_triplet': list(best_ma_triplet),
    'best_ma_lift': best_lift,
    'best_gc_pair': list(best_gc_pair),
    'best_gc_lift': best_gc_lift,
    'best_ob_lookback': best_ob,
    'ma_results': {str(k): v for k, v in ma_results.items()},
    'gc_results': {str(k): v for k, v in gc_results.items()},
    'vol_results': vol_results,
    'ob_results': ob_results,
    'offset_results': offset_results,
}

with open('timeframe_results.json', 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

print("  분석 결과 저장: timeframe_results.json")
print("\n완료.")
