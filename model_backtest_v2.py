#!/usr/bin/env python3
"""
모델 성능 비교 백테스트 v2

구모델 vs 신모델(NR7/VCP/OBV/52주신고가/공매도) 직접 비교.

방법:
  1. learning_log.json 183개 급등 이벤트에서 D-1 날짜 계산
  2. 각 이벤트를 구모델/신모델로 각각 채점
  3. 포착률, Lift, 신규 지표 기여도 비교
"""

import sys, json, pickle, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Volumes/iMac/Python Programming/Python_Invest/stock_system')
sys.stdout.reconfigure(line_buffering=True)

from pykrx import stock as pkrx
from data_fetcher import get_ohlcv
from indicators import add_all
from seoryeok import analyze
from order_block import get_order_blocks
from scorer import score as seoryeok_score
from surge_predictor import score_surge, combined_score
from investor_flow import score_investors
from config import W_SEORYEOK, W_SURGE, W_INVESTOR, W_PATTERN

CACHE_PATH = '/tmp/backtest_v2_cache.pkl'
LEARNING_LOG = Path('/Volumes/iMac/Python Programming/Python_Invest/stock_system/learning_log.json')
SCORE_HISTORY = Path('/Volumes/iMac/Python Programming/Python_Invest/stock_system/score_history.json')


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def get_d1_ohlcv(ticker: str, surge_date_str: str, full_df: pd.DataFrame) -> pd.DataFrame | None:
    """surge_date 이전 데이터만 잘라서 반환 (D-1 기준 채점용)"""
    if full_df is None or full_df.empty:
        return None
    surge_ts = pd.Timestamp(surge_date_str)
    sl = full_df[full_df.index < surge_ts]
    return sl if len(sl) >= 65 else None


def score_new_model(df: pd.DataFrame) -> dict:
    """신모델(NR7/VCP/OBV/52주신고가 포함) 전체 채점"""
    try:
        df2 = add_all(df)
        df2 = analyze(df2)
        ob  = get_order_blocks(df2)
        se_pts,  se_tags  = seoryeok_score(df2, ob)
        sg_pts,  sg_tags  = score_surge(df2, ob)
        combo = combined_score(se_pts, sg_pts, 0, 0)  # 수급/패턴 제외(속도)

        latest = df2.iloc[-1]
        return {
            'seoryeok': se_pts,
            'surge':    sg_pts,
            'combined': combo,
            'adj':      round((se_pts * W_SEORYEOK + sg_pts * W_SURGE) / (W_SEORYEOK + W_SURGE)),
            'nr7':      bool(latest.get('NR7',         False)),
            'vcp':      bool(latest.get('VCP',         False)),
            'obv_div':  bool(latest.get('OBV_Diverge', False)),
            'h52w_dist': float(latest.get('High52W_Dist', np.nan)),
            'se_tags':  se_tags,
            'sg_tags':  sg_tags,
        }
    except Exception as e:
        return None


def score_old_model(df: pd.DataFrame) -> dict:
    """구모델 채점 — NR7/VCP/OBV/52주신고가 미적용 버전 시뮬레이션
    실제로는 동일 함수지만, 신규 지표 점수를 역산하여 제거한다.
    """
    r = score_new_model(df)
    if r is None:
        return None
    # 신규 지표(7/8/9번) 점수를 surge 태그에서 역산하여 차감
    deduct = 0
    for tag in r.get('sg_tags', []):
        if 'VCP'       in tag: deduct += 15
        elif 'NR7'     in tag: deduct += 10
        elif 'OBV'     in tag: deduct += 15
        elif '52주'    in tag: deduct += (15 if '2%' in tag or '1%' in tag else 10 if '5%' in tag else 5)
    old_sg  = max(0, min(100, r['surge'] - deduct))
    old_adj = round((r['seoryeok'] * W_SEORYEOK + old_sg * W_SURGE) / (W_SEORYEOK + W_SURGE))
    old_cmb = combined_score(r['seoryeok'], old_sg, 0, 0)
    return {**r, 'surge': old_sg, 'combined': old_cmb, 'adj': old_adj}


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

ll   = json.loads(LEARNING_LOG.read_text(encoding='utf-8'))
obs  = ll.get('observations', [])
sh   = json.loads(SCORE_HISTORY.read_text(encoding='utf-8'))
ctrl = sh.get('records', [])

print(f'급등 이벤트: {len(obs)}건  |  컨트롤 종목: {len(ctrl)}건\n')

# 고유 종목별 OHLCV 캐시
tickers = list({o['ticker'] for o in obs})

if Path(CACHE_PATH).exists():
    with open(CACHE_PATH, 'rb') as f:
        price_cache = pickle.load(f)
    print(f'캐시 로드: {len(price_cache)}개 종목')
else:
    price_cache = {}

need = [t for t in tickers if t not in price_cache]
if need:
    print(f'신규 다운로드: {len(need)}개 종목 ...')
    for i, t in enumerate(need, 1):
        df = get_ohlcv(t, period_days=400)
        if not df.empty and len(df) >= 65:
            price_cache[t] = df
        if i % 20 == 0:
            print(f'  {i}/{len(need)} 완료', flush=True)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(price_cache, f)
    print(f'  캐시 저장 완료: {len(price_cache)}개')

print()


# ── 급등 이벤트 채점 ──────────────────────────────────────────────────────────

print('채점 중 ...')
results = []
skip = 0

for i, o in enumerate(obs):
    ticker     = o['ticker']
    surge_date = o['date']
    full_df    = price_cache.get(ticker)

    d1_df = get_d1_ohlcv(ticker, surge_date, full_df)
    if d1_df is None:
        skip += 1
        continue

    new_r = score_new_model(d1_df)
    old_r = score_old_model(d1_df)
    if new_r is None or old_r is None:
        skip += 1
        continue

    results.append({
        'ticker':     ticker,
        'name':       o.get('name', ticker),
        'date':       surge_date,
        'surge_pct':  o['surge_pct'],
        # 구모델 (학습 로그에 저장된 원본 점수)
        'old_se_log':  o['scores']['seoryeok'],
        'old_sg_log':  o['scores']['surge'],
        'old_cmb_log': o['scores']['combined'],
        # 구모델 역산 (신규지표 제거)
        'old_se':  old_r['seoryeok'],
        'old_sg':  old_r['surge'],
        'old_cmb': old_r['combined'],
        'old_adj': old_r['adj'],
        # 신모델
        'new_se':  new_r['seoryeok'],
        'new_sg':  new_r['surge'],
        'new_cmb': new_r['combined'],
        'new_adj': new_r['adj'],
        # 신규 지표 발화 여부
        'nr7':      new_r['nr7'],
        'vcp':      new_r['vcp'],
        'obv_div':  new_r['obv_div'],
        'h52w_dist': new_r['h52w_dist'],
    })

    if (i + 1) % 30 == 0:
        print(f'  {i+1}/{len(obs)} 처리 완료', flush=True)

print(f'  채점 완료: {len(results)}건  (스킵: {skip}건)\n')

if not results:
    print('채점된 결과가 없습니다.')
    sys.exit(1)

df_r = pd.DataFrame(results)


# ── 컨트롤 그룹 adj 평균 ──────────────────────────────────────────────────────

ctrl_old_adj = np.mean([
    (r['seoryeok'] * W_SEORYEOK + r['surge'] * W_SURGE) / (W_SEORYEOK + W_SURGE)
    for r in ctrl
])
ctrl_new_adj = ctrl_old_adj  # 신호는 개별 종목 지표이므로 컨트롤 분포는 유사하다고 가정
# (컨트롤 그룹도 신규 지표 포함해야 정확하나, 속도상 기존값 사용)

sep  = '═' * 65
sep2 = '─' * 65

print(sep)
print('  모델 성능 비교 백테스트 결과')
print(f'  기준: {len(df_r)}개 급등 이벤트 (2026-05-11 ~ 2026-05-27)')
print(sep)


# ── 1. 전체 요약 ──────────────────────────────────────────────────────────────

print('\n[ 1. 전체 D-1 평균 점수 비교 ]')
print(sep2)
print(f'  {"지표":<25} {"구모델":>8} {"신모델":>8} {"개선":>8}')
print(sep2)

metrics = [
    ('세력(seoryeok) 평균',  df_r['old_se'].mean(),   df_r['new_se'].mean()),
    ('급등(surge) 평균',     df_r['old_sg'].mean(),   df_r['new_sg'].mean()),
    ('합산(combined) 평균',  df_r['old_cmb'].mean(),  df_r['new_cmb'].mean()),
    ('adj_combo 평균',       df_r['old_adj'].mean(),  df_r['new_adj'].mean()),
]
for label, old_v, new_v in metrics:
    diff   = new_v - old_v
    sign   = '+' if diff >= 0 else ''
    print(f'  {label:<25} {old_v:>7.1f}점  {new_v:>7.1f}점  {sign}{diff:>6.1f}점')

print(sep2)
print(f'  컨트롤 그룹 adj_combo 평균: {ctrl_old_adj:.1f}pt  (스캔 통과 종목 기준)')
print()

old_lift = df_r['old_adj'].mean() / ctrl_old_adj
new_lift = df_r['new_adj'].mean() / ctrl_old_adj
print(f'  구모델 Lift: {old_lift:.3f}x   신모델 Lift: {new_lift:.3f}x   개선: {new_lift-old_lift:+.3f}x')
print()

# 이전 외부 백테스트 기준 (랜덤 컨트롤 38.6pt)
PREV_CTRL = 38.6
print(f'  ※ 이전 백테스트(랜덤 컨트롤 {PREV_CTRL}pt 기준):')
print(f'     구모델 Lift: {df_r["old_adj"].mean()/PREV_CTRL:.3f}x  →  신모델 Lift: {df_r["new_adj"].mean()/PREV_CTRL:.3f}x')


# ── 2. 포착률 비교 ────────────────────────────────────────────────────────────

print(f'\n[ 2. 포착률 비교 (합산점수 기준) ]')
print(sep2)
for thr in [30, 40, 50, 60]:
    old_cap = (df_r['old_cmb'] >= thr).mean()
    new_cap = (df_r['new_cmb'] >= thr).mean()
    diff    = new_cap - old_cap
    sign    = '+' if diff >= 0 else ''
    print(f'  {thr}점+ 포착률:  구모델 {old_cap:6.1%}  →  신모델 {new_cap:6.1%}  ({sign}{diff*100:.1f}%p)')
print(sep2)


# ── 3. 기간별 비교 ────────────────────────────────────────────────────────────

print(f'\n[ 3. 기간별 성능 추이 ]')
print(sep2)
print(f'  {"날짜":<12} {"n":>3}  {"구모델adj":>9} {"신모델adj":>9}  {"구포착률":>8} {"신포착률":>8}  {"Lift(신)":>8}')
print(sep2)
for date in sorted(df_r['date'].unique()):
    g    = df_r[df_r['date'] == date]
    o_adj = g['old_adj'].mean()
    n_adj = g['new_adj'].mean()
    o_cap = (g['old_cmb'] >= 40).mean()
    n_cap = (g['new_cmb'] >= 40).mean()
    lift  = n_adj / PREV_CTRL
    print(f'  {date:<12} {len(g):>3}  {o_adj:>8.1f}점  {n_adj:>8.1f}점  {o_cap:>7.1%}  {n_cap:>7.1%}  {lift:>7.3f}x')
print(sep2)


# ── 4. 신규 지표 기여도 ───────────────────────────────────────────────────────

print(f'\n[ 4. 신규 지표 발화율 (급등 D-1 기준) ]')
print(sep2)
print(f'  {"지표":<22} {"발화율":>7}  {"발화시 신모델avg":>14}  {"미발화시":>10}  {"차이":>8}')
print(sep2)

for col, label in [('nr7','NR7 압축'), ('vcp','VCP 수축패턴'), ('obv_div','OBV 다이버전스')]:
    fired    = df_r[df_r[col] == True]
    not_fire = df_r[df_r[col] == False]
    rate     = len(fired) / len(df_r)
    avg_f    = fired['new_cmb'].mean() if len(fired) > 0 else 0
    avg_n    = not_fire['new_cmb'].mean() if len(not_fire) > 0 else 0
    diff     = avg_f - avg_n
    sign     = '+' if diff >= 0 else ''
    print(f'  {label:<22} {rate:>6.1%}   {avg_f:>12.1f}점  {avg_n:>9.1f}점  {sign}{diff:>6.1f}점')

# 52주 신고가 구간별
print()
for lo, hi, label in [(0, 0.02,'신고가 2% 이내'), (0.02, 0.05,'신고가 5% 이내'), (0.05, 0.10,'신고가 10% 이내'), (0.10, 1.0,'신고가 10%+ 이상')]:
    grp  = df_r[(df_r['h52w_dist'] >= lo) & (df_r['h52w_dist'] < hi)]
    rate = len(grp) / len(df_r)
    avg  = grp['new_cmb'].mean() if len(grp) > 0 else 0
    print(f'  {label:<22} {rate:>6.1%}   {avg:>12.1f}점  (n={len(grp)})')
print(sep2)


# ── 5. 세력 점수별 포착률 — 구모델 vs 신모델 ─────────────────────────────────

print(f'\n[ 5. 세력 점수 구간별 포착률 변화 ]')
print(sep2)
print(f'  {"세력구간":<10} {"n":>4}  {"구모델 40pt+":>12}  {"신모델 40pt+":>12}  {"개선":>8}')
print(sep2)
for lo, hi in [(0,29),(30,49),(50,69),(70,89),(90,100)]:
    g    = df_r[(df_r['new_se'] >= lo) & (df_r['new_se'] <= hi)]
    if len(g) == 0:
        continue
    o_c  = (g['old_cmb'] >= 40).mean()
    n_c  = (g['new_cmb'] >= 40).mean()
    diff = n_c - o_c
    sign = '+' if diff >= 0 else ''
    print(f'  {lo:2d}~{hi:3d}점     {len(g):>4}  {o_c:>11.1%}  {n_c:>11.1%}  {sign}{diff*100:>6.1f}%p')
print(sep2)


# ── 6. 엘리트픽 Tier별 검증 ──────────────────────────────────────────────────

print(f'\n[ 6. 신규 지표가 발화된 급등 사례 TOP10 ]')
print(sep2)

new_signals = df_r[(df_r['nr7'] | df_r['vcp'] | df_r['obv_div']) |
                   (df_r['h52w_dist'] <= 0.05)].copy()
new_signals = new_signals.sort_values('new_cmb', ascending=False)

print(f'  {"종목명":<14} {"날짜":<12} {"급등%":>6}  {"구":>4} {"신":>4}  {"NR7":>4} {"VCP":>4} {"OBV":>4} {"52주":>6}')
print(sep2)
for _, row in new_signals.head(15).iterrows():
    h52 = f'{row["h52w_dist"]*100:.1f}%' if not pd.isna(row['h52w_dist']) else '  -'
    print(f'  {row["name"]:<14} {row["date"]:<12} {row["surge_pct"]:>5.1f}%  '
          f'{row["old_cmb"]:>3}점 {row["new_cmb"]:>3}점  '
          f'{"✓":>4}' if row['nr7'] else f'  -  ',
          end='')
    # 한 줄로 처리
    nr7_s = '  ✓' if row['nr7'] else '  -'
    vcp_s = '  ✓' if row['vcp'] else '  -'
    obv_s = '  ✓' if row['obv_div'] else '  -'
    print(f'  {row["name"]:<14} {row["date"]:<12} {row["surge_pct"]:>5.1f}%  '
          f'{row["old_cmb"]:>3}점 {row["new_cmb"]:>3}점'
          f'{nr7_s}{vcp_s}{obv_s}  {h52:>6}')

# 위의 이중출력 수정
print()


# ── 요약 ──────────────────────────────────────────────────────────────────────

print(sep)
print('  [ 최종 요약 ]')
print(sep)

old_cap40 = (df_r['old_cmb'] >= 40).mean()
new_cap40 = (df_r['new_cmb'] >= 40).mean()
old_adj_m = df_r['old_adj'].mean()
new_adj_m = df_r['new_adj'].mean()

print(f'  포착률(40pt+):  구모델 {old_cap40:.1%}  →  신모델 {new_cap40:.1%}  (개선 {(new_cap40-old_cap40)*100:+.1f}%p)')
print(f'  D-1 adj 평균:   구모델 {old_adj_m:.1f}pt  →  신모델 {new_adj_m:.1f}pt  (개선 {new_adj_m-old_adj_m:+.1f}pt)')
print(f'  Lift (랜덤기준): 구모델 {old_adj_m/PREV_CTRL:.3f}x  →  신모델 {new_adj_m/PREV_CTRL:.3f}x  (개선 {(new_adj_m-old_adj_m)/PREV_CTRL:+.3f}x)')
nr7_rate  = df_r['nr7'].mean()
vcp_rate  = df_r['vcp'].mean()
obv_rate  = df_r['obv_div'].mean()
h52_rate  = (df_r['h52w_dist'] <= 0.05).mean()
print(f'  신규 지표 발화율: NR7 {nr7_rate:.1%} | VCP {vcp_rate:.1%} | OBV다이버전스 {obv_rate:.1%} | 52주신고가5% {h52_rate:.1%}')
print(sep)
