# -*- coding: utf-8 -*-
"""
diag_reversal.json 분석 — 반등 신호 lift/recall/precision + 현재 시스템 대비 증분 회수.
"""
import json

D = json.load(open('/tmp/diag_reversal.json'))
D = [r for r in D if r.get('rsi') is not None and r.get('pma20') is not None]
N = len(D)
P = sum(r['y'] for r in D)
base = P / N
print(f'분석 대상: {N}건  | 급등(5d) {P}건  | 기저율 {base*100:.1f}%\n')


def stats(mask_fn, label):
    sel = [r for r in D if mask_fn(r)]
    n = len(sel)
    if n == 0:
        print(f'  {label:38s} n=0')
        return None
    hit = sum(r['y'] for r in sel)
    prec = hit / n
    lift = prec / base
    recall = hit / P
    print(f'  {label:38s} n={n:4d}  적중 {prec*100:4.1f}%  lift {lift:4.2f}x  recall {recall*100:4.1f}%')
    return {'label': label, 'n': n, 'prec': prec, 'lift': lift, 'recall': recall}


print('── 현재 시스템(combined>=40) 기준선 ──')
cur = stats(lambda r: r['comb'] >= 40, 'combined>=40 (현재 포착)')
stats(lambda r: r['se'] >= 40, 'seoryeok>=40')

print('\n── 단일 반등 인자 ──')
stats(lambda r: r['rsi'] < 30, 'RSI<30 (과매도)')
stats(lambda r: r['rsi'] < 35, 'RSI<35')
stats(lambda r: r['below_ma20'], '종가<MA20 (역배열권)')
stats(lambda r: r['ret5'] < -0.10, '5일 낙폭 >10%')
stats(lambda r: r['ret5'] < -0.15, '5일 낙폭 >15%')
stats(lambda r: (r['vr'] or 0) >= 2, '거래량 2x+')
stats(lambda r: r['pos20'] < 0.25, '20일 저점권(하위25%)')
stats(lambda r: r['lwick'] > 0.4, '긴 아래꼬리(매도소진)')
stats(lambda r: r['inv'] >= 50, '수급>=50')

print('\n── 기계산 반등·압축 신호(보너스) ──')
stats(lambda r: r.get('rsi_recov'), 'RSI_Recovery (과매도 회복)')
stats(lambda r: r.get('vol_recov'), 'VolRecovery (거래량 회복)')
stats(lambda r: r.get('vcp'), 'VCP (변동성 수축)')
stats(lambda r: r.get('bbcompress'), 'BBCompress')
stats(lambda r: r.get('nr7'), 'NR7')

print('\n── 반등 복합 신호 (과매도 × 거래량 × 수급/캔들) ──')
combos = {
    'R1 RSI<35 & vr>=2': lambda r: r['rsi'] < 35 and (r['vr'] or 0) >= 2,
    'R2 RSI<35 & vr>=2 & 양봉': lambda r: r['rsi'] < 35 and (r['vr'] or 0) >= 2 and r['isbull'],
    'R3 RSI<35 & vr>=2 & inv>=50': lambda r: r['rsi'] < 35 and (r['vr'] or 0) >= 2 and r['inv'] >= 50,
    'R4 저점권 & vr>=2 & 양봉': lambda r: r['pos20'] < 0.3 and (r['vr'] or 0) >= 2 and r['isbull'],
    'R5 낙폭>10% & vr>=2 & 양봉': lambda r: r['ret5'] < -0.10 and (r['vr'] or 0) >= 2 and r['isbull'],
    'R6 RSI<40 & vr>=3 & inv>=40': lambda r: r['rsi'] < 40 and (r['vr'] or 0) >= 3 and r['inv'] >= 40,
    'R7 RSI<35 & 저점권 & vr>=1.5': lambda r: r['rsi'] < 35 and r['pos20'] < 0.3 and (r['vr'] or 0) >= 1.5,
    'R8 RSI<35 & 아래꼬리 & vr>=2': lambda r: r['rsi'] < 35 and r['lwick'] > 0.3 and (r['vr'] or 0) >= 2,
}
results = {}
for lab, fn in combos.items():
    results[lab] = stats(fn, lab)

print('\n── 증분 회수: 현재 미포착(combined<40) 중에서만 ──')
miss = [r for r in D if r['comb'] < 40]
miss_p = sum(r['y'] for r in miss)
print(f'  현재 미포착 영역: {len(miss)}건, 그중 급등 {miss_p}건 (이게 놓치는 급등)')


def incr(fn, label):
    sel = [r for r in miss if fn(r)]
    n = len(sel)
    if n == 0:
        print(f'  {label:38s} n=0')
        return
    hit = sum(r['y'] for r in sel)
    prec = hit / n
    lift = prec / base
    rec_of_missed = hit / miss_p if miss_p else 0
    print(f'  {label:30s} n={n:4d} 적중 {prec*100:4.1f}% lift {lift:4.2f}x | 놓친급등 회수 {rec_of_missed*100:4.1f}%')


for lab, fn in combos.items():
    incr(fn, lab)

print('\n── 최선 복합 OR 결합 시 전체 recall/precision ──')
best = max((v for v in results.values() if v and v['lift'] >= 1.3 and v['n'] >= 15),
           key=lambda v: v['recall'], default=None)
if best:
    bl = best['label']; fn = combos[bl]
    both = [r for r in D if r['comb'] >= 40 or fn(r)]
    hit = sum(r['y'] for r in both)
    print(f'  채택 신호: {bl} (lift {best["lift"]:.2f}x)')
    print(f'  현재만(combined>=40): recall {cur["recall"]*100:.1f}% / 적중 {cur["prec"]*100:.1f}%')
    print(f'  +반등 OR결합: recall {hit/P*100:.1f}% / 적중 {hit/len(both)*100:.1f}% (n={len(both)})')
    print(f'  → recall +{(hit/P-cur["recall"])*100:.1f}%p (정밀도 유지 여부 확인)')
else:
    print('  lift>=1.3x & n>=15 충족하는 반등 복합 신호 없음 → 편입 보류 권고')
