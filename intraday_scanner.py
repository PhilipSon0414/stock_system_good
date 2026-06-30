# -*- coding: utf-8 -*-
"""
장중 실시간 점화 스캐너 — '무징후 급등'(전일 EOD 신호 0) 당일 포착용.

배경(2026-06-30 진단): combined<40로 놓치는 급등은 전일 데이터에 정보가 없어
EOD로는 구조적 회수 불가(+0.0%p). 유일한 레버 = 장중 실시간 점화 감지.
이 모듈은 KRX 장중(09:00~15:30) 5분봉으로 거래량 급증·가격 점화를 실시간 탐지하고,
EOD 스캔이 못 뽑은 종목을 우선 표시한다(무징후 회수).

데이터: yfinance 5분봉(개별 KR 종목, .KS/.KQ). 네트워크 의존.

신호(당일 5분봉 누적 기준):
  day_chg   : 전일종가 대비 등락률(이미 얼마나 올랐나)
  vol_pace  : 당일 누적거래량 / (20일평균 × 경과시간비) — 거래량 폭발 페이스
  vwap_dev  : VWAP 대비 이격(+면 매수 우위 장중 수급)
  accel     : 최근 15분(3봉) 모멘텀(가속 여부)
  breakout  : 전일 고가 돌파 여부

점화 판정(Tier):
  T1 강한점화 : day_chg>=5% & vol_pace>=3 & 종가>VWAP & accel>0
  T2 점화     : day_chg>=3% & vol_pace>=2 & 종가>VWAP
  T3 초기징후 : vol_pace>=2 & breakout (가격은 아직, 거래량 선행)

사용법:
  python3 intraday_scanner.py                # 1회 스냅샷(유동 상위 200)
  python3 intraday_scanner.py --top 300      # 유니버스 크기
  python3 intraday_scanner.py --loop 10      # 10분 간격 반복(장중)
  python3 intraday_scanner.py --email        # 점화 포착 시 이메일
"""
import sys
import time
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')

from data_fetcher import get_ticker_list, get_ohlcv

SCRIPT_DIR = Path(__file__).parent
BARS_FULL_DAY = 78           # 09:00~15:30, 5분봉 78개
MIN_PRICE = 1000
MIN_AVG_VOL = 50000
LOG_FILE = SCRIPT_DIR / 'intraday_log.json'   # 점화·대조군 추적 로그(로컬)
CONTROL_RATE = 0.10          # 비점화 대조군 샘플링률(선택편향 제거 → 정확한 base/lift)


def _yf_symbol(code, market):
    return f"{code}.{'KS' if market == 'KOSPI' else 'KQ'}"


def live_session_today():
    """KRX 실시간 세션 여부 — KOSPI 지수 5분봉 최근 캔들이 '오늘'이면 today, 아니면 None.
    휴장/장외에 stale(전 거래일) 데이터를 today로 오기록하는 것을 방지(무인 가드)."""
    import yfinance as yf
    today = datetime.now().strftime('%Y-%m-%d')
    try:
        idx = yf.Ticker('^KS11').history(period='1d', interval='5m')
        if idx is not None and len(idx) and str(idx.index[-1].date()) == today:
            return today
    except Exception:
        pass
    return None


def build_universe(market='ALL', top_n=200):
    """유동성 상위 유니버스 — 가격·거래량 필터 후 거래량 내림차순 top_n."""
    df = get_ticker_list(market)
    if df is None or len(df) == 0:
        return []
    df = df[(df['Close'] >= MIN_PRICE) & (df['Volume'] >= MIN_AVG_VOL)]
    df = df[~df['Code'].str.endswith(('5', '7'))]    # 우선주/스팩 일부 제외
    df = df.sort_values('Volume', ascending=False).head(top_n)
    return [(r['Code'], r['Name'], r['Market']) for _, r in df.iterrows()]


def _daily_baseline(code):
    """전일까지 20일 평균거래량·전일종가·전일고가."""
    try:
        d = get_ohlcv(code, period_days=40)
        if d is None or len(d) < 5:
            return None
        return {
            'avg20_vol': float(d['Volume'].iloc[-20:].mean()),
            'prev_close': float(d['Close'].iloc[-1]),
            'prev_high': float(d['High'].iloc[-1]),
        }
    except Exception:
        return None


def intraday_signals(code, market, base=None):
    """당일 5분봉 점화 신호. 데이터 없으면 None."""
    import yfinance as yf
    sym = _yf_symbol(code, market)
    try:
        bars = yf.Ticker(sym).history(period='1d', interval='5m')
    except Exception:
        return None
    if bars is None or len(bars) < 2:
        return None
    base = base or _daily_baseline(code)
    if not base:
        return None
    o = float(bars['Open'].iloc[0])
    last = float(bars['Close'].iloc[-1])
    hi = float(bars['High'].max())
    cum_vol = float(bars['Volume'].sum())
    n = len(bars)
    elapsed = max(n / BARS_FULL_DAY, 0.05)
    # VWAP
    tp = (bars['High'] + bars['Low'] + bars['Close']) / 3
    vwap = float((tp * bars['Volume']).sum() / bars['Volume'].sum()) if bars['Volume'].sum() else last
    # 15분(3봉) 모멘텀
    accel = (last / float(bars['Close'].iloc[-4]) - 1) if n >= 4 else (last / o - 1)
    pace = cum_vol / (base['avg20_vol'] * elapsed) if base['avg20_vol'] else 0.0
    day_chg = (last / base['prev_close'] - 1) if base['prev_close'] else 0.0
    return {
        'code': code, 'market': market, 'last': last,
        'day_chg': day_chg, 'open_chg': last / o - 1 if o else 0,
        'vol_pace': pace, 'vwap_dev': (last / vwap - 1) if vwap else 0,
        'above_vwap': last >= vwap, 'accel': accel,
        'breakout': last > base['prev_high'], 'bars': n,
    }


def ignition_tier(s):
    """점화 등급. 미점화면 None."""
    if s['day_chg'] >= 0.05 and s['vol_pace'] >= 3 and s['above_vwap'] and s['accel'] > 0:
        return ('T1', '강한점화')
    if s['day_chg'] >= 0.03 and s['vol_pace'] >= 2 and s['above_vwap']:
        return ('T2', '점화')
    if s['vol_pace'] >= 2 and s['breakout']:
        return ('T3', '초기징후(거래량선행)')
    return None


def _eod_combined_map(scan_date):
    """해당일 EOD 스캔 점수(score_history) — 장중 포착 종목의 EOD 미포착 여부 판별."""
    p = SCRIPT_DIR / 'score_history.json'
    if not p.exists():
        return {}
    try:
        recs = json.load(open(p, encoding='utf-8'))['records']
        return {r['ticker']: r.get('combined', 0)
                for r in recs if r.get('scan_date') == scan_date}
    except Exception:
        return {}


def scan_once(market='ALL', top_n=200):
    from data_fetcher import get_name
    uni = build_universe(market, top_n)
    today = datetime.now().strftime('%Y-%m-%d')
    eod = _eod_combined_map(today)
    fired, controls = [], []
    for i, (code, name, mkt) in enumerate(uni):
        s = intraday_signals(code, mkt)
        if not s:
            continue
        tier = ignition_tier(s)
        if tier:
            s['name'] = name
            s['tier'], s['tier_label'] = tier
            s['eod_combined'] = eod.get(code)
            # EOD 미포착(무징후) = 회수 대상
            s['eod_missed'] = (s['eod_combined'] is None or s['eod_combined'] < 40)
            fired.append(s)
        elif random.random() < CONTROL_RATE:
            # 비점화 대조군: 정확한 base rate 산출용(선택편향 제거)
            s['name'] = name
            controls.append(s)
        if (i + 1) % 50 == 0:
            print(f'  …{i+1}/{len(uni)} 스캔 (점화 {len(fired)})', file=sys.stderr, flush=True)
    order = {'T1': 0, 'T2': 1, 'T3': 2}
    fired.sort(key=lambda x: (order[x['tier']], -x['vol_pace']))
    return fired, controls, len(uni)


def log_ignitions(fired, controls, scan_date):
    """점화·대조군을 추적 로그에 기록(outcome=None). 같은 날 같은 종목 1회만(강한 tier 우선)."""
    data = {'records': []}
    if LOG_FILE.exists():
        try:
            data = json.load(open(LOG_FILE, encoding='utf-8'))
        except Exception:
            data = {'records': []}
    seen = {(r['scan_date'], r['ticker']) for r in data['records']}

    def mk(s, is_ig):
        return {
            'scan_date': scan_date,
            'logged_at': datetime.now().strftime('%H:%M'),
            'ticker': s['code'], 'name': s.get('name', s['code']),
            'is_ignition': is_ig,
            'tier': s.get('tier'),
            'ig_price': s['last'],
            'day_chg': round(s['day_chg'], 4),
            'vol_pace': round(s['vol_pace'], 2),
            'vwap_dev': round(s['vwap_dev'], 4),
            'accel': round(s['accel'], 4),
            'breakout': bool(s['breakout']),
            'eod_missed': bool(s.get('eod_missed', True)),
            'surged_by_3d': None, 'surged_by_5d': None,
        }

    added = 0
    for s in fired:
        k = (scan_date, s['code'])
        if k in seen:
            continue
        seen.add(k); data['records'].append(mk(s, True)); added += 1
    for s in controls:
        k = (scan_date, s['code'])
        if k in seen:
            continue
        seen.add(k); data['records'].append(mk(s, False)); added += 1
    data['records'] = data['records'][-20000:]
    json.dump(data, open(LOG_FILE, 'w', encoding='utf-8'), ensure_ascii=False)
    return added


def render(fired, n_uni):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    L = [f"■ 장중 실시간 점화 스캐너 — {now} (유니버스 {n_uni})",
         "  무징후 급등(EOD 미포착) 당일 회수용 · ⚡=EOD 못뽑은 종목"]
    if not fired:
        L.append("  (현재 점화 종목 없음 — 장중 변동성 낮음 또는 장외 시간)")
        return "\n".join(L)
    L.append("─" * 64)
    L.append("  Tier 종목            등락(전일비)  거래량페이스  VWAP  최근15분  EOD")
    for s in fired:
        flag = "⚡미포착" if s['eod_missed'] else f"포착({s['eod_combined']})"
        vw = "▲" if s['above_vwap'] else "▽"
        bo = " 🚀돌파" if s['breakout'] else ""
        L.append(f"  {s['tier']} {s['name'][:10]:10s}  {s['day_chg']*100:+5.1f}%   "
                 f"{s['vol_pace']:4.1f}x  {vw}  {s['accel']*100:+4.1f}%  {flag}{bo}")
    L.append("─" * 64)
    miss = [s for s in fired if s['eod_missed']]
    L.append(f"  점화 {len(fired)}종목 · 그중 EOD 미포착(무징후 회수) {len(miss)}종목")
    L.append("  ※ 장중 점화는 후속 추적으로 정밀도 검증 필요(실시간 신호=참고).")
    return "\n".join(L)


def main():
    args = sys.argv[1:]
    top_n = 200
    if '--top' in args:
        top_n = int(args[args.index('--top') + 1])
    do_email = '--email' in args
    loop_min = None
    if '--loop' in args:
        loop_min = int(args[args.index('--loop') + 1])

    force = '--force' in args   # 신선도 가드 무시(테스트/드라이런)

    def run():
        live = force or live_session_today()
        fired, controls, n = scan_once(top_n=top_n)
        report = render(fired, n)
        print("\n" + report + "\n")
        if live:
            scan_date = datetime.now().strftime('%Y-%m-%d')
            added = log_ignitions(fired, controls, scan_date)
            print(f"  [추적] 로그 기록 {added}건(점화 {len(fired)}+대조 {len(controls)}) "
                  f"→ intraday_eval.py로 정밀도 검증", file=sys.stderr)
        else:
            print("  [추적] 실시간 세션 아님(휴장/장외) — 로깅 생략", file=sys.stderr)
        if do_email and fired:
            try:
                from email_sender import send_report
                send_report(subject=f"[장중점화] {datetime.now():%H:%M} {len(fired)}종목",
                            body=report)
                print("  [이메일] 발송 완료", file=sys.stderr)
            except Exception as e:
                print(f"  [이메일] 스킵: {e}", file=sys.stderr)
        return fired

    if loop_min:
        from datetime import time as _t
        OPEN, CLOSE = _t(9, 0), _t(15, 35)
        print(f"[장중 점화 루프] {loop_min}분 간격 · 09:00~15:35 자동 종료", file=sys.stderr)
        while True:
            now_t = datetime.now().time()
            if now_t > CLOSE:
                print("[장중 점화 루프] 장 마감 — 종료", file=sys.stderr)
                break
            if OPEN <= now_t <= CLOSE:
                run()
            time.sleep(loop_min * 60)
    else:
        run()


if __name__ == '__main__':
    main()
