# -*- coding: utf-8 -*-
"""
장중 점화 정밀도 평가 — intraday_log.json의 점화·대조군을 향후 급등으로 채점.

급등 정의: EOD 시스템과 동일(점화일 이후 N거래일 내 단일일 10%+ 상승).
대조군(비점화) 급등률을 base로 삼아 점화의 lift를 산출 → 선택편향 없는 정밀도.

사용:
  python3 intraday_eval.py            # outcome 채움 + lift 리포트
"""
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
warnings.filterwarnings('ignore')

from data_fetcher import get_ohlcv

LOG_FILE = Path(__file__).parent / 'intraday_log.json'
SURGE_PCT = 0.10
MIN_AGE_DAYS = 5   # outcome 윈도우(5거래일) 경과 확보용 달력일 여유


def _surged_within(ticker, scan_date, ndays):
    """scan_date 이후 ndays 거래일 내 단일일 10%+ 상승 발생 여부. 데이터 부족이면 None."""
    try:
        df = get_ohlcv(ticker, period_days=400)
        if df is None or len(df) < 5:
            return None
        idx = [i for i, d in enumerate(df.index) if str(d.date()) <= scan_date]
        if not idx:
            return None
        pos = idx[-1]
        fut = df.iloc[pos + 1: pos + 1 + ndays]
        if len(fut) < ndays:
            return None   # 아직 윈도우 미완성
        # 단일일 등락률(전일종가 대비)이 윈도우 내 한 번이라도 10%+ 면 급등
        prev = df['Close'].iloc[pos]
        max_ret = -1.0
        for j in range(len(fut)):
            cur = fut['Close'].iloc[j]
            max_ret = max(max_ret, cur / prev - 1)
            prev = cur
        return 1 if max_ret >= SURGE_PCT else 0
    except Exception:
        return None


def fill_outcomes():
    if not LOG_FILE.exists():
        print('로그 없음 — 먼저 intraday_scanner.py 실행 필요')
        return None
    data = json.load(open(LOG_FILE, encoding='utf-8'))
    recs = data['records']
    today = datetime.now().date()
    filled = 0
    for r in recs:
        try:
            d = datetime.strptime(r['scan_date'], '%Y-%m-%d').date()
        except Exception:
            continue
        age = (today - d).days
        for nd, key in ((3, 'surged_by_3d'), (5, 'surged_by_5d')):
            if r.get(key) is None and age >= MIN_AGE_DAYS:
                o = _surged_within(r['ticker'], r['scan_date'], nd)
                if o is not None:
                    r[key] = bool(o); filled += 1
    json.dump(data, open(LOG_FILE, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'outcome 채움 {filled}건')
    return recs


def report(recs):
    def rate(sub, key):
        done = [r for r in sub if r.get(key) is not None]
        if not done:
            return None, 0
        return sum(1 for r in done if r[key]) / len(done), len(done)

    for key, hz in (('surged_by_3d', '3일'), ('surged_by_5d', '5일')):
        ig = [r for r in recs if r['is_ignition']]
        ct = [r for r in recs if not r['is_ignition']]
        ig_r, ig_n = rate(ig, key)
        ct_r, ct_n = rate(ct, key)
        print(f'\n■ {hz} 내 10%+ 급등  (점화 채점 {ig_n} / 대조 {ct_n})')
        if ct_r is None or ig_r is None:
            print('  outcome 미완성 — 5거래일 경과 후 재실행 필요')
            continue
        base = ct_r if ct_r > 0 else 0.001
        print(f'  대조군(비점화) 급등률   {ct_r*100:4.1f}%  (base)')
        print(f'  점화 전체 급등률        {ig_r*100:4.1f}%  lift {ig_r/base:4.2f}x')
        # tier별
        for t in ('T1', 'T2', 'T3'):
            r_, n_ = rate([r for r in ig if r['tier'] == t], key)
            if r_ is not None and n_ >= 3:
                print(f'    {t} 급등률           {r_*100:4.1f}%  lift {r_/base:4.2f}x  (n={n_})')
        # EOD 미포착(무징후 회수) 부분집합
        miss_r, miss_n = rate([r for r in ig if r['eod_missed']], key)
        if miss_r is not None and miss_n >= 3:
            print(f'    ⚡EOD미포착(무징후)   {miss_r*100:4.1f}%  lift {miss_r/base:4.2f}x  (n={miss_n})')


if __name__ == '__main__':
    recs = fill_outcomes()
    if recs:
        n_ig = sum(1 for r in recs if r['is_ignition'])
        n_ct = len(recs) - n_ig
        print(f'\n총 로그 {len(recs)}건 (점화 {n_ig} / 대조 {n_ct})')
        report(recs)
        print('\n※ 표본이 쌓일수록(특히 대조군) lift 신뢰도↑. 매 장중 스캔이 로그 누적.')
