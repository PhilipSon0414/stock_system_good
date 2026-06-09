"""
순방향 성과 추적 (Forward Outcome Tracker)

흐름:
  1. daily_scan.py 실행 → record_scores()  : 오늘 점수를 score_history.json에 저장
  2. daily_learner.py 실행 → update_outcomes(): 오늘 급등 종목으로 과거 예측 결과 업데이트
  3. 20 거래일 경과 후 compute_stats()로 점수대별 적중률 계산
  4. build_tracker_section()으로 이메일 섹션 생성

적중 기준: 점수 기록 이후 N 거래일 내 해당 종목이 단일 거래일 10%+ 급등
거래일 계산: 주말 제외 (공휴일은 미적용, 근사값)
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
HISTORY_FILE = SCRIPT_DIR / 'score_history.json'

WINDOWS = [3, 5, 10, 20]   # 추적 윈도우 (거래일)

BANDS = [
    ('세력80+',  80, 101, 'seoryeok'),
    ('세력70+',  70,  80, 'seoryeok'),
    ('세력60+',  60,  70, 'seoryeok'),
    ('세력40+',  40,  60, 'seoryeok'),
    ('세력~40',   0,  40, 'seoryeok'),
]


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _td_between(d1: str, d2: str) -> int:
    """d1(포함) ~ d2(미포함) 거래일 수 (주말 제외)"""
    dt1 = datetime.strptime(d1, '%Y-%m-%d').date()
    dt2 = datetime.strptime(d2, '%Y-%m-%d').date()
    if dt2 <= dt1:
        return 0
    count, cur = 0, dt1
    while cur < dt2:
        if cur.weekday() < 5:
            count += 1
        cur += timedelta(days=1)
    return count


def _load() -> dict:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {'records': []}


def _save(data: dict):
    # numpy int64/float32 등을 Python 기본 타입으로 변환
    def _convert(obj):
        if hasattr(obj, 'item'):   # numpy scalar
            return obj.item()
        raise TypeError(f'Not serializable: {type(obj)}')

    HISTORY_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_convert),
        encoding='utf-8',
    )


# ── 1. 점수 기록 ──────────────────────────────────────────────────────────────

def record_scores(results: list, scan_date: str | None = None):
    """
    daily_scan.py 결과를 score_history.json에 기록.
    scan_date: 'YYYY-MM-DD' (None이면 오늘)
    results: daily_scan.py의 analyze_one() 반환 dict 목록
    """
    if scan_date is None:
        scan_date = datetime.now().strftime('%Y-%m-%d')

    data = _load()
    existing = {(r['scan_date'], r['ticker']) for r in data['records']}

    new_count = 0
    for r in results:
        if (scan_date, r['ticker']) in existing:
            continue

        # 이진 신호 (ML 학습용 — df에서 직접 추출)
        df = r.get('df')
        latest = df.iloc[-1] if df is not None and len(df) > 0 else {}
        def _flag(col): return bool(latest.get(col, False)) if hasattr(latest, 'get') else False

        data['records'].append({
            'scan_date':     scan_date,
            'ticker':        r['ticker'],
            'name':          r.get('name', r['ticker']),
            'price':         r.get('price', 0),
            'combined':      r.get('combined', 0),
            'raw_combined':  r.get('raw_combined', r.get('combined', 0)),
            'seoryeok':      r.get('seoryeok', 0),
            'surge':         r.get('surge', 0),
            'investor':      r.get('investor', 0),
            'pattern':       r.get('pattern', 0),
            # ── ML 피처 (binary signals) ──────────────────────────
            'vol_ratio':     r.get('vol_ratio', 0),
            'se_entry':      _flag('SE_Entry'),
            'se_accum':      _flag('SE_Accum'),
            'se_exit':       _flag('SE_Exit'),
            'ma_bull':       _flag('MaBull'),
            'golden_cross':  _flag('GoldenCross'),
            'nr7':           _flag('NR7'),
            'vcp':           _flag('VCP'),
            'obv_diverge':   _flag('OBV_Diverge'),
            'high52w_dist':  float(latest.get('High52W_Dist', 1.0)) if hasattr(latest, 'get') else 1.0,
            'consec_days':   r.get('consec_days', 0),
            'sector_rs':     (r.get('sector_rs') or {}).get('rs_score', 0),
            # ── 결과 추적 ─────────────────────────────────────────
            'surged_by_3d':  None,
            'surged_by_5d':  None,
            'surged_by_10d': None,
            'surged_by_20d': None,
        })
        new_count += 1

    _save(data)
    total = len(data['records'])
    print(f'  [추적] {scan_date} 점수 기록: 신규 {new_count}개 (누적 {total}개)')


# ── 2. 결과 업데이트 ──────────────────────────────────────────────────────────

def update_outcomes(surge_tickers: list[str], surge_date: str | None = None):
    """
    daily_learner.py가 수집한 급등 종목으로 과거 예측 기록 업데이트.

    surge_date 날 10%+ 급등한 종목이 과거 5/10/20 거래일 내 스캔된 적 있으면
    해당 기록에 surged_by_Nd = True 마킹.
    윈도우가 만료된 기록(아직 null)은 False 처리.

    surge_date: 'YYYY-MM-DD' (None이면 오늘)
    surge_tickers: 당일 급등 종목 코드 목록
    """
    if surge_date is None:
        surge_date = datetime.now().strftime('%Y-%m-%d')

    surge_set = set(surge_tickers)
    data      = _load()
    updated   = 0

    for rec in data['records']:
        td = _td_between(rec['scan_date'], surge_date)
        if td <= 0:
            continue

        surged = rec['ticker'] in surge_set

        if 0 < td <= 3  and rec.get('surged_by_3d')  is None and surged:
            rec['surged_by_3d']  = True; updated += 1
        if 0 < td <= 5  and rec.get('surged_by_5d')  is None and surged:
            rec['surged_by_5d']  = True; updated += 1
        if 0 < td <= 10 and rec.get('surged_by_10d') is None and surged:
            rec['surged_by_10d'] = True; updated += 1
        if 0 < td <= 20 and rec.get('surged_by_20d') is None and surged:
            rec['surged_by_20d'] = True; updated += 1

        # 윈도우 만료 → False
        if td > 3  and rec.get('surged_by_3d')  is None:
            rec['surged_by_3d']  = False
        if td > 5  and rec.get('surged_by_5d')  is None:
            rec['surged_by_5d']  = False
        if td > 10 and rec.get('surged_by_10d') is None:
            rec['surged_by_10d'] = False
        if td > 20 and rec.get('surged_by_20d') is None:
            rec['surged_by_20d'] = False

    _save(data)
    if updated:
        print(f'  [추적] 급등 결과 반영: {updated}건 업데이트 ({surge_date})')


# ── 3. 통계 계산 ──────────────────────────────────────────────────────────────

def compute_stats() -> dict | None:
    """점수대별 급등 적중률 계산. 완료 기록이 없으면 None."""
    data    = _load()
    records = data['records']

    # surged_by_20d 가 확정된 기록만 사용
    completed = [r for r in records if r['surged_by_20d'] is not None]
    if not completed:
        return None

    def _rate(g, key):
        return sum(1 for r in g if r.get(key)) / len(g) if g else None

    stats = {}
    for label, lo, hi, score_key in BANDS:
        group = [r for r in completed if lo <= r.get(score_key, r.get('combined', 0)) < hi]
        if not group:
            stats[label] = None
            continue

        g3  = [r for r in group if r.get('surged_by_3d')  is not None]
        g5  = [r for r in group if r.get('surged_by_5d')  is not None]
        g10 = [r for r in group if r.get('surged_by_10d') is not None]
        g20 = group

        stats[label] = {
            'count':   len(group),
            'hit_3d':  _rate(g3,  'surged_by_3d'),
            'hit_5d':  _rate(g5,  'surged_by_5d'),
            'hit_10d': _rate(g10, 'surged_by_10d'),
            'hit_20d': _rate(g20, 'surged_by_20d'),
            'n_3d':    len(g3),
            'n_5d':    len(g5),
            'n_10d':   len(g10),
            'n_20d':   len(g20),
        }

    return {
        'bands':           stats,
        'total_completed': len(completed),
        'total_records':   len(records),
        'oldest_scan':     min(r['scan_date'] for r in records) if records else None,
        'latest_scan':     max(r['scan_date'] for r in records) if records else None,
    }


# ── 4. 이메일 섹션 생성 ───────────────────────────────────────────────────────

def build_tracker_section() -> list[str]:
    """daily_scan.py 이메일에 추가할 점수 적중률 섹션 반환."""
    sep  = '═' * 70
    sep2 = '─' * 70
    data = _load()

    stats = compute_stats()

    if stats is None or stats['total_completed'] == 0:
        n = len(data['records'])
        return [
            '',
            sep2,
            '  [ 점수 적중률 추적 — 데이터 누적 중 ]',
            f'  현재 {n}개 기록 중. 20 거래일(약 4주) 경과 후 통계 산출됩니다.',
            sep2,
        ]

    def _fmt(rate, n):
        if rate is None or n == 0:
            return '   -  '
        return f'{rate * 100:4.1f}%({n})'

    lines = [
        '',
        sep,
        '  [ 세력 점수대별 급등(10%+) 적중률 — ML 학습 데이터 누적 현황 ]',
        f'  분석 기간: {stats["oldest_scan"]} ~ {stats["latest_scan"]}'
        f'  |  완료: {stats["total_completed"]}건 / 전체: {stats["total_records"]}건',
        sep2,
        f'  {"세력점수대":<10} {"종목수":>5}  {"3일내":>10} {"5일내":>10} {"10일내":>10} {"20일내":>10}',
        sep2,
    ]

    for label, band_stats in stats['bands'].items():
        if band_stats is None:
            lines.append(f'  {label:<10}    0건  (데이터 없음)')
            continue
        lines.append(
            f'  {label:<10} {band_stats["count"]:>4}건'
            f'  {_fmt(band_stats["hit_3d"],  band_stats["n_3d"]):>10}'
            f'  {_fmt(band_stats["hit_5d"],  band_stats["n_5d"]):>10}'
            f'  {_fmt(band_stats["hit_10d"], band_stats["n_10d"]):>10}'
            f'  {_fmt(band_stats["hit_20d"], band_stats["n_20d"]):>10}'
        )

    lines += [
        sep2,
        '  ※ 기준: 스캔일 이후 N 거래일 내 단일일 10%+ 급등 발생 여부',
        '  ※ 세력80+합산45+ Tier-S: 예상 적중률 18.6% (2.68x lift) — 핵심 필터',
        sep,
    ]
    return lines
