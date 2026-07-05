"""
순방향 성과 추적 (Forward Outcome Tracker)

흐름:
  1. daily_scan.py 실행 → record_scores()  : 오늘 점수를 score_history.json에 저장
  2. daily_learner.py 실행 → update_outcomes(): 미확정 기록을 OHLCV로 직접 라벨링
  3. 20 거래일 경과 후 compute_stats()로 점수대별 적중률 계산
  4. build_tracker_section()으로 이메일 섹션 생성

적중 기준: 점수 기록 이후 N 거래일 내 해당 종목이 단일 거래일 10%+ 급등
           (종가/전일종가, |인접변동|>30%는 분할/권리락 아티팩트로 제외)

라벨링 방식: record-pull — 각 미확정 기록의 스캔일 이후 OHLCV를 직접 조회해 판정.
  기존 push 방식('그날 급등 리스트'에 의존)은 학습기 실행 공백일의 급등을
  놓치고 False로 오기록했으며, 공휴일을 거래일로 세서 명절 근처 윈도우가
  일찍 만료됐다. pull 방식은 실제 거래일 인덱스를 세므로 두 문제가 없다.
"""

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
HISTORY_FILE = SCRIPT_DIR / 'score_history.json'

WINDOWS = [3, 5, 10, 20]   # 추적 윈도우 (거래일)

# 급등 정의 (단일 소스 — relabel_outcomes.py도 이 값을 사용)
SURGE_PCT   = 0.10   # 단일일 종가/전일종가 +10%
PRICE_LIMIT = 0.30   # KRX 일간 상하한 — 초과 인접변동 = 분할/권리락 아티팩트(무수정주가)
HORIZONS    = {'surged_by_3d': 3, 'surged_by_5d': 5,
               'surged_by_10d': 10, 'surged_by_20d': 20}


def _market_regime():
    """시장 레짐(stock_market 브리핑 net 게이지, -1~+1) 1회 조회.
    하위 패키지 config 이름 충돌 방지를 위해 cwd 격리 서브프로세스로 실행.
    실패 시 None (모델이 결측 처리)."""
    smdir = SCRIPT_DIR.parent / 'stock_market'
    code = ('import warnings;warnings.filterwarnings("ignore");'
            'from config import INDICATORS;from fetch import collect;'
            'from scoring import compute_gauge;'
            'print(round(compute_gauge(INDICATORS,collect(INDICATORS))["net"],4))')
    try:
        out = subprocess.run(['python3', '-c', code], cwd=str(smdir),
                             capture_output=True, text=True, timeout=120)
        return float(out.stdout.strip().splitlines()[-1])
    except Exception:
        return None

BANDS = [
    ('세력80+',  80, 101, 'seoryeok'),
    ('세력70+',  70,  80, 'seoryeok'),
    ('세력60+',  60,  70, 'seoryeok'),
    ('세력40+',  40,  60, 'seoryeok'),
    ('세력~40',   0,  40, 'seoryeok'),
]


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def label_from_ohlcv(df, scan_date: str, n: int) -> bool | None:
    """scan_date 이후 n거래일 내 단일일 10%+(종가/전일종가) 급등 여부.

    Returns: True(급등) / False(윈도우 만료, 급등 없음) / None(윈도우 미성숙)
    |인접변동|>PRICE_LIMIT(30%)는 분할/권리락 데이터 아티팩트로 급등 판정에서
    제외하고 가격 스케일만 이월한다. 실제 OHLCV 인덱스를 세므로 공휴일 자동 처리.
    """
    idx = [i for i, d in enumerate(df.index) if str(d.date()) <= scan_date]
    if not idx:
        return None
    pos = idx[-1]
    fut = df.iloc[pos + 1: pos + 1 + n]
    if len(fut) < n:
        return None   # 윈도우 미성숙
    prev = float(df['Close'].iloc[pos])
    for j in range(len(fut)):
        c = float(fut['Close'].iloc[j])
        if prev > 0:
            chg = c / prev - 1
            if abs(chg) > PRICE_LIMIT:   # 아티팩트 → 급등 판정 제외, 스케일만 이월
                prev = c
                continue
            if chg >= SURGE_PCT:
                return True
        prev = c
    return False


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

    regime = _market_regime()   # 시장 레짐 1회 조회(전 종목 공통)
    print(f'  [추적] 시장 레짐(net): {regime}')

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
            'ret5':          r.get('ret5', 0.0),   # 5일 낙폭(GBM 피처, 검증 +14%p AUC)
            'ret20':         r.get('ret20', 0.0),  # 20일 모멘텀(GBM 피처)
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
            'near_bull_ob':  bool((r.get('ob') or {}).get('near_bull', False)),
            'near_bear_ob':  bool((r.get('ob') or {}).get('near_bear', False)),
            'market_regime': regime,   # 시장 레짐(net) — 개별종목×시장국면 학습용
            'below_gate':    bool(r.get('below_gate', False)),  # 탈락 유니버스 샘플 여부
            'surge_prob':    r.get('surge_prob'),               # GBM 급등확률(3일)
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

def update_outcomes(surge_tickers: list[str] | None = None,
                    surge_date: str | None = None):
    """
    미확정(None) 라벨 기록을 record-pull 방식으로 직접 라벨링.

    각 기록의 스캔일 이후 OHLCV를 조회해 N거래일 내 단일일 10%+ 급등 여부를
    판정한다. 실행 공백일이 있어도 놓친 급등을 다음 실행 때 그대로 복원하고,
    공휴일도 실제 거래일 인덱스로 자동 처리한다.

    surge_tickers/surge_date: 구 push 방식 하위호환용 — 더 이상 사용하지 않음.
    """
    data    = _load()
    pending = [r for r in data['records']
               if any(r.get(h) is None for h in HORIZONS)]
    if not pending:
        return

    try:
        from data_fetcher import get_ohlcv
    except Exception as e:
        print(f'  [추적] 라벨링 스킵 (data_fetcher 로드 실패: {e})')
        return

    cache: dict = {}
    updated = 0
    failed  = 0
    for rec in pending:
        ticker = rec.get('ticker')
        scan_d = rec.get('scan_date')
        if not ticker or not scan_d:
            continue
        if ticker not in cache:
            try:
                cache[ticker] = get_ohlcv(ticker, period_days=400)
            except Exception:
                cache[ticker] = None
        df = cache[ticker]
        if df is None or len(df) < 5:
            failed += 1
            continue
        for h, n in HORIZONS.items():
            if rec.get(h) is not None:
                continue
            label = label_from_ohlcv(df, scan_d, n)
            if label is not None:
                rec[h] = label
                updated += 1

    _save(data)
    if updated or failed:
        print(f'  [추적] 라벨 확정: {updated}건 '
              f'(대상 {len(pending)}기록, 조회실패 {failed}종목)')


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
        '  ⚠ 우측절단(right-censoring): 모든 호라이즌 완료(20일 경과) 종목만 집계 →',
        '     표 전체가 과거 코호트 기준. 최근 스캔 제외돼 장기(10·20일) 적중률은 낙관 편향 가능.',
        '     신뢰는 단기(3일) 우선. 세력점수↔적중 단조성도 80+·단기에서만 뚜렷.',
        sep,
    ]
    return lines
