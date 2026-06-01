"""
매일 오후 5시 실행: 당일 급등 종목 학습

흐름:
  1. pykrx로 오늘 20%+ 급등 종목 수집
  2. 각 종목의 전일 상태(OHLCV에서 오늘 캔들 제거) 기준으로
     세력흔적 / 급등임박 / 수급 / 차트패턴 4가지 점수 산출
  3. 관측치를 learning_log.json에 누적 저장
  4. 관측치 10개 이상이면 model_updater로 가중치 자동 업데이트
  5. 이메일 발송

직접 실행:
  python3 daily_learner.py                   # 오늘
  python3 daily_learner.py 20260511          # 특정일
  python3 daily_learner.py 20260511 20260512 # 복수일 (순서대로)
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_fetcher import get_ohlcv, get_name, get_ticker_list
from indicators import add_all
from seoryeok import analyze
from order_block import get_order_blocks
from scorer import score as seoryeok_score
from surge_predictor import score_surge, combined_score
from investor_flow import score_investors
from chart_patterns import score_patterns
from model_updater import update_model, load_learning_log, save_learning_log
from email_sender import send_report
from forward_tracker import update_outcomes

SCRIPT_DIR       = Path(__file__).parent
LOG_DIR          = SCRIPT_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)

MIN_SURGE_PCT        = 15.0   # 학습 대상 최소 급등률 (%)
MIN_OBS_FOR_UPDATE   = 10     # 가중치 업데이트 최소 관측 수
MAX_ANALYZE_STOCKS   = 50     # 분석 최대 종목 수 (과부하 방지)


# ── 1. 급등 종목 수집 ──────────────────────────────────────────────────────────

def get_surge_tickers(min_pct: float = MIN_SURGE_PCT, date_str: str | None = None) -> list[dict]:
    """date_str(YYYYMMDD) 날짜의 min_pct% 이상 급등 종목 목록 반환. None이면 오늘.
    pykrx 실패 시 fdr 병렬 조회로 자동 폴백.
    """
    result = _surge_via_pykrx(min_pct, date_str)
    if result is not None:
        return result
    print('  [학습] pykrx 실패 → fdr 병렬 조회로 전환')
    return _surge_via_fdr(min_pct, date_str)


def _surge_via_pykrx(min_pct: float, date_str: str | None) -> list[dict] | None:
    """pykrx로 당일 등락률 일괄 조회. 실패하면 None 반환."""
    try:
        import pandas as pd
        from pykrx import stock

        target = date_str or datetime.now().strftime('%Y%m%d')
        frames = []
        for market in ('KOSPI', 'KOSDAQ'):
            try:
                df = stock.get_market_ohlcv_by_ticker(target, market=market)
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception:
                pass

        if not frames:
            return None

        all_df = pd.concat(frames)
        all_df.index = all_df.index.astype(str).str.zfill(6)

        pct_col = next(
            (c for c in all_df.columns if '등락' in c or c in ('Change', 'change')),
            None
        )
        if pct_col is None:
            return None

        surge = all_df[all_df[pct_col] >= min_pct]
        surge = surge[~surge.index.str.endswith(('5', '7'))]
        return [
            {'ticker': str(t), 'surge_pct': float(surge.loc[t, pct_col])}
            for t in surge.index
        ]

    except Exception:
        return None


def _surge_via_fdr(min_pct: float, date_str: str | None) -> list[dict]:
    """fdr 병렬 조회로 급등 종목 탐색 (pykrx 폴백용)."""
    import pandas as pd
    import FinanceDataReader as fdr
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    if date_str:
        target_dt = datetime.strptime(date_str, '%Y%m%d').date()
    else:
        target_dt = datetime.now().date()

    # 직전 거래일(전날 기준 OHLCV 필요) 포함 14일치 조회
    start_str = (target_dt - timedelta(days=14)).strftime('%Y-%m-%d')
    end_str   = target_dt.strftime('%Y-%m-%d')

    # 전체 종목 코드 (우선주 제외)
    listing = get_ticker_list('ALL')
    codes   = listing['Code'].astype(str).str.zfill(6).tolist()
    codes   = [c for c in codes if not c.endswith(('5', '7'))]
    print(f'  [학습] fdr 병렬 조회: {len(codes)}종목  {start_str}~{end_str}', flush=True)

    results  = []
    lock     = threading.Lock()
    done     = [0]
    target_ts = pd.Timestamp(target_dt)

    def fetch(code):
        try:
            df = fdr.DataReader(code, start_str, end_str)
            if df is None or df.empty or len(df) < 2:
                return
            df.index = pd.to_datetime(df.index).normalize()
            if target_ts not in df.index:
                return
            pos = df.index.get_loc(target_ts)
            if pos == 0:
                return
            close_t  = float(df['Close'].iloc[pos])
            close_t1 = float(df['Close'].iloc[pos - 1])
            if close_t1 <= 0:
                return
            pct = (close_t - close_t1) / close_t1 * 100.0
            if pct >= min_pct:
                with lock:
                    results.append({'ticker': code, 'surge_pct': round(pct, 2)})
        except Exception:
            pass
        finally:
            with lock:
                done[0] += 1
                if done[0] % 100 == 0:
                    print(f'  [학습] {done[0]}/{len(codes)} 처리... 급등 {len(results)}건', flush=True)

    # 스레드 수를 줄여 네트워크 부하 완화
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(fetch, c) for c in codes]
        for _ in as_completed(futures):
            pass

    print(f'  [학습] fdr 조회 완료: {len(codes)}종목 → 급등 {len(results)}건', flush=True)
    return results


# ── 2. 전일 기준 분석 ─────────────────────────────────────────────────────────

def analyze_prev_day(ticker: str, surge_date_str: str | None = None) -> dict | None:
    """
    급등일 전일 상태로 4가지 점수 분석.
    surge_date_str(YYYYMMDD)이 주어지면 해당일 이전 캔들만 사용,
    없으면 마지막 행(오늘) 제거.
    """
    try:
        df_full = get_ohlcv(ticker, period_days=400)
        if len(df_full) < 62:
            return None

        # 급등일 이전 캔들만 사용 → 전일 시점 재현
        if surge_date_str:
            cutoff = datetime.strptime(surge_date_str, '%Y%m%d').date()
            mask   = [idx.date() < cutoff for idx in df_full.index]
            df = df_full.loc[mask].copy()
            if len(df) < 62:
                return None
        else:
            df = df_full.iloc[:-1].copy()

        df  = add_all(df)
        df  = analyze(df)
        ob  = get_order_blocks(df)

        s_pts,   s_tags   = seoryeok_score(df, ob)
        g_pts,   g_tags   = score_surge(df, ob)
        inv_pts, inv_tags = score_investors(ticker, df)
        pat_pts, pat_tags = score_patterns(df)
        combo             = combined_score(s_pts, g_pts, inv_pts, pat_pts)

        latest = df.iloc[-1]
        w5     = df.iloc[-5:]

        def _any(col):
            return bool(w5[col].any()) if col in w5.columns else False

        features = {
            'SE_Entry':    _any('SE_Entry'),
            'SE_Accum':    _any('SE_Accum'),
            'SE_BearTrap': _any('SE_BearTrap'),
            'SE_Rally':    _any('SE_Rally'),
            'SE_Exit':     _any('SE_Exit'),
            'MaBull':      bool(latest.get('MaBull',     False)),
            'MaBear':      bool(latest.get('MaBear',     False)),
            'MaConverge':  bool(latest.get('MaConverge', False)),
            'GoldenCross': _any('GoldenCross'),
            'near_bull_ob': bool(ob.get('near_bull', False)),
            'near_bear_ob': bool(ob.get('near_bear', False)),
            'VolRatio':    float(latest.get('VolRatio', 0) or 0),
        }

        return {
            'ticker':   ticker,
            'name':     get_name(ticker),
            'scores': {
                'seoryeok': s_pts,
                'surge':    g_pts,
                'investor': inv_pts,
                'pattern':  pat_pts,
                'combined': combo,
            },
            'features': features,
            'tags': {
                'seoryeok': s_tags[:3],
                'surge':    g_tags[:3],
                'investor': [t for t in inv_tags if not t.startswith('수급데이터')][:3],
                'pattern':  pat_tags[:3],
            },
        }

    except Exception as e:
        print(f'  [학습] {ticker} 분석 실패: {e}')
        return None


# ── 3. 학습 메인 루프 ─────────────────────────────────────────────────────────

def run_learning(date_str: str | None = None) -> dict | None:
    """date_str: YYYYMMDD 형식. None이면 오늘."""
    if date_str:
        dt = datetime.strptime(date_str, '%Y%m%d')
        today_str = dt.strftime('%Y-%m-%d')
        pykrx_str = date_str
    else:
        today_str = datetime.now().strftime('%Y-%m-%d')
        pykrx_str = None

    print(f'\n{"="*60}')
    print(f'  일일 급등 학습 시작: {today_str}')
    print(f'{"="*60}')

    # 급등 종목 수집
    print(f'\n  {today_str} {MIN_SURGE_PCT:.0f}%+ 급등 종목 조회 중...')
    surge_list = get_surge_tickers(MIN_SURGE_PCT, date_str=pykrx_str)

    if not surge_list:
        print('  급등 종목 없음 (또는 데이터 조회 실패)')
        update_outcomes([], surge_date=today_str)   # 윈도우 만료 기록 False 처리
        return None

    # 과거 예측 기록 업데이트 (오늘 급등 종목으로 역반영)
    update_outcomes([s['ticker'] for s in surge_list], surge_date=today_str)

    # 최대 MAX_ANALYZE_STOCKS개, 급등률 내림차순
    surge_list.sort(key=lambda x: x['surge_pct'], reverse=True)
    surge_list = surge_list[:MAX_ANALYZE_STOCKS]

    print(f'  급등 종목 {len(surge_list)}개 발견')
    for s in surge_list[:5]:
        print(f'    {s["ticker"]}  +{s["surge_pct"]:.1f}%')
    if len(surge_list) > 5:
        print(f'    ... 외 {len(surge_list)-5}개')

    # 전일 상태 분석
    print(f'\n  전일 지표 분석 중...')
    log      = load_learning_log()
    # 이미 기록된 (날짜, 종목) 중복 방지
    existing = {(o['date'], o['ticker']) for o in log['observations']}
    new_obs  = []

    for i, item in enumerate(surge_list, 1):
        print(f'  [{i:2d}/{len(surge_list)}] {item["ticker"]}  +{item["surge_pct"]:.1f}%  ', end='\r')
        if (today_str, item['ticker']) in existing:
            continue   # 오늘 이미 기록된 종목 건너뜀
        result = analyze_prev_day(item['ticker'], surge_date_str=pykrx_str)
        if result:
            obs = {
                'date':      today_str,
                'ticker':    item['ticker'],
                'name':      result['name'],
                'surge_pct': round(item['surge_pct'], 2),
                'scores':    result['scores'],
                'features':  result['features'],
                'tags':      result['tags'],
            }
            new_obs.append(obs)
            log['observations'].append(obs)
        time.sleep(0.15)

    print(f'\n  분석 완료: {len(new_obs)}/{len(surge_list)}개')

    # 학습 데이터 저장
    save_learning_log(log)
    total_obs = len(log['observations'])
    print(f'  누적 관측치: {total_obs}개 저장')

    # 가중치 업데이트
    update_result = None
    if total_obs >= MIN_OBS_FOR_UPDATE:
        print(f'\n  가중치 업데이트 중... (관측치 {total_obs}개 기반)')
        update_result = update_model(log)
        save_learning_log(log)   # weight_history 포함 재저장
    else:
        remaining = MIN_OBS_FOR_UPDATE - total_obs
        print(f'\n  가중치 업데이트 보류 ({remaining}개 더 필요)')

    # ML 캘리브레이션 자동 갱신 (score_history 기반)
    recal_result = _auto_recalibrate_ml()
    if recal_result.get('updated'):
        print(f'  ML 캘리브레이션 갱신: {recal_result["n"]}건 / {recal_result["n_pos"]}양성 '
              f'/ 기준율 {recal_result["base_rate"]*100:.1f}%')

    # False Positive 분석 출력
    fp_result = _analyze_false_positives()

    return {
        'today':         today_str,
        'surge_count':   len(surge_list),
        'analyzed':      len(new_obs),
        'new_obs':       new_obs,
        'total_obs':     total_obs,
        'update_result': update_result,
        'recal_result':  recal_result,
        'fp_result':     fp_result,
    }


def _auto_recalibrate_ml() -> dict:
    """surge_ml.py의 캘리브레이션 테이블을 score_history 최신 데이터로 자동 갱신."""
    try:
        import surge_ml
        # 캐시 초기화 후 재갱신
        surge_ml._ml_cache = None
        return surge_ml.auto_recalibrate()
    except Exception as e:
        return {'updated': False, 'reason': str(e)}


def _analyze_false_positives() -> dict:
    """스캔 추천 → 급등 미발생(False Positive) 패턴 분석.

    score_history의 surged_by_5d=False 기록을 분석해
    어떤 점수 패턴이 False Positive를 많이 만드는지 파악.
    """
    try:
        import json
        from pathlib import Path
        sh_path = Path(__file__).parent / 'score_history.json'
        if not sh_path.exists():
            return {}
        records = json.loads(sh_path.read_text(encoding='utf-8')).get('records', [])
        tp = [r for r in records if r.get('surged_by_5d') is True]
        fp = [r for r in records if r.get('surged_by_5d') is False]
        if not fp or not tp:
            return {}

        import statistics as st
        tp_se = st.mean(r['seoryeok'] for r in tp)
        fp_se = st.mean(r['seoryeok'] for r in fp)
        tp_co = st.mean(r['combined'] for r in tp)
        fp_co = st.mean(r['combined'] for r in fp)

        # FP가 많은 구간 탐색 (합산 고점 + 세력 낮음)
        fp_high_co_low_se = [r for r in fp
                              if r['combined'] >= 70 and r['seoryeok'] < 60]
        fp_rate_high = len(fp_high_co_low_se) / len(fp) * 100

        result = {
            'tp_count': len(tp), 'fp_count': len(fp),
            'tp_se_avg': round(tp_se, 1), 'fp_se_avg': round(fp_se, 1),
            'tp_co_avg': round(tp_co, 1), 'fp_co_avg': round(fp_co, 1),
            'fp_high_co_low_se_pct': round(fp_rate_high, 1),
        }
        print(f'\n  [False Positive 분석]')
        print(f'  TP(급등성공) {len(tp)}건: 세력평균 {tp_se:.1f}점  합산평균 {tp_co:.1f}점')
        print(f'  FP(급등실패) {len(fp)}건: 세력평균 {fp_se:.1f}점  합산평균 {fp_co:.1f}점')
        print(f'  FP 중 "합산70+&세력<60" 고위험 패턴: {fp_rate_high:.1f}% ({len(fp_high_co_low_se)}건)')
        if fp_se < tp_se - 5:
            print(f'  → 세력점수가 낮으면 합산이 높아도 False Positive 위험')
        return result
    except Exception:
        return {}


# ── 4. 이메일 빌더 ────────────────────────────────────────────────────────────

_COMP_LABEL = {
    'seoryeok': '세력흔적  ',
    'surge':    '급등임박  ',
    'investor': '외국인/기관',
    'pattern':  '차트패턴  ',
}

_SCORE_COMMENT = {
    'seoryeok': ('세력 진입·매집 흔적이 급등 전에 얼마나 포착됐는가',),
    'surge':    ('이평선 수렴·저거래량 압축이 급등 전 잘 나타났는가',),
    'investor': ('외국인·기관 수급이 급등을 사전에 반영했는가',),
    'pattern':  ('W패턴·BB스퀴즈 등 차트 패턴이 얼마나 유효했는가',),
}


def _score_grade(score: float) -> str:
    if score >= 65:  return '우수 ★★★'
    if score >= 50:  return '양호 ★★'
    if score >= 35:  return '보통 ★'
    return '부족 ☆'


def build_email(result: dict) -> tuple[str, str]:
    today = result['today']
    upd   = result.get('update_result')

    subject = (
        f'[세력분석] 급등 학습 {today}'
        f' — {result["surge_count"]}종목'
        + (' / 모델 업데이트 완료' if upd else ' / 모델 유지')
    )

    sep  = '═' * 64
    sep2 = '─' * 64
    L    = []

    # ── 헤더 ──────────────────────────────────────────────────────────────────
    L += [sep, f'  일일 급등 학습 & 모델 업데이트 리포트  {today}', sep, '']
    L += [
        f'  오늘 {MIN_SURGE_PCT:.0f}%+ 급등 종목: {result["surge_count"]}개  |  '
        f'분석 완료: {result["analyzed"]}개  |  누적 관측치: {result["total_obs"]}개',
        '',
    ]

    # ── 급등 종목 전일 점수 테이블 ────────────────────────────────────────────
    L += [
        sep2,
        '  [ 오늘 급등 종목 — 전일(급등 하루 전) 모델 점수 ]',
        sep2,
        f'     {"종목명":<14} {"코드":<8} {"급등률":>7}'
        f'  {"합산":>5} {"세력":>5} {"급등":>5} {"수급":>5} {"패턴":>5}',
        sep2,
    ]
    for obs in sorted(result['new_obs'], key=lambda x: x['surge_pct'], reverse=True):
        sc   = obs['scores']
        flag = '★' if sc['combined'] >= 40 else '  '
        L.append(
            f'  {flag} {obs["name"]:<13} {obs["ticker"]:<8}'
            f' +{obs["surge_pct"]:5.1f}%'
            f'  {sc["combined"]:>5}점 {sc["seoryeok"]:>5}점'
            f' {sc["surge"]:>5}점 {sc["investor"]:>5}점 {sc["pattern"]:>5}점'
        )
    captured = sum(1 for o in result['new_obs'] if o['scores']['combined'] >= 40)
    total_a  = result['analyzed']
    L += [
        sep2,
        f'  ★ 합산 40점+ = 모델이 전날 이미 포착한 종목'
        f'  →  {captured}/{total_a}개 ({captured/total_a*100:.0f}%)' if total_a else '',
        '',
    ]

    # ── 모델 업데이트 결과 (핵심) ─────────────────────────────────────────────
    if upd:
        dw   = upd['default_weights']
        ow   = upd['old_weights']
        nw   = upd['new_weights']
        cs   = upd['component_stats']

        L += [
            sep,
            '  [ 모델 가중치 업데이트 결과 ]',
            sep,
            f'  학습 기준: 최근 {upd["window"]}개 관측치  '
            f'(누적 {upd["total_obs"]}개)',
            f'  전일 평균 합산점수:  {upd["avg_combined"]:.1f}점',
            f'  사전 포착률(40점+):  {upd["hit_rate"]*100:.1f}%',
            '',
            f'  {"컴포넌트":<12} {"기본값":>7}  {"이전값":>7}  {"신규값":>7}  {"변화":>8}  {"전일 평균점수"}',
            sep2,
        ]

        for comp, wkey in [
            ('seoryeok', 'W_SEORYEOK'),
            ('surge',    'W_SURGE'),
            ('investor', 'W_INVESTOR'),
            ('pattern',  'W_PATTERN'),
        ]:
            stat  = cs[comp]
            arrow = ('↑ +' if stat['weight_change'] > 0
                     else ('↓ ' if stat['weight_change'] < 0 else '→  '))
            chg   = f'{arrow}{abs(stat["weight_change"]):.4f}'
            grade = _score_grade(stat['avg_score'])
            L.append(
                f'  {_COMP_LABEL[comp]:<12}'
                f'  {dw[wkey]:.4f}   {stat["old_weight"]:.4f}'
                f'   {stat["new_weight"]:.4f}  {chg:>9}'
                f'  {stat["avg_score"]:5.1f}점 ({grade})'
            )

        L.append(sep2)

        # 변화 해석
        L += ['', '  [ 변화 해석 ]', '']
        for comp, wkey in [
            ('seoryeok', 'W_SEORYEOK'),
            ('surge',    'W_SURGE'),
            ('investor', 'W_INVESTOR'),
            ('pattern',  'W_PATTERN'),
        ]:
            stat  = cs[comp]
            chg   = stat['weight_change']
            score = stat['avg_score']
            arrow = '↑' if chg > 0 else ('↓' if chg < 0 else '→')
            if score >= 65:
                reason = f'급등 전날 평균 {score:.1f}점 — 예측력 우수, 가중치 강화'
            elif score >= 50:
                reason = f'급등 전날 평균 {score:.1f}점 — 예측력 양호, 소폭 증가'
            elif score >= 35:
                reason = f'급등 전날 평균 {score:.1f}점 — 예측력 보통, 현상 유지'
            else:
                reason = f'급등 전날 평균 {score:.1f}점 — 예측력 부족, 가중치 축소'
            L.append(f'  {arrow} {_COMP_LABEL[comp]}: {reason}')

        # 피처 중요도
        fi = upd.get('feature_importance', [])
        if fi:
            L += ['', '  [ 급등 종목 전일 공통 패턴 (피처 출현률) ]', sep2]
            for item in fi[:8]:
                bar_len = int(item['rate'] * 20)
                bar     = '█' * bar_len + '░' * (20 - bar_len)
                L.append(
                    f'  {item["label"]:<18}  [{bar}]  '
                    f'{item["rate"]*100:5.1f}%  ({item["count"]}/{upd["window"]}건)'
                )
            L.append(sep2)

        # 누적 가중치 이력 (최근 5회)
        hist = log_weight_history(result)
        if len(hist) >= 2:
            L += ['', '  [ 가중치 변화 이력 (최근 5회) ]', sep2]
            L.append(
                f'  {"날짜":<12} {"세력":>7} {"급등":>7} {"수급":>7} {"패턴":>7}  {"포착률":>7}'
            )
            L.append(sep2)
            for h in hist[-5:]:
                w = h['weights']
                L.append(
                    f'  {h["date"][:10]:<12}'
                    f'  {w.get("W_SEORYEOK",0):.4f}'
                    f'  {w.get("W_SURGE",0):.4f}'
                    f'  {w.get("W_INVESTOR",0):.4f}'
                    f'  {w.get("W_PATTERN",0):.4f}'
                    f'  {h.get("hit_rate",0)*100:6.1f}%'
                )
            L.append(sep2)

    else:
        remaining = max(0, MIN_OBS_FOR_UPDATE - result['total_obs'])
        L += [
            sep2,
            f'  모델 가중치: 변경 없음 (관측치 {remaining}개 더 필요)',
            f'  현재 가중치 — 세력:{_fmt_w("W_SEORYEOK")}  급등:{_fmt_w("W_SURGE")}'
            f'  수급:{_fmt_w("W_INVESTOR")}  패턴:{_fmt_w("W_PATTERN")}',
            sep2,
        ]

    L += ['', sep, '  ※ 이 리포트는 자동 발송됩니다.', sep]
    return subject, '\n'.join(L)


def log_weight_history(result: dict) -> list:
    """learning_log.json에서 가중치 이력 읽기"""
    try:
        from model_updater import load_learning_log
        log = load_learning_log()
        return log.get('weight_history', [])
    except Exception:
        return []


def _fmt_w(key: str) -> str:
    """learned_weights.json 또는 default에서 현재 가중치 문자열 반환"""
    try:
        from model_updater import load_current_weights
        return f'{load_current_weights().get(key, 0):.4f}'
    except Exception:
        return '?'


# ── 5. GitHub 자동 동기화 ──────────────────────────────────────────────────────

def _git_auto_push():
    """코드 변경 사항을 GitHub에 자동 커밋 & 푸시."""
    import subprocess

    repo_dir = Path(__file__).parent
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=repo_dir, capture_output=True, text=True, timeout=30
        )
        changed = [l for l in result.stdout.splitlines()
                   if l.strip() and not l.endswith('.json') and not l.endswith('.png')]
        if not changed:
            return

        subprocess.run(['git', 'add', '*.py', '.gitignore'],
                       cwd=repo_dir, capture_output=True, timeout=30)

        date_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        commit = subprocess.run(
            ['git', 'commit', '-m', f'auto: daily update {date_str}'],
            cwd=repo_dir, capture_output=True, text=True, timeout=30
        )
        if commit.returncode != 0 and 'nothing to commit' in commit.stdout:
            return

        push = subprocess.run(
            ['git', 'push', 'origin', 'main'],
            cwd=repo_dir, capture_output=True, text=True, timeout=60
        )
        if push.returncode == 0:
            print(f'  [GitHub] ✅ push 완료 → PhilipSon0414/stock_system_good')
        else:
            print(f'  [GitHub] ⚠ push 실패: {push.stderr.strip()[:80]}')
    except Exception as e:
        print(f'  [GitHub] 오류: {e}')


# ── 6. 진입점 ─────────────────────────────────────────────────────────────────

def main():
    # sys.argv[1:] 에 YYYYMMDD 날짜를 넘기면 해당일 학습
    dates = sys.argv[1:] if len(sys.argv) > 1 else [None]

    for date_arg in dates:
        result = run_learning(date_arg)

        if not result:
            label = date_arg or datetime.now().strftime('%Y%m%d')
            dt_label = datetime.strptime(label, '%Y%m%d').strftime('%Y-%m-%d') if date_arg else label
            send_report(
                subject=f'[세력분석] 급등 학습 — {dt_label} 급등 종목 없음',
                body=f'{dt_label} {MIN_SURGE_PCT:.0f}%+ 급등 종목이 없거나 데이터 조회에 실패했습니다.',
            )
            continue

        subject, body = build_email(result)
        print('\n' + body)
        send_report(subject=subject, body=body)
        print('\n  완료.')
        print()

        # GitHub 자동 동기화 (학습 후 코드 변경 사항 push)
        _git_auto_push()


if __name__ == '__main__':
    main()
