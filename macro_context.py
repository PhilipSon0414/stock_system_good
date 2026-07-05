"""
매크로 컨텍스트 — 외부 지표 기반 시장 환경 분석

우선순위 구현:
  1. VIX + USD/KRW        — 한국 시장 방향성 핵심 2개
  2. 미국 지수/선물         — 전야제 심리 반영
  3. 필라델피아 반도체(SOX)  — 코스닥 방향성 상관관계 높음
  4. 환율·금리·원자재       — 업종별 영향
  5. 선물·옵션 만기일 플래그  — 변동성 경보
  6. DART 공시 부정 필터    — 유상증자·CB 발행 종목 자동 제외

매크로 점수 체계:
  +30 ~ -30 범위
  > +10: 강세 환경 → 개별 신호 신뢰도 ↑
  -10 ~ +10: 중립
  < -10: 약세 환경 → 개별 신호 신뢰도 ↓
  < -20: 극도 공포 → 신규 진입 자제
"""

import json
from datetime import datetime, timedelta, date
from pathlib import Path
from functools import lru_cache
from typing import Optional

CACHE_FILE = Path(__file__).parent / '_macro_cache.json'
CACHE_TTL  = 3600   # 1시간 캐시


# ── 1. yfinance 수집 ─────────────────────────────────────────────────────────

_TICKERS = {
    'VIX':       '^VIX',
    'SP500':     '^GSPC',
    'NASDAQ':    '^IXIC',
    'SP500_FUT': 'ES=F',
    'NQ_FUT':    'NQ=F',
    'USD_KRW':   'KRW=X',
    'USD_JPY':   'JPY=X',
    'DXY':       'DX-Y.NYB',
    'SOX':       '^SOX',
    'TNX':       '^TNX',      # 미국 10년물
    'IRX':       '^IRX',      # 미국 2년물
    'OIL_WTI':   'CL=F',
    'COPPER':    'HG=F',
    'GOLD':      'GC=F',
    'NIKKEI':    '^N225',
    'HSI':       '^HSI',
}


def _fetch_yfinance(days: int = 5) -> dict:
    """yfinance로 주요 지표 수집. 실패 시 빈 dict."""
    try:
        import yfinance as yf
        import warnings
        warnings.filterwarnings('ignore')

        result = {}
        symbols = list(_TICKERS.values())
        data = yf.download(symbols, period=f'{days}d', progress=False,
                           auto_adjust=True, group_by='ticker')

        for name, sym in _TICKERS.items():
            try:
                if len(symbols) == 1:
                    df = data
                else:
                    df = data[sym] if sym in data.columns.get_level_values(0) else data
                if df is None or df.empty:
                    continue
                close = df['Close'].dropna()
                if len(close) < 1:
                    continue
                last  = float(close.iloc[-1])
                prev  = float(close.iloc[-2]) if len(close) >= 2 else last
                prev5 = float(close.iloc[-5]) if len(close) >= 5 else prev
                result[name] = {
                    'last':    round(last, 4),
                    'chg1d':   round((last - prev)  / prev  * 100, 2) if prev  else 0,
                    'chg5d':   round((last - prev5) / prev5 * 100, 2) if prev5 else 0,
                }
            except Exception:
                continue
        return result
    except Exception:
        return {}


def _load_cache() -> Optional[dict]:
    if not CACHE_FILE.exists():
        return None
    try:
        data = json.loads(CACHE_FILE.read_text(encoding='utf-8'))
        if datetime.now().timestamp() - data.get('ts', 0) < CACHE_TTL:
            return data['payload']
    except Exception:
        pass
    return None


def _save_cache(payload: dict):
    try:
        CACHE_FILE.write_text(
            json.dumps({'ts': datetime.now().timestamp(), 'payload': payload},
                       ensure_ascii=False),
            encoding='utf-8'
        )
    except Exception:
        pass


def get_external_data(force_refresh: bool = False) -> dict:
    """외부 지표 데이터 (캐시 1시간). 실패 시 빈 dict."""
    if not force_refresh:
        cached = _load_cache()
        if cached:
            return cached
    data = _fetch_yfinance()
    if data:
        _save_cache(data)
    return data


# ── 2. 매크로 점수 계산 ───────────────────────────────────────────────────────

def compute_macro_score(data: dict) -> dict:
    """
    외부 지표 → 매크로 점수 (-30 ~ +30) 계산.

    Returns:
        score        : 종합 매크로 점수
        regime       : 'bull' | 'neutral' | 'bear' | 'panic'
        signals      : 주요 신호 리스트
        gate         : 'go' | 'caution' | 'stop'
        details      : 지표별 값 딕셔너리
        expiry_today : 오늘 선물/옵션 만기일 여부
    """
    score   = 0
    signals = []
    details = {}

    # ── VIX (공포지수) ────────────────────────────────────────────────────
    vix = data.get('VIX', {})
    if vix:
        v = vix.get('last', 20)
        details['VIX'] = v
        if v < 12:
            score += 8;  signals.append(f'VIX {v:.1f} (극도 안정 — 강세 환경)')
        elif v < 15:
            score += 5;  signals.append(f'VIX {v:.1f} (안정)')
        elif v < 20:
            score += 2;  signals.append(f'VIX {v:.1f} (보통)')
        elif v < 25:
            score -= 5;  signals.append(f'⚠ VIX {v:.1f} (불안)')
        elif v < 30:
            score -= 12; signals.append(f'⚠ VIX {v:.1f} (공포 구간)')
        else:
            score -= 22; signals.append(f'🚨 VIX {v:.1f} (극도 공포 — 신규 진입 자제)')

        # VIX 일일 변화
        vc = vix.get('chg1d', 0)
        if vc >= 15:
            score -= 5;  signals.append(f'VIX 급등 {vc:+.1f}%')
        elif vc <= -10:
            score += 3;  signals.append(f'VIX 급락 {vc:+.1f}% (공포 해소)')

    # ── 미국 야간 선물 (다음날 한국 시장 직접 영향) ──────────────────────
    sp_fut = data.get('SP500_FUT', {})
    nq_fut = data.get('NQ_FUT', {})
    if sp_fut:
        chg = sp_fut.get('chg1d', 0)
        details['SP500_FUT'] = chg
        if chg >= 1.0:
            score += 6;  signals.append(f'S&P선물 {chg:+.1f}% (강세 출발 예상)')
        elif chg >= 0.3:
            score += 2
        elif chg <= -1.0:
            score -= 6;  signals.append(f'⚠ S&P선물 {chg:+.1f}% (약세 출발 예상)')
        elif chg <= -0.5:
            score -= 3
    if nq_fut:
        chg = nq_fut.get('chg1d', 0)
        details['NQ_FUT'] = chg
        if chg >= 1.5:
            score += 4;  signals.append(f'나스닥선물 {chg:+.1f}% (기술주 강세)')
        elif chg <= -1.5:
            score -= 4;  signals.append(f'⚠ 나스닥선물 {chg:+.1f}%')

    # ── USD/KRW (원화 가치) ───────────────────────────────────────────────
    krw = data.get('USD_KRW', {})
    if krw:
        last = krw.get('last', 1300)
        chg  = krw.get('chg1d', 0)
        details['USD_KRW'] = last
        if last > 1450:
            score -= 8;  signals.append(f'🚨 원달러 {last:.0f} (외국인 이탈 위험)')
        elif last > 1400:
            score -= 4;  signals.append(f'⚠ 원달러 {last:.0f} (원화 약세)')
        elif last < 1300:
            score += 5;  signals.append(f'원달러 {last:.0f} (원화 강세 — 외국인 유입)')
        # 원화 급격 절상(외국인 복귀 신호)
        if chg <= -0.5:
            score += 3;  signals.append(f'원화 강세 {chg:+.2f}%')
        elif chg >= 1.0:
            score -= 3;  signals.append(f'⚠ 원화 약세 {chg:+.2f}%')

    # ── SOX (반도체 지수 — 코스닥 상관관계) ───────────────────────────────
    sox = data.get('SOX', {})
    if sox:
        chg  = sox.get('chg1d', 0)
        chg5 = sox.get('chg5d', 0)
        details['SOX_1d'] = chg
        details['SOX_5d'] = chg5
        if chg >= 2.0:
            score += 6;  signals.append(f'SOX(반도체) {chg:+.1f}% — 코스닥 강세 기대')
        elif chg >= 1.0:
            score += 3
        elif chg <= -2.0:
            score -= 5;  signals.append(f'⚠ SOX {chg:+.1f}% — 코스닥 약세 우려')
        elif chg <= -1.0:
            score -= 2

    # ── 미국 10년물 금리 (금리 급등 = 성장주 압박) ─────────────────────────
    tnx = data.get('TNX', {})
    if tnx:
        last = tnx.get('last', 4.5)
        chg  = tnx.get('chg1d', 0)  # % 변화 (basis points 아님)
        details['TNX'] = last
        if last >= 5.0:
            score -= 6;  signals.append(f'⚠ 미10년금리 {last:.2f}% (고금리 부담)')
        elif last >= 4.5:
            score -= 2
        # 금리 급등 (하루 5bp+ = 0.05% 이상)
        if chg >= 2.0:
            score -= 4;  signals.append(f'⚠ 금리 급등 {chg:+.2f}%')

    # ── 달러인덱스 (강달러 = 신흥국 자금 이탈) ───────────────────────────
    dxy = data.get('DXY', {})
    if dxy:
        chg = dxy.get('chg1d', 0)
        details['DXY_1d'] = chg
        if chg >= 0.5:
            score -= 3;  signals.append(f'달러강세 DXY {chg:+.2f}% (신흥국 부담)')
        elif chg <= -0.5:
            score += 3;  signals.append(f'달러약세 DXY {chg:+.2f}% (신흥국 유리)')

    # ── 구리 (경기 선행 지표) ────────────────────────────────────────────
    copper = data.get('COPPER', {})
    if copper:
        chg = copper.get('chg1d', 0)
        details['COPPER_1d'] = chg
        if chg >= 2.0:
            score += 3;  signals.append(f'구리 {chg:+.1f}% (경기 강세 신호)')
        elif chg <= -2.0:
            score -= 3;  signals.append(f'⚠ 구리 {chg:+.1f}% (경기 둔화)')

    # ── 유가 (에너지·화학 업종) ─────────────────────────────────────────
    oil = data.get('OIL_WTI', {})
    if oil:
        chg  = oil.get('chg1d', 0)
        last = oil.get('last', 70)
        details['WTI'] = last
        if last > 100:
            score -= 3;  signals.append(f'⚠ WTI {last:.0f}$ (고유가 부담)')
        if chg >= 3.0:
            score += 1;  signals.append(f'유가 {chg:+.1f}% (에너지주 수혜)')
        elif chg <= -3.0:
            score -= 1

    # ── 선물·옵션 만기일 ────────────────────────────────────────────────
    expiry_today = _is_expiry_today()
    details['expiry_today'] = expiry_today
    if expiry_today:
        score -= 3
        signals.append('⚡ 오늘 선물/옵션 만기일 — 변동성 주의')

    # ── 종합 판정 ────────────────────────────────────────────────────────
    score = max(-30, min(30, score))

    if score >= 12:
        regime = 'bull';    gate = 'go'
    elif score >= -5:
        regime = 'neutral'; gate = 'go'
    elif score >= -15:
        regime = 'bear';    gate = 'caution'
    else:
        regime = 'panic';   gate = 'stop'

    return {
        'score':        score,
        'regime':       regime,
        'gate':         gate,
        'signals':      signals,
        'details':      details,
        'expiry_today': expiry_today,
        'fetched_at':   datetime.now().strftime('%H:%M'),
    }


def _is_expiry_today() -> bool:
    """한국 선물/옵션 만기일: 매월 두 번째 목요일"""
    today = date.today()
    first = date(today.year, today.month, 1)
    thu_count = 0
    d = first
    while d.month == today.month:
        if d.weekday() == 3:
            thu_count += 1
            if thu_count == 2:
                return d == today
        d += timedelta(days=1)
    return False


# ── 3. DART 공시 부정 필터 ────────────────────────────────────────────────────

_NEGATIVE_KEYWORDS = [
    '유상증자', '전환사채', '신주인수권부사채', 'CB 발행', 'BW 발행',
    '주식매수선택권', '감자', '주식교환', '합병반대', '상장폐지',
    '불성실공시', '관리종목', '투자주의', '투자경고',
]

_POSITIVE_KEYWORDS = [
    '자사주취득', '자사주매입', '배당', '유무상증자중무상', '실적호조',
]


def get_dart_flags(ticker: str, days: int = 3,
                   end_date: str | None = None) -> dict:
    """dart_utils 모듈 위임 — 공시 긍/부정 분류.
    end_date: 스캔 기준일('YYYY-MM-DD'|'YYYYMMDD') — 지정 시 해당일 기준 조회
              (백데이트 스캔 시 미래 공시 누수 차단)."""
    try:
        from dart_utils import get_dart_disclosures, classify_dart_impact
        if end_date:
            base = end_date if '-' in end_date else f'{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}'
        else:
            base = datetime.now().strftime('%Y-%m-%d')
        info   = get_dart_disclosures(ticker, base, lookback_days=days)
        impact = classify_dart_impact(info)
        return {
            'has_negative': info.get('has_negative', False),
            'has_positive': info.get('has_positive', False),
            'disclosures':  info.get('titles', []),
            'signal':       impact.get('signal', ''),
            'surge_type':   impact.get('surge_type', 'unknown'),
        }
    except Exception:
        return {'has_negative': False, 'has_positive': False,
                'disclosures': [], 'signal': '', 'surge_type': 'unknown'}


# ── 4. 통합 리포트 ────────────────────────────────────────────────────────────

def get_full_macro_context(force_refresh: bool = False) -> dict:
    """
    전체 매크로 컨텍스트 조회 (캐시 적용).
    daily_scan.py의 get_market_context()에 통합되어 사용.
    """
    data  = get_external_data(force_refresh)
    score = compute_macro_score(data)
    return {**score, 'raw': data}


def build_macro_report_lines(ctx: dict) -> list[str]:
    """리포트 출력용 매크로 섹션 라인 생성"""
    lines = []
    score  = ctx.get('score', 0)
    regime = ctx.get('regime', 'neutral')
    gate   = ctx.get('gate', 'go')
    raw    = ctx.get('raw', {})

    gate_icon = {'go': '✅', 'caution': '⚠', 'stop': '🚨'}.get(gate, '')
    regime_kr = {'bull':'강세','neutral':'중립','bear':'약세','panic':'공포'}.get(regime,'?')

    lines.append(f'  매크로점수: {score:+d}점  [{regime_kr}]  {gate_icon}')

    # 핵심 지표 한 줄 요약
    d = ctx.get('details', {})
    parts = []
    if 'VIX' in d:       parts.append(f'VIX {d["VIX"]:.0f}')
    if 'USD_KRW' in d:   parts.append(f'원달러 {d["USD_KRW"]:.0f}')
    if 'SP500_FUT' in d: parts.append(f'S&P선물 {d["SP500_FUT"]:+.1f}%')
    if 'SOX_1d' in d:    parts.append(f'SOX {d["SOX_1d"]:+.1f}%')
    if 'TNX' in d:       parts.append(f'미금리 {d["TNX"]:.2f}%')
    if parts:
        lines.append(f'  {" | ".join(parts)}')

    for sig in ctx.get('signals', [])[:4]:
        lines.append(f'  → {sig}')

    if ctx.get('expiry_today'):
        lines.append('  ⚡ 오늘 선물/옵션 만기일 — 변동성 주의')

    return lines


if __name__ == '__main__':
    print('매크로 컨텍스트 테스트')
    ctx = get_full_macro_context(force_refresh=True)
    for line in build_macro_report_lines(ctx):
        print(line)
    print(f"\n원본 데이터: {list(ctx['raw'].keys())}")
