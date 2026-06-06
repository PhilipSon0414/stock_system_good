"""
2026년 6월 5일 기준 한국 주식시장 학습 및 스캔
─────────────────────────────────────────────────────────
실행:  python market_scan_20260605.py

수집 데이터 기반:
  - KOSPI  종가 8,160.59  (-5.54% / -477.57p)
  - KOSDAQ 종가 ~990선  (-4.50%)
  - 트리거: 브로드컴(AVGO) Q3 AI칩 가이던스 미스 ($16B vs 예상 $17.2B)
  - 매도 사이드카 발동 (2026년 10번째)
  - 외국인 대규모 순매도 (원/달러 17년 최고)
"""

import json
import sys
from datetime import datetime, date
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
LOG_DIR    = SCRIPT_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)

SCAN_DATE = '20260605'

# ── 1. 시장 인텔리전스 (웹 수집) ─────────────────────────────────────────────

MARKET_INTEL = {
    'date': '2026-06-05',
    'day': '목요일',
    'kospi': {
        'open':  8323.20,
        'high':  8382.16,
        'low':   8038.10,
        'close': 8160.59,
        'change_pct': -5.54,
        'change_pts': -477.57,
        'volume': 327_370_000,
        'circuit_breaker': True,   # 매도 사이드카 (올해 10번째)
        'ytd_return_before': 75.0, # 6/5 이전 연초대비 수익률
    },
    'kosdaq': {
        'close': 990.0,            # 근사치 (1000p 하회)
        'change_pct': -4.50,
        'note': '1,000p 지지선 붕괴',
    },
    'won_usd': {
        'rate': 1430.0,            # 근사치
        'note': '원/달러 17년 최고 수준',
    },
    'trigger': {
        'source': 'Broadcom (AVGO)',
        'event': 'Q3 FY2026 AI칩 매출 가이던스 미스',
        'actual': 16.0,            # $B
        'expected': 17.2,          # $B
        'miss_pct': -6.98,
        'full_year_actual': 56.0,  # $B
        'full_year_expected': 57.6,
        'date': '2026-06-03 장후',
    },
    'key_stocks': [
        # 급락 종목
        {'code': '005930', 'name': '삼성전자',     'change': -6.40,  'sector': '반도체', 'side': 'sell'},
        {'code': '000660', 'name': 'SK하이닉스',   'change': -9.92,  'sector': '반도체', 'side': 'sell'},
        {'code': '066570', 'name': 'LG전자',       'change': -16.18, 'sector': 'IT가전/로보틱스', 'side': 'sell'},
        {'code': '005380', 'name': '현대차',        'change': -3.84,  'sector': '자동차', 'side': 'sell'},
        {'code': '373220', 'name': 'LG에너지솔루션','change': -4.07,  'sector': '2차전지', 'side': 'sell'},
        {'code': '000270', 'name': '기아',          'change': -2.55,  'sector': '자동차', 'side': 'sell'},
        {'code': '402340', 'name': 'SK스퀘어',      'change': -7.42,  'sector': '지주/IT', 'side': 'sell'},
        {'code': '012330', 'name': '현대모비스',    'change': -6.42,  'sector': '자동차부품', 'side': 'sell'},
        # 상승/방어 종목
        {'code': '085620', 'name': '미래에셋생명',  'change': +9.50,  'sector': '보험', 'side': 'buy'},
    ],
    'macro_factors': [
        '미국 반도체주 전일 야간 급락 (Nasdaq -4%+)',
        'Broadcom AI 가이던스 미스로 AI 버블 우려 확산',
        '외국인 대규모 순매도 (원/달러 17년 최고)',
        '이란-쿠웨이트 공항 타격 → 중동 긴장 고조',
        '유가 상승 → 인플레 우려',
        '골드만삭스 KOSPI 12개월 목표 12,000p 상향 (AI 어닝 성장 근거)',
    ],
    'sector_performance': {
        '반도체/AI': -8.0,
        'IT가전/로보틱스': -12.0,
        '2차전지': -4.5,
        '자동차': -4.0,
        '지주': -6.0,
        '방산': +1.5,        # 중동 긴장 수혜 추정
        '정유/에너지': +1.2, # 유가 상승 수혜 추정
        '보험/금융': +0.5,   # 방어주 전환 추정
        '바이오/제약': -1.0, # 상대적 선전
    },
    'investor_flow': {
        '외국인': '대규모 순매도 (집중 종목: SK하이닉스)',
        '기관':   '순매도',
        '개인':   '순매수 (저가 매수 유입)',
    },
    'context': {
        'ytd_backdrop': 'KOSPI 2026 연초대비 75%+ 상승 후 첫 대규모 조정',
        'samsung_ytd': 111,   # % YTD
        'skhynix_ytd': 144,   # % YTD
        'lg_elec_peak': '392,500원 (6/2) → 차익 실현 압력',
        'goldman_target': 12000,
    }
}


# ── 2. 학습: 이벤트 분류 및 인사이트 도출 ────────────────────────────────────

def classify_market_event(intel: dict) -> dict:
    """시장 이벤트를 분류하고 학습 인사이트를 도출한다."""
    kospi_chg = intel['kospi']['change_pct']

    if kospi_chg <= -5:
        event_type = 'MAJOR_SELLOFF'       # 대규모 매도 (사이드카 수준)
    elif kospi_chg <= -3:
        event_type = 'SIGNIFICANT_DROP'
    elif kospi_chg <= -1:
        event_type = 'MODERATE_DROP'
    else:
        event_type = 'NORMAL'

    # 촉발 원인 분류
    triggers = []
    t = intel.get('trigger', {})
    if 'AI칩' in t.get('event', ''):
        triggers.append('AI_GUIDANCE_MISS')
    if intel.get('won_usd', {}).get('note'):
        triggers.append('FX_PRESSURE')
    for f in intel.get('macro_factors', []):
        if '이란' in f or '중동' in f:
            triggers.append('GEOPOLITICAL')
            break

    # 섹터 로테이션 파악
    sector_perf = intel.get('sector_performance', {})
    gainers  = [s for s, v in sector_perf.items() if v > 0]
    losers   = [s for s, v in sector_perf.items() if v < -5]

    return {
        'event_type': event_type,
        'triggers': triggers,
        'sector_gainers': gainers,
        'sector_losers': losers,
        'oversold_candidates': [
            s['code'] for s in intel['key_stocks']
            if s['change'] < -8 and s['side'] == 'sell'
        ],
        'resilient_stocks': [
            s['code'] for s in intel['key_stocks']
            if s['change'] > 0
        ],
        'signal': '반도체/AI 고밸류 조정 + 방산/에너지 로테이션',
    }


def learn_event_patterns(intel: dict, classification: dict) -> list[dict]:
    """이벤트에서 실전 학습 패턴을 추출한다."""
    patterns = []

    # 패턴 1: AI 가이던스 미스 → 반도체 과대낙폭
    if 'AI_GUIDANCE_MISS' in classification['triggers']:
        patterns.append({
            'pattern_id': 'AI_GUIDANCE_MISS_SELLOFF',
            'name':       'AI 가이던스 미스 → 반도체 대량 매도',
            'description': (
                f"Broadcom Q3 AI칩 가이던스 {intel['trigger']['actual']}B달러 "
                f"(예상 {intel['trigger']['expected']}B달러 대비 "
                f"{intel['trigger']['miss_pct']:.1f}% 미스) → "
                f"KOSPI {intel['kospi']['change_pct']:.1f}% 급락"
            ),
            'affected_sector': ['반도체', 'AI'],
            'signal_to_watch': [
                '매도 사이드카 발동 후 반등 타이밍',
                '외국인 순매도 소진 시점',
                'SK하이닉스/삼성전자 저점 확인 패턴',
            ],
            'follow_up_scan': '과대낙폭 반도체 눌림목 스캔',
        })

    # 패턴 2: 섹터 로테이션
    if classification['sector_gainers']:
        patterns.append({
            'pattern_id': 'ROTATION_TO_DEFENSIVE',
            'name':       '급락 당일 방어주/실물자산 로테이션',
            'description': (
                f"반도체 -8%, AI가전 -12% 급락 vs "
                f"방산 +1.5%, 정유/에너지 +1.2% 강세"
            ),
            'affected_sector': classification['sector_gainers'],
            'signal_to_watch': [
                '방산주 외국인/기관 순매수 지속 여부',
                '유가 상승 → 정유주 추가 강세',
                '이란 긴장 지속 → 방산 테마 연속성',
            ],
            'follow_up_scan': '방산/에너지 모멘텀 스캔',
        })

    # 패턴 3: LG전자 차익실현 → 과대낙폭
    patterns.append({
        'pattern_id': 'POST_RALLY_PROFIT_TAKING',
        'name':       '단기 급등 후 차익실현 과낙',
        'description': (
            'LG전자 6/2 고가 392,500원 → 2주간 +117% 급등 후 '
            '6/4~6/5 -16.18% 급락 (엔비디아 협력 기대감 과잉 반영)'
        ),
        'affected_sector': ['IT가전/로보틱스'],
        'signal_to_watch': [
            '실질적 엔비디아-LG전자 협력 발표 여부',
            '급락 후 거래량 수렴 → 재진입 시그널',
            '과낙 복귀 패턴 (2~3주 내)',
        ],
        'follow_up_scan': '단기 과낙 반등 스캔',
    })

    return patterns


# ── 3. 스캔 전략: 6/5 이후 주목 종목군 ──────────────────────────────────────

def generate_scan_strategy(intel: dict, classification: dict) -> dict:
    """6/5 시장 상황을 반영한 스캔 전략을 생성한다."""

    strategy = {
        'scan_date': intel['date'],
        'market_regime': 'BEARISH_CORRECTION',  # 조정 국면
        'macro_gate': 'CAUTION',                 # 주의 단계 (사이드카 수준)

        # ── 전략 1: 과대낙폭 반도체 반등 후보 ─────────────────────────
        'strategy_1': {
            'name': '반도체 과대낙폭 반등 스캔',
            'rationale': (
                'KOSPI -5.54% 중 반도체는 -8~10% 과대낙폭. '
                'YTD 111~144% 상승 후 첫 대규모 조정. '
                '골드만삭스 12개월 목표 12,000p 유지 → AI 구조적 성장 지속 전망. '
                '외국인 순매도 소진 확인 후 반등 가능성.'
            ),
            'watch_list': [
                {'code': '000660', 'name': 'SK하이닉스',  'note': '-9.92% 과대낙폭, HBM 공급 독점'},
                {'code': '005930', 'name': '삼성전자',    'note': '-6.40%, PER 저평가 구간 진입 가능'},
                {'code': '402340', 'name': 'SK스퀘어',    'note': '-7.42%, SK하이닉스 지주 할인'},
            ],
            'entry_trigger': [
                '외국인 순매수 전환 2일 연속',
                'VolRatio < 0.5 → 거래량 압축 후 회복 신호 (VolRecovery)',
                'MA20 지지 확인',
                '미국 반도체 지수(SOXX) 반등 확인',
            ],
            'stop_loss': '-7% 이하 추가 하락 시 손절',
            'risk': 'Broadcom 가이던스 미스 재현 시 추가 하락',
        },

        # ── 전략 2: 방산 모멘텀 스캔 ──────────────────────────────────
        'strategy_2': {
            'name': '방산/에너지 섹터 모멘텀 스캔',
            'rationale': (
                '이란-쿠웨이트 공항 타격 → 중동 긴장 고조. '
                '반도체 급락 당일 방산 +1.5%, 에너지 +1.2% 강세. '
                '유가 상승 → 정유주 실적 개선 가시화.'
            ),
            'watch_list': [
                {'code': '012450', 'name': '한화에어로스페이스', 'note': 'K9 자주포 수출 확대, 중동 방산 수혜'},
                {'code': '079550', 'name': 'LIG넥스원',          'note': '미사일/방산 전자장비, 기관 순매수 지속'},
                {'code': '004490', 'name': '세아베스틸',          'note': '방산 소재 공급망'},
                {'code': '010950', 'name': 'S-Oil',              'note': '유가 상승 정유주 직접 수혜'},
                {'code': '096770', 'name': 'SK이노베이션',        'note': '정유+배터리 복합, 유가 헤지'},
            ],
            'entry_trigger': [
                '방산주 SE_Accum 신호 (거래량 수렴 + MA 지지)',
                '거래량 1.5x+ 증가 확인 (기관 매수)',
                '52주 신고가 돌파 후 눌림목',
                '이란 긴장 뉴스 지속 여부 확인',
            ],
            'stop_loss': '-5% (지정학 리스크 조기 해소 시)',
            'risk': '미-이란 협상 타결 → 지정학 프리미엄 소멸',
        },

        # ── 전략 3: LG전자 과낙 반등 스캔 ────────────────────────────
        'strategy_3': {
            'name': 'LG전자 과낙 복귀 스캔',
            'rationale': (
                'LG전자 6/2 고가 392,500원 대비 6/5 종가 ~330,000원 (-16%). '
                '엔비디아-LG 로보틱스 협력 기대감은 현재진행형. '
                '1분기 역대 최고 실적 (매출 23.7조, 영업이익 1.67조) 기초 견고. '
                '2주간 +117% 급등 후 정상 조정으로 볼 수 있음.'
            ),
            'watch_list': [
                {'code': '066570', 'name': 'LG전자',     'note': '과낙 반등 후보, 엔비디아 협력 실질 발표 시 재폭등'},
                {'code': '003550', 'name': 'LG(지주)',   'note': 'LG전자 할인율 축소 가능'},
            ],
            'entry_trigger': [
                '6/5 저가 대비 +3% 이상 반등 + 거래량 0.7x+ 확인',
                'SE_Accum 신호 (거래량 수렴)',
                '엔비디아-LG 공식 협력 발표 뉴스',
                'MA20 회복 (약 310,000~320,000원 구간)',
            ],
            'stop_loss': '-8% (직전 저점 하회 시)',
            'risk': '협력 발표 불발 또는 지연 시 추가 조정',
        },

        # ── 전략 4: 미래에셋생명 모멘텀 스캔 ─────────────────────────
        'strategy_4': {
            'name': '금융/보험 방어주 모멘텀 스캔',
            'rationale': (
                '6/5 미래에셋생명 +9.5% 강세 (52주 신고가 경신). '
                '시장 급락 속 방어적 자금 이동. '
                '금리 인상 기대감 및 보험사 실적 개선 사이클.'
            ),
            'watch_list': [
                {'code': '085620', 'name': '미래에셋생명',  'note': '+9.5% 52주 신고가 경신, 기관 순매수'},
                {'code': '005830', 'name': 'DB손해보험',   'note': '보험 섹터 동반 모멘텀'},
                {'code': '000810', 'name': '삼성화재',     'note': '대형 손보사, 방어주 특성'},
            ],
            'entry_trigger': [
                '신고가 경신 후 눌림목 (SE_Accum + 거래량 수렴)',
                '기관 2일+ 연속 순매수',
                '보험 섹터 지수 상대강도 우위',
            ],
            'stop_loss': '-4% (신고가 돌파 실패 시)',
            'risk': '시장 반등 시 방어주 자금 이탈',
        },
    }

    return strategy


# ── 4. 학습 로그 업데이트 ─────────────────────────────────────────────────────

def update_learning_log(patterns: list[dict]) -> str:
    log_path = SCRIPT_DIR / 'logs' / 'market_events_log.json'

    # 기존 로그 로드
    if log_path.exists():
        try:
            log = json.loads(log_path.read_text(encoding='utf-8'))
        except Exception:
            log = []
    else:
        log = []

    # 6/5 이벤트 항목 추가
    entry = {
        'date':        '2026-06-05',
        'event_type':  'MAJOR_SELLOFF',
        'kospi_chg':   -5.54,
        'kosdaq_chg':  -4.50,
        'trigger':     'Broadcom AI 가이던스 미스',
        'circuit_breaker': True,
        'patterns':    [p['pattern_id'] for p in patterns],
        'learned_at':  datetime.now().isoformat(),
        'source':      'market_scan_20260605.py',
    }

    # 중복 방지
    existing = [e for e in log if e.get('date') == '2026-06-05']
    if not existing:
        log.append(entry)
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'  학습 로그 업데이트: {log_path}')
    else:
        print(f'  학습 로그 이미 존재 (2026-06-05)')

    return str(log_path)


# ── 5. 리포트 출력 ────────────────────────────────────────────────────────────

def print_market_report(intel: dict, classification: dict,
                        patterns: list[dict], strategy: dict):
    SEP = '═' * 70
    sep = '─' * 70

    print(f'\n{SEP}')
    print(f'  📊 한국 주식시장 학습 & 스캔 리포트  [기준일: {intel["date"]}]')
    print(SEP)

    # 시장 현황
    k = intel['kospi']
    q = intel['kosdaq']
    print(f'\n  【 시장 현황 】')
    print(f'  KOSPI : {k["close"]:,.2f}  ({k["change_pct"]:+.2f}%  {k["change_pts"]:+.2f}p)')
    print(f'          시가 {k["open"]:,.0f}  고가 {k["high"]:,.0f}  저가 {k["low"]:,.0f}')
    print(f'          거래량 {k["volume"]:,}주  |  매도 사이드카 발동 ⚠')
    print(f'  KOSDAQ: ~{q["close"]:,.0f}  ({q["change_pct"]:+.2f}%)  ← {q["note"]}')
    print(f'  원/달러: ~{intel["won_usd"]["rate"]:,}원  ({intel["won_usd"]["note"]})')

    # 트리거
    t = intel['trigger']
    print(f'\n  【 급락 트리거 】')
    print(f'  {t["source"]} — {t["event"]}')
    print(f'  Q3 AI칩 가이던스  {t["actual"]}B달러  (예상 {t["expected"]}B달러  {t["miss_pct"]:.1f}%↓)')
    print(f'  연간 AI칩 가이던스 {t["full_year_actual"]}B달러  (예상 {t["full_year_expected"]}B달러)')
    print(f'  발표일: {t["date"]}  →  아시아 시장 장 시작 전 여파')

    # 주요 종목
    print(f'\n  【 주요 종목 등락 】')
    for s in intel['key_stocks']:
        arrow = '▲' if s['change'] > 0 else '▽'
        print(f'  {arrow} {s["name"]:10s} ({s["code"]})  {s["change"]:+.2f}%  [{s["sector"]}]')

    # 섹터
    print(f'\n  【 섹터 성과 】')
    for sector, pct in sorted(intel['sector_performance'].items(),
                               key=lambda x: x[1], reverse=True):
        bar = '▲' if pct > 0 else '▽'
        print(f'  {bar} {sector:15s}  {pct:+.1f}%')

    # 수급
    print(f'\n  【 수급 동향 】')
    for key, val in intel['investor_flow'].items():
        print(f'  {key:6s}: {val}')

    # 학습 패턴
    print(f'\n{sep}')
    print(f'  【 학습된 패턴 】')
    print(sep)
    for i, p in enumerate(patterns, 1):
        print(f'\n  [{i}] {p["name"]}')
        print(f'      {p["description"]}')
        print(f'      → 관찰 포인트:')
        for sig in p['signal_to_watch']:
            print(f'        • {sig}')

    # 스캔 전략
    print(f'\n{sep}')
    print(f'  【 스캔 전략 TOP 4 】')
    print(sep)
    strats = [strategy[k] for k in strategy if k.startswith('strategy_')]
    for i, st in enumerate(strats, 1):
        print(f'\n  [{i}] {st["name"]}')
        print(f'      근거: {st["rationale"][:80]}...')
        print(f'      관심 종목:')
        for w in st['watch_list']:
            print(f'        • {w["name"]:12s} ({w["code"]})  {w["note"]}')
        print(f'      진입 신호:')
        for sig in st['entry_trigger'][:3]:
            print(f'        ✓ {sig}')
        print(f'      손절 기준: {st["stop_loss"]}')
        print(f'      리스크: {st["risk"]}')

    # 매크로 요인
    print(f'\n{sep}')
    print(f'  【 매크로 요인 】')
    for f in intel['macro_factors']:
        print(f'  • {f}')

    # 결론
    print(f'\n{SEP}')
    print(f'  【 결론 및 전략 요약 】')
    print(SEP)
    print(f'''
  ▌ 시장 국면: AI/반도체 주도 랠리(KOSPI +75% YTD) 후 첫 대규모 조정
  ▌ 매크로 게이트: CAUTION (사이드카 발동, 외국인 대규모 순매도)

  ① 단기 (D+1~D+5):
     - 반도체 (SK하이닉스/삼성전자): 과대낙폭 반등 대기
       → 외국인 순매도 소진 + 미국 반도체 반등 확인 후 진입
     - 방산/에너지: 이란 긴장 지속 시 모멘텀 유지
       → 한화에어로스페이스, LIG넥스원 기관 순매수 확인 후 진입

  ② 중기 (D+1주~D+1개월):
     - LG전자: 과낙 복귀 + 엔비디아 협력 발표 카탈리스트 대기
     - 미래에셋생명 등 보험/금융: 시장 변동성 속 방어주 수요 지속

  ③ 리스크 관리:
     - AI 버블 논쟁 지속 → 포지션 규모 축소 권고
     - 원/달러 17년 최고 → 외국인 추가 이탈 가능성
     - Broadcom 가이던스 미스 후 Nvidia/AMD 실적 가이던스 대기
''')
    print(SEP)


# ── 6. 메인 ──────────────────────────────────────────────────────────────────

def main():
    print('2026-06-05 한국 주식시장 학습 및 스캔 시작...\n')

    # 이벤트 분류
    classification = classify_market_event(MARKET_INTEL)
    print(f'이벤트 유형: {classification["event_type"]}')
    print(f'트리거: {classification["triggers"]}')
    print(f'상승 섹터: {classification["sector_gainers"]}')
    print(f'급락 섹터: {classification["sector_losers"]}')

    # 패턴 학습
    patterns = learn_event_patterns(MARKET_INTEL, classification)
    print(f'\n학습 패턴 {len(patterns)}개 도출')

    # 스캔 전략 생성
    strategy = generate_scan_strategy(MARKET_INTEL, classification)

    # 학습 로그 저장
    log_path = update_learning_log(patterns)

    # 리포트 출력
    print_market_report(MARKET_INTEL, classification, patterns, strategy)

    # JSON 스캔 결과 저장
    scan_result = {
        'scan_date':      SCAN_DATE,
        'generated_at':   datetime.now().isoformat(),
        'market_intel':   MARKET_INTEL,
        'classification': classification,
        'patterns':       patterns,
        'strategy':       strategy,
    }
    out_path = LOG_DIR / f'scan_result_{SCAN_DATE}.json'
    out_path.write_text(json.dumps(scan_result, ensure_ascii=False, indent=2),
                        encoding='utf-8')
    print(f'\n스캔 결과 저장: {out_path}')


if __name__ == '__main__':
    main()
