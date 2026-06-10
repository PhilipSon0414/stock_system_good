# 세력 분석 시스템 설정

MA_PERIODS = [5, 10, 20, 60, 120, 240]
VOLUME_AVG_PERIOD = 30           # 거래량 기준 이평 기간 — 백테스트: 30일 D-5 46.8% (vs 20일 41.6%)
VOL_ENTRY_THRESHOLD = 2.0        # 세력 진입 기준 — 백테스트 2.0x Lift 17.03x (3.0x: 10.36x)
VOL_STRONG_THRESHOLD = 5.0       # 세력 강한 진입: 평균 거래량 5배
VOL_QUIET_THRESHOLD = 0.7        # 세력 매집 기준: 평균 거래량 70% 이하

OB_MIN_MOVE = 0.04               # 오더블록 유효 조건: 이후 4% 이상 이동
OB_LOOKBACK = 90                 # 오더블록 탐색 범위 (일) — 백테스트: 90일 63.6% (vs 60일 51.9%)
OB_NEAR_TOLERANCE = 0.02         # 오더블록 근접 기준: 2% 이내

CONSOLIDATION_DAYS = 10          # 매집 횡보 판정 기간
CONSOLIDATION_RANGE = 0.08       # 매집 판정 가격 범위 (8% 이내)
MA_CONVERGE_THRESHOLD = 0.04     # 이평선 수렴 기준: MA5~MA60 간격 4% 이내

PULLBACK_MA_TOLERANCE = 0.05     # 눌림목/매집 이평선 근접 기준 — 백테스트: 5% 36.4% (vs 3% 27.3%)

SCREEN_MIN_SCORE = 40
SCREEN_MAX_RESULTS = 100         # 전체 순위 100개
SCREEN_MIN_VOLUME = 50_000       # 최소 거래량 (저유동성 제외)
SCREEN_MIN_PRICE = 1_000         # 최소 주가 (동전주 제외)

# 합산 점수 가중치 (합계 = 1.0) — learned_weights.json이 있으면 자동 덮어씀
W_SEORYEOK = 0.25                # 세력 흔적
W_SURGE    = 0.30                # 급등 임박
W_INVESTOR = 0.25                # 외국인/기관 수급
W_PATTERN  = 0.20                # 차트 패턴

# 학습된 가중치 로드 (daily_learner.py가 매일 업데이트)
import json as _json
from pathlib import Path as _Path
_learned = _Path(__file__).parent / 'learned_weights.json'
if _learned.exists():
    try:
        _w = _json.loads(_learned.read_text(encoding='utf-8'))
        W_SEORYEOK = _w.get('W_SEORYEOK', W_SEORYEOK)
        W_SURGE    = _w.get('W_SURGE',    W_SURGE)
        W_INVESTOR = _w.get('W_INVESTOR', W_INVESTOR)
        W_PATTERN  = _w.get('W_PATTERN',  W_PATTERN)
    except Exception:
        pass
