"""
DART 전자공시 유틸리티

급등 종목 D-1 공시를 분석해 긍/부정 이벤트를 피처로 변환.
corp_list는 모듈 로드 시 1회 캐시하여 이후 빠르게 조회.

부정 공시: 유상증자·CB·BW → 희석·차입 → 주가 압박
긍정 공시: 자사주 매입·배당 → 주주환원 → 수급 개선
중립 공시: 실적 발표·사업보고서 → 콘텍스트 확인
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

_CFG_PATH = Path(__file__).parent / 'dart_config.json'

NEGATIVE_KEYWORDS = [
    '유상증자', '전환사채', '신주인수권부사채', '교환사채',
    'CB발행', 'BW발행', '감자', '상장폐지', '불성실공시',
    '관리종목', '투자주의', '투자경고', '주식매수선택권',
]
POSITIVE_KEYWORDS = [
    '자기주식취득', '자사주취득', '자사주매입',
    '현금배당', '주식배당', '무상증자',
    '자기주식소각', '주주환원',
]
EARNINGS_KEYWORDS = [
    '실적', '영업이익', '매출', '사업보고서', '반기보고서', '분기보고서',
]


def _load_api_key() -> Optional[str]:
    import os
    key = os.environ.get('DART_API_KEY')
    if key:
        return key
    if _CFG_PATH.exists():
        try:
            return json.loads(_CFG_PATH.read_text())['api_key']
        except Exception:
            pass
    return None


# ── corp_list 캐시 (프로세스당 1회 로드) ────────────────────────────────────

_corp_list_cache = None

def _get_corp_list():
    global _corp_list_cache
    if _corp_list_cache is not None:
        return _corp_list_cache
    try:
        import dart_fss as dart
        key = _load_api_key()
        if not key:
            return None
        dart.set_api_key(key)
        _corp_list_cache = dart.get_corp_list()
        return _corp_list_cache
    except Exception:
        return None


def get_dart_disclosures(ticker: str, date_str: str,
                          lookback_days: int = 3) -> dict:
    """
    ticker 종목의 date_str 기준 최근 lookback_days일 공시 조회.

    Returns:
        has_negative   : 부정 공시 여부 (유상증자·CB·감자 등)
        has_positive   : 긍정 공시 여부 (자사주·배당 등)
        has_earnings   : 실적 공시 여부
        negative_title : 부정 공시 제목 (첫 번째)
        positive_title : 긍정 공시 제목 (첫 번째)
        total_count    : 전체 공시 건수
        titles         : 공시 제목 리스트
        data_ok        : 조회 성공 여부
    """
    result = {
        'has_negative': False, 'has_positive': False, 'has_earnings': False,
        'negative_title': '', 'positive_title': '',
        'total_count': 0, 'titles': [], 'data_ok': False,
    }
    try:
        corp_list = _get_corp_list()
        if not corp_list:
            return result

        corp = corp_list.find_by_stock_code(str(ticker).zfill(6))
        if not corp:
            return result

        end_dt   = datetime.strptime(date_str, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=lookback_days)
        filings  = corp.search_filings(
            bgn_de=start_dt.strftime('%Y%m%d'),
            end_de=end_dt.strftime('%Y%m%d'),
        )

        result['data_ok'] = True

        if not filings:
            return result

        titles = []
        try:
            for f in filings:
                # dart-fss 객체는 dict-like 접근 또는 속성 접근
                t = ''
                if hasattr(f, 'report_nm'):
                    t = f.report_nm
                elif hasattr(f, 'get'):
                    t = f.get('report_nm', '')
                elif isinstance(f, dict):
                    t = f.get('report_nm', '')
                # 문자열로 직렬화된 dict인 경우 파싱
                if not t:
                    fs = str(f)
                    import re
                    m = re.search(r"'report_nm':\s*'([^']+)'", fs)
                    if m:
                        t = m.group(1).strip()
                if t:
                    titles.append(t)
        except Exception:
            pass

        result['total_count'] = len(titles)
        result['titles']      = titles[:5]

        for t in titles:
            if any(kw in t for kw in NEGATIVE_KEYWORDS):
                result['has_negative']   = True
                result['negative_title'] = t[:30]
            if any(kw in t for kw in POSITIVE_KEYWORDS):
                result['has_positive']   = True
                result['positive_title'] = t[:30]
            if any(kw in t for kw in EARNINGS_KEYWORDS):
                result['has_earnings'] = True

    except Exception:
        pass

    return result


def classify_dart_impact(dart_info: dict) -> dict:
    """
    공시 영향도 분류 → 스코어 조정값 반환.

    Returns:
        score_adj  : 합산 점수 조정값 (-20 ~ +10)
        signal     : 표시용 태그
        surge_type : 공시 주도 급등 유형 추정
    """
    adj    = 0
    signal = ''
    surge_type = 'unknown'

    if not dart_info.get('data_ok'):
        return {'score_adj': 0, 'signal': '', 'surge_type': surge_type}

    if dart_info.get('has_negative'):
        adj    = -20
        signal = f'⚠ 부정공시: {dart_info["negative_title"]}'
        surge_type = 'negative_disclosure'
    elif dart_info.get('has_positive'):
        adj    = +8
        signal = f'✅ 긍정공시: {dart_info["positive_title"]}'
        surge_type = 'positive_disclosure'
    elif dart_info.get('has_earnings'):
        adj    = +5
        signal = '📊 실적 공시 동반'
        surge_type = 'earnings_driven'
    elif dart_info.get('total_count', 0) > 0:
        signal = f'공시 {dart_info["total_count"]}건'
        surge_type = 'disclosure_driven'
    else:
        surge_type = 'technical'   # 공시 없음 = 기술적 급등

    return {'score_adj': adj, 'signal': signal, 'surge_type': surge_type}


def get_bulk_disclosures_for_date(date_str: str) -> dict:
    """
    특정 날짜의 전체 부정 공시 종목 코드 집합 반환 (스캔 사전 제외용).
    date_str: 'YYYY-MM-DD'

    Returns: {ticker: [titles], ...} — 부정 공시 있는 종목만
    """
    result = {}
    try:
        import dart_fss as dart
        key = _load_api_key()
        if not key:
            return result
        dart.set_api_key(key)

        end_dt   = datetime.strptime(date_str, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=1)

        # 전체 공시 목록 (DART search_filings는 종목별이라 전체 조회는 별도 API 필요)
        # 대신 pykrx로 당일 공시 종목 목록을 가져올 수 없으므로
        # 스캔 결과에서 개별 조회하는 방식 사용
        return result
    except Exception:
        return result
