"""
업종 모멘텀 대비 상대강도 분석

pykrx 업종 지수로 종목의 시장 대비 상대강도(RS) 계산.
강한 업종 + 강한 종목 = 세력 작업 최적 환경.

RS 점수:
  종목 20일 수익률 - 업종 지수 20일 수익률
  → 양수: 업종 대비 강세 (아웃퍼폼)
  → 음수: 업종 대비 약세 (언더퍼폼)
"""

from datetime import datetime, timedelta
from functools import lru_cache
import pandas as pd

# KRX 업종 지수 코드 (코스피 업종)
_SECTOR_INDICES = {
    '음식료품':   '1001',
    '섬유의복':   '1002',
    '종이목재':   '1003',
    '화학':      '1004',
    '의약품':    '1005',
    '비금속광물':  '1006',
    '철강금속':   '1007',
    '기계':      '1008',
    '전기전자':   '1009',
    '의료정밀':   '1010',
    '운수장비':   '1011',
    '유통업':    '1012',
    '전기가스업':  '1013',
    '건설업':    '1014',
    '운수창고':   '1015',
    '통신업':    '1016',
    '금융업':    '1017',
    '은행':      '1018',
    '증권':      '1019',
    '보험':      '1020',
    '서비스업':   '1021',
}


@lru_cache(maxsize=1)
def _get_sector_map(date_key: str) -> dict:
    """종목코드 → 업종명 매핑 (날짜별 캐시)"""
    try:
        from pykrx import stock
        df = stock.get_market_sector_classifications(date_key, market='KOSPI')
        if df is None or df.empty:
            return {}
        if '섹터' in df.columns:
            return dict(zip(df.index.astype(str).str.zfill(6), df['섹터']))
        if '업종명' in df.columns:
            return dict(zip(df.index.astype(str).str.zfill(6), df['업종명']))
    except Exception:
        pass
    return {}


def get_sector_rs(ticker: str, stock_20d_return: float) -> dict:
    """
    종목의 업종 대비 상대강도 계산.

    Args:
        ticker           : 종목코드
        stock_20d_return : 종목 20일 수익률 (%)

    Returns:
        sector_name   : 업종명
        sector_20d    : 업종 지수 20일 수익률 (%)
        rs_vs_sector  : 종목 - 업종 (%) ← 핵심 지표
        sector_strong : 업종 자체가 강세인지 (>0%)
        rs_strong     : 업종 대비 강세인지 (rs>3%)
        signal        : 리포트 태그
    """
    result = {
        'sector_name': '', 'sector_20d': None,
        'rs_vs_sector': None, 'sector_strong': False,
        'rs_strong': False, 'signal': '',
    }
    try:
        from pykrx import stock
        today     = datetime.now().strftime('%Y%m%d')
        sector_map = _get_sector_map(today)
        sector_name = sector_map.get(str(ticker).zfill(6), '')
        result['sector_name'] = sector_name

        # 업종 지수 코드 찾기 (부분 매칭)
        sector_code = None
        for name, code in _SECTOR_INDICES.items():
            if name in sector_name or sector_name in name:
                sector_code = code
                break

        if not sector_code:
            return result

        # 업종 지수 20일 수익률
        end   = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=35)).strftime('%Y%m%d')
        idx_df = stock.get_index_ohlcv_by_date(start, end, sector_code)
        if idx_df is None or idx_df.empty or '종가' not in idx_df.columns:
            return result

        idx_df = idx_df.tail(21)
        if len(idx_df) < 20:
            return result

        sector_20d = (idx_df['종가'].iloc[-1] - idx_df['종가'].iloc[-20]) / idx_df['종가'].iloc[-20] * 100
        rs         = round(stock_20d_return - sector_20d, 1)

        result['sector_20d']   = round(sector_20d, 1)
        result['rs_vs_sector'] = rs
        result['sector_strong'] = sector_20d > 0
        result['rs_strong']     = rs >= 3.0

        # 신호 태그
        if rs >= 5.0 and sector_20d > 0:
            result['signal'] = f'★ 업종강세({sector_name} +{sector_20d:.1f}%) + 상대강도 +{rs:.1f}%'
        elif rs >= 3.0:
            result['signal'] = f'업종대비 강세 (RS +{rs:.1f}%, {sector_name})'
        elif rs >= 0:
            result['signal'] = f'업종동행 (RS {rs:+.1f}%)'
        else:
            result['signal'] = f'업종 언더퍼폼 (RS {rs:+.1f}%)'

    except Exception:
        pass
    return result


def get_market_sector_summary() -> dict:
    """전체 업종 지수 5일 등락 요약 (강세 업종 파악용)"""
    summary = {}
    try:
        from pykrx import stock
        end   = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=15)).strftime('%Y%m%d')
        for name, code in _SECTOR_INDICES.items():
            try:
                df = stock.get_index_ohlcv_by_date(start, end, code)
                if df is not None and not df.empty and len(df) >= 5 and '종가' in df.columns:
                    chg5 = (df['종가'].iloc[-1] - df['종가'].iloc[-5]) / df['종가'].iloc[-5] * 100
                    summary[name] = round(chg5, 1)
            except Exception:
                continue
    except Exception:
        pass
    return summary
