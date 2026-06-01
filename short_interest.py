"""
5순위: 공매도 잔고 분석

공매도 잔고 비율 + 추세로 쇼트 스퀴즈 가능성 판단.
pykrx 기반으로 최근 20거래일 공매도 잔고 데이터를 조회.

높은 공매도 잔고 + 세력 진입 조합 = 쇼트 스퀴즈 가능성 ↑
공매도 감소 추세 + 세력 진입 = 세력이 숏 포지션 정리를 유도하는 패턴
"""

from datetime import datetime, timedelta
import pandas as pd


def get_short_interest(ticker: str) -> dict:
    """공매도 잔고 비율 및 추세 조회

    Returns:
        ratio     : 최근 공매도잔고 비율 (%)
        trend     : 'increasing' | 'decreasing' | 'stable'
        squeeze   : 쇼트 스퀴즈 가능성 (ratio > 1% + decreasing)
        signal    : 리포트 출력용 태그 문자열
        data_ok   : 데이터 정상 조회 여부
    """
    try:
        from pykrx import stock

        end   = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')

        df = stock.get_shorting_balance_by_date(start, end, ticker)
        if df is None or df.empty or '공매도잔고' not in df.columns:
            return {'data_ok': False}

        # 거래 데이터가 있는 행만 사용
        df = df[df['상장주식수'] > 0].copy()
        if len(df) < 3:
            return {'data_ok': False}

        # 비중(%) 재계산: 공매도잔고 / 상장주식수 * 100
        df['ratio_pct'] = df['공매도잔고'] / df['상장주식수'] * 100
        ratio = float(df['ratio_pct'].iloc[-1])

        # 추세: 최근 5일 평균 vs 이전 5일 평균
        if len(df) >= 10:
            recent = df['ratio_pct'].iloc[-5:].mean()
            prev   = df['ratio_pct'].iloc[-10:-5].mean()
            if prev > 0:
                chg = (recent - prev) / prev
                if chg > 0.10:
                    trend = 'increasing'
                elif chg < -0.10:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
        else:
            trend = 'stable'

        # 5일 변화율 (압박 해소 강도)
        change_rate_5d = None
        if len(df) >= 5:
            old_ratio = float(df['ratio_pct'].iloc[-5])
            if old_ratio > 0:
                change_rate_5d = round((ratio - old_ratio) / old_ratio * 100, 1)

        # 쇼트 스퀴즈 가능성: 잔고 비율 1%+ & 감소 추세
        squeeze = ratio >= 1.0 and trend == 'decreasing'

        # 급격한 공매도 압박 해소: 5일 -20%+ 감소 = 쇼트커버 신호
        rapid_cover = (change_rate_5d is not None and change_rate_5d <= -20.0)

        trend_str = {'increasing': '증가↑', 'decreasing': '감소↓', 'stable': '유지→'}.get(trend, '')
        chg_str   = f' ({change_rate_5d:+.0f}%/5일)' if change_rate_5d is not None else ''

        if rapid_cover and ratio >= 1.0:
            signal = f'★ 공매도 급감 — 쇼트커버 가능 ({ratio:.1f}%{chg_str})'
        elif squeeze:
            signal = f'쇼트스퀴즈 가능 (공매도 {ratio:.1f}% {trend_str}{chg_str})'
        elif ratio >= 2.0:
            signal = f'공매도 과다 ({ratio:.1f}% {trend_str})'
        elif ratio >= 1.0:
            signal = f'공매도 주의 ({ratio:.1f}% {trend_str})'
        else:
            signal = f'공매도 낮음 ({ratio:.1f}%)'

        return {
            'ratio':          ratio,
            'trend':          trend,
            'squeeze':        squeeze,
            'rapid_cover':    rapid_cover,
            'change_rate_5d': change_rate_5d,
            'signal':         signal,
            'data_ok':        True,
        }

    except Exception:
        return {'data_ok': False}
