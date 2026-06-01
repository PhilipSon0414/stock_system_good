#!/usr/bin/env python3
"""
세력 분석 주식 투자 시스템
────────────────────────────────────────────────────────────────
사용법:
  python main.py                      # 대화형 메뉴
  python main.py -t 005930            # 삼성전자 단일 종목 분석
  python main.py -s                   # 코스피+코스닥 스크리닝
  python main.py -s -m KOSDAQ         # 코스닥만 스크리닝
  python main.py -t 005930 --no-chart # 차트 없이 텍스트만
────────────────────────────────────────────────────────────────
"""

import argparse
import sys

from data_fetcher import get_ohlcv, get_name
from indicators import add_all
from seoryeok import analyze
from order_block import get_order_blocks
from scorer import score
from visualizer import plot
from screener import run as screen_run


def analyze_ticker(ticker: str, show_chart: bool = True):
    print(f"\n분석 중: {ticker} ...")
    name = get_name(ticker)
    df = get_ohlcv(ticker, period_days=400)

    if df.empty or len(df) < 60:
        print(f"  데이터 부족: {ticker}")
        return None

    df = add_all(df)
    df = analyze(df)
    ob = get_order_blocks(df)
    pts, tags = score(df, ob)

    _print_report(df, ob, ticker, name, pts, tags)

    if show_chart:
        plot(df, ob, ticker, name, pts, tags)

    return {'ticker': ticker, 'name': name, 'score': pts, 'tags': tags, 'df': df, 'ob': ob}


def _print_report(df, ob, ticker, name, pts, tags):
    sep = '=' * 65
    latest = df.iloc[-1]
    w5 = df.iloc[-5:]

    print(f'\n{sep}')
    print(f'  {name} ({ticker})')
    print(sep)
    print(f'  현재가 : {latest["Close"]:>12,.0f} 원')
    print(f'  투자점수: {pts:>3d} / 100')

    print(f'\n  [거래량]')
    vr = latest.get('VolRatio', 0)
    print(f'  오늘 거래량 배율: {vr:.1f}x  (평균 대비)')
    if vr >= 5:   print('  → 세력급 거래량 폭증!')
    elif vr >= 3: print('  → 거래량 주목 구간')
    elif vr < 0.5: print('  → 매우 조용 (세력 대기 또는 관심 없음)')

    print(f'\n  [이평선]')
    for p in [5, 20, 60, 120]:
        key = f'MA{p}'
        if key in latest.index and latest[key] > 0:
            diff = (latest['Close'] - latest[key]) / latest[key] * 100
            flag = '▲' if diff > 0 else '▽'
            print(f'  MA{p:3d}: {latest[key]:>10,.0f} 원  ({flag}{abs(diff):.1f}%)')
    if latest.get('MaBull', False):    print('  → ✅ 이평선 정배열 (상승 추세)')
    elif latest.get('MaBear', False):  print('  → ❌ 이평선 역배열 (하락 추세)')
    if latest.get('MaConverge', False): print('  → 🔄 이평선 수렴 중 (폭발 준비 가능)')
    if w5['GoldenCross'].any() if 'GoldenCross' in w5 else False:
        print('  → ⭐ 골든크로스 발생 (최근 5일)')

    print(f'\n  [세력 패턴 (최근 5일)]')
    signals = {
        'SE_Entry':    '세력 진입 신호',
        'SE_Accum':    '세력 매집 구간',
        'SE_BearTrap': '베어트랩 (하락 유도 후 회복)',
        'SE_Rally':    '상승 시작 신호',
        'SE_Exit':     '⚠ 세력 이탈 신호',
    }
    detected = False
    for col, label in signals.items():
        if col in w5.columns and w5[col].any():
            print(f'  • {label}')
            detected = True
    if latest.get('Pullback', False):
        print('  • 눌림목 매수 기회')
        detected = True
    if not detected:
        print('  • 특이 신호 없음')

    print(f'\n  [오더블록]')
    if ob.get('bull'):
        last = ob['bull'][-1]
        print(f'  강세 OB: {last["Low"]:,.0f} ~ {last["High"]:,.0f} 원  ({last["Date"].date()})')
        if ob['near_bull']: print('  → ✅ 현재 강세 OB 근접! 매수 고려 구간')
    else:
        print('  강세 OB: 없음')

    if ob.get('bear'):
        last = ob['bear'][-1]
        print(f'  약세 OB: {last["Low"]:,.0f} ~ {last["High"]:,.0f} 원  ({last["Date"].date()})')
        if ob['near_bear']: print('  → ⚠  현재 약세 OB 근접! 저항 주의')

    print(f'\n  [투자 판단]')
    for t in tags:
        print(f'  • {t}')
    if pts >= 70:
        verdict = '매수 적극 검토 ★★★'
    elif pts >= 55:
        verdict = '매수 검토 ★★'
    elif pts >= 40:
        verdict = '관망 / 모니터링 ★'
    else:
        verdict = '매수 보류'
    print(f'\n  → {verdict}  ({pts}점)')
    print(sep)


def interactive():
    print('\n세력 분석 주식 투자 시스템')
    print('─' * 40)
    print('1. 단일 종목 분석')
    print('2. 전체 종목 스크리닝')
    print('q. 종료')
    choice = input('\n선택: ').strip()

    if choice == '1':
        ticker = input('종목 코드 (예: 005930): ').strip()
        analyze_ticker(ticker)
    elif choice == '2':
        mkt = input('시장 선택 (KOSPI / KOSDAQ / ALL, 기본 ALL): ').strip().upper()
        if mkt not in ('KOSPI', 'KOSDAQ', 'ALL'):
            mkt = 'ALL'
        results = screen_run(mkt)
        if results:
            chart_input = input('\n상위 종목 차트 출력? (y/n): ').strip().lower()
            if chart_input == 'y':
                for r in results[:3]:
                    plot(r['df'], r['ob'], r['ticker'], r['name'],
                         r['score'], r['tags'])
    elif choice == 'q':
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='세력 분석 주식 투자 시스템')
    parser.add_argument('-t', '--ticker', help='종목 코드 (예: 005930)')
    parser.add_argument('-s', '--screen', action='store_true', help='종목 스크리닝 실행')
    parser.add_argument('-m', '--market', default='ALL',
                        choices=['KOSPI', 'KOSDAQ', 'ALL'])
    parser.add_argument('--no-chart', action='store_true', help='차트 출력 안 함')
    args = parser.parse_args()

    if args.ticker:
        analyze_ticker(args.ticker, show_chart=not args.no_chart)
    elif args.screen:
        results = screen_run(args.market)
        if results and not args.no_chart:
            for r in results[:3]:
                plot(r['df'], r['ob'], r['ticker'], r['name'],
                     r['score'], r['tags'])
    else:
        interactive()


if __name__ == '__main__':
    main()
