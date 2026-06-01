#!/usr/bin/env python3
"""
매일 저녁 8시 세력 분석 스캔 자동 실행 스케줄러

실행 방법:
  python3 scheduler.py          # 백그라운드에서 계속 실행, 매일 20:00에 스캔

종료:
  Ctrl+C 또는 프로세스 종료

참고:
  macOS 재시작 후에도 자동 실행되려면 setup_launchd.sh 를 사용하세요.
"""

import schedule
import time
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
SCAN_SCRIPT = SCRIPT_DIR / 'daily_scan.py'
PYTHON      = sys.executable
SCAN_TIME   = '20:00'   # 24시간 형식
MARKET      = 'ALL'     # KOSPI / KOSDAQ / ALL


def run_daily_scan():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'\n[{now}] 정기 스캔 시작 → {SCAN_SCRIPT}')
    try:
        subprocess.run([PYTHON, str(SCAN_SCRIPT), MARKET], check=True)
    except subprocess.CalledProcessError as e:
        print(f'  스캔 오류: {e}')
    except Exception as e:
        print(f'  예외 발생: {e}')


def main():
    print('=' * 55)
    print('  세력 분석 자동 스케줄러 시작')
    print(f'  매일 {SCAN_TIME}에 코스피+코스닥 전체 스캔')
    print(f'  Python: {PYTHON}')
    print('  종료: Ctrl+C')
    print('=' * 55)

    schedule.every().day.at(SCAN_TIME).do(run_daily_scan)

    # 오늘 이미 8시가 지났으면 다음 실행까지 시간 출력
    next_run = schedule.next_run()
    print(f'\n  다음 실행 예정: {next_run.strftime("%Y-%m-%d %H:%M")}')
    print('  대기 중...\n')

    while True:
        schedule.run_pending()
        time.sleep(30)  # 30초마다 체크


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\n  스케줄러 종료.')
