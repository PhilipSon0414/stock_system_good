#!/bin/bash
# 장중 실시간 점화 스캐너 — LaunchAgent 래퍼
# 평일 09:00 launchd 트리거 → --loop이 10분 간격 스캔, 15:35 자동 종료.
# 휴장/장외엔 신선도 가드가 로깅을 건너뜀(stale 미기록).
cd "/Volumes/iMac/Python Programming/Python_Invest/stock_system" || exit 1
mkdir -p logs
LOG="logs/intraday.log"
echo "===== $(date '+%Y-%m-%d %H:%M:%S') 장중 점화 루프 시작 =====" >> "$LOG"
python3 intraday_scanner.py --loop 10 --top 250 >> "$LOG" 2>&1
echo "===== $(date '+%Y-%m-%d %H:%M:%S') 종료 =====" >> "$LOG"
