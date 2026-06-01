#!/bin/bash
# macOS LaunchAgent 등록 스크립트
# 이 스크립트를 실행하면 Mac 재시작 후에도 매일 20:00 자동 실행됩니다.
#
# 사용법: bash setup_launchd.sh
#         bash setup_launchd.sh remove   ← 삭제

LABEL="com.stocksystem.dailyscan"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
PYTHON=$(which python3)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCAN_SCRIPT="${SCRIPT_DIR}/daily_scan.py"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "$LOG_DIR"

# ── 삭제 모드 ──────────────────────────────────────────────────────────────
if [ "$1" = "remove" ]; then
    launchctl unload "$PLIST" 2>/dev/null
    rm -f "$PLIST"
    echo "LaunchAgent 제거 완료."
    exit 0
fi

# ── 등록 모드 ──────────────────────────────────────────────────────────────
echo "LaunchAgent 등록 중..."
echo "  Python:  $PYTHON"
echo "  스크립트: $SCAN_SCRIPT"
echo "  실행 시각: 매일 20:00"

cat > "$PLIST" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON}</string>
        <string>${SCAN_SCRIPT}</string>
        <string>ALL</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>20</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>${LOG_DIR}/daily_scan.log</string>

    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/daily_scan_error.log</string>

    <key>WorkingDirectory</key>
    <string>${SCRIPT_DIR}</string>

    <key>RunAtLoad</key>
    <false/>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
EOF

# 이전 등록이 있으면 제거 후 재등록
launchctl unload "$PLIST" 2>/dev/null
launchctl load "$PLIST"

echo ""
echo "✅ 완료! 매일 저녁 20:00에 자동 실행됩니다."
echo ""
echo "  로그 파일: ${LOG_DIR}/daily_scan.log"
echo "  삭제하려면: bash setup_launchd.sh remove"
echo ""
