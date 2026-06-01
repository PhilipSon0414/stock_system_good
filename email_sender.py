"""
이메일 발송 모듈

설정 방법:
  1. email_config.json 파일을 열고 본인 Gmail 정보를 입력
  2. Gmail → 계정 관리 → 보안 → 앱 비밀번호 생성 (2단계 인증 필요)
  3. 생성된 16자리 앱 비밀번호를 'app_password' 에 입력

보안 주의: email_config.json 을 git에 올리지 마세요.
"""

import smtplib
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / 'email_config.json'


def load_config() -> dict:
    if not _CONFIG_PATH.exists():
        _create_default_config()
        raise FileNotFoundError(
            f'email_config.json 이 없어 기본 파일을 생성했습니다.\n'
            f'파일을 열고 Gmail 정보를 입력해주세요: {_CONFIG_PATH}'
        )
    with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def _create_default_config():
    default = {
        "sender_email": "your_gmail@gmail.com",
        "app_password": "xxxx xxxx xxxx xxxx",
        "recipient_email": "your_gmail@gmail.com",
        "enabled": False,
        "_note": "Gmail 앱 비밀번호 발급: myaccount.google.com/apppasswords (2단계 인증 필요)"
    }
    with open(_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(default, f, ensure_ascii=False, indent=2)


def send_report(subject: str, body: str, attachment_path: str = None) -> bool:
    try:
        cfg = load_config()
    except FileNotFoundError as e:
        print(f'  [이메일] 설정 없음: {e}')
        return False

    if not cfg.get('enabled', False):
        print('  [이메일] 비활성화 상태 (email_config.json → "enabled": true 로 변경)')
        return False

    sender = cfg['sender_email']
    password = cfg['app_password'].replace(' ', '')
    recipient = cfg['recipient_email']

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
        encoders.encode_base64(part)
        filename = os.path.basename(attachment_path)
        part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
        msg.attach(part)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender, password)
            smtp.sendmail(sender, recipient, msg.as_string())
        print(f'  [이메일] 발송 완료 → {recipient}')
        return True
    except smtplib.SMTPAuthenticationError:
        print('  [이메일] 인증 실패: 앱 비밀번호를 확인하세요')
        return False
    except Exception as e:
        print(f'  [이메일] 발송 실패: {e}')
        return False
