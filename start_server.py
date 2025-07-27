#!/usr/bin/env python3
"""
Gunicorn 서버 시작 스크립트
"""
import os
import sys
import subprocess

def main():
    """Gunicorn 서버를 시작합니다."""
    print("Gunicorn 서버를 시작합니다...")
    
    # Gunicorn 명령어 구성
    cmd = [
        "gunicorn",
        "--bind", "0.0.0.0:5000",
        "--workers", "2",
        "--timeout", "30",
        "--access-logfile", "-",
        "--error-logfile", "-",
        "--log-level", "info",
        "server.app:app"
    ]
    
    try:
        # Gunicorn 서버 시작
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Gunicorn 서버 시작 실패: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("서버가 중단되었습니다.")
        sys.exit(0)

if __name__ == "__main__":
    main() 