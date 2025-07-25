#!/usr/bin/env python3
"""
TensorFlow Transformer 챗봇 실행 스크립트
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """명령어를 실행하고 결과를 출력합니다."""
    print(f"\n{'='*50}")
    print(f"실행 중: {description}")
    print(f"명령어: {command}")
    print('='*50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True, encoding='utf-8')
        print("성공!")
        if result.stdout:
            print("출력:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"오류 발생: {e}")
        if e.stdout:
            print("표준 출력:")
            print(e.stdout)
        if e.stderr:
            print("오류 출력:")
            print(e.stderr)
        return False

def check_dependencies():
    """필요한 의존성이 설치되어 있는지 확인합니다."""
    print("의존성 확인 중...")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
    except ImportError:
        print("✗ TensorFlow가 설치되지 않았습니다.")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy가 설치되지 않았습니다.")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError:
        print("✗ Pandas가 설치되지 않았습니다.")
        return False
    
    return True

def main():
    """메인 실행 함수"""
    print("TensorFlow Transformer 챗봇 시스템")
    print("="*50)
    
    # 1. 의존성 확인
    if not check_dependencies():
        print("\n의존성 설치가 필요합니다:")
        print("pip install -r requirements.txt")
        return
    
    # 2. 데이터 전처리
    print("\n1단계: 데이터 전처리")
    if not run_command("python utils/data_preprocessing.py", "데이터 전처리"):
        print("데이터 전처리에 실패했습니다.")
        return
    
    # 3. 모델 학습
    print("\n2단계: 모델 학습")
    print("학습은 시간이 오래 걸릴 수 있습니다...")
    
    # 학습 파라미터 설정 (빠른 테스트를 위해 작은 값들 사용)
    train_command = "python training/train.py"
    if not run_command(train_command, "모델 학습"):
        print("모델 학습에 실패했습니다.")
        return
    
    # 4. 챗봇 실행
    print("\n3단계: 챗봇 실행")
    print("대화를 시작합니다. 'quit'를 입력하면 종료됩니다.")
    
    chatbot_command = "python inference/chatbot.py"
    if not run_command(chatbot_command, "챗봇 실행"):
        print("챗봇 실행에 실패했습니다.")
        return

def interactive_mode():
    """대화형 모드로 실행"""
    print("TensorFlow Transformer 챗봇 - 대화형 모드")
    print("="*50)
    
    while True:
        print("\n선택하세요:")
        print("1. 데이터 전처리")
        print("2. 모델 학습")
        print("3. 챗봇 실행")
        print("4. 전체 과정 실행")
        print("5. 종료")
        
        choice = input("\n선택 (1-5): ").strip()
        
        if choice == '1':
            run_command("python utils/data_preprocessing.py", "데이터 전처리")
        elif choice == '2':
            run_command("python training/train.py", "모델 학습")
        elif choice == '3':
            run_command("python inference/chatbot.py", "챗봇 실행")
        elif choice == '4':
            main()
        elif choice == '5':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 1-5 중에서 선택해주세요.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main() 