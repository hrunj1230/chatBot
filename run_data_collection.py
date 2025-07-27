#!/usr/bin/env python3
"""
자동 데이터 수집 실행 스크립트
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("자동 데이터 수집을 시작합니다...")
    
    try:
        from utils.auto_data_collector import run_auto_data_collection
        
        # 자동 데이터 수집 실행
        collected_data = run_auto_data_collection()
        
        print(f"\n✅ 데이터 수집 완료!")
        print(f"📊 수집된 데이터: {len(collected_data)}개")
        
        # 데이터 파일 확인
        if os.path.exists('data/auto_collected_data.json'):
            print("📁 저장된 파일: data/auto_collected_data.json")
            
            # 파일 크기 확인
            file_size = os.path.getsize('data/auto_collected_data.json')
            print(f"📏 파일 크기: {file_size} bytes")
            
            # 데이터 샘플 확인
            import json
            with open('data/auto_collected_data.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"📋 데이터 샘플:")
                for i, item in enumerate(data[:3]):
                    print(f"  {i+1}. Q: {item['question'][:50]}...")
                    print(f"     A: {item['answer'][:50]}...")
                    print(f"     소스: {item['source']}")
                    print()
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터 수집 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 자동 데이터 수집이 성공적으로 완료되었습니다!")
    else:
        print("💥 자동 데이터 수집 중 오류가 발생했습니다.")
        sys.exit(1) 