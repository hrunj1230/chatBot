#!/usr/bin/env python3
"""
모델 재학습 스크립트
새로운 학습 데이터로 모델을 재학습시킵니다.
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_pytorch import TransformerModel
from utils.tokenizer import ChatbotTokenizer
from training.train_manual import ManualChatbotTrainer

def load_training_data():
    """새로운 학습 데이터를 로드합니다."""
    training_file = 'data/training_data.json'
    
    if not os.path.exists(training_file):
        print("학습 데이터 파일이 없습니다.")
        return []
    
    try:
        with open(training_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"로드된 학습 데이터: {len(data)}개")
        return data
    except Exception as e:
        print(f"학습 데이터 로드 실패: {e}")
        return []

def prepare_training_data(training_data):
    """학습 데이터를 모델 학습 형식으로 변환합니다."""
    conversations = []
    
    for item in training_data:
        question = item.get('question', '').strip()
        answer = item.get('answer', '').strip()
        
        if question and answer:
            conversations.append({
                'input': question,
                'output': answer
            })
    
    print(f"준비된 대화 데이터: {len(conversations)}개")
    return conversations

def save_conversations(conversations):
    """대화 데이터를 파일에 저장합니다."""
    output_file = 'data/new_conversations.json'
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        print(f"대화 데이터 저장 완료: {output_file}")
        return True
    except Exception as e:
        print(f"대화 데이터 저장 실패: {e}")
        return False

def retrain_model():
    """모델을 재학습시킵니다."""
    print("=== 모델 재학습 시작 ===")
    
    # 1. 새로운 학습 데이터 로드
    training_data = load_training_data()
    if not training_data:
        print("학습할 데이터가 없습니다.")
        return False
    
    # 2. 학습 데이터 준비
    conversations = prepare_training_data(training_data)
    if not conversations:
        print("준비된 대화 데이터가 없습니다.")
        return False
    
    # 3. 대화 데이터 저장
    if not save_conversations(conversations):
        return False
    
    # 4. 토크나이저 업데이트
    try:
        tokenizer = ChatbotTokenizer()
        tokenizer.load('utils/tokenizer.pkl')
        
        # 새로운 데이터로 토크나이저 업데이트
        all_texts = []
        for conv in conversations:
            all_texts.append(conv['input'])
            all_texts.append(conv['output'])
        
        tokenizer.fit_on_texts(all_texts)
        tokenizer.save('utils/updated_tokenizer.pkl')
        print("토크나이저 업데이트 완료")
        
    except Exception as e:
        print(f"토크나이저 업데이트 실패: {e}")
        return False
    
    # 5. 모델 재학습
    try:
        trainer = ManualChatbotTrainer()
        
        # 새로운 데이터로 학습
        trainer.train_with_new_data(conversations)
        
        print("모델 재학습 완료!")
        return True
        
    except Exception as e:
        print(f"모델 재학습 실패: {e}")
        return False

def main():
    """메인 함수"""
    print(f"모델 재학습 시작: {datetime.now()}")
    
    success = retrain_model()
    
    if success:
        print("✅ 모델 재학습이 성공적으로 완료되었습니다!")
        print("새로운 모델이 'models/retrained_model/' 폴더에 저장되었습니다.")
    else:
        print("❌ 모델 재학습에 실패했습니다.")
    
    print(f"완료 시간: {datetime.now()}")

if __name__ == '__main__':
    main() 