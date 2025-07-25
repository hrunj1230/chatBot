import pandas as pd
import numpy as np
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tokenizer import ChatbotTokenizer

def load_sample_data():
    """샘플 대화 데이터를 생성합니다."""
    conversations = [
        ["안녕하세요", "안녕하세요! 무엇을 도와드릴까요?"],
        ["오늘 날씨가 어때요?", "오늘은 맑고 화창한 날씨입니다."],
        ["감사합니다", "천만에요! 더 필요한 것이 있으시면 언제든 말씀해 주세요."],
        ["뭐해?", "채팅하고 있어요. 재미있네요!"],
        ["이름이 뭐야?", "저는 AI 챗봇입니다. 반갑습니다!"],
        ["시간이 몇 시야?", "죄송하지만 저는 실시간 시계에 접근할 수 없어요."],
        ["재미있는 이야기 해줘", "옛날 옛적에 한 마을에 착한 아이가 살았어요..."],
        ["수학 문제 도와줘", "어떤 수학 문제인가요? 최선을 다해 도와드리겠습니다."],
        ["음악 추천해줘", "클래식, 팝, 재즈 중 어떤 장르를 좋아하시나요?"],
        ["책 추천해줘", "소설, 자기계발서, 과학책 중 어떤 종류를 원하시나요?"]
    ]
    return conversations

def preprocess_conversations(conversations):
    """대화 데이터를 전처리합니다."""
    questions = []
    answers = []
    
    for conv in conversations:
        if len(conv) >= 2:
            questions.append(conv[0])
            answers.append(conv[1])
    
    return questions, answers

def create_training_data(questions, answers, tokenizer):
    """학습용 데이터를 생성합니다."""
    # 입력과 타겟 시퀀스 생성
    input_sequences = tokenizer.encode(questions)
    target_sequences = tokenizer.encode(answers)
    
    # 타겟 시퀀스에서 다음 토큰 예측을 위한 시프트
    target_sequences_shifted = np.roll(target_sequences, -1, axis=1)
    target_sequences_shifted[:, -1] = 0  # 마지막 토큰은 패딩
    
    return input_sequences, target_sequences_shifted

def save_processed_data(input_sequences, target_sequences, tokenizer, data_dir='data'):
    """전처리된 데이터를 저장합니다."""
    os.makedirs(data_dir, exist_ok=True)
    
    # 데이터 저장
    np.save(os.path.join(data_dir, 'input_sequences.npy'), input_sequences)
    np.save(os.path.join(data_dir, 'target_sequences.npy'), target_sequences)
    
    # 토크나이저 저장
    tokenizer.save(os.path.join(data_dir, 'tokenizer.pkl'))
    
    print(f"데이터가 {data_dir} 폴더에 저장되었습니다.")
    print(f"입력 시퀀스 형태: {input_sequences.shape}")
    print(f"타겟 시퀀스 형태: {target_sequences.shape}")
    print(f"어휘 크기: {tokenizer.get_vocab_size()}")

def main():
    """메인 전처리 함수"""
    print("데이터 전처리를 시작합니다...")
    
    # 샘플 데이터 로드
    conversations = load_sample_data()
    print(f"로드된 대화 수: {len(conversations)}")
    
    # 대화 데이터 전처리
    questions, answers = preprocess_conversations(conversations)
    print(f"전처리된 질문 수: {len(questions)}")
    print(f"전처리된 답변 수: {len(answers)}")
    
    # 토크나이저 초기화 및 학습
    tokenizer = ChatbotTokenizer(max_vocab_size=1000, max_length=20)
    all_texts = questions + answers
    tokenizer.fit(all_texts)
    print("토크나이저 학습 완료")
    
    # 학습 데이터 생성
    input_sequences, target_sequences = create_training_data(questions, answers, tokenizer)
    
    # 데이터 저장
    save_processed_data(input_sequences, target_sequences, tokenizer)
    
    print("데이터 전처리가 완료되었습니다!")

if __name__ == "__main__":
    main() 