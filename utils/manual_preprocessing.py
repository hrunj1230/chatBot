import pandas as pd
import numpy as np
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tokenizer import ChatbotTokenizer
from utils.manual_data_collector import ManualDataCollector

class ManualPreprocessor:
    def __init__(self):
        self.tokenizer = None
        self.manual_data = []
    
    def load_manual_dataset(self, dataset_path: str = 'data/manual_dataset.json'):
        """메뉴얼 데이터셋을 로드합니다."""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            qa_pairs = []
            for item in data:
                question = item['question']
                answer = item['answer']
                qa_pairs.append([question, answer])
            
            self.manual_data = qa_pairs
            print(f"메뉴얼 데이터셋 {len(qa_pairs)}개를 로드했습니다.")
            return qa_pairs
            
        except FileNotFoundError:
            print(f"메뉴얼 데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
            return []
        except Exception as e:
            print(f"메뉴얼 데이터셋 로드 실패: {e}")
            return []
    
    def preprocess_manual_data(self, qa_pairs):
        """메뉴얼 데이터를 전처리합니다."""
        questions = []
        answers = []
        
        for qa in qa_pairs:
            if len(qa) >= 2:
                question = qa[0].strip()
                answer = qa[1].strip()
                
                # 빈 데이터 제거
                if question and answer:
                    questions.append(question)
                    answers.append(answer)
        
        print(f"전처리된 질문 수: {len(questions)}")
        print(f"전처리된 답변 수: {len(answers)}")
        
        return questions, answers
    
    def update_tokenizer(self, questions, answers, max_vocab_size=2000, max_length=100):
        """토크나이저를 메뉴얼 데이터로 업데이트합니다."""
        # 기존 토크나이저 로드 시도
        try:
            self.tokenizer = ChatbotTokenizer(max_vocab_size=max_vocab_size, max_length=max_length)
            self.tokenizer.load('data/tokenizer.pkl')
            print("기존 토크나이저를 로드했습니다.")
        except:
            print("기존 토크나이저를 찾을 수 없습니다. 새로 생성합니다.")
            self.tokenizer = ChatbotTokenizer(max_vocab_size=max_vocab_size, max_length=max_length)
        
        # 모든 텍스트로 토크나이저 학습
        all_texts = questions + answers
        self.tokenizer.fit(all_texts)
        
        print(f"토크나이저 어휘 크기: {self.tokenizer.get_vocab_size()}")
        print("토크나이저 학습이 완료되었습니다.")
        
        return self.tokenizer
    
    def create_training_data(self, questions, answers):
        """학습용 데이터를 생성합니다."""
        if not self.tokenizer:
            raise ValueError("토크나이저가 초기화되지 않았습니다.")
        
        # 입력과 타겟 시퀀스 생성
        input_sequences = self.tokenizer.encode(questions)
        target_sequences = self.tokenizer.encode(answers)
        
        # 타겟 시퀀스에서 다음 토큰 예측을 위한 시프트
        target_sequences_shifted = np.roll(target_sequences, -1, axis=1)
        target_sequences_shifted[:, -1] = 0  # 마지막 토큰은 패딩
        
        return input_sequences, target_sequences_shifted
    
    def save_processed_data(self, input_sequences, target_sequences, data_dir='data'):
        """전처리된 데이터를 저장합니다."""
        os.makedirs(data_dir, exist_ok=True)
        
        # 데이터 저장
        np.save(os.path.join(data_dir, 'manual_input_sequences.npy'), input_sequences)
        np.save(os.path.join(data_dir, 'manual_target_sequences.npy'), target_sequences)
        
        # 토크나이저 저장
        self.tokenizer.save(os.path.join(data_dir, 'manual_tokenizer.pkl'))
        
        print(f"메뉴얼 데이터가 {data_dir} 폴더에 저장되었습니다.")
        print(f"입력 시퀀스 형태: {input_sequences.shape}")
        print(f"타겟 시퀀스 형태: {target_sequences.shape}")
        print(f"어휘 크기: {self.tokenizer.get_vocab_size()}")
    
    def create_combined_dataset(self, include_general=True):
        """일반 대화와 메뉴얼 데이터를 결합합니다."""
        combined_qa = []
        
        # 메뉴얼 데이터 추가
        combined_qa.extend(self.manual_data)
        
        # 일반 대화 데이터 추가 (선택사항)
        if include_general:
            try:
                with open('data/conversations.json', 'r', encoding='utf-8') as f:
                    general_data = json.load(f)
                combined_qa.extend(general_data)
                print(f"일반 대화 데이터 {len(general_data)}개를 추가했습니다.")
            except:
                print("일반 대화 데이터를 찾을 수 없습니다.")
        
        print(f"총 {len(combined_qa)}개의 Q&A 쌍을 결합했습니다.")
        return combined_qa

def main():
    """메인 전처리 함수"""
    print("메뉴얼 데이터 전처리를 시작합니다...")
    
    # 1. 메뉴얼 데이터 수집 (없으면 기본 세무회계 데이터 사용)
    collector = ManualDataCollector()
    qa_pairs = collector.create_manual_dataset([], accounting_data=True)
    
    # 2. 전처리기 초기화
    preprocessor = ManualPreprocessor()
    
    # 3. 데이터 전처리
    questions, answers = preprocessor.preprocess_manual_data(qa_pairs)
    
    # 4. 토크나이저 업데이트
    tokenizer = preprocessor.update_tokenizer(questions, answers, max_vocab_size=2000, max_length=100)
    
    # 5. 학습 데이터 생성
    input_sequences, target_sequences = preprocessor.create_training_data(questions, answers)
    
    # 6. 데이터 저장
    preprocessor.save_processed_data(input_sequences, target_sequences)
    
    print("메뉴얼 데이터 전처리가 완료되었습니다!")

if __name__ == "__main__":
    main() 