#!/usr/bin/env python3
"""
실시간 학습 시스템
서버에서 대화하면서 실시간으로 모델을 학습시킵니다.
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from collections import deque
import threading
import time

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_pytorch import TransformerModel
from utils.tokenizer import ChatbotTokenizer

class RealtimeLearner:
    def __init__(self, buffer_size=100, learning_threshold=50):
        """
        실시간 학습기 초기화
        
        Args:
            buffer_size: 학습 버퍼 크기
            learning_threshold: 학습 실행 임계값
        """
        self.buffer_size = buffer_size
        self.learning_threshold = learning_threshold
        self.conversation_buffer = deque(maxlen=buffer_size)
        self.learning_lock = threading.Lock()
        self.is_learning = False
        
        # 모델과 토크나이저
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 학습 통계
        self.total_conversations = 0
        self.learning_count = 0
        self.last_learning_time = None
        
        # 세무회계 전문 데이터
        self.accounting_data = self.load_accounting_data()
        
        print(f"실시간 학습기 초기화 완료 (버퍼 크기: {buffer_size}, 임계값: {learning_threshold})")
    
    def load_accounting_data(self):
        """세무회계 전문 데이터 로드"""
        accounting_data = [
            # 세무 관련
            {
                "question": "부가가치세 신고는 언제 하나요?",
                "answer": "부가가치세는 매 분기 말 다음 달 25일까지 신고합니다. 예를 들어 1분기(1-3월)는 4월 25일까지 신고해야 합니다."
            },
            {
                "question": "종합소득세 신고 기간은?",
                "answer": "종합소득세는 매년 5월 1일부터 5월 31일까지 신고합니다. 이 기간을 '이번 달'이라고 합니다."
            },
            {
                "question": "법인세 신고는 언제 하나요?",
                "answer": "법인세는 사업연도 종료일로부터 3개월 이내에 신고합니다. 예를 들어 12월 31일이 사업연도 종료일이면 다음 해 3월 31일까지 신고해야 합니다."
            },
            {
                "question": "부가가치세 면세 대상은?",
                "answer": "부가가치세 면세 대상에는 의료서비스, 교육서비스, 도서류, 신문, 여객운송, 금융보험서비스 등이 있습니다."
            },
            {
                "question": "간이과세자는 무엇인가요?",
                "answer": "간이과세자는 연매출 8,000만원 미만인 사업자로, 부가가치세를 간이하게 계산하여 신고합니다."
            },
            
            # 회계 관련
            {
                "question": "복식부기란 무엇인가요?",
                "answer": "복식부기는 모든 거래를 차변과 대변으로 기록하는 회계방식입니다. 자산=부채+자본의 등식을 항상 유지합니다."
            },
            {
                "question": "대차대조표는 무엇인가요?",
                "answer": "대차대조표는 특정 시점의 재무상태를 보여주는 재무제표입니다. 자산, 부채, 자본을 표시합니다."
            },
            {
                "question": "손익계산서는 무엇인가요?",
                "answer": "손익계산서는 일정 기간의 경영성과를 보여주는 재무제표입니다. 수익, 비용, 당기순이익을 표시합니다."
            },
            {
                "question": "현금흐름표는 무엇인가요?",
                "answer": "현금흐름표는 일정 기간의 현금 유입과 유출을 보여주는 재무제표입니다. 영업활동, 투자활동, 재무활동으로 구분합니다."
            },
            {
                "question": "감가상각이란 무엇인가요?",
                "answer": "감가상각은 고정자산의 사용에 따른 가치 감소를 회계적으로 인식하는 방법입니다. 매년 일정 비율로 비용을 인식합니다."
            },
            
            # 챗봇 관련
            {
                "question": "챗봇이란 무엇인가요?",
                "answer": "챗봇은 인공지능을 활용한 대화형 프로그램입니다. 사용자의 질문에 자동으로 답변하고 대화를 나눌 수 있습니다."
            },
            {
                "question": "AI 챗봇의 장점은?",
                "answer": "AI 챗봇의 장점은 24시간 서비스 가능, 빠른 응답, 일관된 답변, 비용 절약, 고객 만족도 향상 등이 있습니다."
            },
            {
                "question": "챗봇은 어떻게 학습하나요?",
                "answer": "챗봇은 대화 데이터를 통해 학습합니다. 질문과 답변 쌍을 많이 제공할수록 더 정확하고 유용한 답변을 할 수 있습니다."
            },
            {
                "question": "챗봇의 응답 품질을 높이는 방법은?",
                "answer": "챗봇의 응답 품질을 높이려면 다양한 질문 패턴, 정확한 답변, 맥락 이해, 지속적인 학습 데이터 추가가 필요합니다."
            },
            {
                "question": "챗봇이 할 수 있는 일은?",
                "answer": "챗봇은 고객 문의 응답, 정보 제공, 예약 관리, 주문 처리, 상담 안내, FAQ 답변 등 다양한 업무를 수행할 수 있습니다."
            }
        ]
        
        print(f"세무회계 전문 데이터 로드 완료: {len(accounting_data)}개")
        return accounting_data
    
    def load_model(self):
        """모델과 토크나이저 로드"""
        try:
            # 토크나이저 로드
            self.tokenizer = ChatbotTokenizer()
            self.tokenizer.load('utils/tokenizer.pkl')
            
            # 모델 로드
            checkpoint = torch.load('models/checkpoints/best_model.pth', map_location=self.device)
            
            vocab_size = checkpoint.get('vocab_size', self.tokenizer.get_vocab_size())
            d_model = checkpoint.get('d_model', 256)
            num_layers = checkpoint.get('num_layers', 6)
            num_heads = checkpoint.get('num_heads', 8)
            d_ff = checkpoint.get('d_ff', 1024)
            
            self.model = TransformerModel(
                vocab_size=vocab_size,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"모델 로드 완료: {self.device}")
            return True
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return False
    
    def add_conversation(self, user_input, bot_response, quality_score=1.0):
        """
        대화 데이터 추가
        
        Args:
            user_input: 사용자 입력
            bot_response: 챗봇 응답
            quality_score: 대화 품질 점수 (0.0 ~ 1.0)
        """
        conversation = {
            'user_input': user_input,
            'bot_response': bot_response,
            'quality_score': quality_score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.conversation_buffer.append(conversation)
        self.total_conversations += 1
        
        # 임계값에 도달하면 학습 실행
        if len(self.conversation_buffer) >= self.learning_threshold:
            self.trigger_learning()
    
    def add_accounting_data(self, question, answer):
        """세무회계 데이터 추가"""
        accounting_data = {
            'question': question,
            'answer': answer,
            'category': 'accounting',
            'timestamp': datetime.now().isoformat()
        }
        
        # 세무회계 데이터 파일에 저장
        accounting_file = 'data/accounting_data.json'
        os.makedirs(os.path.dirname(accounting_file), exist_ok=True)
        
        try:
            if os.path.exists(accounting_file):
                with open(accounting_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            existing_data.append(accounting_data)
            
            with open(accounting_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            print(f"세무회계 데이터 추가: {question}")
            
        except Exception as e:
            print(f"세무회계 데이터 저장 실패: {e}")
    
    def trigger_learning(self):
        """학습 실행 트리거"""
        if self.is_learning:
            return
        
        with self.learning_lock:
            if self.is_learning:
                return
            
            self.is_learning = True
        
        # 별도 스레드에서 학습 실행
        learning_thread = threading.Thread(target=self._perform_learning)
        learning_thread.daemon = True
        learning_thread.start()
    
    def _perform_learning(self):
        """실제 학습 수행"""
        try:
            print(f"실시간 학습 시작: {len(self.conversation_buffer)}개 대화")
            
            # 현재 버퍼의 대화 데이터 가져오기
            conversations = list(self.conversation_buffer)
            self.conversation_buffer.clear()
            
            # 학습 데이터 준비
            training_data = []
            
            # 대화 데이터 추가
            for conv in conversations:
                if conv['quality_score'] > 0.5:  # 품질이 좋은 대화만
                    training_data.append({
                        'input': conv['user_input'],
                        'output': conv['bot_response']
                    })
            
            # 세무회계 전문 데이터 추가 (랜덤 선택)
            import random
            num_accounting = min(10, len(self.accounting_data))
            selected_accounting = random.sample(self.accounting_data, num_accounting)
            
            for data in selected_accounting:
                training_data.append({
                    'input': data['question'],
                    'output': data['answer']
                })
            
            if not training_data:
                print("학습할 데이터가 없습니다.")
                return
            
            # 토크나이저 업데이트
            all_texts = []
            for data in training_data:
                all_texts.append(data['input'])
                all_texts.append(data['output'])
            
            self.tokenizer.fit_on_texts(all_texts)
            
            # 간단한 온라인 학습 (미니 배치)
            self._online_learning(training_data)
            
            # 학습 통계 업데이트
            self.learning_count += 1
            self.last_learning_time = datetime.now()
            
            print(f"실시간 학습 완료: {len(training_data)}개 데이터")
            
        except Exception as e:
            print(f"실시간 학습 실패: {e}")
        finally:
            self.is_learning = False
    
    def _online_learning(self, training_data):
        """온라인 학습 수행"""
        if not self.model or not self.tokenizer:
            print("모델이나 토크나이저가 로드되지 않았습니다.")
            return
        
        try:
            # 간단한 온라인 학습 (실제로는 더 복잡한 방법 사용)
            # 여기서는 모델 가중치를 약간 조정하는 방식으로 구현
            
            self.model.train()
            
            # 미니 배치 학습
            batch_size = 4
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                
                # 배치 데이터 준비
                inputs = []
                targets = []
                
                for data in batch:
                    input_seq = self.tokenizer.encode([data['input']])
                    target_seq = self.tokenizer.encode([data['output']])
                    
                    inputs.append(input_seq[0])
                    targets.append(target_seq[0])
                
                # 패딩 처리
                max_input_len = max(len(seq) for seq in inputs)
                max_target_len = max(len(seq) for seq in targets)
                
                padded_inputs = []
                padded_targets = []
                
                for input_seq, target_seq in zip(inputs, targets):
                    # 입력 패딩
                    padded_input = input_seq + [0] * (max_input_len - len(input_seq))
                    padded_inputs.append(padded_input)
                    
                    # 타겟 패딩
                    padded_target = target_seq + [0] * (max_target_len - len(target_seq))
                    padded_targets.append(padded_target)
                
                # 텐서 변환
                input_tensor = torch.tensor(padded_inputs, dtype=torch.long).to(self.device)
                target_tensor = torch.tensor(padded_targets, dtype=torch.long).to(self.device)
                
                # 순전파
                output = self.model(input_tensor)
                
                # 손실 계산 (간단한 MSE)
                loss = torch.nn.functional.mse_loss(output.view(-1, output.size(-1)), target_tensor.view(-1))
                
                # 역전파 (매우 작은 학습률)
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            self.model.eval()
            
            # 모델 저장
            self._save_model()
            
        except Exception as e:
            print(f"온라인 학습 실패: {e}")
    
    def _save_model(self):
        """모델 저장"""
        try:
            # 실시간 학습된 모델 저장
            save_path = 'models/realtime_model.pth'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'vocab_size': self.tokenizer.get_vocab_size(),
                'd_model': self.model.d_model,
                'num_layers': self.model.num_layers,
                'num_heads': self.model.num_heads,
                'd_ff': self.model.d_ff,
                'learning_count': self.learning_count,
                'total_conversations': self.total_conversations,
                'last_learning_time': self.last_learning_time.isoformat() if self.last_learning_time else None
            }, save_path)
            
            # 토크나이저 저장
            self.tokenizer.save('utils/realtime_tokenizer.pkl')
            
            print(f"실시간 학습 모델 저장 완료: {save_path}")
            
        except Exception as e:
            print(f"모델 저장 실패: {e}")
    
    def get_stats(self):
        """학습 통계 반환"""
        return {
            'total_conversations': self.total_conversations,
            'learning_count': self.learning_count,
            'buffer_size': len(self.conversation_buffer),
            'last_learning_time': self.last_learning_time.isoformat() if self.last_learning_time else None,
            'is_learning': self.is_learning
        }
    
    def get_accounting_suggestions(self, user_input):
        """세무회계 관련 제안 반환"""
        suggestions = []
        
        # 사용자 입력과 유사한 세무회계 질문 찾기
        for data in self.accounting_data:
            if any(keyword in user_input.lower() for keyword in ['세무', '회계', '세금', '부가가치세', '소득세', '법인세']):
                suggestions.append(data['question'])
        
        return suggestions[:3]  # 최대 3개 제안

# 전역 실시간 학습기 인스턴스
realtime_learner = None

def initialize_realtime_learner():
    """실시간 학습기 초기화"""
    global realtime_learner
    realtime_learner = RealtimeLearner()
    return realtime_learner.load_model()

def get_realtime_learner():
    """실시간 학습기 인스턴스 반환"""
    return realtime_learner 