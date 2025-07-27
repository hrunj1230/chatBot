#!/usr/bin/env python3
"""
경량 파인튜닝 모듈
효율적인 한국어 챗봇 학습을 위한 경량화된 파인튜닝 방식을 구현합니다.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from collections import deque
import threading
import time

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_pytorch import TransformerModel
from utils.tokenizer import ChatbotTokenizer

class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) 레이어"""
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # LoRA 가중치 (저랭크 행렬)
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        
        # 스케일링 팩터
        self.scaling = alpha / rank
        
    def forward(self, x):
        return x @ self.lora_A.T @ self.lora_B.T * self.scaling

class AdapterLayer(nn.Module):
    """Adapter 레이어"""
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.up_proj(self.activation(self.down_proj(x)))

class LightweightFineTuner:
    def __init__(self, model_path='models/checkpoints/best_model.pth'):
        """
        경량 파인튜닝 초기화
        
        Args:
            model_path: 기본 모델 경로
        """
        self.model_path = 'models/manual_chatbot_model/best_model.pth'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델과 토크나이저
        self.base_model = None
        self.tokenizer = None
        
        # 경량화 컴포넌트
        self.lora_layers = {}
        self.adapter_layers = {}
        
        # 학습 설정
        self.learning_rate = 1e-4
        self.batch_size = 4
        self.max_length = 128
        
        # 학습 데이터 버퍼
        self.training_buffer = deque(maxlen=1000)
        self.is_training = False
        self.training_lock = threading.Lock()
        
        # 성능 통계
        self.training_count = 0
        self.last_training_time = None
        
        print(f"경량 파인튜닝 초기화 완료 (장치: {self.device})")
    
    def load_base_model(self):
        """기본 모델 로드"""
        try:
            # 토크나이저 로드
            self.tokenizer = ChatbotTokenizer()
            self.tokenizer.load('data/manual_tokenizer.pkl')
            
            # 기본 모델 로드
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            vocab_size = checkpoint.get('vocab_size', self.tokenizer.get_vocab_size())
            d_model = checkpoint.get('d_model', 256)
            num_layers = checkpoint.get('num_layers', 6)
            num_heads = checkpoint.get('num_heads', 8)
            d_ff = checkpoint.get('d_ff', 1024)
            
            self.base_model = TransformerModel(
                vocab_size=vocab_size,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff
            ).to(self.device)
            
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.base_model.eval()
            
            # 모델 파라미터 고정 (경량 파인튜닝을 위해)
            for param in self.base_model.parameters():
                param.requires_grad = False
            
            print(f"기본 모델 로드 완료: {self.device}")
            return True
            
        except Exception as e:
            print(f"기본 모델 로드 실패: {e}")
            return False
    
    def add_lora_to_model(self, target_modules=['attention', 'ffn']):
        """모델에 LoRA 레이어 추가"""
        if not self.base_model:
            print("기본 모델이 로드되지 않았습니다.")
            return False
        
        try:
            for name, module in self.base_model.named_modules():
                if any(target in name for target in target_modules):
                    if isinstance(module, nn.Linear):
                        # LoRA 레이어 추가
                        lora_layer = LoRALayer(
                            module.in_features, 
                            module.out_features,
                            rank=8,
                            alpha=16
                        ).to(self.device)
                        
                        self.lora_layers[name] = lora_layer
                        
                        # 원본 forward 함수 저장
                        original_forward = module.forward
                        
                        # LoRA가 적용된 forward 함수
                        def lora_forward(x, original_forward=original_forward, lora_layer=lora_layer):
                            base_output = original_forward(x)
                            lora_output = lora_layer(x)
                            return base_output + lora_output
                        
                        module.forward = lora_forward
            
            print(f"LoRA 레이어 추가 완료: {len(self.lora_layers)}개")
            return True
            
        except Exception as e:
            print(f"LoRA 추가 실패: {e}")
            return False
    
    def add_adapters_to_model(self, target_modules=['attention', 'ffn']):
        """모델에 Adapter 레이어 추가"""
        if not self.base_model:
            print("기본 모델이 로드되지 않았습니다.")
            return False
        
        try:
            for name, module in self.base_model.named_modules():
                if any(target in name for target in target_modules):
                    if hasattr(module, 'd_model'):
                        # Adapter 레이어 추가
                        adapter_layer = AdapterLayer(
                            module.d_model,
                            adapter_size=64
                        ).to(self.device)
                        
                        self.adapter_layers[name] = adapter_layer
                        
                        # 원본 forward 함수 저장
                        original_forward = module.forward
                        
                        # Adapter가 적용된 forward 함수
                        def adapter_forward(x, original_forward=original_forward, adapter_layer=adapter_layer):
                            base_output = original_forward(x)
                            adapter_output = adapter_layer(base_output)
                            return base_output + adapter_output
                        
                        module.forward = adapter_forward
            
            print(f"Adapter 레이어 추가 완료: {len(self.adapter_layers)}개")
            return True
            
        except Exception as e:
            print(f"Adapter 추가 실패: {e}")
            return False
    
    def add_training_data(self, question, answer, quality_score=1.0):
        """학습 데이터 추가"""
        training_data = {
            'question': question,
            'answer': answer,
            'quality_score': quality_score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_buffer.append(training_data)
        
        # 버퍼가 충분히 차면 학습 실행
        if len(self.training_buffer) >= 50:
            self.trigger_training()
    
    def trigger_training(self):
        """학습 실행 트리거"""
        if self.is_training:
            return
        
        with self.training_lock:
            if self.is_training:
                return
            
            self.is_training = True
        
        # 별도 스레드에서 학습 실행
        training_thread = threading.Thread(target=self._perform_training)
        training_thread.daemon = True
        training_thread.start()
    
    def _perform_training(self):
        """실제 경량 파인튜닝 수행"""
        try:
            print(f"경량 파인튜닝 시작: {len(self.training_buffer)}개 데이터")
            
            # 현재 버퍼의 데이터 가져오기
            training_data = list(self.training_buffer)
            self.training_buffer.clear()
            
            # 품질이 좋은 데이터만 선택
            high_quality_data = [
                data for data in training_data 
                if data['quality_score'] > 0.5
            ]
            
            if not high_quality_data:
                print("학습할 고품질 데이터가 없습니다.")
                return
            
            # 경량 파인튜닝 실행
            self._lightweight_finetune(high_quality_data)
            
            # 학습 통계 업데이트
            self.training_count += 1
            self.last_training_time = datetime.now()
            
            print(f"경량 파인튜닝 완료: {len(high_quality_data)}개 데이터")
            
        except Exception as e:
            print(f"경량 파인튜닝 실패: {e}")
        finally:
            self.is_training = False
    
    def _lightweight_finetune(self, training_data):
        """경량 파인튜닝 실행"""
        if not self.base_model or not self.tokenizer:
            print("모델이나 토크나이저가 로드되지 않았습니다.")
            return
        
        try:
            # 학습 모드로 전환
            self.base_model.train()
            
            # LoRA와 Adapter 파라미터만 학습 가능하게 설정
            trainable_params = []
            
            # LoRA 파라미터
            for lora_layer in self.lora_layers.values():
                trainable_params.extend(lora_layer.parameters())
            
            # Adapter 파라미터
            for adapter_layer in self.adapter_layers.values():
                trainable_params.extend(adapter_layer.parameters())
            
            # 옵티마이저 설정 (경량 파라미터만)
            optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
            
            # 미니 배치 학습
            for i in range(0, len(training_data), self.batch_size):
                batch = training_data[i:i+self.batch_size]
                
                # 배치 데이터 준비
                questions = [data['question'] for data in batch]
                answers = [data['answer'] for data in batch]
                
                # 토크나이징
                question_tokens = self.tokenizer.encode(questions)
                answer_tokens = self.tokenizer.encode(answers)
                
                # 패딩 처리
                max_q_len = max(len(tokens) for tokens in question_tokens)
                max_a_len = max(len(tokens) for tokens in answer_tokens)
                
                padded_questions = []
                padded_answers = []
                
                for q_tokens, a_tokens in zip(question_tokens, answer_tokens):
                    # 질문 패딩
                    padded_q = q_tokens + [0] * (max_q_len - len(q_tokens))
                    padded_questions.append(padded_q)
                    
                    # 답변 패딩
                    padded_a = a_tokens + [0] * (max_a_len - len(a_tokens))
                    padded_answers.append(padded_a)
                
                # 텐서 변환
                question_tensor = torch.tensor(padded_questions, dtype=torch.long).to(self.device)
                answer_tensor = torch.tensor(padded_answers, dtype=torch.long).to(self.device)
                
                # 순전파
                optimizer.zero_grad()
                
                # 질문을 입력으로 사용하여 답변 생성
                output = self.base_model(question_tensor)
                
                # 손실 계산 (교차 엔트로피)
                loss = F.cross_entropy(
                    output.view(-1, output.size(-1)), 
                    answer_tensor.view(-1),
                    ignore_index=0  # 패딩 토큰 무시
                )
                
                # 역전파
                loss.backward()
                optimizer.step()
                
                print(f"배치 {i//self.batch_size + 1} 손실: {loss.item():.4f}")
            
            # 평가 모드로 전환
            self.base_model.eval()
            
            # 경량 파인튜닝된 모델 저장
            self._save_finetuned_model()
            
        except Exception as e:
            print(f"경량 파인튜닝 실행 실패: {e}")
    
    def _save_finetuned_model(self):
        """경량 파인튜닝된 모델 저장"""
        try:
            # LoRA 가중치 저장
            lora_weights = {}
            for name, lora_layer in self.lora_layers.items():
                lora_weights[name] = {
                    'lora_A': lora_layer.lora_A.data.cpu(),
                    'lora_B': lora_layer.lora_B.data.cpu(),
                    'scaling': lora_layer.scaling
                }
            
            # Adapter 가중치 저장
            adapter_weights = {}
            for name, adapter_layer in self.adapter_layers.items():
                adapter_weights[name] = {
                    'down_proj': adapter_layer.down_proj.state_dict(),
                    'up_proj': adapter_layer.up_proj.state_dict()
                }
            
            # 전체 모델 저장
            save_path = 'models/lightweight_finetuned_model.pth'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            torch.save({
                'base_model_state_dict': self.base_model.state_dict(),
                'lora_weights': lora_weights,
                'adapter_weights': adapter_weights,
                'training_count': self.training_count,
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'vocab_size': self.tokenizer.get_vocab_size(),
                'd_model': self.base_model.d_model,
                'num_layers': self.base_model.num_layers,
                'num_heads': self.base_model.num_heads,
                'd_ff': self.base_model.d_ff
            }, save_path)
            
            print(f"경량 파인튜닝 모델 저장 완료: {save_path}")
            
        except Exception as e:
            print(f"모델 저장 실패: {e}")
    
    def generate_response(self, user_input):
        """경량 파인튜닝된 모델로 응답 생성"""
        if not self.base_model or not self.tokenizer:
            return "모델이 로드되지 않았습니다."
        
        try:
            # 입력 토크나이징
            input_tokens = self.tokenizer.encode([user_input])
            input_tensor = torch.tensor(input_tokens, dtype=torch.long).to(self.device)
            
            # 추론
            with torch.no_grad():
                output = self.base_model(input_tensor)
                
                # 다음 토큰 예측
                next_token = torch.argmax(output[0, -1, :])
                
                # 간단한 응답 생성 (실제로는 더 복잡한 디코딩 필요)
                response_tokens = [next_token.item()]
                
                # 토큰을 텍스트로 변환
                response = self.tokenizer.decode([response_tokens])
                
                return response if response else "응답을 생성할 수 없습니다."
                
        except Exception as e:
            print(f"응답 생성 실패: {e}")
            return "응답 생성 중 오류가 발생했습니다."
    
    def get_stats(self):
        """학습 통계 반환"""
        return {
            'training_count': self.training_count,
            'buffer_size': len(self.training_buffer),
            'lora_layers': len(self.lora_layers),
            'adapter_layers': len(self.adapter_layers),
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'is_training': self.is_training
        }

# 전역 인스턴스
lightweight_finetuner = None

def initialize_lightweight_finetuner():
    """경량 파인튜너 초기화"""
    global lightweight_finetuner
    lightweight_finetuner = LightweightFineTuner()
    
    # 기본 모델 로드
    if lightweight_finetuner.load_base_model():
        # LoRA 추가
        lightweight_finetuner.add_lora_to_model()
        # Adapter 추가
        lightweight_finetuner.add_adapters_to_model()
        return True
    else:
        return False

def get_lightweight_finetuner():
    """경량 파인튜너 인스턴스 반환"""
    return lightweight_finetuner 