import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('..')

from models.transformer import TransformerModel
from utils.tokenizer import ChatbotTokenizer

class ChatbotInference:
    def __init__(self, model_path='models/chatbot_model', data_dir='data'):
        self.model_path = model_path
        self.data_dir = data_dir
        
        # 토크나이저 로드
        self.tokenizer = ChatbotTokenizer()
        self.tokenizer.load(os.path.join(data_dir, 'tokenizer.pkl'))
        
        # 모델 초기화 및 로드
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.model = TransformerModel(
            vocab_size=self.vocab_size,
            d_model=128,
            num_layers=4,
            num_heads=8,
            dff=512
        )
        
        # 모델 가중치 로드
        try:
            self.model.load_weights(os.path.join(model_path, 'final_model'))
            print("모델이 성공적으로 로드되었습니다.")
        except:
            print("모델 로드에 실패했습니다. 먼저 모델을 학습해주세요.")
            return
    
    def generate_response(self, input_text, max_length=20, temperature=1.0):
        """입력 텍스트에 대한 응답을 생성합니다."""
        # 입력 텍스트를 토큰화
        input_sequence = self.tokenizer.encode([input_text])
        
        # 응답 생성
        response_tokens = self._generate_sequence(input_sequence[0], max_length, temperature)
        
        # 토큰을 텍스트로 변환
        response_text = self.tokenizer.decode([response_tokens])[0]
        
        return response_text
    
    def _generate_sequence(self, input_sequence, max_length, temperature):
        """시퀀스를 생성합니다."""
        # 입력 시퀀스 준비
        encoder_input = tf.expand_dims(input_sequence, 0)
        
        # 시작 토큰 (예: <START>)
        output = tf.expand_dims([1], 0)  # 시작 토큰 ID
        
        for i in range(max_length):
            # 예측
            predictions = self.model(encoder_input, False, None)
            
            # 마지막 토큰의 예측만 사용
            predictions = predictions[:, -1:, :]
            
            # temperature 적용
            predictions = predictions / temperature
            
            # 다음 토큰 선택
            predicted_id = tf.random.categorical(predictions, num_samples=1)
            
            # 종료 토큰이면 중단
            if predicted_id == 0:  # <PAD> 토큰
                break
            
            # 출력에 추가
            output = tf.concat([output, predicted_id], axis=-1)
        
        return output[0].numpy()
    
    def chat(self):
        """대화형 인터페이스를 시작합니다."""
        print("=== AI 챗봇 ===")
        print("대화를 시작합니다. 'quit'를 입력하면 종료됩니다.")
        print("-" * 30)
        
        while True:
            user_input = input("사용자: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("챗봇: 안녕히 가세요!")
                break
            
            if not user_input:
                continue
            
            try:
                response = self.generate_response(user_input, max_length=20, temperature=0.8)
                print(f"챗봇: {response}")
            except Exception as e:
                print(f"챗봇: 죄송합니다. 응답을 생성하는 중 오류가 발생했습니다. ({str(e)})")
            
            print("-" * 30)

def main():
    """메인 함수"""
    print("챗봇을 시작합니다...")
    
    # 챗봇 초기화
    chatbot = ChatbotInference()
    
    # 대화 시작
    chatbot.chat()

if __name__ == "__main__":
    main() 