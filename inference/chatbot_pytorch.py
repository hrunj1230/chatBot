import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_pytorch import TransformerModel
from utils.tokenizer import ChatbotTokenizer

class ChatbotInference:
    def __init__(self, model_path='models/chatbot_model', data_dir='data'):
        self.model_path = model_path
        self.data_dir = data_dir
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 디바이스: {self.device}")
        
        # 토크나이저 로드
        self.tokenizer = ChatbotTokenizer()
        self.tokenizer.load(os.path.join(data_dir, 'tokenizer.pkl'))
        
        # 모델 초기화 및 로드
        self.load_model()
    
    def load_model(self):
        """모델을 로드합니다."""
        try:
            # 체크포인트 로드
            checkpoint = torch.load(os.path.join(self.model_path, 'best_model.pth'), 
                                  map_location=self.device)
            
            # 모델 파라미터 추출
            vocab_size = checkpoint['vocab_size']
            d_model = checkpoint['d_model']
            num_layers = checkpoint['num_layers']
            num_heads = checkpoint['num_heads']
            d_ff = checkpoint['d_ff']
            
            # 모델 초기화
            self.model = TransformerModel(
                vocab_size=vocab_size,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff
            ).to(self.device)
            
            # 모델 가중치 로드
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print("모델이 성공적으로 로드되었습니다.")
            print(f"모델 파라미터: vocab_size={vocab_size}, d_model={d_model}, "
                  f"num_layers={num_layers}, num_heads={num_heads}, d_ff={d_ff}")
            
        except Exception as e:
            print(f"모델 로드에 실패했습니다: {str(e)}")
            print("먼저 모델을 학습해주세요.")
            return
    
    def generate_response(self, input_text, max_length=20, temperature=1.0):
        """입력 텍스트에 대한 응답을 생성합니다."""
        # 입력 텍스트를 토큰화
        input_sequence = self.tokenizer.encode([input_text])
        input_tensor = torch.LongTensor(input_sequence).to(self.device)
        
        # 응답 생성
        response_tokens = self._generate_sequence(input_tensor, max_length, temperature)
        
        # 토큰을 텍스트로 변환
        response_text = self.tokenizer.decode([response_tokens])[0]
        
        return response_text
    
    def _generate_sequence(self, input_sequence, max_length, temperature):
        """시퀀스를 생성합니다."""
        with torch.no_grad():
            # 패딩 마스크 생성
            mask = self.model.create_padding_mask(input_sequence)
            
            # 인코더 출력
            encoder_output = self.model(input_sequence, mask)
            
            # 시작 토큰 (예: <START>)
            output = torch.tensor([[1]], device=self.device)  # 시작 토큰 ID
            
            for i in range(max_length):
                # 현재 시퀀스로 예측
                current_input = torch.cat([input_sequence, output], dim=1)
                current_mask = self.model.create_padding_mask(current_input)
                
                predictions = self.model(current_input, current_mask)
                
                # 마지막 토큰의 예측만 사용
                next_token_logits = predictions[:, -1, :] / temperature
                
                # 다음 토큰 선택
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 종료 토큰이면 중단
                if next_token.item() == 0:  # <PAD> 토큰
                    break
                
                # 출력에 추가
                output = torch.cat([output, next_token], dim=1)
        
        return output[0].cpu().numpy()
    
    def chat(self):
        """대화형 인터페이스를 시작합니다."""
        print("=== PyTorch AI 챗봇 ===")
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
    print("PyTorch 챗봇을 시작합니다...")
    
    # 챗봇 초기화
    chatbot = ChatbotInference()
    
    # 대화 시작
    chatbot.chat()

if __name__ == "__main__":
    main() 