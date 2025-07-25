import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_pytorch import TransformerModel
from utils.tokenizer import ChatbotTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm

class ChatbotTrainer:
    def __init__(self, vocab_size, d_model=128, num_layers=4, num_heads=8, d_ff=512):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 디바이스: {self.device}")
        
        # 모델 초기화
        self.model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff
        ).to(self.device)
        
        # 손실 함수와 옵티마이저
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 패딩 토큰 무시
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # 학습 히스토리
        self.train_losses = []
        self.train_accuracies = []
    
    def create_padding_mask(self, seq):
        """패딩 마스크 생성"""
        return (seq != 0).unsqueeze(1).unsqueeze(2)
    
    def train_epoch(self, dataloader):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, (input_seq, target_seq) in enumerate(progress_bar):
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            
            # 패딩 마스크 생성
            mask = self.create_padding_mask(input_seq)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(input_seq, mask)
            
            # 손실 계산 (타겟 시퀀스의 다음 토큰 예측)
            output = output[:, :-1, :].contiguous().view(-1, self.vocab_size)
            target = target_seq[:, 1:].contiguous().view(-1)
            
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # 통계 계산
            total_loss += loss.item()
            
            # 정확도 계산
            pred = output.argmax(dim=-1)
            correct = (pred == target).sum().item()
            total_correct += correct
            total_tokens += (target != 0).sum().item()
            
            # 진행률 업데이트
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct/(target != 0).sum().item():.4f}'
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, avg_accuracy
    
    def train(self, dataloader, epochs=50, save_path='models/chatbot_model'):
        """모델 학습"""
        print("학습을 시작합니다...")
        
        # 저장 디렉토리 생성
        os.makedirs(save_path, exist_ok=True)
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # 학습
            train_loss, train_acc = self.train_epoch(dataloader)
            
            # 히스토리 저장
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # 모델 저장
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': train_loss,
                    'vocab_size': self.vocab_size,
                    'd_model': self.d_model,
                    'num_layers': self.num_layers,
                    'num_heads': self.num_heads,
                    'd_ff': self.d_ff
                }, os.path.join(save_path, 'best_model.pth'))
                print(f"새로운 최고 모델 저장됨 (Loss: {train_loss:.4f})")
            
            # 주기적 저장
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': train_loss
                }, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # 최종 모델 저장
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': train_loss,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff
        }, os.path.join(save_path, 'final_model.pth'))
        
        print(f"학습 완료! 모델이 {save_path}에 저장되었습니다.")
        
        # 학습 히스토리 플롯
        self.plot_training_history()
    
    def plot_training_history(self):
        """학습 히스토리를 시각화합니다."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 손실 그래프
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 정확도 그래프
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history_pytorch.png')
        plt.show()

def load_training_data(data_dir='data'):
    """학습 데이터를 로드합니다."""
    input_sequences = np.load(os.path.join(data_dir, 'input_sequences.npy'))
    target_sequences = np.load(os.path.join(data_dir, 'target_sequences.npy'))
    
    # 토크나이저 로드
    tokenizer = ChatbotTokenizer()
    tokenizer.load(os.path.join(data_dir, 'tokenizer.pkl'))
    
    return input_sequences, target_sequences, tokenizer

def create_dataloader(input_sequences, target_sequences, batch_size=32):
    """PyTorch DataLoader를 생성합니다."""
    # NumPy 배열을 PyTorch 텐서로 변환
    input_tensor = torch.LongTensor(input_sequences)
    target_tensor = torch.LongTensor(target_sequences)
    
    # 데이터셋 생성
    dataset = TensorDataset(input_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def main():
    """메인 학습 함수"""
    print("PyTorch 챗봇 모델 학습을 시작합니다...")
    
    # 데이터 로드
    input_sequences, target_sequences, tokenizer = load_training_data()
    print(f"입력 데이터 형태: {input_sequences.shape}")
    print(f"타겟 데이터 형태: {target_sequences.shape}")
    
    # 데이터로더 생성
    dataloader = create_dataloader(input_sequences, target_sequences, batch_size=16)
    
    # 트레이너 초기화
    vocab_size = tokenizer.get_vocab_size()
    trainer = ChatbotTrainer(
        vocab_size=vocab_size,
        d_model=128,
        num_layers=4,
        num_heads=8,
        d_ff=512
    )
    
    # 모델 학습
    trainer.train(dataloader, epochs=100)
    
    print("학습이 완료되었습니다!")

if __name__ == "__main__":
    main() 