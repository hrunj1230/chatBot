import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_pytorch import TransformerModel
from utils.tokenizer import ChatbotTokenizer

class ManualChatbotTrainer:
    def __init__(self, vocab_size, d_model=256, num_layers=6, num_heads=8, d_ff=1024):
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, factor=0.5)
        
        # 학습 히스토리
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
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
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
                'Acc': f'{correct/(target != 0).sum().item():.4f}' if (target != 0).sum().item() > 0 else '0.0000'
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, avg_accuracy
    
    def validate_epoch(self, dataloader):
        """검증 에포크"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for input_seq, target_seq in dataloader:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                # 패딩 마스크 생성
                mask = self.create_padding_mask(input_seq)
                
                # Forward pass
                output = self.model(input_seq, mask)
                
                # 손실 계산
                output = output[:, :-1, :].contiguous().view(-1, self.vocab_size)
                target = target_seq[:, 1:].contiguous().view(-1)
                
                loss = self.criterion(output, target)
                
                # 통계 계산
                total_loss += loss.item()
                
                # 정확도 계산
                pred = output.argmax(dim=-1)
                correct = (pred == target).sum().item()
                total_correct += correct
                total_tokens += (target != 0).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, avg_accuracy
    
    def train(self, train_dataloader, val_dataloader=None, epochs=100, save_path='models/manual_chatbot_model'):
        """모델 학습"""
        print("메뉴얼 챗봇 학습을 시작합니다...")
        print(f"총 에포크: {epochs}")
        print(f"학습 데이터 배치 수: {len(train_dataloader)}")
        if val_dataloader:
            print(f"검증 데이터 배치 수: {len(val_dataloader)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            print(f"\n에포크 {epoch+1}/{epochs}")
            
            # 학습
            train_loss, train_acc = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            print(f"학습 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            
            # 검증
            if val_dataloader:
                val_loss, val_acc = self.validate_epoch(val_dataloader)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                print(f"검증 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
                
                # 학습률 스케줄러 업데이트
                self.scheduler.step(val_loss)
                
                # 최고 성능 모델 저장
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model(save_path, is_best=True)
                    print(f"새로운 최고 성능 모델 저장 (Val Loss: {val_loss:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping: {patience} 에포크 동안 개선 없음")
                        break
            else:
                # 검증 데이터가 없으면 주기적으로 저장
                if (epoch + 1) % 10 == 0:
                    self.save_model(save_path, is_best=False, epoch=epoch+1)
        
        # 최종 모델 저장
        self.save_model(save_path, is_best=False, epoch='final')
        
        # 학습 히스토리 플롯
        self.plot_training_history()
        
        print("메뉴얼 챗봇 학습이 완료되었습니다!")
    
    def save_model(self, save_path, is_best=False, epoch=None):
        """모델 저장"""
        os.makedirs(save_path, exist_ok=True)
        
        if is_best:
            filename = 'best_model.pth'
        elif epoch:
            filename = f'model_epoch_{epoch}.pth'
        else:
            filename = 'final_model.pth'
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        torch.save(checkpoint, os.path.join(save_path, filename))
        print(f"모델이 {os.path.join(save_path, filename)}에 저장되었습니다.")
    
    def plot_training_history(self):
        """학습 히스토리 플롯"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 손실 플롯
        ax1.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 정확도 플롯
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        if self.val_accuracies:
            ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

def load_manual_training_data(data_dir='data'):
    """메뉴얼 학습 데이터를 로드합니다."""
    try:
        input_sequences = np.load(os.path.join(data_dir, 'manual_input_sequences.npy'))
        target_sequences = np.load(os.path.join(data_dir, 'manual_target_sequences.npy'))
        
        print(f"메뉴얼 학습 데이터 로드 완료")
        print(f"입력 시퀀스 형태: {input_sequences.shape}")
        print(f"타겟 시퀀스 형태: {target_sequences.shape}")
        
        return input_sequences, target_sequences
        
    except FileNotFoundError:
        print("메뉴얼 학습 데이터를 찾을 수 없습니다.")
        print("먼저 utils/manual_preprocessing.py를 실행해주세요.")
        return None, None

def create_manual_dataloader(input_sequences, target_sequences, batch_size=16, train_ratio=0.8):
    """메뉴얼 데이터 로더를 생성합니다."""
    # 데이터 분할
    n_samples = len(input_sequences)
    n_train = int(n_samples * train_ratio)
    
    # 학습/검증 데이터 분할
    train_input = input_sequences[:n_train]
    train_target = target_sequences[:n_train]
    val_input = input_sequences[n_train:]
    val_target = target_sequences[n_train:]
    
    # 텐서 변환
    train_input_tensor = torch.LongTensor(train_input)
    train_target_tensor = torch.LongTensor(train_target)
    val_input_tensor = torch.LongTensor(val_input)
    val_target_tensor = torch.LongTensor(val_target)
    
    # 데이터셋 생성
    train_dataset = TensorDataset(train_input_tensor, train_target_tensor)
    val_dataset = TensorDataset(val_input_tensor, val_target_tensor)
    
    # 데이터 로더 생성
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"학습 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(val_dataset)}개")
    
    return train_dataloader, val_dataloader

def main():
    """메인 학습 함수"""
    print("메뉴얼 챗봇 학습을 시작합니다...")
    
    # 1. 토크나이저 로드
    try:
        tokenizer = ChatbotTokenizer()
        tokenizer.load('data/manual_tokenizer.pkl')
        vocab_size = tokenizer.get_vocab_size()
        print(f"토크나이저 로드 완료 - 어휘 크기: {vocab_size}")
    except:
        print("토크나이저를 찾을 수 없습니다. 먼저 utils/manual_preprocessing.py를 실행해주세요.")
        return
    
    # 2. 학습 데이터 로드
    input_sequences, target_sequences = load_manual_training_data()
    if input_sequences is None:
        return
    
    # 3. 데이터 로더 생성
    train_dataloader, val_dataloader = create_manual_dataloader(
        input_sequences, target_sequences, batch_size=8, train_ratio=0.8
    )
    
    # 4. 트레이너 초기화
    trainer = ManualChatbotTrainer(
        vocab_size=vocab_size,
        d_model=256,
        num_layers=6,
        num_heads=8,
        d_ff=1024
    )
    
    # 5. 모델 학습
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=100,
        save_path='models/manual_chatbot_model'
    )
    
    print("메뉴얼 챗봇 학습이 완료되었습니다!")

if __name__ == "__main__":
    main() 