import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('..')

from models.transformer import TransformerModel
from utils.tokenizer import ChatbotTokenizer
import matplotlib.pyplot as plt

class ChatbotTrainer:
    def __init__(self, vocab_size, d_model=128, num_layers=4, num_heads=8, dff=512):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        
        # 모델 초기화
        self.model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dff=dff
        )
        
        # 옵티마이저와 손실 함수
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
        )
        
        # 메트릭
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
    
    def accuracy_function(self, real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))
        
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)
        
        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)
    
    @tf.function
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        with tf.GradientTape() as tape:
            predictions = self.model(inp, True, None)
            loss = self.loss_function(tar_real, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)
    
    def train(self, dataset, epochs=50, save_path='models/chatbot_model'):
        """모델 학습"""
        print("학습을 시작합니다...")
        
        # 체크포인트 설정
        checkpoint_path = save_path
        ckpt = tf.train.Checkpoint(transformer=self.model, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        
        # 체크포인트가 있으면 로드
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print(f'최신 체크포인트가 복원되었습니다: {ckpt_manager.latest_checkpoint}')
        
        # 학습 루프
        for epoch in range(epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            
            for (batch, (inp, tar)) in enumerate(dataset):
                self.train_step(inp, tar)
                
                if batch % 10 == 0:
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}')
            
            # 체크포인트 저장
            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print(f'체크포인트가 저장되었습니다: {ckpt_save_path}')
            
            print(f'Epoch {epoch + 1} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}')
        
        # 최종 모델 저장
        self.model.save_weights(f'{save_path}/final_model')
        print(f"모델이 {save_path}에 저장되었습니다.")

def load_training_data(data_dir='data'):
    """학습 데이터를 로드합니다."""
    input_sequences = np.load(os.path.join(data_dir, 'input_sequences.npy'))
    target_sequences = np.load(os.path.join(data_dir, 'target_sequences.npy'))
    
    # 토크나이저 로드
    tokenizer = ChatbotTokenizer()
    tokenizer.load(os.path.join(data_dir, 'tokenizer.pkl'))
    
    return input_sequences, target_sequences, tokenizer

def create_dataset(input_sequences, target_sequences, batch_size=32):
    """TensorFlow 데이터셋을 생성합니다."""
    dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences))
    dataset = dataset.shuffle(1000).batch(batch_size, drop_remainder=True)
    return dataset

def main():
    """메인 학습 함수"""
    print("챗봇 모델 학습을 시작합니다...")
    
    # 데이터 로드
    input_sequences, target_sequences, tokenizer = load_training_data()
    print(f"입력 데이터 형태: {input_sequences.shape}")
    print(f"타겟 데이터 형태: {target_sequences.shape}")
    
    # 데이터셋 생성
    dataset = create_dataset(input_sequences, target_sequences, batch_size=16)
    
    # 트레이너 초기화
    vocab_size = tokenizer.get_vocab_size()
    trainer = ChatbotTrainer(
        vocab_size=vocab_size,
        d_model=128,
        num_layers=4,
        num_heads=8,
        dff=512
    )
    
    # 모델 학습
    trainer.train(dataset, epochs=100)
    
    print("학습이 완료되었습니다!")

if __name__ == "__main__":
    main() 