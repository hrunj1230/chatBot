import numpy as np
import pickle
import os
import re
from collections import Counter

class ChatbotTokenizer:
    def __init__(self, max_vocab_size=10000, max_length=50):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.word_index = {}
        self.index_word = {}
        self.fitted = False
        
    def fit(self, texts):
        """텍스트 데이터로 토크나이저를 학습시킵니다."""
        # 모든 텍스트를 단어로 분리
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        # 단어 빈도 계산
        word_counts = Counter(all_words)
        
        # 가장 빈도가 높은 단어들 선택
        most_common = word_counts.most_common(self.max_vocab_size - 2)  # <PAD>, <OOV> 제외
        
        # 단어 인덱스 생성
        self.word_index = {'<PAD>': 0, '<OOV>': 1}
        for i, (word, _) in enumerate(most_common, 2):
            self.word_index[word] = i
        
        # 인덱스-단어 매핑 생성
        self.index_word = {v: k for k, v in self.word_index.items()}
        
        self.fitted = True
        
    def encode(self, texts):
        """텍스트를 시퀀스로 변환합니다."""
        if not self.fitted:
            raise ValueError("토크나이저가 아직 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        sequences = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            sequence = [self.word_index.get(word, self.word_index['<OOV>']) for word in words]
            sequences.append(sequence)
        
        # 패딩
        padded_sequences = []
        for sequence in sequences:
            if len(sequence) > self.max_length:
                sequence = sequence[:self.max_length]
            else:
                sequence = sequence + [self.word_index['<PAD>']] * (self.max_length - len(sequence))
            padded_sequences.append(sequence)
        
        return np.array(padded_sequences)
    
    def decode(self, sequences):
        """시퀀스를 텍스트로 변환합니다."""
        if not self.fitted:
            raise ValueError("토크나이저가 아직 학습되지 않았습니다.")
        
        texts = []
        for sequence in sequences:
            # 패딩 토큰 제거
            sequence = [token for token in sequence if token != self.word_index['<PAD>']]
            # 인덱스를 단어로 변환
            words = [self.index_word.get(token, '<OOV>') for token in sequence]
            text = ' '.join(words)
            texts.append(text)
        return texts
    
    def get_vocab_size(self):
        """어휘 크기를 반환합니다."""
        return len(self.word_index)
    
    def save(self, filepath):
        """토크나이저를 파일로 저장합니다."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word_index': self.word_index,
                'index_word': self.index_word,
                'max_vocab_size': self.max_vocab_size,
                'max_length': self.max_length,
                'fitted': self.fitted
            }, f)
    
    def load(self, filepath):
        """파일에서 토크나이저를 로드합니다."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.word_index = data['word_index']
            self.index_word = data['index_word']
            self.max_vocab_size = data['max_vocab_size']
            self.max_length = data['max_length']
            self.fitted = data['fitted'] 