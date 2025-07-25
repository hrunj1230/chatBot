import pandas as pd
import numpy as np
import json
import re
from collections import Counter
import matplotlib.pyplot as plt

class DataQualityChecker:
    def __init__(self):
        self.quality_metrics = {}
    
    def check_conversation_quality(self, conversations):
        """대화 데이터의 품질을 검사합니다."""
        print("대화 데이터 품질 검사를 시작합니다...")
        
        total_conversations = len(conversations)
        quality_conversations = []
        
        for i, conv in enumerate(conversations):
            if len(conv) != 2:
                continue
                
            question, answer = conv[0], conv[1]
            
            # 품질 검사 조건
            if self._is_high_quality(question, answer):
                quality_conversations.append(conv)
        
        # 품질 메트릭 계산
        self.quality_metrics = {
            'total_conversations': total_conversations,
            'quality_conversations': len(quality_conversations),
            'quality_ratio': len(quality_conversations) / total_conversations,
            'avg_question_length': np.mean([len(conv[0]) for conv in quality_conversations]),
            'avg_answer_length': np.mean([len(conv[1]) for conv in quality_conversations])
        }
        
        print(f"품질 검사 완료:")
        print(f"- 전체 대화: {total_conversations}개")
        print(f"- 고품질 대화: {len(quality_conversations)}개")
        print(f"- 품질 비율: {self.quality_metrics['quality_ratio']:.2%}")
        print(f"- 평균 질문 길이: {self.quality_metrics['avg_question_length']:.1f}자")
        print(f"- 평균 답변 길이: {self.quality_metrics['avg_answer_length']:.1f}자")
        
        return quality_conversations
    
    def _is_high_quality(self, question, answer):
        """대화의 품질을 판단합니다."""
        # 1. 길이 검사
        if len(question) < 2 or len(answer) < 5:
            return False
        
        if len(question) > 100 or len(answer) > 500:
            return False
        
        # 2. 특수문자 비율 검사
        special_chars_question = len(re.findall(r'[^\w\s가-힣]', question))
        special_chars_answer = len(re.findall(r'[^\w\s가-힣]', answer))
        
        if special_chars_question / len(question) > 0.3:
            return False
        
        if special_chars_answer / len(answer) > 0.2:
            return False
        
        # 3. 반복 패턴 검사
        if self._has_repetition(answer):
            return False
        
        # 4. 의미있는 내용 검사
        if not self._has_meaningful_content(question, answer):
            return False
        
        return True
    
    def _has_repetition(self, text):
        """반복 패턴이 있는지 검사합니다."""
        words = text.split()
        if len(words) < 3:
            return False
        
        word_counts = Counter(words)
        for word, count in word_counts.items():
            if count > len(words) * 0.3:  # 30% 이상 반복
                return True
        
        return False
    
    def _has_meaningful_content(self, question, answer):
        """의미있는 내용인지 검사합니다."""
        # 질문에 의문사나 질문 형태가 있는지
        question_indicators = ['뭐', '어떻게', '언제', '어디', '왜', '누가', '?', '요', '까']
        has_question = any(indicator in question for indicator in question_indicators)
        
        # 답변이 질문에 관련된 내용인지
        common_words = set(question.split()) & set(answer.split())
        has_related_content = len(common_words) > 0
        
        return has_question and has_related_content
    
    def analyze_diversity(self, conversations):
        """대화 데이터의 다양성을 분석합니다."""
        print("대화 데이터 다양성 분석...")
        
        # 주제별 분류
        topics = {
            '인사': ['안녕', '반가워', '만나서'],
            '날씨': ['날씨', '맑', '흐림', '비', '눈'],
            '음식': ['먹', '음식', '맛', '식사', '점심', '저녁'],
            '영화': ['영화', '드라마', '시리즈', '넷플릭스'],
            '음악': ['음악', '노래', '가수', '앨범'],
            '운동': ['운동', '헬스', '요가', '달리기'],
            '학습': ['공부', '학습', '시험', '책', '독서'],
            '여행': ['여행', '휴가', '관광', '여행지'],
            '일': ['일', '업무', '회사', '직장'],
            '취미': ['취미', '관심', '재미', '즐거움']
        }
        
        topic_counts = {topic: 0 for topic in topics.keys()}
        
        for conv in conversations:
            question = conv[0].lower()
            for topic, keywords in topics.items():
                if any(keyword in question for keyword in keywords):
                    topic_counts[topic] += 1
                    break
        
        # 다양성 점수 계산
        total = len(conversations)
        diversity_score = len([count for count in topic_counts.values() if count > 0]) / len(topics)
        
        print(f"주제별 분포:")
        for topic, count in topic_counts.items():
            percentage = count / total * 100
            print(f"- {topic}: {count}개 ({percentage:.1f}%)")
        
        print(f"다양성 점수: {diversity_score:.2f}")
        
        return topic_counts, diversity_score
    
    def generate_quality_report(self, conversations, output_path='data/quality_report.txt'):
        """품질 검사 보고서를 생성합니다."""
        quality_conversations = self.check_conversation_quality(conversations)
        topic_counts, diversity_score = self.analyze_diversity(quality_conversations)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== 대화 데이터 품질 검사 보고서 ===\n\n")
            f.write(f"검사 일시: {pd.Timestamp.now()}\n")
            f.write(f"전체 대화 수: {self.quality_metrics['total_conversations']}\n")
            f.write(f"고품질 대화 수: {self.quality_metrics['quality_conversations']}\n")
            f.write(f"품질 비율: {self.quality_metrics['quality_ratio']:.2%}\n")
            f.write(f"평균 질문 길이: {self.quality_metrics['avg_question_length']:.1f}자\n")
            f.write(f"평균 답변 길이: {self.quality_metrics['avg_answer_length']:.1f}자\n")
            f.write(f"다양성 점수: {diversity_score:.2f}\n\n")
            
            f.write("주제별 분포:\n")
            for topic, count in topic_counts.items():
                percentage = count / self.quality_metrics['quality_conversations'] * 100
                f.write(f"- {topic}: {count}개 ({percentage:.1f}%)\n")
        
        print(f"품질 검사 보고서가 {output_path}에 저장되었습니다.")
        
        return quality_conversations

def main():
    """메인 품질 검사 함수"""
    # 대화 데이터 로드
    try:
        with open('data/conversations.json', 'r', encoding='utf-8') as f:
            conversations = json.load(f)
    except FileNotFoundError:
        print("대화 데이터 파일을 찾을 수 없습니다. 먼저 데이터를 수집해주세요.")
        return
    
    # 품질 검사 실행
    checker = DataQualityChecker()
    quality_conversations = checker.generate_quality_report(conversations)
    
    # 고품질 데이터 저장
    with open('data/quality_conversations.json', 'w', encoding='utf-8') as f:
        json.dump(quality_conversations, f, ensure_ascii=False, indent=2)
    
    print(f"고품질 대화 데이터가 data/quality_conversations.json에 저장되었습니다.")

if __name__ == "__main__":
    main() 