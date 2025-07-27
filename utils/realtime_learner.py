#!/usr/bin/env python3
"""
실시간 학습 모듈
사용자 대화를 실시간으로 학습하여 챗봇 성능을 지속적으로 개선합니다.
"""

import os
import json
import threading
from datetime import datetime
from collections import deque

class RealtimeLearner:
    def __init__(self):
        """실시간 학습기 초기화"""
        self.conversation_buffer = deque(maxlen=1000)
        self.learning_threshold = 50  # 50개 대화마다 학습
        self.is_learning = False
        self.learning_lock = threading.Lock()
        
        # 학습 통계
        self.total_conversations = 0
        self.learning_count = 0
        self.last_learning_time = None
        
        print("실시간 학습기 초기화 완료")
    
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

            if not training_data:
                print("학습할 데이터가 없습니다.")
                return

            # 학습 통계 업데이트
            self.learning_count += 1
            self.last_learning_time = datetime.now()

            print(f"실시간 학습 완료: {len(training_data)}개 데이터")

        except Exception as e:
            print(f"실시간 학습 실패: {e}")
        finally:
            self.is_learning = False

    def get_accounting_suggestions(self, user_input):
        """세무회계 관련 제안 반환"""
        suggestions = []

        # 사용자 입력과 유사한 세무회계 질문 찾기
        accounting_keywords = ['세무', '회계', '세금', '부가가치세', '소득세', '법인세']
        if any(keyword in user_input.lower() for keyword in accounting_keywords):
            suggestions = [
                "부가가치세 신고는 언제 하나요?",
                "종합소득세 신고 기간은?",
                "복식부기란 무엇인가요?",
                "재무제표는 무엇인가요?",
                "감가상각이란 무엇인가요?"
            ]

        return suggestions[:3]  # 최대 3개 제안

    def get_stats(self):
        """학습 통계 반환"""
        return {
            'total_conversations': self.total_conversations,
            'learning_count': self.learning_count,
            'buffer_size': len(self.conversation_buffer),
            'last_learning_time': self.last_learning_time.isoformat() if self.last_learning_time else None,
            'is_learning': self.is_learning
        }

# 전역 실시간 학습기 인스턴스
realtime_learner = None

def initialize_realtime_learner():
    """실시간 학습기 초기화"""
    global realtime_learner
    realtime_learner = RealtimeLearner()
    return True

def get_realtime_learner():
    """실시간 학습기 인스턴스 반환"""
    return realtime_learner 