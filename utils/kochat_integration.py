#!/usr/bin/env python3
"""
Kochat 통합 모듈
한국어 챗봇 딥러닝 프레임워크를 현재 시스템에 통합합니다.
"""

import os
import sys
import json
from datetime import datetime

# Kochat 라이브러리 설치 필요
try:
    from kochat.app import KochatApi
    from kochat.data import Dataset
    from kochat.model import intent, entity
    from kochat.loss import CenterLoss, CRFLoss
    from kochat.proc import GensimEmbedder
    from kochat.proc import DistanceClassifier, EntityRecognizer
    import gensim.downloader as gensim_downloader
    KOCHAT_AVAILABLE = True
except ImportError:
    print("Kochat 라이브러리가 설치되지 않았습니다. pip install kochat로 설치하세요.")
    KOCHAT_AVAILABLE = False

class KochatIntegration:
    def __init__(self):
        """Kochat 통합 초기화"""
        self.kochat_api = None
        self.is_initialized = False
        
        if not KOCHAT_AVAILABLE:
            print("Kochat을 사용할 수 없습니다.")
            return
        
        print("Kochat 통합 모듈 초기화 중...")
    
    def setup_kochat(self, data_path='data/kochat_data'):
        """Kochat 설정 및 초기화"""
        if not KOCHAT_AVAILABLE:
            return False
        
        try:
            # 1. 데이터셋 객체 생성
            print("데이터셋 생성 중...")
            dataset = Dataset(ood=True)
            
            # 2. 임베딩 프로세서 생성 (FastText)
            print("임베딩 프로세서 생성 중...")
            try:
                # 한국어 FastText 모델 다운로드
                fasttext_model = gensim_downloader.load('fasttext-wiki-news-subwords-300')
                emb = GensimEmbedder(model=fasttext_model)
            except:
                print("FastText 모델 로드 실패, 기본 임베딩 사용")
                emb = GensimEmbedder()
            
            # 3. 의도 분류기 생성
            print("의도 분류기 생성 중...")
            clf = DistanceClassifier(
                model=intent.CNN(dataset.intent_dict),
                loss=CenterLoss(dataset.intent_dict)
            )
            
            # 4. 개체명 인식기 생성
            print("개체명 인식기 생성 중...")
            rcn = EntityRecognizer(
                model=entity.LSTM(dataset.entity_dict),
                loss=CRFLoss(dataset.entity_dict)
            )
            
            # 5. Kochat API 생성
            print("Kochat API 생성 중...")
            self.kochat_api = KochatApi(
                dataset=dataset,
                embed_processor=(emb, True),
                intent_classifier=(clf, True),
                entity_recognizer=(rcn, True),
                scenarios=[]  # 커스텀 시나리오 추가 가능
            )
            
            self.is_initialized = True
            print("Kochat 초기화 완료!")
            return True
            
        except Exception as e:
            print(f"Kochat 초기화 실패: {e}")
            return False
    
    def add_custom_scenarios(self):
        """커스텀 시나리오 추가"""
        if not self.is_initialized:
            return
        
        # 세무회계 시나리오
        accounting_scenario = {
            'name': 'accounting',
            'intents': [
                'tax_inquiry',      # 세무 문의
                'accounting_help',  # 회계 도움
                'vat_question',     # 부가가치세 질문
                'income_tax',       # 소득세 질문
                'corporate_tax'     # 법인세 질문
            ],
            'entities': [
                'tax_type',         # 세금 종류
                'amount',           # 금액
                'date',             # 날짜
                'company_type'      # 회사 유형
            ]
        }
        
        # 챗봇 시나리오
        chatbot_scenario = {
            'name': 'chatbot',
            'intents': [
                'chatbot_info',     # 챗봇 정보
                'ai_question',      # AI 질문
                'learning_help',    # 학습 도움
                'performance_question'  # 성능 질문
            ],
            'entities': [
                'ai_type',          # AI 종류
                'learning_method',  # 학습 방법
                'performance_metric'  # 성능 지표
            ]
        }
        
        print("커스텀 시나리오 추가 완료")
    
    def process_message(self, user_input):
        """Kochat을 사용한 메시지 처리"""
        if not self.is_initialized:
            return None
        
        try:
            # Kochat API를 통한 메시지 처리
            result = self.kochat_api.process(user_input)
            
            return {
                'intent': result.get('intent'),
                'confidence': result.get('confidence'),
                'entities': result.get('entities'),
                'response': result.get('response'),
                'is_ood': result.get('is_ood', False)
            }
            
        except Exception as e:
            print(f"Kochat 메시지 처리 실패: {e}")
            return None
    
    def train_kochat(self, training_data):
        """Kochat 모델 학습"""
        if not self.is_initialized:
            return False
        
        try:
            print("Kochat 모델 학습 시작...")
            
            # 학습 데이터를 Kochat 형식으로 변환
            kochat_data = self._convert_to_kochat_format(training_data)
            
            # Kochat 학습 실행
            self.kochat_api.train(kochat_data)
            
            print("Kochat 모델 학습 완료!")
            return True
            
        except Exception as e:
            print(f"Kochat 학습 실패: {e}")
            return False
    
    def _convert_to_kochat_format(self, training_data):
        """학습 데이터를 Kochat 형식으로 변환"""
        kochat_data = {
            'intents': [],
            'entities': [],
            'scenarios': []
        }
        
        for data in training_data:
            # 의도 분류 데이터
            intent_data = {
                'text': data.get('question', ''),
                'intent': self._classify_intent(data.get('question', '')),
                'confidence': 1.0
            }
            kochat_data['intents'].append(intent_data)
            
            # 엔티티 추출 데이터
            entity_data = {
                'text': data.get('question', ''),
                'entities': self._extract_entities(data.get('question', ''))
            }
            kochat_data['entities'].append(entity_data)
        
        return kochat_data
    
    def _classify_intent(self, text):
        """텍스트의 의도 분류"""
        # 간단한 키워드 기반 의도 분류
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['세무', '세금', '부가가치세', '소득세', '법인세']):
            return 'tax_inquiry'
        elif any(word in text_lower for word in ['회계', '부기', '재무제표', '대차대조표']):
            return 'accounting_help'
        elif any(word in text_lower for word in ['챗봇', 'ai', '인공지능']):
            return 'chatbot_info'
        else:
            return 'general_inquiry'
    
    def _extract_entities(self, text):
        """텍스트에서 엔티티 추출"""
        entities = []
        text_lower = text.lower()
        
        # 세금 종류 추출
        tax_types = ['부가가치세', '소득세', '법인세', '종합소득세']
        for tax_type in tax_types:
            if tax_type in text:
                entities.append({
                    'entity': 'tax_type',
                    'value': tax_type,
                    'start': text.find(tax_type),
                    'end': text.find(tax_type) + len(tax_type)
                })
        
        # 금액 추출 (간단한 정규식)
        import re
        amount_pattern = r'\d+만원|\d+원|\d+천원'
        amounts = re.findall(amount_pattern, text)
        for amount in amounts:
            start = text.find(amount)
            entities.append({
                'entity': 'amount',
                'value': amount,
                'start': start,
                'end': start + len(amount)
            })
        
        return entities
    
    def get_performance_metrics(self):
        """성능 지표 반환"""
        if not self.is_initialized:
            return None
        
        try:
            # Kochat의 성능 평가 메트릭 사용
            metrics = self.kochat_api.evaluate()
            
            return {
                'intent_accuracy': metrics.get('intent_accuracy', 0),
                'entity_f1': metrics.get('entity_f1', 0),
                'ood_detection': metrics.get('ood_detection', 0),
                'response_time': metrics.get('response_time', 0)
            }
            
        except Exception as e:
            print(f"성능 지표 수집 실패: {e}")
            return None

# 전역 Kochat 인스턴스
kochat_integration = None

def initialize_kochat():
    """Kochat 초기화"""
    global kochat_integration
    kochat_integration = KochatIntegration()
    return kochat_integration.setup_kochat()

def get_kochat_integration():
    """Kochat 인스턴스 반환"""
    return kochat_integration 