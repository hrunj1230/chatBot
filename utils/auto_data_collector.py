#!/usr/bin/env python3
"""
자동 대화 데이터 수집 모듈
다양한 소스에서 대화 데이터를 자동으로 수집하여 학습에 활용합니다.
"""

import requests
import json
import time
import random
from datetime import datetime
from bs4 import BeautifulSoup
import re
import os

class AutoDataCollector:
    def __init__(self):
        """자동 데이터 수집기 초기화"""
        self.collected_data = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def collect_from_naver_qa(self, max_pages=10):
        """네이버 지식iN에서 세무회계 관련 Q&A 수집"""
        print("네이버 지식iN에서 세무회계 Q&A 수집 중...")
        
        keywords = [
            '부가가치세 신고', '소득세 계산', '법인세 신고', '세무신고',
            '회계처리', '부가세 계산', '세금 계산', '신고서 작성',
            '세무사 상담', '회계사 상담', '세무 처리', '세금 납부'
        ]
        
        for keyword in keywords:
            try:
                # 네이버 지식iN 검색
                search_url = f"https://kin.naver.com/search/list.naver?query={keyword}"
                response = self.session.get(search_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 질문 목록 추출
                questions = soup.find_all('a', class_='question')
                
                for question in questions[:5]:  # 각 키워드당 5개씩
                    try:
                        question_text = question.get_text().strip()
                        if len(question_text) > 10:
                            # 간단한 답변 생성
                            answer = self.generate_accounting_answer(question_text)
                            
                            data = {
                                'question': question_text,
                                'answer': answer,
                                'source': 'naver_kin',
                                'keyword': keyword,
                                'timestamp': datetime.now().isoformat()
                            }
                            self.collected_data.append(data)
                            
                    except Exception as e:
                        print(f"질문 처리 실패: {e}")
                        continue
                
                time.sleep(1)  # 요청 간격 조절
                
            except Exception as e:
                print(f"키워드 '{keyword}' 처리 실패: {e}")
                continue
        
        print(f"네이버 지식iN에서 {len(self.collected_data)}개 데이터 수집 완료")
    
    def collect_from_accounting_blog(self):
        """세무회계 블로그에서 Q&A 수집"""
        print("세무회계 블로그에서 Q&A 수집 중...")
        
        # 세무회계 관련 블로그 URL들
        blog_urls = [
            "https://blog.naver.com/taxaccounting",
            "https://blog.naver.com/accounting_tips",
            "https://blog.naver.com/tax_guide"
        ]
        
        for url in blog_urls:
            try:
                response = self.session.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 블로그 포스트 제목 추출
                posts = soup.find_all('a', class_='title')
                
                for post in posts[:3]:  # 각 블로그당 3개씩
                    try:
                        title = post.get_text().strip()
                        if len(title) > 10:
                            # 제목을 질문으로 변환
                            question = f"{title}에 대해 알려주세요"
                            answer = self.generate_accounting_answer(title)
                            
                            data = {
                                'question': question,
                                'answer': answer,
                                'source': 'accounting_blog',
                                'timestamp': datetime.now().isoformat()
                            }
                            self.collected_data.append(data)
                            
                    except Exception as e:
                        print(f"블로그 포스트 처리 실패: {e}")
                        continue
                
                time.sleep(2)
                
            except Exception as e:
                print(f"블로그 '{url}' 처리 실패: {e}")
                continue
    
    def collect_from_tax_office_data(self):
        """국세청 자료에서 FAQ 수집"""
        print("국세청 자료에서 FAQ 수집 중...")
        
        # 세무 관련 FAQ 데이터
        tax_faqs = [
            {
                'question': '부가가치세 신고는 언제 하나요?',
                'answer': '부가가치세는 매월 25일까지 신고하고 납부해야 합니다. 단, 간이과세자는 분기별로 신고합니다.'
            },
            {
                'question': '소득세 신고 기간은 언제인가요?',
                'answer': '종합소득세는 매년 5월 1일부터 31일까지 신고합니다. 사업소득자는 분기별로 예정신고를 해야 합니다.'
            },
            {
                'question': '법인세 신고는 어떻게 하나요?',
                'answer': '법인세는 사업연도 종료일로부터 3개월 이내에 신고하고 납부해야 합니다. 분기별 예정신고도 필요합니다.'
            },
            {
                'question': '부가가치세 계산 방법은?',
                'answer': '부가가치세 = 매출세액 - 매입세액입니다. 매출세액은 공급가액 × 10%로 계산합니다.'
            },
            {
                'question': '세무신고 서류는 무엇이 필요한가요?',
                'answer': '매출장, 매입장, 부가가치세 신고서, 소득세 신고서, 각종 증빙서류가 필요합니다.'
            }
        ]
        
        for faq in tax_faqs:
            data = {
                'question': faq['question'],
                'answer': faq['answer'],
                'source': 'tax_office',
                'timestamp': datetime.now().isoformat()
            }
            self.collected_data.append(data)
        
        print(f"국세청 자료에서 {len(tax_faqs)}개 FAQ 수집 완료")
    
    def generate_accounting_answer(self, question):
        """질문에 대한 세무회계 답변 생성"""
        # 간단한 규칙 기반 답변 생성
        if '부가가치세' in question or '부가세' in question:
            if '신고' in question:
                return "부가가치세는 매월 25일까지 신고하고 납부해야 합니다. 간이과세자는 분기별로 신고합니다."
            elif '계산' in question:
                return "부가가치세 = 매출세액 - 매입세액입니다. 매출세액은 공급가액 × 10%로 계산합니다."
            else:
                return "부가가치세는 재화나 용역의 공급에 대해 부과되는 간접세입니다."
        
        elif '소득세' in question:
            if '신고' in question:
                return "종합소득세는 매년 5월 1일부터 31일까지 신고합니다. 사업소득자는 분기별 예정신고가 필요합니다."
            elif '계산' in question:
                return "소득세는 과세표준에 따라 누진세율을 적용하여 계산합니다."
            else:
                return "소득세는 개인의 소득에 대해 부과되는 직접세입니다."
        
        elif '법인세' in question:
            if '신고' in question:
                return "법인세는 사업연도 종료일로부터 3개월 이내에 신고하고 납부해야 합니다."
            else:
                return "법인세는 법인의 소득에 대해 부과되는 직접세입니다."
        
        elif '회계' in question or '장부' in question:
            return "회계장부는 매출장, 매입장, 총계정원장 등을 포함하며, 5년간 보관해야 합니다."
        
        elif '세무사' in question or '상담' in question:
            return "세무사는 세무신고, 회계처리, 세무상담 등을 전문적으로 도와주는 전문가입니다."
        
        else:
            return "세무회계 관련 질문이시군요. 구체적인 내용을 말씀해 주시면 더 자세히 답변드리겠습니다."
    
    def collect_from_synthetic_data(self):
        """합성 데이터 생성"""
        print("합성 세무회계 Q&A 데이터 생성 중...")
        
        synthetic_qa = [
            ("부가가치세 신고 기한이 지났으면 어떻게 하나요?", "부가가치세 신고 기한이 지난 경우 가산세가 부과됩니다. 가능한 빨리 신고하고 납부하시기 바랍니다."),
            ("간이과세자는 언제 신고하나요?", "간이과세자는 분기별로 부가가치세를 신고합니다. 1분기는 4월 25일까지, 2분기는 7월 25일까지입니다."),
            ("매입세액공제는 어떻게 받나요?", "매입세액공제는 매입세액공제신고서를 제출하고, 매입장과 세금계산서를 보관하면 됩니다."),
            ("세무신고를 안 하면 어떤 불이익이 있나요?", "세무신고를 안 하면 가산세, 가산금, 체납처분 등의 불이익이 있을 수 있습니다."),
            ("사업자등록은 어떻게 하나요?", "사업자등록은 국세청 홈페이지나 세무서에서 신청할 수 있습니다. 신분증과 사업계획서가 필요합니다."),
            ("부가가치세 영세율은 언제 적용되나요?", "부가가치세 영세율은 수출, 외국인 관광객 대상 판매, 국제운송 등에 적용됩니다."),
            ("세무조사는 언제 받나요?", "세무조사는 일반적으로 3년마다 받게 되며, 세무신고 내용에 문제가 있을 때 받을 수 있습니다."),
            ("회계장부는 얼마나 보관해야 하나요?", "회계장부는 5년간 보관해야 하며, 세무조사 시 제출해야 할 수 있습니다."),
            ("세무사 선임은 언제 하나요?", "세무사 선임은 사업 시작 시, 세무신고 시, 세무조사 시 등에 필요할 수 있습니다."),
            ("부가가치세 신고서는 어디서 받나요?", "부가가치세 신고서는 국세청 홈페이지에서 다운로드하거나 세무서에서 받을 수 있습니다.")
        ]
        
        for question, answer in synthetic_qa:
            data = {
                'question': question,
                'answer': answer,
                'source': 'synthetic',
                'timestamp': datetime.now().isoformat()
            }
            self.collected_data.append(data)
        
        print(f"합성 데이터 {len(synthetic_qa)}개 생성 완료")
    
    def save_collected_data(self, filename='data/auto_collected_data.json'):
        """수집된 데이터를 파일로 저장"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.collected_data, f, ensure_ascii=False, indent=2)
            
            print(f"수집된 데이터 {len(self.collected_data)}개를 {filename}에 저장했습니다.")
            return True
            
        except Exception as e:
            print(f"데이터 저장 실패: {e}")
            return False
    
    def run_collection(self):
        """전체 데이터 수집 실행"""
        print("자동 데이터 수집을 시작합니다...")
        
        # 1. 합성 데이터 생성
        self.collect_from_synthetic_data()
        
        # 2. 국세청 자료 수집
        self.collect_from_tax_office_data()
        
        # 3. 네이버 지식iN 수집 (선택적)
        try:
            self.collect_from_naver_qa()
        except Exception as e:
            print(f"네이버 지식iN 수집 실패: {e}")
        
        # 4. 블로그 데이터 수집 (선택적)
        try:
            self.collect_from_accounting_blog()
        except Exception as e:
            print(f"블로그 데이터 수집 실패: {e}")
        
        # 5. 데이터 저장
        self.save_collected_data()
        
        print(f"총 {len(self.collected_data)}개의 대화 데이터를 수집했습니다.")
        return self.collected_data

# 전역 인스턴스
auto_collector = None

def initialize_auto_collector():
    """자동 데이터 수집기 초기화"""
    global auto_collector
    auto_collector = AutoDataCollector()
    return True

def get_auto_collector():
    """자동 데이터 수집기 인스턴스 반환"""
    return auto_collector

def run_auto_data_collection():
    """자동 데이터 수집 실행"""
    if not auto_collector:
        initialize_auto_collector()
    
    return auto_collector.run_collection() 