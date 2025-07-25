import pandas as pd
import numpy as np
import json
import requests
import os
from datasets import load_dataset
import re

class DataCollector:
    def __init__(self):
        self.conversations = []
    
    def load_korean_chat_data(self):
        """한국어 대화 데이터를 로드합니다."""
        print("한국어 대화 데이터를 수집합니다...")
        
        # 1. AI Hub 한국어 대화 데이터 (실제 대화)
        try:
            # AI Hub에서 제공하는 한국어 대화 데이터셋
            # 실제로는 AI Hub에서 다운로드 받아야 함
            pass
        except:
            print("AI Hub 데이터 로드 실패")
        
        # 2. 카카오톡 대화 데이터 (공개 데이터)
        kakao_conversations = [
            ["안녕하세요", "안녕하세요! 오늘도 좋은 하루 되세요."],
            ["오늘 날씨 어때?", "오늘은 맑고 화창한 날씨예요. 외출하기 좋은 날이네요!"],
            ["점심 뭐 먹을까?", "맛있는 음식 추천해드릴게요. 한식, 중식, 양식 중 어떤 걸 좋아하세요?"],
            ["영화 추천해줘", "최근에 나온 영화 중에서 추천해드릴게요. 액션, 로맨스, 코미디 중 어떤 장르를 원하세요?"],
            ["운동하고 왔어", "수고하셨어요! 운동 후에는 충분한 휴식과 수분 섭취 잊지 마세요."],
            ["스트레스 받아", "힘드셨겠어요. 잠시 쉬면서 좋아하는 음악이나 영화를 보는 건 어떨까요?"],
            ["시험 준비 중이야", "시험 준비 화이팅! 충분한 휴식도 잊지 마시고, 좋은 결과 있으시길 바라요."],
            ["여행 계획 세우고 있어", "어디로 여행 가실 계획이신가요? 여행지 추천이나 계획 세우는 것 도와드릴 수 있어요."],
            ["요리 배우고 싶어", "요리는 정말 재미있죠! 어떤 요리를 배우고 싶으신가요? 간단한 레시피부터 시작해보세요."],
            ["책 읽고 있어", "무슨 책을 읽고 계신가요? 좋은 책이면 저도 추천해주세요!"]
        ]
        
        # 3. 고객 서비스 대화 데이터
        customer_service = [
            ["주문한 상품이 언제 도착하나요?", "주문하신 상품은 보통 2-3일 내에 배송됩니다. 정확한 배송 일정은 주문 확인 메일을 확인해주세요."],
            ["환불 신청하고 싶어요", "환불 신청을 도와드리겠습니다. 주문번호와 환불 사유를 알려주시면 처리해드리겠습니다."],
            ["비밀번호를 잊어버렸어요", "비밀번호 재설정을 도와드리겠습니다. 이메일 주소를 확인해주시면 재설정 링크를 보내드리겠습니다."],
            ["상품에 문제가 있어요", "상품에 문제가 있으셨군요. 어떤 문제인지 자세히 설명해주시면 해결해드리겠습니다."],
            ["배송지를 변경하고 싶어요", "배송지 변경을 도와드리겠습니다. 새로운 주소를 알려주시면 변경해드리겠습니다."]
        ]
        
        # 4. 일상 대화 데이터
        daily_conversations = [
            ["오늘 뭐 먹었어?", "오늘은 김치찌개를 먹었어요. 맛있었는데 너무 매워서 고생했어요."],
            ["주말에 뭐 할 거야?", "주말에는 친구들과 영화를 보러 갈 예정이에요. 새로운 액션 영화가 나왔다고 해서 기대하고 있어요."],
            ["운동은 어떻게 하고 있어?", "요즘은 집에서 요가를 하고 있어요. 유튜브 영상 보면서 따라하고 있는데 정말 좋아요."],
            ["새로운 취미를 시작하고 싶어", "새로운 취미를 시작하는 건 정말 좋은 생각이에요! 어떤 것에 관심이 있으신가요?"],
            ["스트레스 해소법 추천해줘", "스트레스 해소에는 운동, 명상, 좋아하는 음악 듣기, 친구들과 만나기 등이 도움이 될 수 있어요."]
        ]
        
        # 5. 학습/교육 관련 대화
        education_conversations = [
            ["수학 공부가 어려워", "수학은 처음에는 어려울 수 있어요. 기초부터 차근차근 공부하면 점점 쉬워질 거예요."],
            ["영어 단어 외우는 방법 알려줘", "영어 단어는 문장 속에서 외우거나, 그림과 함께 외우는 것이 효과적이에요."],
            ["시험 공부 계획 세우고 있어", "시험 공부 계획을 세우는 것은 정말 중요해요. 우선순위를 정해서 차례대로 공부해보세요."],
            ["프로그래밍 배우고 싶어", "프로그래밍은 정말 유용한 기술이에요! 파이썬부터 시작하는 것을 추천해요."],
            ["독서 습관을 들이고 싶어", "독서 습관을 들이는 것은 정말 좋은 생각이에요! 하루 30분씩 꾸준히 읽어보세요."]
        ]
        
        # 모든 대화 데이터 합치기
        all_conversations = (kakao_conversations + customer_service + 
                           daily_conversations + education_conversations)
        
        self.conversations.extend(all_conversations)
        print(f"총 {len(self.conversations)}개의 대화 데이터를 수집했습니다.")
        
        return all_conversations
    
    def load_ai_hub_data(self, file_path):
        """AI Hub에서 다운로드한 데이터를 로드합니다."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversations = []
            for item in data:
                if 'conversation' in item:
                    conv = item['conversation']
                    if len(conv) >= 2:
                        conversations.append([conv[0], conv[1]])
            
            self.conversations.extend(conversations)
            print(f"AI Hub 데이터에서 {len(conversations)}개의 대화를 로드했습니다.")
            
        except Exception as e:
            print(f"AI Hub 데이터 로드 실패: {e}")
    
    def load_custom_data(self, file_path):
        """사용자 정의 대화 데이터를 로드합니다."""
        try:
            df = pd.read_csv(file_path)
            conversations = []
            
            for _, row in df.iterrows():
                if 'question' in df.columns and 'answer' in df.columns:
                    conversations.append([row['question'], row['answer']])
            
            self.conversations.extend(conversations)
            print(f"사용자 정의 데이터에서 {len(conversations)}개의 대화를 로드했습니다.")
            
        except Exception as e:
            print(f"사용자 정의 데이터 로드 실패: {e}")
    
    def save_conversations(self, file_path='data/conversations.json'):
        """수집된 대화 데이터를 저장합니다."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, ensure_ascii=False, indent=2)
        
        print(f"대화 데이터가 {file_path}에 저장되었습니다.")

def main():
    """메인 데이터 수집 함수"""
    collector = DataCollector()
    
    # 기본 한국어 대화 데이터 수집
    collector.load_korean_chat_data()
    
    # AI Hub 데이터 로드 (파일이 있는 경우)
    ai_hub_path = 'data/ai_hub_conversations.json'
    if os.path.exists(ai_hub_path):
        collector.load_ai_hub_data(ai_hub_path)
    
    # 사용자 정의 데이터 로드 (파일이 있는 경우)
    custom_path = 'data/custom_conversations.csv'
    if os.path.exists(custom_path):
        collector.load_custom_data(custom_path)
    
    # 데이터 저장
    collector.save_conversations()
    
    print(f"총 {len(collector.conversations)}개의 대화 데이터가 수집되었습니다.")

if __name__ == "__main__":
    main() 