import pandas as pd
import numpy as np
import json
import os
import re
from typing import List, Tuple

class ManualDataCollector:
    def __init__(self):
        self.manual_data = []
        self.accounting_data = []
    
    def load_manual_from_text(self, file_path: str) -> List[Tuple[str, str]]:
        """텍스트 파일에서 메뉴얼 데이터를 로드합니다."""
        qa_pairs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 섹션별로 분리 (예: "## 질문" 형식)
            sections = re.split(r'##\s+', content)
            
            for section in sections:
                if '질문:' in section and '답변:' in section:
                    lines = section.split('\n')
                    question = ""
                    answer = ""
                    in_question = False
                    in_answer = False
                    
                    for line in lines:
                        if '질문:' in line:
                            in_question = True
                            in_answer = False
                            question = line.replace('질문:', '').strip()
                        elif '답변:' in line:
                            in_question = False
                            in_answer = True
                            answer = line.replace('답변:', '').strip()
                        elif in_question and line.strip():
                            question += " " + line.strip()
                        elif in_answer and line.strip():
                            answer += " " + line.strip()
                    
                    if question and answer:
                        qa_pairs.append((question, answer))
            
            print(f"텍스트 파일에서 {len(qa_pairs)}개의 Q&A 쌍을 추출했습니다.")
            return qa_pairs
            
        except Exception as e:
            print(f"텍스트 파일 로드 실패: {e}")
            return []
    
    def load_manual_from_csv(self, file_path: str) -> List[Tuple[str, str]]:
        """CSV 파일에서 메뉴얼 데이터를 로드합니다."""
        try:
            df = pd.read_csv(file_path)
            qa_pairs = []
            
            # 컬럼명 확인 및 매핑
            question_col = None
            answer_col = None
            
            for col in df.columns:
                if '질문' in col or 'question' in col.lower():
                    question_col = col
                elif '답변' in col or 'answer' in col.lower():
                    answer_col = col
            
            if question_col and answer_col:
                for _, row in df.iterrows():
                    question = str(row[question_col]).strip()
                    answer = str(row[answer_col]).strip()
                    if question and answer and question != 'nan' and answer != 'nan':
                        qa_pairs.append((question, answer))
            
            print(f"CSV 파일에서 {len(qa_pairs)}개의 Q&A 쌍을 추출했습니다.")
            return qa_pairs
            
        except Exception as e:
            print(f"CSV 파일 로드 실패: {e}")
            return []
    
    def load_accounting_data(self) -> List[Tuple[str, str]]:
        """세무회계 관련 데이터를 로드합니다."""
        accounting_qa = [
            # 세무 관련
            ["법인세 신고 기한이 언제인가요?", "법인세 신고 기한은 사업연도 종료일로부터 3개월 이내입니다. 예를 들어 12월 31일이 사업연도 종료일이라면 다음해 3월 31일까지 신고해야 합니다."],
            ["부가가치세 신고는 언제 하나요?", "부가가치세는 매분기 종료일로부터 25일 이내에 신고합니다. 1분기는 4월 25일, 2분기는 7월 25일, 3분기는 10월 25일, 4분기는 다음해 1월 25일까지입니다."],
            ["개인사업자의 소득세 신고 기한은?", "개인사업자의 소득세 신고는 다음해 5월 31일까지입니다. 단, 종합소득세 신고와 별도로 분리하여 신고할 수 있습니다."],
            ["세무조정이란 무엇인가요?", "세무조정은 회계상 손익과 세법상 손익의 차이를 조정하는 것을 말합니다. 예를 들어 업무상 접대비, 기부금, 감가상각비 등의 차이를 조정합니다."],
            ["법인세율은 어떻게 되나요?", "법인세율은 과세표준 2억원 이하 10%, 2억원 초과 22%입니다. 중소기업의 경우 2억원 이하 10%, 2억원 초과 20%입니다."],
            
            # 회계 관련
            ["복식부기와 단식부기의 차이는?", "복식부기는 거래의 이중성을 기록하는 방식으로, 자산=부채+자본의 등식을 유지합니다. 단식부기는 현금의 수입과 지출만 기록하는 방식입니다."],
            ["대차대조표란 무엇인가요?", "대차대조표는 특정 시점의 재무상태를 나타내는 재무제표입니다. 자산, 부채, 자본을 분류하여 표시하며, 자산=부채+자본의 등식이 성립합니다."],
            ["손익계산서의 구성요소는?", "손익계산서는 매출액, 매출원가, 판매비와 관리비, 영업외손익, 법인세비용 등으로 구성됩니다. 최종적으로 당기순이익을 계산합니다."],
            ["감가상각이란 무엇인가요?", "감가상각은 고정자산의 가치 감소를 회계적으로 인식하는 방법입니다. 정액법, 정률법, 생산량비례법 등이 있으며, 자산의 종류에 따라 상각방법이 정해집니다."],
            ["재고자산 평가방법은?", "재고자산 평가방법에는 개별법, 선입선출법, 후입선출법, 평균법이 있습니다. 기업은 한 번 선택한 방법을 계속 사용해야 하며, 변경 시 세무서에 신고해야 합니다."],
            
            # 실무 관련
            ["전표 작성 시 주의사항은?", "전표 작성 시 거래일자, 계정과목, 금액, 거래내용을 정확히 기록해야 합니다. 특히 계정과목은 세무서에서 정한 표준계정과목을 사용하는 것이 좋습니다."],
            ["연말정산 시 필요한 서류는?", "연말정산 시 급여명세서, 각종 공제증빙서류, 의료비 영수증, 교육비 영수증, 기부금 영수증 등이 필요합니다. 각 공제항목별로 최대 공제한도가 정해져 있습니다."],
            ["부가가치세 계산 방법은?", "부가가치세는 매출세액에서 매입세액을 차감하여 계산합니다. 매출세액 = 매출액 × 10%, 매입세액은 매입 시 부가세를 포함한 금액에서 계산합니다."],
            ["세무조사 대비 방법은?", "세무조사 대비를 위해서는 전표와 장부를 정확히 작성하고, 각종 증빙서류를 체계적으로 보관해야 합니다. 특히 현금거래는 반드시 영수증을 받아 보관하세요."],
            ["법인세 신고 시 주의사항은?", "법인세 신고 시 세무조정사항을 정확히 반영하고, 각종 공제요건을 충족하는지 확인해야 합니다. 또한 신고서와 함께 필요한 서류를 첨부해야 합니다."]
        ]
        
        self.accounting_data = accounting_qa
        print(f"세무회계 데이터 {len(accounting_qa)}개를 로드했습니다.")
        return accounting_qa
    
    def create_manual_dataset(self, manual_files: List[str], accounting_data: bool = True) -> List[Tuple[str, str]]:
        """메뉴얼 데이터셋을 생성합니다."""
        all_qa_pairs = []
        
        # 메뉴얼 파일들 로드
        for file_path in manual_files:
            if file_path.endswith('.txt'):
                qa_pairs = self.load_manual_from_text(file_path)
            elif file_path.endswith('.csv'):
                qa_pairs = self.load_manual_from_csv(file_path)
            else:
                print(f"지원하지 않는 파일 형식: {file_path}")
                continue
            
            all_qa_pairs.extend(qa_pairs)
        
        # 세무회계 데이터 추가
        if accounting_data:
            accounting_qa = self.load_accounting_data()
            all_qa_pairs.extend(accounting_qa)
        
        # 중복 제거 (튜플로 변환 후 set 사용)
        qa_tuples = [tuple(qa) for qa in all_qa_pairs]
        unique_qa_tuples = list(set(qa_tuples))
        unique_qa_pairs = [list(qa) for qa in unique_qa_tuples]
        
        print(f"총 {len(unique_qa_pairs)}개의 고유한 Q&A 쌍을 생성했습니다.")
        return unique_qa_pairs
    
    def save_manual_dataset(self, qa_pairs: List[Tuple[str, str]], output_path: str = 'data/manual_dataset.json'):
        """메뉴얼 데이터셋을 저장합니다."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # JSON 형태로 저장
        data = []
        for question, answer in qa_pairs:
            data.append({
                'question': question,
                'answer': answer,
                'category': 'manual'
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"메뉴얼 데이터셋이 {output_path}에 저장되었습니다.")
        
        # CSV 형태로도 저장
        csv_path = output_path.replace('.json', '.csv')
        df = pd.DataFrame(qa_pairs, columns=['question', 'answer'])
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"메뉴얼 데이터셋이 {csv_path}에도 저장되었습니다.")

def main():
    """메인 함수"""
    collector = ManualDataCollector()
    
    # 메뉴얼 파일 경로들 (실제 파일 경로로 수정)
    manual_files = [
        'data/manual1.txt',  # 예시 파일
        'data/manual2.csv'   # 예시 파일
    ]
    
    # 실제 존재하는 파일만 필터링
    existing_files = [f for f in manual_files if os.path.exists(f)]
    
    if not existing_files:
        print("메뉴얼 파일이 없습니다. 기본 세무회계 데이터만 사용합니다.")
        existing_files = []
    
    # 데이터셋 생성
    qa_pairs = collector.create_manual_dataset(existing_files, accounting_data=True)
    
    # 데이터셋 저장
    collector.save_manual_dataset(qa_pairs)
    
    print("메뉴얼 데이터 수집이 완료되었습니다!")

if __name__ == "__main__":
    main() 