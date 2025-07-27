#!/usr/bin/env python3
"""
자동 스케줄러 모듈
정기적으로 데이터를 수집하고 학습을 실행합니다.
"""

import time
import threading
from datetime import datetime, timedelta
import schedule

class AutoScheduler:
    def __init__(self):
        """자동 스케줄러 초기화"""
        self.is_running = False
        self.scheduler_thread = None
        
        # 스케줄 설정
        self.data_collection_interval = 0.25  # 15분마다 데이터 수집 (0.25시간)
        self.learning_trigger_interval = 0.5  # 30분마다 학습 트리거 (0.5시간)
        
        print("자동 스케줄러 초기화 완료")
    
    def start_scheduler(self):
        """스케줄러 시작"""
        if self.is_running:
            print("스케줄러가 이미 실행 중입니다.")
            return
        
        self.is_running = True
        
        # 스케줄 설정
        schedule.every(self.data_collection_interval).hours.do(self.run_data_collection)
        schedule.every(self.learning_trigger_interval).hours.do(self.trigger_learning)
        
        # 스케줄러 스레드 시작
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        print(f"자동 스케줄러 시작: 데이터 수집 {self.data_collection_interval*60}분마다, 학습 트리거 {self.learning_trigger_interval*60}분마다")
    
    def stop_scheduler(self):
        """스케줄러 중지"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        print("자동 스케줄러 중지")
    
    def _run_scheduler(self):
        """스케줄러 실행 루프"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 체크
            except Exception as e:
                print(f"스케줄러 실행 오류: {e}")
                time.sleep(60)
    
    def run_data_collection(self):
        """데이터 수집 실행"""
        try:
            print(f"[{datetime.now()}] 자동 데이터 수집 시작...")
            
            from utils.auto_data_collector import run_auto_data_collection
            collected_data = run_auto_data_collection()
            
            # 수집된 데이터를 학습 시스템에 추가
            self._add_to_learning_systems(collected_data)
            
            print(f"[{datetime.now()}] 자동 데이터 수집 완료: {len(collected_data)}개 데이터")
            
        except Exception as e:
            print(f"[{datetime.now()}] 자동 데이터 수집 실패: {e}")
    
    def trigger_learning(self):
        """학습 트리거 실행"""
        try:
            print(f"[{datetime.now()}] 학습 트리거 실행...")
            
            # 실시간 학습기 트리거
            try:
                from utils.realtime_learner import get_realtime_learner
                realtime_learner = get_realtime_learner()
                if realtime_learner and len(realtime_learner.conversation_buffer) >= 10:
                    realtime_learner.trigger_learning()
                    print(f"[{datetime.now()}] 실시간 학습 트리거 완료")
            except Exception as e:
                print(f"[{datetime.now()}] 실시간 학습 트리거 실패: {e}")
            
            # 경량 파인튜닝 트리거
            try:
                from utils.lightweight_finetuning import get_lightweight_finetuner
                lightweight_finetuner = get_lightweight_finetuner()
                if lightweight_finetuner and len(lightweight_finetuner.training_buffer) >= 10:
                    lightweight_finetuner.trigger_training()
                    print(f"[{datetime.now()}] 경량 파인튜닝 트리거 완료")
            except Exception as e:
                print(f"[{datetime.now()}] 경량 파인튜닝 트리거 실패: {e}")
            
        except Exception as e:
            print(f"[{datetime.now()}] 학습 트리거 실패: {e}")
    
    def _add_to_learning_systems(self, collected_data):
        """수집된 데이터를 학습 시스템에 추가"""
        try:
            from utils.realtime_learner import get_realtime_learner
            from utils.lightweight_finetuning import get_lightweight_finetuner
            
            realtime_learner = get_realtime_learner()
            lightweight_finetuner = get_lightweight_finetuner()
            
            for data in collected_data:
                if realtime_learner:
                    realtime_learner.add_conversation(data['question'], data['answer'], 0.8)
                
                if lightweight_finetuner:
                    lightweight_finetuner.add_training_data(data['question'], data['answer'], 0.8)
            
            print(f"수집된 데이터 {len(collected_data)}개를 학습 시스템에 추가했습니다.")
            
        except Exception as e:
            print(f"학습 시스템에 데이터 추가 실패: {e}")
    
    def get_schedule_info(self):
        """스케줄 정보 반환"""
        return {
            'is_running': self.is_running,
            'data_collection_interval_minutes': int(self.data_collection_interval * 60),
            'learning_trigger_interval_minutes': int(self.learning_trigger_interval * 60),
            'next_data_collection': schedule.next_run(),
            'next_learning_trigger': schedule.next_run()
        }

# 전역 인스턴스
auto_scheduler = None

def initialize_auto_scheduler():
    """자동 스케줄러 초기화"""
    global auto_scheduler
    auto_scheduler = AutoScheduler()
    return True

def get_auto_scheduler():
    """자동 스케줄러 인스턴스 반환"""
    return auto_scheduler

def start_auto_scheduler():
    """자동 스케줄러 시작"""
    if not auto_scheduler:
        initialize_auto_scheduler()
    
    auto_scheduler.start_scheduler()
    return True

def stop_auto_scheduler():
    """자동 스케줄러 중지"""
    if auto_scheduler:
        auto_scheduler.stop_scheduler()
    return True 