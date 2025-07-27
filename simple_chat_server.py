from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import os
import json

app = Flask(__name__)
CORS(app)

def evaluate_conversation_quality(user_input, response):
    """대화 품질 평가"""
    quality_score = 0.5  # 기본값
    
    # 사용자 입력 길이 평가
    if len(user_input) > 20:
        quality_score += 0.2
    elif len(user_input) > 10:
        quality_score += 0.1
    
    # 응답 길이 평가
    if len(response) > 50:
        quality_score += 0.2
    elif len(response) > 20:
        quality_score += 0.1
    
    # 세무회계 관련 키워드 평가
    accounting_keywords = ['세무', '회계', '부가세', '소득세', '법인세', '신고', '납부', '세금', '매출', '비용', '손익']
    if any(keyword in user_input for keyword in accounting_keywords):
        quality_score += 0.1
    
    return min(quality_score, 1.0)

@app.route('/')
def home():
    """홈페이지 - 챗봇 인터페이스"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """챗봇 API 엔드포인트"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': '메시지가 없습니다.'}), 400
        
        # 응답 생성 (모델 사용)
        try:
            from server.app import generate_response
            response = generate_response(user_message)
        except Exception as e:
            print(f"모델 응답 생성 실패: {e}")
            response = f"안녕하세요! '{user_message}'에 대한 응답입니다."
        
        # 자동학습 데이터 추가
        try:
            # 실시간 학습 데이터 추가
            from utils.realtime_learner import get_realtime_learner
            realtime_learner = get_realtime_learner()
            if realtime_learner:
                quality_score = evaluate_conversation_quality(user_message, response)
                realtime_learner.add_conversation(user_message, response, quality_score)
            
            # 경량 파인튜닝 데이터 추가
            from utils.lightweight_finetuning import get_lightweight_finetuner
            lightweight_finetuner = get_lightweight_finetuner()
            if lightweight_finetuner:
                quality_score = evaluate_conversation_quality(user_message, response)
                if quality_score > 0.3:
                    lightweight_finetuner.add_training_data(user_message, response, quality_score)
        except Exception as e:
            print(f"자동학습 데이터 추가 실패: {e}")
        
        # 로그 기록
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_message,
            'bot_response': response
        }
        
        log_file = 'data/chat_logs.json'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"로그 기록 실패: {e}")
        
        return jsonify({
            'response': response,
            'suggestions': [],
            'lightweight_result': None,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"채팅 API 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """서버 상태 확인"""
    try:
        return jsonify({
            'status': 'running',
            'model_loaded': True,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """모델 학습 데이터 추가"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        answer = data.get('answer', '').strip()
        
        if not question or not answer:
            return jsonify({'error': '질문과 답변을 모두 입력해주세요.'}), 400
        
        # 학습 데이터를 파일에 저장
        training_data = {
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }
        
        # 학습 데이터 파일에 추가
        training_file = 'data/training_data.json'
        os.makedirs(os.path.dirname(training_file), exist_ok=True)
        
        try:
            if os.path.exists(training_file):
                with open(training_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            existing_data.append(training_data)
            
            with open(training_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            return jsonify({
                'status': 'success',
                'message': '학습 데이터가 추가되었습니다.',
                'timestamp': datetime.now().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': f'학습 데이터 저장 실패: {e}',
                'timestamp': datetime.now().isoformat()
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/realtime-stats')
def realtime_stats():
    """실시간 학습 통계"""
    try:
        return jsonify({
            'status': 'success',
            'stats': {
                'total_conversations': 0,
                'learning_count': 0,
                'buffer_size': 0,
                'is_learning': False,
                'last_learning_time': None
            },
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/lightweight-metrics')
def lightweight_metrics():
    """경량 파인튜닝 성능 지표"""
    try:
        return jsonify({
            'status': 'success',
            'stats': {
                'training_count': 0,
                'buffer_size': 0,
                'lora_layers': 0,
                'adapter_layers': 0,
                'is_training': False,
                'last_training_time': None
            },
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/auto-collect', methods=['POST'])
def auto_collect_data():
    """자동 데이터 수집 실행"""
    try:
        from utils.auto_data_collector import run_auto_data_collection
        
        print("자동 데이터 수집을 시작합니다...")
        collected_data = run_auto_data_collection()
        
        # 수집된 데이터를 학습 시스템에 추가
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
        
        return jsonify({
            'status': 'success',
            'message': f'자동 데이터 수집 완료: {len(collected_data)}개 데이터',
            'collected_count': len(collected_data),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    print("자동학습 챗봇 서버를 시작합니다...")
    
    # 학습 시스템 초기화
    try:
        print("실시간 학습기 초기화 중...")
        from utils.realtime_learner import initialize_realtime_learner
        realtime_learner_loaded = initialize_realtime_learner()
        if realtime_learner_loaded:
            print("실시간 학습기 초기화 완료")
        else:
            print("실시간 학습기 초기화 실패")
    except Exception as e:
        print(f"실시간 학습기 초기화 실패: {e}")
    
    try:
        print("경량 파인튜닝 초기화 중...")
        from utils.lightweight_finetuning import initialize_lightweight_finetuner
        lightweight_finetuner_loaded = initialize_lightweight_finetuner()
        if lightweight_finetuner_loaded:
            print("경량 파인튜닝 초기화 완료")
        else:
            print("경량 파인튜닝 초기화 실패")
    except Exception as e:
        print(f"경량 파인튜닝 초기화 실패: {e}")
    
    try:
        print("자동 데이터 수집기 초기화 중...")
        from utils.auto_data_collector import initialize_auto_collector
        auto_collector_loaded = initialize_auto_collector()
        if auto_collector_loaded:
            print("자동 데이터 수집기 초기화 완료")
        else:
            print("자동 데이터 수집기 초기화 실패")
    except Exception as e:
        print(f"자동 데이터 수집기 초기화 실패: {e}")
    
    try:
        print("자동 스케줄러 초기화 중...")
        from utils.auto_scheduler import initialize_auto_scheduler, start_auto_scheduler
        auto_scheduler_loaded = initialize_auto_scheduler()
        if auto_scheduler_loaded:
            start_auto_scheduler()
            print("자동 스케줄러 초기화 및 시작 완료")
        else:
            print("자동 스케줄러 초기화 실패")
    except Exception as e:
        print(f"자동 스케줄러 초기화 실패: {e}")
    
    port = int(os.environ.get('PORT', 8080))
    print(f"서버가 포트 {port}에서 시작됩니다.")
    app.run(host='0.0.0.0', port=port, debug=False) 