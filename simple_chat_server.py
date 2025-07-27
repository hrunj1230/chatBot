from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import os
import json

app = Flask(__name__)
CORS(app)

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
        
        # 간단한 응답 생성
        response = f"안녕하세요! '{user_message}'에 대한 응답입니다."
        
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

if __name__ == '__main__':
    print("간단한 챗봇 서버를 시작합니다...")
    port = int(os.environ.get('PORT', 8080))
    print(f"서버가 포트 {port}에서 시작됩니다.")
    app.run(host='0.0.0.0', port=port, debug=False) 