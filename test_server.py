from flask import Flask, request, jsonify
from datetime import datetime
import os
import json

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({'status': 'running', 'message': 'Test server is running'})

@app.route('/api/chat', methods=['POST'])
def chat():
    """간단한 테스트 챗봇 API"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': '메시지가 없습니다.'}), 400
        
        # 간단한 응답 생성
        response = f"테스트 응답: {user_message}"
        
        # 로그 기록
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_message,
            'bot_response': response
        }
        
        log_file = 'data/test_chat_logs.json'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            if len(logs) > 100:
                logs = logs[-100:]
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"로그 기록 실패: {e}")
        
        return jsonify({
            'response': response,
            'suggestions': [],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"채팅 API 오류: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("테스트 서버를 시작합니다...")
    app.run(host='0.0.0.0', port=8080, debug=True) 