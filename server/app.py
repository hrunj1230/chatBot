from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys
import json
import torch
import numpy as np
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_pytorch import TransformerModel
from utils.tokenizer import ChatbotTokenizer
from utils.realtime_learner import initialize_realtime_learner, get_realtime_learner
from utils.lightweight_finetuning import initialize_lightweight_finetuner, get_lightweight_finetuner

app = Flask(__name__)
CORS(app)

# 전역 변수로 모델과 토크나이저 저장
model = None
tokenizer = None

def load_model():
    """학습된 모델을 로드합니다."""
    global model, tokenizer
    
    try:
        # 토크나이저 로드
        tokenizer = ChatbotTokenizer()
        tokenizer.load('utils/tokenizer.pkl')
        
        # 모델 로드
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load('models/checkpoints/best_model.pth', map_location=device)
        
        # 체크포인트에서 모델 설정 가져오기
        vocab_size = checkpoint.get('vocab_size', tokenizer.get_vocab_size())
        d_model = checkpoint.get('d_model', 256)
        num_layers = checkpoint.get('num_layers', 6)
        num_heads = checkpoint.get('num_heads', 8)
        d_ff = checkpoint.get('d_ff', 1024)
        
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff
        ).to(device)
        
        # 모델 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"모델이 {device}에 로드되었습니다.")
        return True
        
    except FileNotFoundError as e:
        print(f"모델 파일을 찾을 수 없습니다: {e}")
        print("서버는 실행되지만 모델이 로드되지 않았습니다.")
        return False
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("서버는 실행되지만 모델이 로드되지 않았습니다.")
        return False

def generate_response(user_input, max_length=50):
    """사용자 입력에 대한 응답을 생성합니다."""
    if model is None or tokenizer is None:
        return "모델이 로드되지 않았습니다."
    
    try:
        # 입력 텍스트를 토큰화
        input_sequence = tokenizer.encode([user_input])
        input_tensor = torch.tensor(input_sequence, dtype=torch.long)
        
        # GPU 사용 가능시 GPU로 이동
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # 응답 생성
        with torch.no_grad():
            output = model(input_tensor)
            
            # 다음 토큰 예측
            next_token = torch.argmax(output[0, -1, :]).unsqueeze(0)
            
            # 시퀀스 생성
            generated_sequence = [next_token.item()]
            
            for _ in range(max_length - 1):
                # 현재 시퀀스로 다음 토큰 예측
                current_sequence = torch.tensor([generated_sequence], dtype=torch.long).to(device)
                output = model(current_sequence)
                next_token = torch.argmax(output[0, -1, :]).unsqueeze(0)
                generated_sequence.append(next_token.item())
                
                # EOS 토큰이면 중단
                if next_token.item() == tokenizer.word_index.get('<EOS>', 0):
                    break
        
        # 토큰을 텍스트로 변환
        response = tokenizer.decode([generated_sequence])[0]
        return response
        
    except Exception as e:
        print(f"응답 생성 실패: {e}")
        return "죄송합니다. 응답을 생성할 수 없습니다."

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
        
        # 응답 생성
        response = generate_response(user_message)
        
        # 경량 파인튜닝 처리
        lightweight_result = None
        lightweight_finetuner = get_lightweight_finetuner()
        if lightweight_finetuner:
            # 경량 파인튜닝된 모델로 응답 생성
            lightweight_response = lightweight_finetuner.generate_response(user_message)
            lightweight_result = {
                'response': lightweight_response,
                'method': 'lightweight_finetuning'
            }
            
            # 대화 품질 평가 및 학습 데이터 추가
            conversation_quality = self._evaluate_conversation_quality(user_message, response)
            if conversation_quality > 0.3:  # 품질이 좋은 대화만 학습
                lightweight_finetuner.add_training_data(user_message, response, conversation_quality)
        
        # 실시간 학습 데이터 추가
        realtime_learner = get_realtime_learner()
        if realtime_learner:
            # 대화 품질 평가
            quality_score = _evaluate_conversation_quality(user_message, response)
            realtime_learner.add_conversation(user_message, response, quality_score)
            
            # 세무회계 관련 제안
            suggestions = realtime_learner.get_accounting_suggestions(user_message)
        else:
            suggestions = []
        
        # 로그 기록
        log_chat(user_message, response)
        
        return jsonify({
            'response': response,
            'suggestions': suggestions,
            'lightweight_result': lightweight_result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """서버 상태 확인"""
    try:
        return jsonify({
            'status': 'running',
            'model_loaded': model is not None,
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
            
            print(f"학습 데이터 추가: {question} -> {answer}")
            
            return jsonify({
                'status': 'success',
                'message': '학습 데이터가 성공적으로 추가되었습니다.',
                'data_count': len(existing_data),
                'timestamp': datetime.now().isoformat()
            }), 200
            
        except Exception as e:
            print(f"학습 데이터 저장 실패: {e}")
            return jsonify({'error': '학습 데이터 저장에 실패했습니다.'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """모델 재학습 실행"""
    try:
        print("모델 재학습 요청 받음...")
        
        # 재학습 스크립트 실행
        import subprocess
        result = subprocess.run([
            sys.executable, 
            'training/retrain_model.py'
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            return jsonify({
                'status': 'success',
                'message': '모델 재학습이 완료되었습니다.',
                'output': result.stdout,
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'error': '모델 재학습에 실패했습니다.',
                'output': result.stderr,
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
        realtime_learner = get_realtime_learner()
        if realtime_learner:
            stats = realtime_learner.get_stats()
            return jsonify({
                'status': 'success',
                'stats': stats,
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'error': '실시간 학습기가 초기화되지 않았습니다.',
                'timestamp': datetime.now().isoformat()
            }), 500
            
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
        lightweight_finetuner = get_lightweight_finetuner()
        if lightweight_finetuner:
            stats = lightweight_finetuner.get_stats()
            return jsonify({
                'status': 'success',
                'stats': stats,
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'error': '경량 파인튜너가 초기화되지 않았습니다.',
                'timestamp': datetime.now().isoformat()
            }), 500

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def _evaluate_conversation_quality(user_input, response):
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

def log_chat(user_input, response):
    """대화 로그를 기록합니다."""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user_input': user_input,
        'bot_response': response
    }
    
    log_file = 'data/chat_logs.json'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    try:
        # 기존 로그 로드
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # 새 로그 추가
        logs.append(log_entry)
        
        # 로그 저장 (최근 1000개만 유지)
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"로그 기록 실패: {e}")

# 모델 로드 함수 (Gunicorn에서 호출)
def initialize_model():
    global model_loaded
    model_loaded = load_model()
    
    if model_loaded:
        print("챗봇 서버가 시작되었습니다. (모델 로드 완료)")
    else:
        print("챗봇 서버가 시작되었습니다. (모델 로드 실패 - 기본 응답만 가능)")

# 모델 초기화 (안전하게)
print("서버 초기화 중...")
try:
    initialize_model()
    print("모델 초기화 완료")
    
    # 실시간 학습기 초기화
    print("실시간 학습기 초기화 중...")
    realtime_learner_loaded = initialize_realtime_learner()
    if realtime_learner_loaded:
        print("실시간 학습기 초기화 완료")
    else:
        print("실시간 학습기 초기화 실패 (기본 모드로 실행)")
    
    # 경량 파인튜닝 초기화
    print("경량 파인튜닝 초기화 중...")
    lightweight_finetuner_loaded = initialize_lightweight_finetuner()
    if lightweight_finetuner_loaded:
        print("경량 파인튜닝 초기화 완료")
    else:
        print("경량 파인튜닝 초기화 실패 (기본 모드로 실행)")
        
except Exception as e:
    print(f"모델 초기화 실패: {e}")
    print("기본 응답 모드로 서버를 시작합니다.")
    model_loaded = False

# Flask 서버 실행 (모든 환경에서)
if __name__ == '__main__':
    print("Flask 서버를 시작합니다...")
    # Railway 환경 변수에서 포트 가져오기
    port = int(os.environ.get('PORT', 8080))
    print(f"서버가 포트 {port}에서 시작됩니다.")
    app.run(host='0.0.0.0', port=port, debug=False) 