# Gunicorn 설정 파일
import multiprocessing

# 서버 설정
bind = "0.0.0.0:8080"
workers = 1
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# 타임아웃 설정
timeout = 30
keepalive = 2

# 로깅 설정
accesslog = "-"
errorlog = "-"
loglevel = "info"

# 프로세스 설정
preload_app = True
daemon = False

# 애플리케이션 초기화 함수
def on_starting(server):
    """서버 시작 시 모델 초기화"""
    print("Gunicorn 서버 시작 중...")
    
def when_ready(server):
    """워커 준비 시 모델 초기화"""
    print("워커 준비 완료, 모델 초기화 중...")
    try:
        from server.app import initialize_model
        initialize_model()
        print("모델 초기화 완료")
    except Exception as e:
        print(f"모델 초기화 실패: {e}")
        print("기본 응답 모드로 서버를 시작합니다.")

# 헬스체크 설정
check_config = True 