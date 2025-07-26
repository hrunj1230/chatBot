# Gunicorn 설정 파일
import multiprocessing

# 서버 설정
bind = "0.0.0.0:5000"
workers = multiprocessing.cpu_count() * 2 + 1
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

# 헬스체크 설정
check_config = True 