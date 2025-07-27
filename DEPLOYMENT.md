# 🤖 AI 챗봇 배포 가이드

## 📋 **배포 옵션**

### 1. **로컬 Docker 배포** (가장 간단)

```bash
# 1. Docker 설치 확인
docker --version
docker-compose --version

# 2. 프로젝트 빌드 및 실행
docker-compose up --build

# 3. 백그라운드 실행
docker-compose up -d --build

# 4. 로그 확인
docker-compose logs -f chatbot

# 5. 서비스 중지
docker-compose down
```

### 2. **클라우드 서비스 배포**

#### **A. Heroku (무료 티어 종료)**
```bash
# 1. Heroku CLI 설치
# 2. 로그인
heroku login

# 3. 앱 생성
heroku create your-chatbot-name

# 4. 배포
git push heroku main

# 5. 앱 실행
heroku ps:scale web=1
```

#### **B. Railway (추천)**
```bash
# 1. Railway CLI 설치
npm install -g @railway/cli

# 2. 로그인
railway login

# 3. 프로젝트 초기화
railway init

# 4. 배포
railway up

# 5. 도메인 확인
railway domain
```

#### **C. Render**
```bash
# 1. Render.com에서 새 Web Service 생성
# 2. GitHub 저장소 연결
# 3. 빌드 명령어: pip install -r requirements.txt
# 4. 시작 명령어: python server/app.py
# 5. 환경 변수 설정
```

#### **D. AWS EC2**
```bash
# 1. EC2 인스턴스 생성 (Ubuntu 20.04)
# 2. 보안 그룹 설정 (포트 22, 80, 443, 5000)
# 3. SSH 연결
ssh -i your-key.pem ubuntu@your-ip

# 4. Docker 설치
sudo apt update
sudo apt install docker.io docker-compose

# 5. 프로젝트 클론
git clone https://github.com/your-username/your-chatbot.git
cd your-chatbot

# 6. 배포
sudo docker-compose up -d --build
```

### 3. **GitHub Actions 자동 배포**

#### **Railway 자동 배포 설정**
```yaml
# .github/workflows/deploy.yml
name: Deploy to Railway

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Railway
      uses: railway/deploy@v1
      with:
        railway_token: ${{ secrets.RAILWAY_TOKEN }}
```

## 🔧 **환경 변수 설정**

### **필수 환경 변수**
```bash
# .env 파일 생성
PYTHONPATH=/app
FLASK_ENV=production
FLASK_APP=server/app.py
```

### **선택적 환경 변수**
```bash
# 데이터베이스 설정
DATABASE_URL=postgresql://user:pass@host:port/db

# Redis 설정
REDIS_URL=redis://localhost:6379

# 로깅 설정
LOG_LEVEL=INFO
LOG_FILE=/app/logs/chatbot.log
```

## 📊 **모니터링 및 로그**

### **로그 확인**
```bash
# Docker 로그
docker-compose logs -f chatbot

# 애플리케이션 로그
tail -f logs/chatbot.log

# 시스템 리소스
docker stats
```

### **헬스체크**
```bash
# 서버 상태 확인
curl http://localhost:8080/api/status

# 응답 시간 측정
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8080/api/status
```

## 🔄 **자동 재학습 설정**

### **GitHub Actions 스케줄링**
```yaml
# .github/workflows/retrain.yml
name: Auto Retrain

on:
  schedule:
    - cron: '0 2 * * 1'  # 매주 월요일 오전 2시
  workflow_dispatch:     # 수동 실행

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Retrain model
      run: python training/train_pytorch.py
    
    - name: Deploy to Railway
      uses: railway/deploy@v1
      with:
        railway_token: ${{ secrets.RAILWAY_TOKEN }}
```

## 🚀 **성능 최적화**

### **1. 모델 최적화**
```python
# 모델 양자화
import torch.quantization as quantization

# 모델을 INT8로 양자화
quantized_model = quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### **2. 캐싱 설정**
```python
# Redis 캐싱 추가
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_response(user_input):
    cache_key = f"response:{hash(user_input)}"
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    return None
```

### **3. 로드 밸런싱**
```nginx
# nginx.conf
upstream chatbot {
    server chatbot:5000;
    server chatbot:5001;
    server chatbot:5002;
}

server {
    listen 80;
    location / {
        proxy_pass http://chatbot;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📈 **확장성 고려사항**

### **1. 수평 확장**
- 여러 인스턴스 실행
- 로드 밸런서 설정
- 세션 공유 (Redis)

### **2. 수직 확장**
- 더 큰 인스턴스 사용
- GPU 가속 활용
- 메모리 최적화

### **3. 데이터베이스 확장**
- PostgreSQL 클러스터
- 읽기 전용 복제본
- 샤딩 전략

## 🔒 **보안 설정**

### **1. HTTPS 설정**
```bash
# Let's Encrypt 인증서
sudo certbot --nginx -d your-domain.com

# 자동 갱신
sudo crontab -e
0 12 * * * /usr/bin/certbot renew --quiet
```

### **2. 방화벽 설정**
```bash
# UFW 설정
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

### **3. API 보안**
```python
# API 키 인증
from functools import wraps
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != 'your-secret-key':
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function
```

## 📞 **문제 해결**

### **일반적인 문제들**

1. **포트 충돌**
   ```bash
   # 포트 확인
   netstat -tulpn | grep :5000
   
   # 프로세스 종료
   sudo kill -9 <PID>
   ```

2. **메모리 부족**
   ```bash
   # 메모리 사용량 확인
   free -h
   
   # 스왑 파일 생성
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. **Docker 이미지 크기**
   ```dockerfile
   # 멀티스테이지 빌드
   FROM python:3.11-slim as builder
   COPY requirements.txt .
   RUN pip install --user -r requirements.txt
   
   FROM python:3.11-slim
   COPY --from=builder /root/.local /root/.local
   ```

## 📚 **추가 리소스**

- [Docker 공식 문서](https://docs.docker.com/)
- [Flask 배포 가이드](https://flask.palletsprojects.com/en/2.3.x/deploying/)
- [Railway 문서](https://docs.railway.app/)
- [Render 문서](https://render.com/docs) 