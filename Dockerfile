FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p data models/checkpoints server/templates

# 포트 노출
EXPOSE 5000

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV FLASK_APP=server/app.py
ENV FLASK_ENV=production

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/status || exit 1

# 애플리케이션 실행
CMD ["python", "server/app.py"] 