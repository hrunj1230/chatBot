# 빌드 스테이지
FROM python:3.11-slim as builder

# 빌드 도구 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# 프로덕션 스테이지
FROM python:3.11-slim

# 런타임 패키지만 설치
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 작업 디렉토리 설정
WORKDIR /app

# 사용자 Python 패키지 복사
COPY --from=builder /root/.local /root/.local

# PATH에 사용자 Python 패키지 추가
ENV PATH=/root/.local/bin:$PATH

# 애플리케이션 코드 복사
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p data models/checkpoints server/templates logs

# 포트 노출
EXPOSE 5000

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV FLASK_APP=server/app.py
ENV FLASK_ENV=production

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/status || exit 1

# 애플리케이션 실행
CMD ["python", "server/app.py"] 