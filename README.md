# 🤖 자동학습 세무회계 챗봇

딥러닝 기반의 자동학습 세무회계 챗봇입니다. 실시간 학습과 경량 파인튜닝을 통해 지속적으로 성능을 개선합니다.

## 🚀 주요 기능

- **실시간 학습**: 사용자 대화를 실시간으로 학습
- **경량 파인튜닝**: LoRA + Adapter를 사용한 효율적 학습
- **자동 데이터 수집**: 웹 크롤링을 통한 세무회계 데이터 수집
- **자동 스케줄링**: 24시간마다 데이터 수집, 6시간마다 학습 트리거
- **웹 인터페이스**: 실시간 학습 통계 및 모니터링

## 🛠️ 기술 스택

- **Backend**: Python 3.11, Flask, PyTorch
- **AI/ML**: Transformer 모델, LoRA, Adapter
- **Deployment**: Docker, Railway
- **Data Collection**: BeautifulSoup4, Requests
- **Scheduling**: Schedule

## 📦 설치 및 실행

### 로컬 개발

```bash
# 1. 저장소 클론
git clone <repository-url>
cd chatbot

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 서버 실행
python simple_chat_server.py

# 4. 웹 접속
http://localhost:8080
```

### Railway 배포

#### 1. Railway 계정 생성
- [Railway](https://railway.app)에서 계정 생성
- GitHub 계정으로 로그인

#### 2. 프로젝트 생성
```bash
# Railway CLI 설치
npm install -g @railway/cli

# 로그인
railway login

# 프로젝트 초기화
railway init
```

#### 3. 환경 변수 설정
Railway 대시보드에서 다음 환경 변수 설정:
```
PYTHONPATH=/app
FLASK_APP=simple_chat_server.py
FLASK_ENV=production
PRODUCTION=true
```

#### 4. 배포
```bash
# 자동 배포 (GitHub 연동)
git push origin main

# 또는 수동 배포
railway up
```

#### 5. 도메인 설정
- Railway 대시보드에서 Custom Domain 설정
- 또는 Railway 제공 도메인 사용

## 🔧 API 엔드포인트

### 채팅
- `POST /api/chat` - 챗봇과 대화
- `GET /api/status` - 서버 상태 확인

### 학습 관리
- `POST /api/train` - 수동 학습 데이터 추가
- `POST /api/auto-collect` - 자동 데이터 수집 실행
- `GET /api/realtime-stats` - 실시간 학습 통계
- `GET /api/lightweight-metrics` - 경량 파인튜닝 지표

## 📊 학습 시스템

### 실시간 학습기
- **주기**: 50개 대화마다 자동 학습
- **품질 필터**: 0.5점 이상 대화만 학습
- **저장**: 대화 데이터 자동 저장

### 경량 파인튜닝
- **주기**: 50개 데이터마다 자동 학습
- **방식**: LoRA + Adapter 레이어 학습
- **품질 필터**: 0.3점 이상 데이터만 학습

### 자동 데이터 수집
- **주기**: 24시간마다 자동 실행
- **소스**: 
  - 합성 세무회계 Q&A (10개)
  - 국세청 FAQ (5개)
  - 네이버 지식iN 크롤링
  - 세무회계 블로그 수집

## 🔄 자동화 프로세스

1. **서버 시작** → 모든 시스템 초기화
2. **자동 스케줄러 시작** → 24시간마다 데이터 수집
3. **웹 크롤링** → 세무회계 관련 데이터 수집
4. **학습 시스템 추가** → 수집된 데이터 자동 학습
5. **모델 업데이트** → 지속적인 성능 개선

## 📈 모니터링

- **웹 대시보드**: 실시간 학습 통계 확인
- **API 엔드포인트**: 학습 상태 및 지표 조회
- **로그**: 상세한 학습 및 오류 로그

## 🐛 문제 해결

### 일반적인 문제

1. **모델 로드 실패**
   - `models/checkpoints/best_model.pth` 파일 확인
   - `utils/tokenizer.pkl` 파일 확인

2. **의존성 오류**
   - `pip install -r requirements.txt` 재실행
   - Python 3.11 버전 확인

3. **Railway 배포 실패**
   - Docker 빌드 로그 확인
   - 환경 변수 설정 확인
   - 포트 설정 확인 (8080)

### 로그 확인

```bash
# Railway 로그 확인
railway logs

# 로컬 로그 확인
tail -f logs/chatbot.log
```

## 📝 라이선스

MIT License

## 🤝 기여

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 지원

문제가 발생하면 GitHub Issues에 등록해 주세요. 