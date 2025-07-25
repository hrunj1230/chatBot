# PyTorch Transformer 챗봇

PyTorch와 Transformer 아키텍처를 이용한 대화형 챗봇입니다.

## 프로젝트 구조

```
chatbot/
├── data/           # 데이터 파일들
├── models/         # 모델 정의 및 저장
├── training/       # 학습 스크립트
├── inference/      # 추론 및 서빙
├── utils/          # 유틸리티 함수들
├── requirements.txt
└── README.md
```

## 설치 및 실행

1. 의존성 설치:
```bash
pip install -r requirements.txt
```

2. 데이터 준비:
```bash
python utils/data_preprocessing.py
```

3. 모델 학습:
```bash
python training/train_pytorch.py
```

4. 챗봇 실행:
```bash
python inference/chatbot_pytorch.py
```

## 주요 기능

- PyTorch Transformer 기반 대화 생성
- 한국어 대화 데이터 지원
- 실시간 대화 인터페이스
- 모델 성능 모니터링
- GPU/CPU 자동 감지 및 사용 