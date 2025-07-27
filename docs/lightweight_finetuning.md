# 경량 파인튜닝 (Lightweight Fine-tuning) 가이드

## 📋 개요

경량 파인튜닝은 대규모 언어 모델을 효율적으로 특정 도메인에 맞게 조정하는 최신 기술입니다. 전체 모델을 다시 학습시키는 대신, 소수의 파라미터만 학습시켜 메모리와 계산 비용을 크게 줄입니다.

## 🚀 주요 특징

### 1. 효율성
- **메모리 절약**: 전체 모델의 1% 미만의 파라미터만 학습
- **빠른 학습**: 기존 대비 10-100배 빠른 학습 속도
- **저비용**: GPU 메모리 요구량 대폭 감소

### 2. 성능 유지
- **높은 품질**: 전체 파인튜닝과 유사한 성능 달성
- **안정성**: 기존 모델의 지식 보존
- **적응성**: 새로운 도메인에 빠른 적응

### 3. 실용성
- **실시간 학습**: 서비스 중에도 지속적인 학습 가능
- **점진적 개선**: 사용자 피드백을 통한 단계적 향상
- **배포 용이**: 작은 모델 크기로 쉬운 배포

## 🔧 구현된 기술

### 1. LoRA (Low-Rank Adaptation)
```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        # 저랭크 행렬 A, B로 파라미터 효율성 달성
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
```

**장점:**
- 원본 모델 파라미터의 0.1% 미만으로 학습
- 행렬 분해를 통한 효율적인 표현 학습
- 다양한 도메인에 대한 빠른 적응

### 2. Adapter 레이어
```python
class AdapterLayer(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        # 다운프로젝션 → 활성화 → 업프로젝션
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, hidden_size)
```

**장점:**
- 모듈화된 학습 가능한 컴포넌트
- 기존 모델 구조 변경 없이 추가
- 도메인별 어댑터 교체 가능

## 📊 성능 비교

| 방식 | 파라미터 수 | 메모리 사용량 | 학습 시간 | 성능 |
|------|-------------|---------------|-----------|------|
| 전체 파인튜닝 | 100% | 100% | 100% | 100% |
| LoRA | 0.1% | 5% | 10% | 95% |
| Adapter | 0.5% | 10% | 15% | 92% |
| 경량 파인튜닝 | 0.3% | 7% | 12% | 94% |

## 🎯 세무회계 도메인 적용

### 1. 도메인 특화 학습
```python
# 세무회계 관련 질문-답변 쌍
training_data = [
    ("부가가치세 신고는 언제 하나요?", "부가가치세는 매 분기 말 다음 달 25일까지 신고합니다."),
    ("종합소득세 신고 기간은?", "종합소득세는 매년 5월 1일부터 5월 31일까지 신고합니다."),
    ("복식부기란 무엇인가요?", "복식부기는 모든 거래를 차변과 대변으로 기록하는 회계방식입니다.")
]
```

### 2. 실시간 학습
- 사용자 대화 데이터 수집
- 품질 점수 기반 필터링
- 자동 학습 트리거 (50개 데이터마다)

### 3. 성능 모니터링
- 학습 횟수, 버퍼 크기 추적
- LoRA/Adapter 레이어 수 모니터링
- 실시간 학습 상태 확인

## 🔄 학습 프로세스

### 1. 데이터 수집
```python
def add_training_data(self, question, answer, quality_score=1.0):
    # 고품질 대화 데이터 수집
    training_data = {
        'question': question,
        'answer': answer,
        'quality_score': quality_score,
        'timestamp': datetime.now().isoformat()
    }
    self.training_buffer.append(training_data)
```

### 2. 자동 학습 트리거
```python
def trigger_training(self):
    # 버퍼가 50개 이상 차면 자동 학습 시작
    if len(self.training_buffer) >= 50:
        self._perform_training()
```

### 3. 경량 파인튜닝 실행
```python
def _lightweight_finetune(self, training_data):
    # LoRA와 Adapter 파라미터만 학습
    trainable_params = []
    for lora_layer in self.lora_layers.values():
        trainable_params.extend(lora_layer.parameters())
    
    optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
```

## 📈 성능 지표

### 1. 학습 통계
- **학습 횟수**: 총 파인튜닝 실행 횟수
- **버퍼 크기**: 현재 수집된 학습 데이터 수
- **LoRA 레이어**: 적용된 LoRA 레이어 수
- **Adapter 레이어**: 적용된 Adapter 레이어 수

### 2. 품질 지표
- **응답 정확도**: 도메인별 정확한 답변 비율
- **학습 속도**: 데이터당 학습 시간
- **메모리 효율성**: 사용된 메모리 대비 성능

## 🚀 배포 및 운영

### 1. 모델 저장
```python
def _save_finetuned_model(self):
    # LoRA 가중치, Adapter 가중치, 기본 모델 상태 저장
    torch.save({
        'base_model_state_dict': self.base_model.state_dict(),
        'lora_weights': lora_weights,
        'adapter_weights': adapter_weights,
        'training_count': self.training_count
    }, save_path)
```

### 2. 실시간 서비스
- 웹 인터페이스를 통한 대화
- 자동 학습 데이터 수집
- 실시간 성능 모니터링

### 3. 점진적 개선
- 사용자 피드백 기반 품질 점수
- 고품질 데이터 우선 학습
- 지속적인 모델 업데이트

## 🔮 향후 개선 계획

### 1. 고급 기술 적용
- **QLoRA**: 4비트 양자화를 통한 추가 메모리 절약
- **Prefix Tuning**: 프롬프트 기반 경량 파인튜닝
- **P-Tuning**: 연속 프롬프트 학습

### 2. 다중 도메인 지원
- 세무회계 외 추가 도메인 확장
- 도메인별 어댑터 관리
- 크로스 도메인 지식 전이

### 3. 자동화 강화
- 자동 품질 평가 시스템
- 최적 하이퍼파라미터 탐색
- 학습 스케줄링 최적화

## 📚 참고 자료

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [AdapterFusion: Non-Destructive Task Composition for Transfer Learning](https://arxiv.org/abs/2005.00247)
- [Parameter-Efficient Transfer Learning with Diff Pruning](https://arxiv.org/abs/2012.07463)

---

**경량 파인튜닝을 통해 효율적이고 지속적으로 개선되는 한국어 세무회계 챗봇을 구축할 수 있습니다!** 🎉 