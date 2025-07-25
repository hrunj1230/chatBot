import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

def evaluate_model_performance(model, test_dataset, tokenizer):
    """모델 성능을 평가합니다."""
    print("모델 성능 평가를 시작합니다...")
    
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    predictions = []
    true_labels = []
    
    start_time = time.time()
    
    for batch, (inp, tar) in enumerate(test_dataset):
        # 예측
        predictions_batch = model(inp, False, None)
        
        # 손실 계산
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
        )(tar, predictions_batch)
        
        # 패딩 토큰 제외
        mask = tf.math.logical_not(tf.math.equal(tar, 0))
        loss = tf.boolean_mask(loss, mask)
        total_loss += tf.reduce_mean(loss)
        
        # 정확도 계산
        predicted_ids = tf.argmax(predictions_batch, axis=-1)
        true_ids = tar
        
        # 패딩 토큰 제외하고 정확도 계산
        mask = tf.math.logical_not(tf.math.equal(true_ids, 0))
        predicted_ids = tf.boolean_mask(predicted_ids, mask)
        true_ids = tf.boolean_mask(true_ids, mask)
        
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_ids, true_ids), tf.float32))
        total_accuracy += accuracy
        
        num_batches += 1
        
        # 예측 결과 저장
        predictions.extend(predicted_ids.numpy())
        true_labels.extend(true_ids.numpy())
    
    end_time = time.time()
    
    # 평균 계산
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    # 추가 메트릭 계산
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted'
    )
    
    print(f"평가 완료 (소요시간: {end_time - start_time:.2f}초)")
    print(f"평균 손실: {avg_loss:.4f}")
    print(f"평균 정확도: {avg_accuracy:.4f}")
    print(f"정밀도: {precision:.4f}")
    print(f"재현율: {recall:.4f}")
    print(f"F1 점수: {f1:.4f}")
    
    return {
        'loss': avg_loss.numpy(),
        'accuracy': avg_accuracy.numpy(),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
        'true_labels': true_labels
    }

def plot_training_history(history):
    """학습 히스토리를 시각화합니다."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 손실 그래프
    ax1.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 정확도 그래프
    ax2.plot(history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def generate_sample_responses(model, tokenizer, sample_inputs, max_length=20):
    """샘플 입력에 대한 응답을 생성합니다."""
    print("샘플 응답 생성:")
    print("-" * 50)
    
    for input_text in sample_inputs:
        try:
            # 입력 토큰화
            input_sequence = tokenizer.encode([input_text])
            
            # 응답 생성
            predictions = model(input_sequence, False, None)
            predicted_ids = tf.argmax(predictions, axis=-1)
            
            # 응답 디코딩
            response = tokenizer.decode(predicted_ids.numpy())[0]
            
            print(f"입력: {input_text}")
            print(f"응답: {response}")
            print("-" * 30)
            
        except Exception as e:
            print(f"입력 '{input_text}'에 대한 응답 생성 실패: {str(e)}")

def benchmark_inference_speed(model, tokenizer, test_inputs, num_runs=100):
    """추론 속도를 벤치마크합니다."""
    print(f"추론 속도 벤치마크 ({num_runs}회 실행)...")
    
    # 워밍업
    warmup_input = tokenizer.encode(["안녕하세요"])
    for _ in range(10):
        _ = model(warmup_input, False, None)
    
    # 실제 벤치마크
    times = []
    for _ in range(num_runs):
        input_text = np.random.choice(test_inputs)
        input_sequence = tokenizer.encode([input_text])
        
        start_time = time.time()
        _ = model(input_sequence, False, None)
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"평균 추론 시간: {avg_time:.4f}초 (±{std_time:.4f})")
    print(f"초당 처리 가능한 요청 수: {1/avg_time:.2f}")
    
    return avg_time, std_time

def save_evaluation_results(results, filename='evaluation_results.json'):
    """평가 결과를 파일로 저장합니다."""
    import json
    
    # numpy 배열을 리스트로 변환
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        else:
            results_serializable[key] = value
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"평가 결과가 {filename}에 저장되었습니다.") 