# Phase 6. Learning Rate 스케줄러를 포함한 훈련 파이프라인과 Greedy Decoding을 사용한 추론(Inference)
- 여기서는 트랜스포머 논문에서 제안한 독특한 학습률(Learning Rate) 스케줄링 기법을 적용하여 모델을 훈련하는 방법과, 훈련된 모델이 실제로 번역(혹은 텍스트 생성)을 수행할 때 사용하는 그리디 디코딩(Greedy Decoding) 기반의 추론 코드를 완성해 보겠습니다.

## 1. 웜업(Warm-up) 학습률 스케줄러
(이론적 배경)
- 트랜스포머 논문에서는 옵티마이저로 Adam을 사용하되, 학습률을 고정하지 않고 학습 단계(Step)에 따라 동적으로 변화시키는 방식을 사용합니다(일명 Noam Scheduler).
  - Warm-up 단계: 처음 지정된 횟수(warmup_steps) 동안은 학습률을 선형적으로 빠르게 증가시킵니다. (초기 불안정한 학습을 방지)
  - Decay 단계: 웜업이 끝나면 학습 단계의 역제곱근(inverse square root)에 비례하여 서서히 학습률을 감소시킵니다.

$$\text{LR} = d_{model}^{-0.5} \cdot \min(\text{step\_num}^{-0.5}, \text{step\_num} \cdot \text{warmup\_steps}^{-1.5})$$

- PyTorch의 optim.lr_scheduler.LambdaLR을 활용하여 이 수식을 간단히 적용할 수 있습니다.

```python
import torch
import torch.nn as nn
import torch.optim as optim

def get_lr_multiplier(step_num, d_model, warmup_steps=4000):
    if step_num == 0:
        step_num = 1
    # 논문의 수식 적용
    arg1 = step_num ** -0.5
    arg2 = step_num * (warmup_steps ** -1.5)
    return (d_model ** -0.5) * min(arg1, arg2)

# 옵티마이저와 스케줄러 설정 예시 (이전 단계의 model이 있다고 가정)
# optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
# lr_lambda = lambda step: get_lr_multiplier(step, d_model=256, warmup_steps=4000)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

## 2. 훈련 루프 (Training Pipeline)
- 훈련 시 디코더에는 정답 문장(Target)이 통째로 들어갑니다. 하지만 마스크(Causal Mask) 덕분에 미래의 단어는 보지 못하므로, "현재 단어까지 보고 다음 단어를 맞추는" 학습이 병렬로 한 번에 이루어집니다. 이를 교사 강요(Teacher Forcing)라고 합니다.

```python
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        
        # 디코더의 입력과 정답(Target)을 분리합니다.
        # 예: 원래 문장이 [<SOS>, I, am, a, student, <EOS>] 라면
        # 입력(tgt_input) : [<SOS>, I, am, a, student]
        # 정답(tgt_expect): [I, am, a, student, <EOS>]
        tgt_input = tgt[:, :-1]
        tgt_expect = tgt[:, 1:]
        
        optimizer.zero_grad()
        
        # 모델 순전파
        # output 형태: (batch_size, tgt_seq_len - 1, tgt_vocab_size)
        output = model(src, tgt_input)
        
        # CrossEntropyLoss 적용을 위해 텐서를 2차원으로 펼칩니다.
        output = output.reshape(-1, output.shape[-1])
        tgt_expect = tgt_expect.reshape(-1)
        
        # 손실 계산 (패딩 토큰은 ignore_index로 설정하여 손실 계산에서 제외됨)
        loss = criterion(output, tgt_expect)
        
        # 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()
        scheduler.step() # 스텝마다 학습률 업데이트
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)
```

## 3. 추론 (Inference / Greedy Decoding)
(이론적 배경)
- 학습이 끝난 후 실제로 문장을 생성할 때는 정답(Target)이 없습니다. 따라서 다음과 같은 순서로 단어를 하나씩 예측합니다.

1. 원본 문장(src)을 인코더에 통과시켜 문맥 정보(enc_out)를 얻어냅니다. (이 연산은 한 번만 수행)
2. 디코더의 첫 입력으로 문장의 시작을 알리는 <SOS> 토큰을 넣습니다.
3. 디코더가 다음 단어를 예측(가장 확률이 높은 단어 선택 = Greedy)합니다.
4. 예측된 단어를 입력 시퀀스에 추가하여 다시 디코더에 넣습니다.
5. 문장의 끝을 알리는 <EOS> 토큰이 나오거나 최대 길이에 도달할 때까지 3~4번을 반복합니다.

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    model.eval()
    
    # 1. 인코더 연산 (단 한 번만 수행)
    enc_out = model.encoder(src, mask=src_mask)
    
    # 2. 디코더의 초기 입력 생성 (처음에는 start_symbol 하나만 존재)
    # 차원: (1, 1) - batch_size=1, seq_len=1
    tgt = torch.tensor([[start_symbol]], dtype=torch.long).to(device)
    
    for _ in range(max_len):
        # 3. 타겟 마스크 생성
        tgt_mask = model.make_tgt_mask(tgt)
        
        # 4. 디코더 연산
        out = model.decoder(tgt, enc_out, src_mask, tgt_mask)
        
        # 5. 최종 선형 레이어 통과
        prob = model.fc_out(out[:, -1, :]) # 마지막으로 생성된 토큰에 대한 로짓
        
        # 6. 가장 높은 확률을 가진 단어 인덱스 선택 (Greedy)
        _, next_word = torch.max(prob, dim=1)
        next_word_item = next_word.item()
        
        # 7. 예측된 단어를 기존 tgt 시퀀스에 이어붙임(Concatenation)
        tgt = torch.cat([tgt, torch.tensor([[next_word_item]], device=device)], dim=1)
        
        # 8. 종료 토큰을 만나면 생성 중단
        if next_word_item == end_symbol:
            break
            
    return tgt
```

## 4. 전체 테스트 코드

```python
# --- Mac OS MPS 환경 설정 ---
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# --- 하이퍼파라미터 및 특수 토큰 설정 ---
PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2
SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = 100, 100 # 단순 테스트용 아주 작은 사전
D_MODEL = 64
MAX_LEN = 20

# 1. 모델, 손실함수, 옵티마이저, 스케줄러 초기화
model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, PAD_IDX, PAD_IDX,
                    d_model=D_MODEL, num_heads=2, d_ff=128, num_layers=2).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
lr_lambda = lambda step: get_lr_multiplier(step, d_model=D_MODEL, warmup_steps=40)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 2. 더미 DataLoader 생성 (임의의 1배치 훈련)
from torch.utils.data import TensorDataset, DataLoader
dummy_src = torch.randint(3, SRC_VOCAB_SIZE, (16, 10))
dummy_tgt = torch.randint(3, TGT_VOCAB_SIZE, (16, 12))
# 첫 토큰은 SOS, 마지막 토큰은 EOS로 임의 설정
dummy_tgt[:, 0] = SOS_IDX 
dummy_tgt[:, -1] = EOS_IDX

dataloader = DataLoader(TensorDataset(dummy_src, dummy_tgt), batch_size=16)

# 3. 훈련 루프 실행 (1 에포크)
print("=== 훈련 1 Epoch 실행 ===")
loss = train_epoch(model, dataloader, optimizer, scheduler, criterion, device)
print(f"Train Loss: {loss:.4f}\n")

# 4. 추론(Greedy Decoding) 테스트
print("=== 추론(인퍼런스) 테스트 ===")
# 번역할 가상의 입력 문장 (배치 1, 시퀀스 5)
test_src = torch.tensor([[5, 12, 45, 8, 20]]).to(device)
test_src_mask = model.make_src_mask(test_src)

generated_tgt = greedy_decode(model, test_src, test_src_mask, max_len=15, 
                              start_symbol=SOS_IDX, end_symbol=EOS_IDX, device=device)

print(f"입력 시퀀스: {test_src.tolist()}")
print(f"생성된 타겟 시퀀스: {generated_tgt.tolist()}")
```

- 지금까지 6단계에 걸쳐 데이터의 입력부터 어텐션 구조, 모델 조립, 그리고 학습과 추론 파이프라인까지 트랜스포머의 전체 생애주기를 Mac 환경의 PyTorch로 완벽하게 구현해 보았습니다. 이 베이스라인 코드를 바탕으로 향후 빔 서치(Beam Search)를 도입하거나 실제 데이터셋(예: Multi30k 등)을 연동하시면 멋진 기계 번역기를 완성하실 수 있습니다.

