# Phase 4. 서브 레이어(Feed Forward Network, Layer Normalization)와 인코더/디코더 레이어 조립
- 이제 트랜스포머의 뼈대를 이루는 각 레이어를 조립할 차례입니다. 
- 어텐션의 결과를 후처리하는 피드 포워드 신경망(FFN)과 학습을 안정적으로 만들어주는 잔차 연결(Residual Connection) 및 층 정규화(Layer Normalization)를 알아보고, 이를 바탕으로 인코더와 디코더의 단일 레이어를 완성해 보겠습니다.

## 1. 포지션 와이즈 피드 포워드 신경망 (Position-wise FFN)
(이론적 배경)
- 어텐션 메커니즘이 단어와 단어 사이의 '관계'를 파악하는 역할이라면, 피드 포워드 신경망은 그 파악된 정보를 바탕으로 각 단어(토큰) 자체의 특징을 더 깊게 학습하는 역할을 합니다. 'Position-wise'라는 이름이 붙은 이유는 시퀀스 내의 각 위치(단어)마다 동일한 신경망이 독립적으로 적용되기 때문입니다.

- 구조는 매우 단순하게 두 개의 선형(Linear) 레이어와 그 사이의 ReLU 활성화 함수로 이루어져 있습니다. 
- 일반적으로 첫 번째 레이어에서 은닉층의 크기($d_{ff}$)를 $d_{model}$의 4배 정도로 늘렸다가 두 번째 레이어에서 다시 원래 차원으로 복구합니다.

```python
import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # d_model 차원을 받아 d_ff 차원으로 확장 (보통 d_model의 4배)
        self.fc1 = nn.Linear(d_model, d_ff)
        # 다시 d_model 차원으로 축소
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x 차원: (batch_size, seq_len, d_model)
        return self.fc2(self.dropout(self.relu(self.fc1(x))))
```

## 2. 잔차 연결과 층 정규화 (Add & Norm)
(이론적 배경)
- 딥러닝 모델이 깊어질수록 역전파 시 기울기가 소실되거나 학습이 불안정해지는 문제가 발생합니다. 
- 트랜스포머는 이를 해결하기 위해 ResNet에서 유명해진 잔차 연결(Residual Connection)과 층 정규화(Layer Normalization)를 사용합니다.
  - Add (잔차 연결): 서브 레이어(어텐션 또는 FFN)를 통과한 출력값에 통과하기 전의 원래 입력값을 더해줍니다. $Output = Sublayer(x) + x$
  - Norm (층 정규화): 더해진 결과의 평균과 분산을 구해서 정규화(Normalization)를 수행하여 학습 밸런스를 맞춥니다.

## 3. 인코더 레이어 (Encoder Layer) 조립
- 이제 앞서 만든 `MultiHeadAttention`과 `PositionwiseFeedForward`를 결합하여 인코더의 단일 레이어를 만듭니다. 이 레이어가 $N$번 반복되는 것이 전체 인코더가 됩니다.

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 1. 멀티 헤드 어텐션 (Self-Attention)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 2. 피드 포워드 신경망
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 3. 층 정규화 레이어 2개
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        # --- 1. Multi-Head Attention + Add & Norm ---
        # Self-Attention이므로 Query, Key, Value 모두 x를 사용합니다.
        attn_out = self.self_attn(q=x, k=x, v=x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out)) # Add & Norm
        
        # --- 2. Feed Forward + Add & Norm ---
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out)) # Add & Norm
        
        return x
```

## 4. 디코더 레이어 (Decoder Layer) 조립
- 디코더 레이어는 인코더 레이어와 유사하지만, 중간에 인코더-디코더 어텐션 (Cross-Attention) 단계가 하나 더 추가되어 총 3개의 서브 레이어를 가집니다.

1. Masked Self-Attention: 디코더 자신이 생성한 이전 단어들만 보도록 마스킹된 상태에서 어텐션을 수행합니다.
2. Cross-Attention (Encoder-Decoder Attention): Query는 디코더에서, Key와 Value는 인코더의 최종 출력값에서 가져옵니다. 디코더가 다음 단어를 예측하기 위해 원본 문장(인코더)의 어느 부분에 집중해야 할지 결정하는 핵심 단계입니다.
3. Feed Forward Network: 인코더와 동일합니다.

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 1. 마스크가 적용되는 디코더 셀프 어텐션
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 2. 인코더의 출력을 참조하는 크로스 어텐션
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 3. 피드 포워드 신경망
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 3개의 서브 레이어에 대한 층 정규화
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # --- 1. Masked Multi-Head Attention + Add & Norm ---
        # 미래를 보지 못하게 하는 tgt_mask 적용
        attn1_out = self.self_attn(q=x, k=x, v=x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn1_out))
        
        # --- 2. Cross-Attention + Add & Norm ---
        # Query: 디코더에서 나온 x / Key, Value: 인코더에서 나온 enc_out
        # src_mask를 적용해 인코더 입력의 패딩 부분은 무시하도록 합니다.
        attn2_out = self.cross_attn(q=x, k=enc_out, v=enc_out, mask=src_mask)
        x = self.norm2(x + self.dropout(attn2_out))
        
        # --- 3. Feed Forward + Add & Norm ---
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        
        return x
```

## 5. 테스트 코드
- 지금까지 만든 인코더 레이어와 디코더 레이어에 임의의 텐서를 통과시켜, 차원이 꼬이지 않고 정상적으로 출력되는지 검증합니다.

```python
# --- Mac OS MPS 환경 설정 ---
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# --- 하이퍼파라미터 ---
D_MODEL = 512
NUM_HEADS = 8
D_FF = 2048
BATCH_SIZE = 4
SRC_SEQ_LEN = 10 # 원본 문장 길이
TGT_SEQ_LEN = 8  # 번역될 문장 길이

# 레이어 인스턴스화 및 디바이스 이동
enc_layer = EncoderLayer(d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF).to(device)
dec_layer = DecoderLayer(d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF).to(device)

# --- 더미 데이터 ---
# 인코더와 디코더의 입력 시퀀스
src_input = torch.randn(BATCH_SIZE, SRC_SEQ_LEN, D_MODEL).to(device)
tgt_input = torch.randn(BATCH_SIZE, TGT_SEQ_LEN, D_MODEL).to(device)

# 더미 마스크 (마스킹 로직이 차원 오류를 일으키지 않는지 확인하기 위한 1 행렬)
# src_mask: 패딩 마스크 역할
src_mask = torch.ones(BATCH_SIZE, 1, 1, SRC_SEQ_LEN).bool().to(device)
# tgt_mask: 디코더의 (패딩 + 인과적) 마스크 역할
tgt_mask = torch.ones(BATCH_SIZE, 1, TGT_SEQ_LEN, TGT_SEQ_LEN).bool().to(device)

print("=== 인코더/디코더 레이어 테스트 ===")
print(f"인코더 입력 형태: {src_input.shape}")

# 1. 인코더 레이어 통과
enc_output = enc_layer(src_input, mask=src_mask)
print(f"인코더 출력 형태: {enc_output.shape}") # (4, 10, 512) 기대

print(f"\n디코더 입력 형태: {tgt_input.shape}")

# 2. 디코더 레이어 통과 (인코더의 출력 결과물을 참조값으로 전달)
dec_output = dec_layer(tgt_input, enc_out=enc_output, src_mask=src_mask, tgt_mask=tgt_mask)
print(f"디코더 출력 형태: {dec_output.shape}") # (4, 8, 512) 기대
```

