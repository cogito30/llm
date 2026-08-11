# Phase 5. Transformer
- 지금까지 만든 인코더 레이어(Encoder Layer)와 디코더 레이어(Decoder Layer)를 논문에서 제안한 대로 각각 $N$번(기본 6번) 반복하여 쌓아 올린 후, 이를 하나로 묶어 최종 Transformer 클래스를 완성해 보겠습니다.

## 1. 전체 인코더 (Encoder) 묶기
(이론적 배경)
- 전체 인코더는 1단계에서 만든 `TokenEmbedding`과 `PositionalEncoding`을 통과한 뒤, 4단계에서 만든 `EncoderLayer`를 $N$개 통과하는 구조입니다. 논문에서는 각 레이어의 결과가 다음 레이어의 입력으로 순차적으로 들어가며 더 깊은 문맥 정보를 학습하게 됩니다.


```python
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 d_ff: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        
        # 1. 입력 처리 모듈
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        
        # 2. 인코더 레이어를 num_layers(N)개 만큼 생성하여 리스트로 묶음
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        # 마지막 출력에 적용할 층 정규화
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x 차원: (batch_size, src_seq_len)
        x = self.embed(x)
        x = self.pe(x)
        
        # N개의 인코더 레이어를 순차적으로 통과
        for layer in self.layers:
            x = layer(x, mask)
            
        return self.norm(x) # (batch_size, src_seq_len, d_model)
```
## 2. 전체 디코더 (Decoder) 묶기
(이론적 배경)
- 디코더 역시 1단계의 임베딩과 위치 인코딩을 거친 뒤, DecoderLayer를 $N$번 통과합니다. 이때 중요한 점은 매 레이어마다 인코더의 최종 출력값(enc_out)을 입력으로 함께 전달받는다는 것입니다.

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 d_ff: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        
        # 디코더 레이어 N개 생성
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # x 차원: (batch_size, tgt_seq_len)
        x = self.embed(x)
        x = self.pe(x)
        
        # 디코더 레이어를 순차적으로 통과하며 enc_out과 어텐션 수행
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
            
        return self.norm(x) # (batch_size, tgt_seq_len, d_model)
```

## 3. 최종 Transformer 모델 조립

- 이제 인코더와 디코더를 연결하고, 입력 시퀀스에 맞춰 마스크(Mask)를 자동으로 생성하는 로직, 그리고 디코더의 출력을 우리가 원하는 단어 사전 크기(tgt_vocab_size)로 변환하는 최종 선형 레이어(Linear Layer)를 추가합니다.

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, src_pad_idx: int, tgt_pad_idx: int,
                 d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, 
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        
        # 패딩 인덱스 저장 (마스크 생성에 사용)
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        
        # 인코더와 디코더 초기화
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        
        # 디코더의 출력 (d_model)을 타겟 단어 사전의 크기 (tgt_vocab_size)로 변환
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
    def make_src_mask(self, src):
        # 소스(원본) 문장용 패딩 마스크 생성: (batch_size, 1, 1, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        # 1. 타겟 문장용 패딩 마스크: (batch_size, 1, 1, tgt_len)
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 2. 미래 정보를 가리는 인과적 마스크: (tgt_len, tgt_len)
        # 입력된 텐서(tgt)와 동일한 디바이스(MPS 또는 CPU)에 마스크를 생성합니다.
        tgt_len = tgt.shape[1]
        tgt_causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        
        # 3. 패딩 마스크와 인과적 마스크를 논리곱(&)으로 결합
        # 브로드캐스팅을 통해 결과 차원은 (batch_size, 1, tgt_len, tgt_len)이 됩니다.
        tgt_mask = tgt_pad_mask & tgt_causal_mask
        return tgt_mask
    
    def forward(self, src, tgt):
        # 1. 마스크 생성
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # 2. 인코더 통과
        enc_out = self.encoder(src, mask=src_mask)
        
        # 3. 디코더 통과
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        
        # 4. 최종 단어 확률 분포를 위한 선형 변환
        out = self.fc_out(dec_out)
        
        return out # (batch_size, tgt_seq_len, tgt_vocab_size)
```

## 4. 최종 통합 테스트

- 모든 컴포넌트를 결합한 Transformer 객체를 생성하고, Mac 환경(MPS)에서 한국어-영어 번역과 같은 상황을 가정한 더미 입력값을 넣어 결과를 확인합니다.

```python
# --- Mac OS MPS 환경 설정 ---
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"현재 디바이스: {device}")

# --- 하이퍼파라미터 (기계번역 가정) ---
SRC_VOCAB_SIZE = 8000 # 한국어 사전 크기
TGT_VOCAB_SIZE = 6500 # 영어 사전 크기
SRC_PAD_IDX = 0
TGT_PAD_IDX = 0
BATCH_SIZE = 64

# --- 모델 초기화 및 디바이스 이동 ---
model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE, 
    tgt_vocab_size=TGT_VOCAB_SIZE, 
    src_pad_idx=SRC_PAD_IDX, 
    tgt_pad_idx=TGT_PAD_IDX,
    d_model=256,   # 빠른 테스트를 위해 논문(512)보다 축소
    num_heads=8, 
    d_ff=1024, 
    num_layers=3   # 빠른 테스트를 위해 논문(6)보다 축소
).to(device)

# --- 더미 입력 생성 ---
# 소스 문장 (한국어) - 시퀀스 길이 15
src_data = torch.randint(1, SRC_VOCAB_SIZE, (BATCH_SIZE, 15)).to(device)
# 타겟 문장 (영어) - 시퀀스 길이 20
# (디코더 학습 시 입력으로는 보통 문장의 시작 토큰인 <SOS>가 포함된 시퀀스가 들어갑니다)
tgt_data = torch.randint(1, TGT_VOCAB_SIZE, (BATCH_SIZE, 20)).to(device)

# 패딩 토큰 일부 임의 생성 (테스트용)
src_data[:, 12:] = SRC_PAD_IDX 
tgt_data[:, 17:] = TGT_PAD_IDX 

print(f"\n입력 src 형태: {src_data.shape}")
print(f"입력 tgt 형태: {tgt_data.shape}")

# --- 순전파(Forward) 실행 ---
output = model(src_data, tgt_data)

print(f"\n최종 출력 형태: {output.shape}") 
# 기대 결과: torch.Size([64, 20, 6500])
# 즉, 배치 내 64개의 문장 각각의 20개 단어 위치에 대해, 6500개의 영어 단어 중 어떤 단어가 올지(로짓 값)를 출력함
```

- 이제 우리는 밑바닥부터 PyTorch를 활용해 모든 내부가 어떻게 동작하는지 완벽히 이해하면서 트랜스포머 모델을 조립해 냈습니다!
