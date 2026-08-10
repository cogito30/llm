import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # vocab_size: 단어 사전의 크기, d_model: 임베딩 벡터의 차원
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x 차원: (배치 크기, 시퀀스 길이)
        # 임베딩 후 차원: (배치 크기, 시퀀스 길이, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 1. 0으로 채워진 (max_len, d_model) 크기의 행렬 생성
        pe = torch.zeros(max_len, d_model)
        
        # 2. pos를 위한 열 벡터 생성 (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3. 10000^(2i/d_model) 항 계산 (exp와 log를 사용해 수치적 안정성 확보)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 4. 짝수 인덱스(0, 2, 4...)에는 sin 적용
        pe[:, 0::2] = torch.sin(position * div_term)
        # 5. 홀수 인덱스(1, 3, 5...)에는 cos 적용
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 배치 연산을 위해 첫 번째 차원 추가: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 6. 모델 파라미터가 아닌 버퍼로 등록 
        # (optimizer가 업데이트하지 않지만 state_dict에는 저장됨)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x는 임베딩을 통과한 텐서: (batch_size, seq_len, d_model)
        # 입력 시퀀스 길이(x.size(1))만큼 pe를 잘라서 더해줌
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":
    # --- Mac OS MPS 환경 설정 ---
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 하이퍼파라미터 ---
    VOCAB_SIZE = 10000 # 단어 사전 크기
    D_MODEL = 512      # 논문과 동일한 임베딩 차원
    BATCH_SIZE = 4     # 4개의 문장
    SEQ_LEN = 10       # 각 문장은 10개의 단어로 구성

    # --- 모듈 초기화 및 디바이스 이동 ---
    embed = TokenEmbedding(vocab_size=VOCAB_SIZE, d_model=D_MODEL).to(device)
    pe = PositionalEncoding(d_model=D_MODEL).to(device)

    # --- 더미 입력 생성 ---
    # 0 ~ 9999 사이의 무작위 정수로 이루어진 (4, 10) 형태의 텐서
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
    print(f"입력 x 형태: {x.shape}") 

    # --- 순전파(Forward Pass) 실행 ---
    embedded_x = embed(x)
    print(f"임베딩 통과 후 형태: {embedded_x.shape}")

    encoded_x = pe(embedded_x)
    print(f"위치 인코딩(PE) 통과 후 최종 형태: {encoded_x.shape}")

    # 결과물 중 첫 번째 배치, 첫 번째 단어의 처음 5개 차원 값 확인
    print("\n첫 번째 단어의 임베딩 + 위치 정보 벡터 (앞 5개 차원):")
    print(encoded_x[0, 0, :5])