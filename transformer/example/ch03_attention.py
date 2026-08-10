import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None, dropout_layer=None):
    """
    q, k, v 차원: (batch_size, num_heads, seq_len, d_k)
    """
    # Key의 마지막 두 차원을 전치(Transpose)하여 Query와 행렬 곱(Dot Product)을 수행
    # 결과 차원: (batch_size, num_heads, seq_len, seq_len)
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 마스크가 존재할 경우, 마스크가 False(또는 0)인 부분을 -1e9(매우 작은 수)로 채움
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # Softmax를 적용하여 확률값(Attention Weights) 생성
    p_attn = F.softmax(scores, dim=-1)
    
    if dropout_layer is not None:
        p_attn = dropout_layer(p_attn)
        
    # 최종적으로 Value와 행렬 곱
    # 결과 차원: (batch_size, num_heads, seq_len, d_k)
    return torch.matmul(p_attn, v), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # d_model은 num_heads로 나누어 떨어져야 합니다.
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        # Q, K, V를 위한 선형 변환 레이어
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 여러 헤드의 결과를 합친 후 통과시킬 최종 선형 레이어
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Q, K, V에 각각 선형 변환을 적용하고, 헤드 수만큼 차원을 분리합니다.
        # view 적용 후: (batch_size, seq_len, num_heads, d_k)
        # transpose 적용 후: (batch_size, num_heads, seq_len, d_k)
        query = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key   = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 스케일드 닷 프로덕트 어텐션 수행
        x, attn_weights = scaled_dot_product_attention(query, key, value, mask, self.dropout)
        
        # 3. 쪼개졌던 헤드들을 다시 하나로 합칩니다.
        # transpose 및 contiguous 수행 후: (batch_size, seq_len, num_heads, d_k)
        # view 수행 후: (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        # 4. 최종 선형 변환
        return self.W_o(x)

if __name__ == "__main__":
    # --- Mac OS MPS 환경 설정 ---
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # --- 하이퍼파라미터 ---
    D_MODEL = 512
    NUM_HEADS = 8
    BATCH_SIZE = 2
    SEQ_LEN = 5

    # 모듈 초기화 및 디바이스 이동
    mha = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS).to(device)

    # --- 더미 데이터 ---
    # 셀프 어텐션(Self-Attention)의 경우 Q, K, V가 모두 동일한 입력에서 옵니다.
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL).to(device)

    # 2단계에서 구현한 패딩 마스크의 단순화 버전 (앞의 3개 단어만 유효, 뒤 2개는 패딩이라 가정)
    # 마스크 차원: (batch_size, 1, 1, seq_len) - 헤드 차원 브로드캐스팅을 위해 1, 1 추가
    dummy_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ], dtype=torch.bool, device=device).unsqueeze(1).unsqueeze(2)

    print(f"입력 차원: {dummy_input.shape}")
    print(f"마스크 차원: {dummy_mask.shape}")

    # 순전파 실행
    output = mha(dummy_input, dummy_input, dummy_input, dummy_mask)

    print(f"출력 차원: {output.shape}") 
    # 기대 결과: torch.Size([2, 5, 512]) -> 입력과 동일한 차원을 유지해야 합니다.
