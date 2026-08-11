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

if __name__ == "__main__":
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