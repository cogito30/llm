import torch

def create_padding_mask(seq, pad_idx=0):
    """
    입력 시퀀스(seq)에서 pad_idx와 동일한 위치를 0으로, 유효한 단어를 1로 반환합니다.
    (PyTorch 버전에 따라 True/False 형태의 boolean 마스크를 쓰기도 합니다.)
    """
    # seq 형태: (batch_size, seq_len)
    
    # 패딩이 아닌 유효한 토큰의 위치를 True(또는 1)로 만듭니다.
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2) 
    # mask 형태: (batch_size, 1, 1, seq_len)
    # 중간에 1차원 두 개를 추가하는 이유는 이후 Multi-Head Attention 연산 시
    # 헤드(Head) 차원과 쿼리(Query)의 시퀀스 길이에 맞게 브로드캐스팅(Broadcasting)하기 위함입니다.
    
    return mask

def create_causal_mask(seq_len, device):
    """
    미래의 토큰을 보지 못하도록 대각선 아래쪽만 1(또는 True)로 채워진 하삼각행렬 마스크를 생성합니다.
    """
    # torch.tril은 행렬의 대각선 윗부분을 0으로 만듭니다 (하삼각행렬)
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
    
    # mask 형태: (seq_len, seq_len)
    # 디코더의 모든 배치와 모든 헤드에 동일하게 적용되므로 배치 차원은 생략 가능합니다.
    return mask

if __name__ == "__main__":
    # --- Mac OS MPS 환경 설정 ---
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # --- 더미 데이터 설정 ---
    PAD_IDX = 0
    # 2개의 문장, 시퀀스 길이는 5라고 가정 (0은 패딩 토큰)
    decoder_input = torch.tensor([
        [5, 12, 4, 0, 0],  # 첫 번째 문장은 뒤에 2개의 패딩이 있음
        [7, 9, 15, 3, 0]   # 두 번째 문장은 뒤에 1개의 패딩이 있음
    ], device=device)

    batch_size, seq_len = decoder_input.size()

    print("=== 1. 패딩 마스크 (Padding Mask) ===")
    pad_mask = create_padding_mask(decoder_input, pad_idx=PAD_IDX)
    # 가독성을 위해 마지막 2차원(seq_len, seq_len) 형태로 변환해서 출력
    print(pad_mask[0, 0, :]) # 첫 번째 문장의 마스크 모양: [True, True, True, False, False]

    print("\n=== 2. 인과적 마스크 (Causal Mask) ===")
    causal_mask = create_causal_mask(seq_len, device=device)
    print(causal_mask)

    print("\n=== 3. 디코더 최종 마스크 (Padding + Causal) ===")
    # 디코더에서는 이 둘을 논리적 AND(&) 연산으로 결합하여 사용합니다.
    decoder_mask = pad_mask & causal_mask
    print("첫 번째 문장의 최종 디코더 마스크:")
    print(decoder_mask[0, 0])