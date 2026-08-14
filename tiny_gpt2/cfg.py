GPT_CONFIG_124M = {
    "vocab_size": 50257,       # 토크나이저에서 사용할 어휘사전 크기
    "context_length": 256,    # 위침 임베딩으로 모델이 다룰 수 있는 입력 토큰의 최대 개수
    "emb_dim": 768,            # 토큰의 임베딩 크기
    "n_heads": 12,             # 어텐션 헤드의 개수
    "n_layers": 12,            # 트랜스포머 블록 개수
    "drop_rate": 0.1,          # 과대 적합을 막기 위한 드롭아웃 비율
    "qkv_bias": False          # 멀티 헤드 어텐션의 W_q,W_k, W_v에서 bias 지정 여부
}