import torch
import torch.nn as nn
import torch.optim as optim

from ch05_transformer import Transformer

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

if __name__ == "__main__":
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