# Project: GPT-2 구현하기

- Reference: [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)


## 1단계. 텍스트 데이터 처리
- Tokenizer 구현하기
1) 텍스트 읽어오기
2) 텍스트 전처리: 특수토큰, 토큰으로 나누기
- 특수 토큰 처리: `<unk>`, `<endoftext>`, `<bos>`, `<eos>`, `<pad>`
3) 어휘사전 생성하기
4) 토큰화된 텍스트를 토큰 ID로 변환하기
5) BPE 사용하기

## 2단계. Dataset && DataLoader 구현하기
- input && target 쌍으로 나누기
- sliding window 방식으로 구현하기
- 배치 사이에 중첩이 있으면 overfitting 증가

## 3단계. 단어 임베딩하기
- 임베딩은 LUT로 동작함을 확인하기
- position embedding 인코딩하기
  - 절대 위치 임베딩 사용하기

## 4단계. Attention 계층 구현
- self-attention 구현하기
1) 입력 임베딩 벡터에서 query 선택
2) query와 입력 임베딩 벡터를 dot product해서 attention_score 구하기
3) attention_score를 층 정규화해서 attention_weight로 만들기
4) attention_wieght에 각 임베딩 벡터(key)를 곱하고 더해서 context_vector 만들기

- 정규화시 attention_score를 key의 임베딩 차원의 제곱근으로 나누기
- multi-head attention 구현하기
- causal mask 처리

## 5단계 GPT 구현하기
1) 전체 구조 잡기
2) 각 Layer 구현하기
- LayerNorm 구현
- GELU, SwiGLU 활성화 함수 구현
- FeedForward 모듈 구현
- Transformer Block 구현하기
  - Shortcut Connection 구현
3) GPT-2 모델 구현
- 토큰 임베딩 층과 출력 층의 가중치 묶기