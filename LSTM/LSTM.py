import os
import time
import math
import random
import requests
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

# ===============================================
# 0) 환경 설정 및 하이퍼파라미터
# ===============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
print(f"Using device: {device}")

# 하이퍼파라미터
batch_size = 64        # 배치 크기
block_size = 128       # 한 시퀀스 길이 (Truncated BPTT 길이)
embedding_dim = 256    # 임베딩 차원
hidden_size = 512      # LSTM hidden
num_layers = 1         # LSTM 층 수
dropout = 0.1          # LSTM dropout
learning_rate = 3e-3   # Adam 학습률
max_iters = 50         # 학습 스텝 수
eval_interval = 5      # eval 주기
eval_iters = 1         # eval에서 배치 반복 수
ckpt_path = "char_rnn_tinyshakespeare.pt" # 체크포인트 저장 경로

# ===============================================
# 1) 데이터 다운로드 & 전처리
# ===============================================

# Tiny Shakespeare (Karpathy) 데이터셋 다운로드
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_file = "tinyshakespeare.txt"
if not os.path.exists(data_file):
    print("Downloading Tiny Shakespeare...")
    with open(data_file, "wb") as f:
        f.write(requests.get(url, timeout=30).content)

with open(data_file, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Dataset length (chars): {len(text):,}")

# 문자 집합 & 인코딩/디코딩
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab_size: {vocab_size}")
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

def encode(s):
    """문자열을 텐서(인덱스)로 인코딩"""
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(t):
    """텐서(인덱스)를 문자열로 디코딩"""
    # t가 스칼라가 아닐 경우만 반복 처리
    if t.ndim > 0:
        return "".join([itos[int(i)] for i in t])
    else:
        return itos[int(t)]


data = encode(text)
# train/val split (90/10)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ===============================================
# 2) 배치 생성 함수 (언어모델용)
# ===============================================

def get_batch(split, batch_size=batch_size, block_size=block_size):
    """언어 모델 학습을 위한 배치(x, y) 생성"""
    source = train_data if split == "train" else val_data
    # 무작위 시작 인덱스 선택
    ix = torch.randint(len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ===============================================
# 3) 모델 정의
# ===============================================

class CharRNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM 레이어 정의
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        
        self.ln = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        
        # Xavier 초기화(선택)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, idx, h_c=None):
        # idx: (B, T)
        x = self.embed(idx) # (B, T, C)
        
        # h_c는 (h, c) 튜플 형태
        out, h_c_out = self.lstm(x, h_c) # out: (B, T, H), h_c_out: (h, c)
        
        out = self.ln(out)
        logits = self.head(out) # (B, T, V)
        return logits, h_c_out

# 모델 초기화
model = CharRNNLM(vocab_size, embedding_dim, hidden_size, num_layers, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# ===============================================
# 4) 손실 추정 함수
# ===============================================

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            logits, _ = model(xb)
            
            # CrossEntropyLoss를 위한 shape 변환: (B*T, V) vs (B*T)
            B, T, V = logits.shape
            loss = criterion(logits.view(B*T, V), yb.view(B*T))
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# ===============================================
# 5) 샘플링(텍스트 생성) 함수
# ===============================================

@torch.no_grad()
def generate(model, start_text="\n", max_new_tokens=400, temperature=1.0):
    model.eval()
    # 시작 텍스트 인코딩 및 배치 차원 추가: (1, T0)
    idx = encode(start_text).unsqueeze(0).to(device) 
    h_c = None # 초기 은닉/셀 상태는 None
    out_chars = [*start_text]
    
    for _ in range(max_new_tokens):
        # 모델에 입력할 때 마지막 block_size만 유지 (메모리/속도 관리)
        logits, h_c = model(idx[:, -block_size:], h_c) 
        
        # 마지막 시점의 로짓만 사용하고, temperature로 나누어 분포 조절
        logits = logits[:, -1, :] / max(1e-6, temperature) 
        
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1) # 다음 토큰 샘플링
        
        # 생성된 토큰을 입력 시퀀스에 추가
        idx = torch.cat([idx, next_id], dim=1)
        out_chars.append(decode(next_id))
        
    return "".join(out_chars)

# ===============================================
# 6) 학습 루프
# ===============================================

best_val = float("inf")
start = time.time()
print("Start training...")

for it in range(1, max_iters + 1):
    xb, yb = get_batch("train")
    
    logits, _ = model(xb)
    
    B, T, V = logits.shape
    loss = criterion(logits.view(B*T, V), yb.view(B*T))
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Gradient Clipping
    clip_grad_norm_(model.parameters(), max_norm=1.0) 
    optimizer.step()
    
    # 주기적 평가
    if it % eval_interval == 0 or it == 1:
        metrics = estimate_loss()
        elapsed = time.time() - start
        
        ppl_train = math.exp(metrics["train"])
        ppl_val = math.exp(metrics["val"])
        
        print(f"[{it:5d}/{max_iters}] "
              f"train loss={metrics['train']:.3f} (ppl={ppl_train:.1f}) "
              f"val loss={metrics['val']:.3f} (ppl={ppl_val:.1f}) "
              f"elapsed={elapsed/60:.1f} min")
        
        # 베스트 모델 저장
        if metrics["val"] < best_val:
            best_val = metrics["val"]
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "vocab_size": vocab_size,
                    "embedding_dim": embedding_dim,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                },
                "stoi": stoi, "itos": itos
            }, ckpt_path)
            print(f" -> Saved checkpoint to {ckpt_path}")


# ===============================================
# 7) 체크포인트 로드 (옵션)
# ===============================================

def load_checkpoint(path=ckpt_path):
    """저장된 체크포인트를 로드하여 모델을 반환"""
    # 체크포인트 파일이 없다면 현재 학습된 모델을 사용
    if not os.path.exists(path):
        print(f"Warning: Checkpoint file not found at {path}. Using currently trained model.")
        return model 
        
    ckpt = torch.load(path, map_location=device)
    m = CharRNNLM(
        ckpt["config"]["vocab_size"],
        ckpt["config"]["embedding_dim"],
        ckpt["config"]["hidden_size"],
        ckpt["config"]["num_layers"],
        ckpt["config"]["dropout"],
    ).to(device)
    m.load_state_dict(ckpt["model_state"])
    return m

print("\n=== Sample (temperature=0.8) ===")
print(generate(model, start_text="ROMEO:\n", max_new_tokens=500, temperature=0.8))

print("\n=== Sample (temperature=1.2) ===")
print(generate(model, start_text="JULIET:\n", max_new_tokens=500, temperature=1.2))