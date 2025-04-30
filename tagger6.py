# tagger6.py - Character-level Language Model with N-gram context

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ---------------- Hyperparameters ----------------
EMBED_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.003
FIG_DIR = "figures"
SAMPLES_DIR = "samples"

# ---------------- Dataset ----------------
class CharDataset(Dataset):
    def __init__(self, text, k, char2idx):
        self.k = k
        self.char2idx = char2idx
        self.data = [char2idx[c] for c in text if c in char2idx]

    def __len__(self):
        return len(self.data) - self.k

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.k], dtype=torch.long)
        y = torch.tensor(self.data[idx + self.k], dtype=torch.long)
        return x, y

# ---------------- Model ----------------
class NgramCharLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, k):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * k, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.emb(x)          # [B, k, D]
        flat = e.view(e.size(0), -1)  # [B, k*D]
        h = self.act(self.fc1(flat))
        return self.fc2(h)

# ---------------- Training & Eval ----------------
def train_model(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

# ---------------- Sampling ----------------
def emphsample(model, char2idx, idx2char, prefix, num_chars, k):
    model.eval()
    result = list(prefix)
    context = [char2idx.get(c, 0) for c in prefix.lower()][-k:]
    context = [0]*(k - len(context)) + context

    for _ in range(num_chars):
        x = torch.tensor([context], dtype=torch.long).to(device)
        out = model(x)
        probs = torch.softmax(out[0], dim=0).cpu()
        next_idx = torch.multinomial(probs, 1).item()
        next_char = idx2char[next_idx]
        result.append(next_char)
        context = context[1:] + [next_idx]

    return ''.join(result)

# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True, help="Path to text file")
    parser.add_argument("--k", type=int, default=10, help="Context length")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    with open(args.corpus, encoding="utf-8") as f:
        raw = f.read()
    chars = sorted(set(raw))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    vocab_size = len(chars)
    k = args.k

    dataset = CharDataset(raw, k, char2idx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = NgramCharLM(vocab_size, EMBED_DIM, HIDDEN_DIM, k).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []
    for epoch in range(1, EPOCHS+1):
        train_loss = train_model(model, loader, criterion, optimizer)
        val_loss = evaluate_model(model, loader, criterion)
        losses.append(val_loss)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
        # Sample every few epochs
        gen = emphsample(model, char2idx, idx2char, prefix="", num_chars=100, k=k)
        with open(f"{SAMPLES_DIR}/sample_k{k}_ep{epoch}_without_prefix.txt", "w", encoding="utf-8") as f:
            f.write(gen)

    # Save final plot
    plt.plot(range(1, EPOCHS+1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(f"Char LM (k={k})")
    #plt.savefig(f"{FIG_DIR}/part6_k{k}_loss.png")
    plt.savefig(f"{FIG_DIR}/part6_k{k}_loss_without_prefix.png")

    print(f"Saved loss plot and samples for k={k}")