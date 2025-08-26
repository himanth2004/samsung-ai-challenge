import os
import glob
import csv
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Model definition (kept local so this script is standalone)
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        H = D // self.d_k
        Q = self.q(x).view(B, T, H, self.d_k).transpose(1, 2)
        K = self.k(x).view(B, T, H, self.d_k).transpose(1, 2)
        V = self.v(x).view(B, T, H, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ V
        output = output.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(output)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim=128):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


class MouseDynamicsClassifier(nn.Module):
    def __init__(self, input_dim=2, d_model=64, num_heads=4, num_layers=2, num_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer_blocks(x)
        x = x.mean(dim=1)
        return self.classifier(x)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_session_csv(path: str) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            try:
                x = int(float(row[-2]))
                y = int(float(row[-1]))
                coords.append((x, y))
            except Exception:
                continue
    return coords


def make_windows(coords: List[Tuple[int, int]], window_size: int, stride: int) -> List[List[Tuple[int, int]]]:
    windows: List[List[Tuple[int, int]]] = []
    if len(coords) < window_size:
        return windows
    for start in range(0, len(coords) - window_size + 1, stride):
        windows.append(coords[start:start + window_size])
    return windows


class MouseDataset(Dataset):
    def __init__(self, human_dirs: List[str], robot_dirs: List[str], window_size: int = 50, stride: int = 25):
        self.samples: List[Tuple[torch.Tensor, int]] = []

        def collect_samples(dirs: List[str], label: int):
            for d in dirs:
                for csv_path in glob.glob(os.path.join(d, "*.csv")):
                    coords = read_session_csv(csv_path)
                    for w in make_windows(coords, window_size, stride):
                        # Normalize by first coordinate to reduce absolute position bias
                        if not w:
                            continue
                        x0, y0 = w[0]
                        norm = [(x - x0, y - y0) for (x, y) in w]
                        tensor = torch.tensor(norm, dtype=torch.float32)
                        self.samples.append((tensor, label))

        collect_samples(human_dirs, 0)
        collect_samples(robot_dirs, 1)
        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return x, y


def train():
    random.seed(42)
    torch.manual_seed(42)

    human_dirs = [
        os.path.join(BASE_DIR, "User7"),
    ]
    robot_dirs = [
        os.path.join(BASE_DIR, "User9"),
    ]

    dataset = MouseDataset(human_dirs, robot_dirs, window_size=50, stride=25)
    if len(dataset) == 0:
        raise RuntimeError("No training samples found. Check CSV paths.")

    # Split train/val
    val_ratio = 0.1
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=False)

    model = MouseDynamicsClassifier()
    device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                v_correct += (preds == y).sum().item()
                v_total += y.size(0)
        val_acc = v_correct / max(1, v_total)
        print(f"Epoch {epoch}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")

    # Save CPU-only state_dict
    out_path = os.path.join(BASE_DIR, "behavior_model.pth")
    state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict_cpu, out_path)
    print(f"Saved trained behavior model to: {out_path}")


if __name__ == "__main__":
    train()


