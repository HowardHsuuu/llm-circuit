import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple

class FeatureDataset(Dataset):
    def __init__(self, features_path: str, logits_path: str, layers: List[int]):
        feat = torch.load(features_path)
        log  = torch.load(logits_path)
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in layers:
            Z = feat[layer]  # [N, F]
            L = log[layer]   # [N, V]
            for i in range(Z.size(0)):
                self.samples.append((Z[i], L[i]))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]  # (z: Tensor[F], logits: Tensor[V])

class CrossLayerTranscoder(nn.Module):
    def __init__(self, feature_dim: int, num_layers: int, vocab_size: int):
        super().__init__()
        self.num_layers = num_layers
        self.W_dec = nn.Parameter(
            torch.zeros(num_layers, num_layers, feature_dim, feature_dim)
        )
        self.W_out = nn.Parameter(torch.zeros(num_layers, feature_dim, vocab_size))

    def forward(self, feat_list: List[torch.Tensor]) -> torch.Tensor:
        h = [torch.zeros_like(feat_list[0]) for _ in range(self.num_layers)]
        for src in range(self.num_layers):
            f = feat_list[src]  # [F]
            for tgt in range(self.num_layers):
                W = self.W_dec[src, tgt]  # [F, F]
                h[tgt] = h[tgt] + W.transpose(0,1) @ f
        logits = torch.zeros(self.W_out.size(2), device=feat_list[0].device)
        for l in range(self.num_layers):
            logits = logits + feat_list[l] @ self.W_out[l]
        return logits

def train_clt(
    features_path: str,
    logits_path: str,
    layers: List[int],
    feature_dim: int,
    vocab_size: int,
    device: str,
    save_path: str
):
    ds = FeatureDataset(features_path, logits_path, layers)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    clt = CrossLayerTranscoder(feature_dim, len(layers), vocab_size).to(device)
    opt = optim.Adam(clt.parameters(), lr=1e-3)
    for epoch in range(20):
        total_loss = 0.0
        for z, target in dl:
            z, target = z.to(device), target.to(device)
            feat_list = [z for _ in layers]
            pred = clt(feat_list)
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"[CLT] Epoch {epoch+1}/20 loss {total_loss/len(dl):.4f}")

    torch.save({
        "W_dec": clt.W_dec.detach().cpu(),  # shape [L,L,F,F]
        "W_out": clt.W_out.detach().cpu()   # shape [L,F,V]
    }, save_path)
    print(f"[CLT] Saved weights → {save_path}")
