import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from typing import List

class ResidualDataset(Dataset):
    def __init__(self, activations_path: str, layers: List[int]):
        data = torch.load(activations_path)
        self.samples = []
        for layer in layers:
            for vec in data[layer]:
                self.samples.append(vec)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, feature_dim, bias=False)
        self.relu    = nn.ReLU()
        self.decoder = nn.Linear(feature_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [batch, input_dim]
        z = self.relu(self.encoder(x))
        rec = self.decoder(z)
        return z, rec

def train_autoencoder(
    activations_path: str,
    layers: List[int],
    input_dim: int,
    feature_dim: int,
    device: str,
    save_path: str
):
    ds = ResidualDataset(activations_path, layers)
    dl = DataLoader(ds, batch_size=128, shuffle=True)
    ae = SparseAutoencoder(input_dim, feature_dim).to(device)
    opt = optim.Adam(ae.parameters(), lr=1e-3)
    for epoch in range(10):
        total_loss = 0.0
        for x in dl:
            x = x.to(device)
            z, rec = ae(x)
            loss = nn.functional.mse_loss(rec, x) + 1e-3 * z.abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"[AE] Epoch {epoch+1}/10 loss {total_loss/len(dl):.4f}")
    torch.save(ae.encoder.state_dict(), save_path)
    print(f"[AE] Saved encoder weights → {save_path}")
