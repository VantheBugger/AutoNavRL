import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Uncertainty-aware PVO using MC Dropout
class UncertaintyPVONetwork(nn.Module):
    def __init__(self, seq_len=5, scan_dim=42, action_dim=2, hidden_dim=64, dropout=0.2):
        super(UncertaintyPVONetwork, self).__init__()
        self.spatial_mlp = nn.Sequential(
            nn.Linear(scan_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.gru = nn.GRU(input_size=32, hidden_size=hidden_dim, batch_first=True)
        # output both mean probability and a learnt log-variance (optional)
        self.prob_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, scan_seq, action):
        batch_size, seq_len, _ = scan_seq.size()
        scan_flat = scan_seq.view(batch_size * seq_len, -1)
        spatial_features = self.spatial_mlp(scan_flat)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)
        gru_out, hidden = self.gru(spatial_features)
        env_context = gru_out[:, -1, :]
        fused = torch.cat([env_context, action], dim=1)
        prob = self.prob_head(fused)
        return prob

# helper training function (small runs)
def train_network(model, dataset_path, epochs=10, batch_size=128, out_path=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = np.load(dataset_path)
    X_scans = torch.tensor(data['scans'], dtype=torch.float32)
    X_actions = torch.tensor(data['actions'], dtype=torch.float32)
    Y_labels = torch.tensor(data['labels'], dtype=torch.float32).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(X_scans, X_actions, Y_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    criterion = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xs, xa, y in loader:
            xs = xs.to(device)
            xa = xa.to(device)
            y = y.to(device)
            opt.zero_grad()
            preds = model(xs, xa)
            loss = criterion(preds, y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch+1}/{epochs} loss={total/len(loader):.4f}")
    if out_path is not None:
        torch.save(model.state_dict(), out_path)
    return model
