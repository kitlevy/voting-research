#2D vector version

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score

#2D data generation
def generate_turnout_dataset(num_samples=100000, n=5, gamma=1.0, seed=None, use_log=True):
    rng = np.random.RandomState(seed)
    uA = rng.rand(num_samples, n)   #utilities for candidate A
    uB = rng.rand(num_samples, n)   #utilities for candidate B

    SWA = uA.sum(axis=1)
    SWB = uB.sum(axis=1)
    a_star_is_A = (SWA >= SWB)

    d = np.abs(uA - uB)
    p = d ** gamma                      #turnout probs
    preferA = (uA > uB).astype(float)
    preferB = (uB > uA).astype(float)

    votesA = (p * preferA).sum(axis=1)
    votesB = (p * preferB).sum(axis=1)
    f_is_A = (votesA >= votesB)        #expected-vote winner (tie -> A)

    sw_a_star = np.where(a_star_is_A, SWA, SWB)
    sw_f = np.where(f_is_A, SWA, SWB)
    #numeric stability: sw_f > 0 since Uniform(0,1) draws
    y = sw_a_star / sw_f
    if use_log:
        y = np.log(y)

    #features shape (num_samples, n, 2)
    X = np.stack([uA, uB], axis=-1)
    return X.astype(np.float32), y.astype(np.float32)

class TurnoutDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)    #(N, n, 2)
        self.y = torch.from_numpy(y)    #(N,)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

#DeepSets model
class DeepSet(nn.Module):
    def __init__(self, in_dim=2, phi_dim=64, rho_dim=64):
        super().__init__()
        #phi: maps each voter's utilities (2 numbers) -> embedding vector
        self.phi = nn.Sequential(
            nn.Linear(in_dim, phi_dim),
            nn.ReLU(),
            nn.Linear(phi_dim, phi_dim),
            nn.ReLU()
        )
        #rho: maps the pooled embedding (sum across voters) -> final prediction
        self.rho = nn.Sequential(
            nn.Linear(phi_dim, rho_dim),
            nn.ReLU(),
            nn.Linear(rho_dim, 1)
        )
    def forward(self, x):
        #x: (batch, n, 2)
        batch, n, d = x.shape
        h = self.phi(x.view(-1, d)).view(batch, n, -1)  #(batch, n, phi_dim)
        s = h.sum(dim=1)                                #sum-pool -> (batch, phi_dim)
        out = self.rho(s).squeeze(-1)                   #(batch,)
        return out


def train_model(model, train_loader, val_loader=None, epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)                                #forward pass
            loss = criterion(pred, yb)
            opt.zero_grad()                                 #reset gradients
            loss.backward()                                 #backpropagate
            opt.step()                                      #update weights
            total_loss += loss.item() * xb.size(0)          
        avg_loss = total_loss / len(train_loader.dataset)
        if val_loader is not None:
            model.eval()
            ys, ps = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    p = model(xb).cpu().numpy()
                    ps.append(p); ys.append(yb.numpy())
            ys = np.concatenate(ys); ps = np.concatenate(ps)
            mse = mean_squared_error(ys, ps)
            print(f"Epoch {ep+1}/{epochs}: train_loss={avg_loss:.4e}, val_mse={mse:.4e}")
        else:
            print(f"Epoch {ep+1}/{epochs}: train_loss={avg_loss:.4e}")
    return model

if __name__ == "__main__":
    #params
    N = 80000
    n = 5
    gamma = 1.0
    X, y = generate_turnout_dataset(num_samples=N+20000, n=n, gamma=gamma, seed=0, use_log=True)
    Xtrain, ytrain = X[:N], y[:N]
    Xval, yval = X[N:], y[N:]
    train_ds = TurnoutDataset(Xtrain, ytrain)
    val_ds = TurnoutDataset(Xval, yval)
    tr = DataLoader(train_ds, batch_size=512, shuffle=True)
    va = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = DeepSet(in_dim=2, phi_dim=64, rho_dim=64)
    model = train_model(model, tr, va, epochs=20, lr=1e-3, device='cpu')

    #evaluate on validation set (convert back from log if needed)
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in va:
            p = model(xb).numpy()
            preds.append(p); ys.append(yb.numpy())
    ys = np.concatenate(ys); preds = np.concatenate(preds)
    #metrics on log-target
    print("Val MSE (log-target):", mean_squared_error(ys, preds))
    print("Val R2 (log-target):", r2_score(ys, preds))

    real_y = np.exp(ys); real_pred = np.exp(preds)
    print("Val MSE (original y):", mean_squared_error(real_y, real_pred))
    print("Max true distortion in val set:", real_y.max())
