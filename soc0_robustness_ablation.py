#!/usr/bin/env python3
"""
soc0_robustness_ablation.py

Standalone script: loads pretrained checkpoints (all N_TRAIN_SEEDS seeds)
for CAMEL, CoDA, CoDA-Efficient, Baseline, and MAML-Windowed, and runs ONLY
the SOC_0 initialization robustness ablation on sim-to-real data --
entirely post-hoc, no retraining.

Extracted verbatim from plot_sim_to_real_2.py's model classes, KF, and
adaptation logic, so results are guaranteed consistent with the main
pipeline. Unlike the main script, this one does NOT window train/val/test
splits or build training DataLoaders -- only sim2real_prepared is built,
since that's all this ablation needs, making startup much faster.

Tested at TWO calibration fractions -- 10% (the paper's main operating
point) and 1% (a harder point, to check the interaction between a noisier
raw meta-model signal and SOC_0 recovery) -- not the full calib_fracs
sweep, since this ablation is answering a KF-tuning question (does the
filter let the meta-model correct a bad initial belief?), not a
data-efficiency question (already covered elsewhere).

REQUIRES: ./checkpoints/seed_{0..N_TRAIN_SEEDS-1}/ containing, for every
seed: camel_best.pt, coda_best.pt, coda_efficient_best.pt,
maml_windowed_best.pt -- all previously trained and saved by the main
pipeline. This script does NOT retrain anything.
"""

import os
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
tex_fonts = {
    "text.usetex": False,
    "font.family": "serif",
    "axes.labelsize": 16,
    "font.size": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
}
plt.rcParams.update(tex_fonts)

# ═════════════════════════════════════════════════════════════════════════
# 1. Hyperparameters (identical to the main script)
# ═════════════════════════════════════════════════════════════════════════
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", device)

WINDOW_SIZE   = 60
FEATURE_DIM   = 5
HIDDEN_DIM    = 64
NUM_LAYERS    = 3
R_DIM         = 10
LR            = 2e-3
WEIGHT_DECAY  = 1e-3
RIDGE_LAMBDA  = 1e-3
CALIB_FRAC    = 0.10     # main operating point
L1_WEIGHT     = 1e-4
BATCH_SIZE    = 1024

KF_Q = 1e-5; KF_R = 1e-2

N_TRAIN_SEEDS = 5

feature_cols = ["v", "i", "Ts", "delta_v", "delta_i"]
target_col   = "SOC"

CHECKPOINT_DIR = "./checkpoints"; PLOT_DIR = "./plots"
os.makedirs(CHECKPOINT_DIR, exist_ok=True); os.makedirs(PLOT_DIR, exist_ok=True)

def get_seed_dir(seed):
    path = os.path.join(CHECKPOINT_DIR, f"seed_{seed}")
    os.makedirs(path, exist_ok=True)
    return path

print("Config loaded.")


# ═════════════════════════════════════════════════════════════════════════
# 2. Data Loading  (USER: adjust to your actual CSV layout, same as main script)
# ═════════════════════════════════════════════════════════════════════════
DATA_DIR = Path("./dataset_battery")

battery_folders = sorted([f for f in DATA_DIR.iterdir() if f.is_dir()])
dataset_runs   = {}
all_data_list  = []

for battery_folder in battery_folders:
    batt_name = battery_folder.name
    dataset_runs[batt_name] = {}
    for csv_file in sorted(battery_folder.glob("*.csv")):
        file_name = csv_file.stem
        df = pd.read_csv(csv_file)

        if "timestamp" not in df.columns:
            df["timestamp"] = np.arange(len(df), dtype=np.float32)

        if " - " in file_name:
            parts = file_name.split(" - ", maxsplit=1)
            profile, temp_str = parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "Unknown")
        else:
            profile, temp_str = file_name, "Unknown"
        if not temp_str:
            temp_str = "Unknown"

        df["Battery"] = batt_name
        df["Profile"] = profile
        df["Temperature"] = temp_str

        dataset_runs[batt_name][file_name] = df
        all_data_list.append(df)

df_all = pd.concat(all_data_list, ignore_index=True)
print(f"Loaded {len(all_data_list)} files across {len(dataset_runs)} batteries.")

for col in ["v", "i", "Ts", "SOC", "timestamp"]:
    assert col in df_all.columns, f"Missing column: {col}. Check your CSV column names."

for b in dataset_runs:
    for run_name in dataset_runs[b]:
        df = dataset_runs[b][run_name]
        df["delta_v"] = df["v"].diff().fillna(0)
        df["delta_i"] = df["i"].diff().fillna(0)

all_tasks = [
    {"battery": b, "run_name": rn, "df": df}
    for b, runs in dataset_runs.items()
    for rn, df in runs.items()
]
print(f"Total tasks: {len(all_tasks)}")


# ═════════════════════════════════════════════════════════════════════════
# 3. Battery-Level Split (50 / 15 / 35) + Sim-to-Real Holdout
# ═════════════════════════════════════════════════════════════════════════
SIM2REAL_BATTERY_NAMES = ["INR21700-50E - simtoreal"]

all_batteries_full = list(dataset_runs.keys())
sim2real_batteries  = [b for b in all_batteries_full if b in SIM2REAL_BATTERY_NAMES]
missing_s2r = [b for b in SIM2REAL_BATTERY_NAMES if b not in all_batteries_full]
if missing_s2r:
    print(f"  WARNING: sim-to-real battery name(s) not found: {missing_s2r}")

all_batteries = [b for b in all_batteries_full if b not in sim2real_batteries]
random.shuffle(all_batteries)   # must match the SAME seed/order the main script used

n_batt       = len(all_batteries)
n_train_batt = int(0.50 * n_batt)
n_val_batt   = int(0.15 * n_batt)

train_batteries = all_batteries[:n_train_batt]
val_batteries   = all_batteries[n_train_batt : n_train_batt + n_val_batt]
test_batteries  = all_batteries[n_train_batt + n_val_batt:]

# We only need train_tasks' LENGTH (to size CAMEL/CoDA/CoDA-Efficient's
# per-task parameter tensors so load_state_dict matches the checkpoint
# shape) -- never its actual windowed data, so no windowing is done for it.
train_tasks    = [t for t in all_tasks if t["battery"] in train_batteries]
sim2real_tasks = [t for t in all_tasks if t["battery"] in sim2real_batteries]

print(f"Train tasks (count only, for model sizing): {len(train_tasks)}")
print(f"Sim2Real tasks: {len(sim2real_tasks)}  batteries: {sim2real_batteries}")

assert len(sim2real_tasks) > 0, (
    "sim2real_tasks is empty -- check SIM2REAL_BATTERY_NAMES matches a key "
    "in dataset_runs."
)


# ═════════════════════════════════════════════════════════════════════════
# 4. Preprocessing Helpers
# ═════════════════════════════════════════════════════════════════════════
# NOTE: the scaler MUST be fit on the exact same train_tasks the main script
# used, since it directly affects every downstream prediction.
global_scaler = StandardScaler()
global_scaler.fit(np.vstack([t["df"][feature_cols].values.astype(np.float32) for t in train_tasks]))

def make_windows_numpy(df, scaler, window_size=WINDOW_SIZE):
    X_raw = scaler.transform(df[feature_cols].values.astype(np.float32))
    Y_raw = df[target_col].values.astype(np.float32)
    T, d  = X_raw.shape
    n     = T - window_size
    idx   = np.arange(n)[:, None] + np.arange(window_size)[None, :]
    X_out = X_raw[idx]
    Y_out = Y_raw[window_size:window_size + n].reshape(n, 1)
    t_out = df["timestamp"].values[window_size:window_size + n]
    return np.ascontiguousarray(X_out), np.ascontiguousarray(Y_out), t_out

def build_task_tensors(task_list):
    out = []
    for t in task_list:
        X, Y, ts = make_windows_numpy(t["df"], global_scaler)
        out.append({**t, "X": X, "Y": Y, "time": ts})
    return out

print("Preprocessing helpers ready.")


# ═════════════════════════════════════════════════════════════════════════
# 5. Model Architectures  (extracted verbatim from plot_sim_to_real_2.py)
# ═════════════════════════════════════════════════════════════════════════
class BasisMLP(nn.Module):
    """Shared windowed MLP: [B, W, d] -> [B, r]. Used by CAMEL and CoDA."""
    def __init__(self, window=WINDOW_SIZE, feat=FEATURE_DIM,
                 hidden=HIDDEN_DIM, r=R_DIM, n_layers=NUM_LAYERS, dropout=0.1):
        super().__init__()
        in_dim = window * feat
        layers = [nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, r))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class BaselineMLP(nn.Module):
    """Plain MLP: [B, W, d] -> [B, 1]. Backbone for CoDA/CoDA-Efficient/MAML-Windowed/Baseline."""
    def __init__(self, window=WINDOW_SIZE, feat=FEATURE_DIM,
                 hidden=HIDDEN_DIM, n_layers=NUM_LAYERS, dropout=0.1):
        super().__init__()
        in_dim = window * feat
        layers = [nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class CAMEL(nn.Module):
    def __init__(self, n_tasks, r=R_DIM):
        super().__init__()
        self.v_net  = BasisMLP(r=r)
        self.omegas = nn.Parameter(torch.randn(n_tasks, r, 1) * 0.05)

    def forward(self, x, task_idx):
        V = self.v_net(x)
        omega = self.omegas[task_idx]
        return torch.bmm(V.unsqueeze(1), omega).squeeze(1)

    def adapt(self, x_calib, y_calib, ridge=RIDGE_LAMBDA):
        with torch.no_grad():
            V = self.v_net(x_calib)
            A = V.T @ V + ridge * torch.eye(V.shape[1], device=V.device)
            return torch.linalg.solve(A, V.T @ y_calib)

    @torch.no_grad()
    def predict_with_omega(self, x, omega):
        return (self.v_net(x) @ omega).clamp(0, 1)


adaptation_step_coda = 40   # L-BFGS typically converges by step 10-15

class CoDaModel(nn.Module):
    """Original CoDA: full-rank flat hypernetwork lift."""
    def __init__(self, n_tasks, r=R_DIM, emb_size=None):
        super().__init__()
        emb_size = emb_size or r
        self.emb_size = emb_size
        self.backbone = BaselineMLP(dropout=0.1)
        self.embeddings = nn.Parameter(torch.randn(n_tasks, emb_size) * 0.01)

        self._backbone_param_shapes = [
            (name, p.shape, p.numel()) for name, p in self.backbone.named_parameters()
        ]
        total_backbone_params = sum(s for _, _, s in self._backbone_param_shapes)
        self.lifting = nn.Linear(emb_size, total_backbone_params, bias=False)
        nn.init.normal_(self.lifting.weight, std=1e-3)

    def _apply_delta(self, delta_flat):
        result = {}; offset = 0
        for name, shape, numel in self._backbone_param_shapes:
            result[name] = delta_flat[offset:offset + numel].view(shape)
            offset += numel
        return result

    def _adapted_forward(self, x_flat, e_1d):
        delta_flat = self.lifting(e_1d)
        delta_dict = self._apply_delta(delta_flat)
        adapted_params = {name: p + delta_dict[name] for name, p in self.backbone.named_parameters()}
        from torch.func import functional_call
        return functional_call(self.backbone, adapted_params, (x_flat,))

    def adapt(self, x_calib, y_calib, n_steps=adaptation_step_coda, lr=1.0, optimizer="lbfgs"):
        e = nn.Parameter(torch.zeros(self.emb_size, device=x_calib.device))
        x_flat = x_calib.view(x_calib.size(0), -1) if x_calib.dim() == 3 else x_calib
        loss_history = []

        def _loss():
            pred = self._adapted_forward(x_flat, e)
            l1 = self.lifting(e).abs().mean()
            return F.mse_loss(pred, y_calib) + L1_WEIGHT * l1

        if optimizer == "lbfgs":
            opt = optim.LBFGS([e], lr=lr, max_iter=1, history_size=10, line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad(); loss = _loss(); loss.backward(); return loss
            for _ in range(n_steps):
                loss = opt.step(closure)
                loss_history.append(loss.item())
        else:
            opt = optim.Adam([e], lr=lr)
            for _ in range(n_steps):
                opt.zero_grad(); loss = _loss(); loss.backward(); opt.step()
                loss_history.append(loss.item())
        return e.detach(), loss_history

    @torch.no_grad()
    def predict_with_emb(self, x, e):
        e_1d = e.view(-1)
        x_flat = x.view(x.size(0), -1) if x.dim() == 3 else x
        return self._adapted_forward(x_flat, e_1d).clamp(0, 1)


class CoDaModelEfficient(nn.Module):
    """CoDA-Efficient: low-rank per-layer lift, emb_size unchanged at r=10."""
    def __init__(self, n_tasks, r=R_DIM, emb_size=None):
        super().__init__()
        emb_size = emb_size or r
        self.emb_size = emb_size
        self.backbone = BaselineMLP(dropout=0.1)
        self.embeddings = nn.Parameter(torch.randn(n_tasks, emb_size) * 0.01)

        self._weight_specs = []
        self._bias_specs   = []
        for name, p in self.backbone.named_parameters():
            if p.dim() == 2:
                out_dim, in_dim = p.shape
                self._weight_specs.append((name, out_dim, in_dim))
            else:
                self._bias_specs.append((name, p.numel()))

        self.U = nn.ParameterDict()
        self.V = nn.ParameterDict()
        for name, out_dim, in_dim in self._weight_specs:
            key = name.replace(".", "_")
            self.U[key] = nn.Parameter(torch.randn(out_dim, emb_size) * 0.01)
            self.V[key] = nn.Parameter(torch.randn(in_dim,  emb_size) * 0.01)

        total_bias_params = sum(n for _, n in self._bias_specs)
        if total_bias_params > 0:
            self.bias_lift = nn.Linear(emb_size, total_bias_params, bias=False)
            nn.init.normal_(self.bias_lift.weight, std=1e-3)
        else:
            self.bias_lift = None

    def _apply_delta(self, e_1d):
        result = {}
        for name, out_dim, in_dim in self._weight_specs:
            key = name.replace(".", "_")
            U_l, V_l = self.U[key], self.V[key]
            result[name] = (U_l * e_1d) @ V_l.T
        if self.bias_lift is not None:
            bias_flat = self.bias_lift(e_1d)
            offset = 0
            for name, numel in self._bias_specs:
                result[name] = bias_flat[offset:offset + numel]
                offset += numel
        return result

    def _adapted_forward(self, x_flat, e_1d):
        delta_dict = self._apply_delta(e_1d)
        adapted_params = {name: p + delta_dict[name] for name, p in self.backbone.named_parameters()}
        from torch.func import functional_call
        return functional_call(self.backbone, adapted_params, (x_flat,))

    def adapt(self, x_calib, y_calib, n_steps=adaptation_step_coda, lr=1.0, optimizer="lbfgs"):
        e = nn.Parameter(torch.zeros(self.emb_size, device=x_calib.device))
        x_flat = x_calib.view(x_calib.size(0), -1) if x_calib.dim() == 3 else x_calib
        loss_history = []

        def _loss():
            pred = self._adapted_forward(x_flat, e)
            delta_dict = self._apply_delta(e)
            l1 = torch.cat([d.flatten() for d in delta_dict.values()]).abs().mean()
            return F.mse_loss(pred, y_calib) + L1_WEIGHT * l1

        if optimizer == "lbfgs":
            opt = optim.LBFGS([e], lr=lr, max_iter=1, history_size=10, line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad(); loss = _loss(); loss.backward(); return loss
            for _ in range(n_steps):
                loss = opt.step(closure)
                loss_history.append(loss.item())
        else:
            opt = optim.Adam([e], lr=lr)
            for _ in range(n_steps):
                opt.zero_grad(); loss = _loss(); loss.backward(); opt.step()
                loss_history.append(loss.item())
        return e.detach(), loss_history

    @torch.no_grad()
    def predict_with_emb(self, x, e):
        e_1d = e.view(-1)
        x_flat = x.view(x.size(0), -1) if x.dim() == 3 else x
        return self._adapted_forward(x_flat, e_1d).clamp(0, 1)

print("Model architectures ready.")


# ═════════════════════════════════════════════════════════════════════════
# 6. True Capacity Lookup + Kalman Filter (with x0/p0 support)
# ═════════════════════════════════════════════════════════════════════════
TRUE_CAPACITY_MAH = {
    "ANR26650M1":    2300,
    "PD3032":         180,
    "INR21700-50E - simtoreal":4500,
    "NCA103450":     2200,
    "NCA463436A":     680,
    "NCA593446":     1260,
    "NCA623535":     1050,
    "NCA673440":     1220,
    "NCA793540":     1515,
    "NCA843436":     1275,
    "NCR18500A":     1900,
    "NCR18650BD":    3030,
    "NCR18650BF":    2835,
    "NCR18650PF":    2700,
    "UF103450P":     1880,
    "UF463450F":      960,
    "UF553443ZU":    1000,
    "UF653450S":     1250,
    "UR18650A":      2150,
    "UR18650ZTA":    2900,
    "UR18650F":      2300,
    "T18650":        2200,
    "A123 - simtoreal": 2300,
}

def get_true_c_nom(battery_name):
    mah = TRUE_CAPACITY_MAH.get(battery_name)
    if mah is None:
        return None
    return (mah / 1000.0) * 3600.0   # Coulombs

def get_c_nom_for_eval(battery_name, df_task=None, calib_idx=None):
    """No estimation fallback -- raises if the battery isn't in the lookup table."""
    c_true = get_true_c_nom(battery_name)
    if c_true is None:
        raise ValueError(
            f"No true capacity found for battery '{battery_name}' in "
            f"TRUE_CAPACITY_MAH. Add its datasheet capacity before evaluating it."
        )
    return c_true


def kalman_filter_coulomb(z, current, dt, q=KF_Q, r=KF_R, c_nom=3600.0,
                          x0=None, p0=1.0):
    """
    1D Kalman filter with Coulomb-counting state dynamics.
    x0: initial SOC belief. None -> falls back to z[0] (raw NN's own first
        prediction). Pass an explicit value (e.g. 0.0, 0.5) to test recovery
        from a deliberately wrong initial belief.
    p0: initial error covariance. Higher p0 -> the filter starts trusting
        the "measurement" (raw NN prediction) more, correcting a bad x0
        faster, then settles into the normal Q/R-driven balance as p decays.
    """
    n = len(z)
    z       = np.nan_to_num(z, nan=0.0, posinf=1.0, neginf=0.0)
    current = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(c_nom) or c_nom <= 0:
        c_nom = 3600.0

    x      = np.zeros(n);  p = np.zeros(n)
    x[0]   = float(x0) if x0 is not None else z[0]
    p[0]   = p0
    dt_arr = np.full(n, dt) if np.isscalar(dt) else np.asarray(dt, dtype=np.float64)
    dt_arr = np.nan_to_num(dt_arr, nan=0.0, posinf=0.0, neginf=0.0)

    for k in range(1, n):
        x_pred = np.clip(x[k-1] - (current[k] * dt_arr[k]) / c_nom, 0.0, 1.0)
        p_pred = p[k-1] + q
        K      = p_pred / (p_pred + r)
        x[k]   = np.clip(x_pred + K * (z[k] - x_pred), 0.0, 1.0)
        p[k]   = (1 - K) * p_pred
    return x

print(f"True-capacity lookup and KF ready. Q={KF_Q} R={KF_R}")


# ═════════════════════════════════════════════════════════════════════════
# 7. Inference Helpers
# ═════════════════════════════════════════════════════════════════════════
def run_inference(X, model_fn, chunk=2048):
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            preds.append(model_fn(X[i:i+chunk]))
    return torch.cat(preds).cpu().numpy().flatten()


def get_calib_idx(N, N_c, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    idx, _ = torch.sort(torch.randperm(N, generator=g)[:N_c])
    return idx


def train_baseline(X_train, Y_train, epochs=1000):
    """Train BaselineMLP from scratch (Baseline has no meta-trained checkpoint)."""
    model = BaselineMLP(dropout=0.1).to(device)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=50, min_lr=1e-5)
    dl    = DataLoader(TensorDataset(X_train, Y_train), batch_size=min(512, len(X_train)),
                       shuffle=True, pin_memory=False)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        model.train()
        ep_loss = 0.0
        for X_b, Y_b in dl:
            opt.zero_grad()
            l = loss_fn(model(X_b), Y_b)
            l.backward(); opt.step()
            ep_loss += l.item() * len(X_b)
        epoch_loss = ep_loss / len(X_train)
        sched.step(epoch_loss)
        losses.append(epoch_loss)
    return model, losses

print("Inference helpers ready.")


# ═════════════════════════════════════════════════════════════════════════
# 8. MAML Architecture
# ═════════════════════════════════════════════════════════════════════════
MAML_LR_INNER = 2e-2
MAML_N_INNER  = 25
MAML_STATIC_IN = 3

class MAMLStaticMLP(nn.Module):
    """Paper-faithful single-timestep architecture -- unused here (only
    MAML-Windowed is evaluated), kept only so functional_forward's
    isinstance check resolves correctly."""
    def __init__(self, in_dim=MAML_STATIC_IN, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)

MAMLWindowedMLP = BaselineMLP   # identical architecture, different training

class MAMLPaperExactMLP(nn.Module):
    """Unused here -- kept only for functional_forward's isinstance check."""
    def __init__(self, in_dim=MAML_STATIC_IN, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


def functional_forward(model, x, params):
    """Run model.forward using `params` instead of model's own weights."""
    if isinstance(model, (MAMLStaticMLP, MAMLPaperExactMLP)):
        if x.dim() == 3:
            x = x[:, -1, :MAML_STATIC_IN]
        elif x.dim() == 2 and x.size(1) != MAML_STATIC_IN:
            x = x[:, :MAML_STATIC_IN]
    else:
        if x.dim() == 3:
            x = x.view(x.size(0), -1)

    param_iter = iter(params.values())
    for layer in list(model.net.children()):
        if isinstance(layer, nn.Linear):
            w, b = next(param_iter), next(param_iter)
            x = F.linear(x, w, b)
        elif isinstance(layer, nn.SiLU):
            x = F.silu(x)
        elif isinstance(layer, nn.ReLU):
            x = F.relu(x)
        elif isinstance(layer, nn.Dropout):
            pass
    return x

print("MAML architecture ready.")


# ═════════════════════════════════════════════════════════════════════════
# 9. Pre-compute Window Tensors (sim2real ONLY -- nothing else is needed)
# ═════════════════════════════════════════════════════════════════════════
print("Pre-computing sliding windows for sim2real_tasks only...")
sim2real_prepared = build_task_tensors(sim2real_tasks)
print(f"  Sim2Real: {sum(len(t['X']) for t in sim2real_prepared):,} windows, "
      f"{len(sim2real_prepared)} tasks")


# ═════════════════════════════════════════════════════════════════════════
# 10. Pre-flight checkpoint check
# ═════════════════════════════════════════════════════════════════════════
_required_ckpts = ["camel_best.pt"]

_missing_ckpts = []
for ts in range(N_TRAIN_SEEDS):
    seed_dir = get_seed_dir(ts)
    for fname in _required_ckpts:
        fpath = os.path.join(seed_dir, fname)
        if not os.path.exists(fpath):
            _missing_ckpts.append(fpath)

if _missing_ckpts:
    raise FileNotFoundError(
        f"Missing {len(_missing_ckpts)} checkpoint file(s):\n" +
        "\n".join(f"  {p}" for p in _missing_ckpts) +
        f"\n\nExpected all {N_TRAIN_SEEDS} seeds x {len(_required_ckpts)} "
        f"checkpoints under {CHECKPOINT_DIR}/seed_{{0..{N_TRAIN_SEEDS-1}}}/."
    )
print(f"All {N_TRAIN_SEEDS * len(_required_ckpts)} required checkpoints found.")


# ═════════════════════════════════════════════════════════════════════════
# 11. SOC_0 Initialization Robustness Ablation (post-hoc only)
# ═════════════════════════════════════════════════════════════════════════
# Tested at TWO calibration fractions: 10% (paper's main operating point)
# and 1% (a harder point, checking whether a noisier raw meta-model signal
# makes SOC_0 recovery harder). NOT the full calib_fracs sweep -- this
# ablation targets KF-tuning behaviour, not data efficiency.
#
# Only p0 (initial covariance) is varied, NOT the paper's tuned steady-state
# Q/R -- p0 controls how fast the filter starts trusting the "measurement"
# (raw NN prediction) after a bad x0, without touching normal-case accuracy.

ABLATION_FRACS = [0.1]
SOC0_VARIANTS = {
    # "default (raw-NN)": None,
    "SOC0=0.0":         0.0,
    # "SOC0=0.5":         0.5,
}
P0_VARIANTS = {
    "p0=0.0001 (low, confident-but-wrong x0)": 0.0001,   # K~0.011 at step 1 -- slow correction
    "p0=1 (current default)":                  1.0,       # K~0.99 at step 1 -- near-instant correction
}


def _adapt_and_raw_soc0(method, seed, task_list, frac):
    seed_dir = get_seed_dir(seed)
    out = []

    if method == "CAMEL":
        ck = torch.load(os.path.join(seed_dir, "camel_best.pt"), map_location=device)
        m = CAMEL(n_tasks=len(train_tasks), r=R_DIM).to(device)
        m.load_state_dict(ck["state_dict"]); m.eval()
    # elif method == "CoDA":
    #     ck = torch.load(os.path.join(seed_dir, "coda_best.pt"), map_location=device)
    #     m = CoDaModel(n_tasks=len(train_tasks), r=R_DIM, emb_size=R_DIM).to(device)
    #     m.load_state_dict(ck["state_dict"]); m.eval()
    # elif method == "CoDA-Efficient":
    #     ck = torch.load(os.path.join(seed_dir, "coda_efficient_best.pt"), map_location=device)
    #     m = CoDaModelEfficient(n_tasks=len(train_tasks), r=R_DIM, emb_size=R_DIM).to(device)
    #     m.load_state_dict(ck["state_dict"]); m.eval()
    # elif method == "MAML":
    #     maml_state = torch.load(os.path.join(seed_dir, "maml_windowed_best.pt"), map_location=device)
    #     m = MAMLWindowedMLP().to(device)

    for t in task_list:
        X = torch.from_numpy(t["X"]).float().to(device)
        Y = torch.from_numpy(t["Y"]).float().to(device)
        Ytrue = Y.cpu().numpy().flatten()
        N = len(X)

        if method == "Baseline":
            Nc  = max(10, int(frac * N))
            idx = get_calib_idx(N, Nc, seed=seed)
            bl, _ = train_baseline(X[idx], Y[idx], epochs=1000)
            bl.eval()
            Yraw = run_inference(X, lambda x: bl(x).clamp(0, 1))
        elif method == "CAMEL":
            Nc  = max(R_DIM + 3, int(frac * N))
            idx = get_calib_idx(N, Nc, seed=seed)
            omega = m.adapt(X[idx], Y[idx])
            Yraw = run_inference(X, lambda x: m.predict_with_omega(x, omega))
        # elif method in ("CoDA", "CoDA-Efficient"):
        #     Nc  = max(16, int(frac * N))
        #     idx = get_calib_idx(N, Nc, seed=seed)
        #     e, _ = m.adapt(X[idx], Y[idx], n_steps=adaptation_step_coda, lr=5e-3)
        #     Yraw = run_inference(X, lambda x: m.predict_with_emb(x, e))
        # elif method == "MAML":
        #     Nc  = max(10, int(frac * N))
        #     idx = get_calib_idx(N, Nc, seed=seed)
        #     fast_weights = {n: p.clone().requires_grad_(True) for n, p in maml_state.items()}
        #     for _ in range(MAML_N_INNER):
        #         pred = functional_forward(m, X[idx], fast_weights)
        #         loss = F.mse_loss(pred, Y[idx])
        #         if torch.isnan(loss):
        #             fast_weights = {n: p.clone().requires_grad_(True) for n, p in maml_state.items()}
        #             break
        #         grads = torch.autograd.grad(loss, fast_weights.values(), allow_unused=True)
        #         fast_weights = {n: (w - MAML_LR_INNER * g).detach().requires_grad_(True)
        #                         for n, w, g in zip(fast_weights.keys(), fast_weights.values(), grads)}
        #         if any(torch.isnan(w).any() for w in fast_weights.values()):
        #             fast_weights = {n: p.clone().requires_grad_(True) for n, p in maml_state.items()}
        #             break
        #     with torch.no_grad():
        #         Yraw = run_inference(X, lambda x: functional_forward(m, x, fast_weights).clamp(0, 1))

        I_arr  = t["df"]["i"].values[WINDOW_SIZE:].astype(np.float64)
        dt_arr = np.diff(t["time"].astype(np.float64), prepend=t["time"][0])
        n_min  = min(len(Yraw), len(I_arr), len(dt_arr))
        Yraw, I_arr, dt_arr = Yraw[:n_min], I_arr[:n_min], dt_arr[:n_min]
        Ytrue_t = Ytrue[:n_min]
        c_nom = get_c_nom_for_eval(t["battery"], t["df"], idx)
        out.append(dict(battery=t["battery"], run=t.get("run_name", ""), Yraw=Yraw,
                        I_arr=I_arr, dt_arr=dt_arr, Ytrue=Ytrue_t, c_nom=c_nom))
    return out


print("\n" + "=" * 78)
print("SOC_0 INITIALIZATION ROBUSTNESS ABLATION (sim-to-real, post-hoc, "
      f"{N_TRAIN_SEEDS} train-seed-paired evaluations, fracs={ABLATION_FRACS})")
print("=" * 78)

# soc0_methods = ["CAMEL", "CoDA", "CoDA-Efficient", "Baseline", "MAML"]
soc0_methods = ["CAMEL", "Baseline"]
soc0_rows = []

for frac in ABLATION_FRACS:
    for method in soc0_methods:
        per_seed_adapted = [
            _adapt_and_raw_soc0(method, seed, sim2real_prepared, frac)
            for seed in range(N_TRAIN_SEEDS)
        ]
        for soc0_label, x0 in SOC0_VARIANTS.items():
            for p0_label, p0 in P0_VARIANTS.items():
                seed_rmses = []
                for adapted in per_seed_adapted:
                    task_rmses = []
                    for a in adapted:
                        Ykf = kalman_filter_coulomb(a["Yraw"], a["I_arr"], a["dt_arr"],
                                                    c_nom=a["c_nom"], x0=x0, p0=p0)
                        task_rmses.append(float(np.sqrt(mean_squared_error(a["Ytrue"], Ykf))))
                    seed_rmses.append(float(np.mean(task_rmses)))
                soc0_rows.append(dict(frac=frac, method=method, soc0=soc0_label, p0=p0_label,
                                      rmse_mean=float(np.mean(seed_rmses)),
                                      rmse_std=float(np.std(seed_rmses))))
        print(f"  frac={frac*100:.0f}%  {method}: done ({N_TRAIN_SEEDS} seeds x "
              f"{len(sim2real_prepared)} tasks adapted once, re-filtered under "
              f"{len(SOC0_VARIANTS)}x{len(P0_VARIANTS)} KF configs)")

df_soc0 = pd.DataFrame(soc0_rows)
print("\n" + df_soc0.to_string(index=False))
df_soc0.to_csv(f"{PLOT_DIR}/soc0_robustness_ablation.csv", index=False)
print("\nSaved soc0_robustness_ablation.csv")

# ── Plot: grouped bars, one figure per calib_frac, one panel per p0 ─────────
soc0_labels = list(SOC0_VARIANTS.keys())
bar_w = 0.8 / len(soc0_labels)
x_pos = np.arange(len(soc0_methods))
soc0_colors = {"default (raw-NN)": "#333333", "SOC0=0.0": "#d62728", "SOC0=0.5": "#ff7f0e"}

for frac in ABLATION_FRACS:
    fig, axes = plt.subplots(1, len(P0_VARIANTS), figsize=(7 * len(P0_VARIANTS), 5), sharey=True)
    if len(P0_VARIANTS) == 1:
        axes = [axes]
    frac_df = df_soc0[df_soc0["frac"] == frac]

    for ax, (p0_label, p0) in zip(axes, P0_VARIANTS.items()):
        sub = frac_df[frac_df["p0"] == p0_label]
        for i, soc0_label in enumerate(soc0_labels):
            vals = [sub[(sub["method"] == m) & (sub["soc0"] == soc0_label)]["rmse_mean"].values[0]
                    for m in soc0_methods]
            errs = [sub[(sub["method"] == m) & (sub["soc0"] == soc0_label)]["rmse_std"].values[0]
                    for m in soc0_methods]
            offset = (i - (len(soc0_labels) - 1) / 2) * bar_w
            ax.bar(x_pos + offset, vals, yerr=errs, width=bar_w, capsize=3,
                  color=soc0_colors[soc0_label], label=soc0_label, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(soc0_methods, fontsize=9, rotation=20)
        ax.set_title(p0_label, fontsize=11)
        ax.set_ylabel("Mean RMSE (sim-to-real)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"SOC_0 Initialization Robustness — Sim-to-Real, calib={frac*100:.0f}%",
                 fontsize=13, y=1.03)
    plt.tight_layout()
    fname = f"{PLOT_DIR}/soc0_robustness_ablation_calib{int(frac*100)}pct.pdf"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {fname}")

# ── Illustrative time-domain plot: one sim-to-real task, SOC_0 in {1.0, 0.5, 0.0} ─
# Three initial beliefs shown together per method, all using the SAME p0
# (the current default, p0=1.0 -- near-instant correction per the math
# above) so the comparison isolates "how wrong was the initial guess",
# not "how was the filter tuned".
TRAJ_RUN_NAME = "BJDST - 15"   # specific INR21700 run to plot; None -> falls back to the first run found
traj_task = next((t for t in sim2real_prepared if t.get("run_name") == TRAJ_RUN_NAME), None)
if traj_task is None:
    available = [t.get("run_name") for t in sim2real_prepared]
    raise ValueError(
        f"Run '{TRAJ_RUN_NAME}' not found in sim2real_prepared. "
        f"Available runs: {available}"
    )
print(f"\nTrajectory illustration task: {traj_task.get('run_name', '')}")

TRAJ_SOC0_VARIANTS = {
    # "SOC0=1.0 (true)": 1.0,
    # "SOC0=0.5":         0.5,
    "SOC0=0.0":         0.0,
}
# Both p0 values now shown together: line STYLE encodes SOC_0, color SHADE
# encodes p0 (full color = p0=1/fast correction, faded = p0=0.0001/slow),
# so all 6 combinations stay distinguishable within one method's color family.
TRAJ_P0_VARIANTS = {
    "p0=1 (fast)":    dict(value=1.0,    alpha=1.0,  lw=1.8),
    "p0=0.0001 (slow)": dict(value=0.0001, alpha=0.55, lw=1.4),
}
soc0_line_styles = {"SOC0=0.0": "-"}

for frac in ABLATION_FRACS:
    fig, axes = plt.subplots(1,len(soc0_methods), figsize=(16, 5), sharex=False)
    # method_colors = {"CAMEL": "#01696f", "CoDA": "#006494", "CoDA-Efficient": "#8e44ad",
    #                  "Baseline": "#964219", "MAML": "#d62728"}
    method_colors = {"CAMEL": "#01696f","Baseline": "#964219"}

    for row_idx, method in enumerate(soc0_methods):
        ax = axes[row_idx]
        adapted = _adapt_and_raw_soc0(method, 0, [traj_task], frac)[0]
        t_axis = traj_task["time"][:len(adapted["Yraw"])]
        ax.plot(t_axis, adapted["Ytrue"], "k-", lw=2.2, label="True SOC", zorder=10)

        for p0_label, p0_cfg in TRAJ_P0_VARIANTS.items():
            for soc0_label, x0 in TRAJ_SOC0_VARIANTS.items():
                Ykf = kalman_filter_coulomb(adapted["Yraw"], adapted["I_arr"], adapted["dt_arr"],
                                            c_nom=adapted["c_nom"], x0=x0, p0=p0_cfg["value"])
                rmse = np.sqrt(mean_squared_error(adapted["Ytrue"], Ykf))
                ax.plot(t_axis, Ykf, soc0_line_styles[soc0_label], color=method_colors[method],
                       lw=p0_cfg["lw"], alpha=p0_cfg["alpha"],
                       label=f"{soc0_label}, {p0_label}, RMSE={rmse:.4f}")

        ax.set_title(f"{method} — recovery from SOC_0=0.0, "
                    f"p0 in {{1, 0.0001}}, calib={frac*100:.0f}%", fontsize=16)
        ax.set_ylabel("SOC", fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=12, loc="upper right", ncol=1)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Time (s)", fontsize=14)

    plt.suptitle(f"SOC_0 Recovery, both p0 settings — {traj_task.get('run_name','')}, "
                f"calib={frac*100:.0f}%", fontsize=18, y=1.01)
    plt.tight_layout()
    fname = f"{PLOT_DIR}/soc0_recovery_trajectories_calib{int(frac*100)}pct.pdf"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {fname}")

print("\n" + "=" * 78)
print("SOC_0 ROBUSTNESS ABLATION COMPLETE.")
print(f"Outputs written to: {PLOT_DIR}/")
print("=" * 78)