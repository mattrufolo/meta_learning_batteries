# plot_sim_to_real_2.py
# Evaluation-only script (no training): loads the meta-trained checkpoints and
#   (1) evaluates them on the two REAL batteries (A123 and INR21700): trajectory
#       plots, sim-to-real sweep, per-method figure with one line per battery;
#   (2) evaluates them on the held-out SIMULATED test batteries, with the very
#       same sweep (methods, seeds, calibration fractions, true-capacity KF),
#       and produces data_efficiency_final.png.
# Everything before the "EVALUATION" section is the original code with the
# targeted changes marked in the comments (split, capacities, windows).

import sys, os
from datetime import datetime

LOG_PATH = "./full_notebook_output_log.txt"
import matplotlib.pyplot as plt

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


class _Tee:
    """Duplicates every write to both the real stdout and a log file."""
    def __init__(self, filepath, stream):
        self.file = open(filepath, "a", buffering=1)   # line-buffered
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
    def flush(self):
        self.stream.flush()
        self.file.flush()
    def isatty(self):
        return False

# Guard against re-running this cell wrapping stdout multiple times.
if not isinstance(sys.stdout, _Tee):
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    with open(LOG_PATH, "w") as _f:
        _f.write(f"{'='*70}\n")
        _f.write(f"Notebook run started: {datetime.now().isoformat()}\n")
        _f.write(f"{'='*70}\n\n")
    sys.stdout = _Tee(LOG_PATH, sys.stdout)
    sys.stderr = _Tee(LOG_PATH, sys.stderr)   # also capture warnings/tracebacks
    print(f"[Logging active] All print() output from here on is also being "
          f"saved to: {LOG_PATH}")
else:
    print(f"[Logging already active] Still writing to: {LOG_PATH}")

# ── Suppress the torch.load "weights_only" FutureWarning ────────────────────
# Every torch.load(...) call in this notebook loads checkpoints WE generated
# ourselves earlier in this same pipeline -- there is no untrusted-source
# risk here (the warning exists to flag loading pickle data from elsewhere).
# With dozens of torch.load calls inside loops (the comprehensive sweep alone
# calls it hundreds of times at runtime), leaving this on floods both the
# console and the log file with an identical, non-actionable warning.
import warnings
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
print("[Warnings] Suppressed torch.load's weights_only FutureWarning "
      "(safe -- all checkpoints are self-generated).")


import time
import os, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd


SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ─── SHARED HYPERPARAMETERS — identical across CAMEL, CoDA, and Baseline ────
WINDOW_SIZE   = 60       # timesteps per input window
FEATURE_DIM   = 5        # [v, i, Ts, delta_v, delta_i]
HIDDEN_DIM    = 64       # MLP hidden width (all models)
NUM_LAYERS    = 3        # MLP depth (all models)
R_DIM         = 10       # basis dim (CAMEL omega) = embedding dim (CoDA)
LR            = 2e-3
WEIGHT_DECAY  = 1e-3
EPOCHS        = 2000
# EPOCHS        = 2
RIDGE_LAMBDA  = 1e-3     # CAMEL ridge regularizer
CALIB_FRAC    = 0.10     # calibration fraction (CAMEL/CoDA adaptation)
L1_WEIGHT     = 1e-4     # CoDA lifting sparsity
BATCH_SIZE    = 1024


# KF — shared (same Q/R for all models, for the primary fair comparison)
KF_Q = 1e-5; KF_R = 1e-2


# KF — ad-hoc per-model values (Issue #6: optional separate tuning per model)
# These are used in the *ad-hoc* variant reported alongside the shared-KF results.
# Rationale: meta-learned models produce smoother raw predictions (lower intrinsic noise)
# so they can afford a lower R (trust the NN more); the baseline MLP trained from scratch
# on CALIB_FRAC data is noisier, so a higher R makes the KF smooth more aggressively.
KF_Q_CAMEL = 1e-5; KF_R_CAMEL = 5e-3   # trust CAMEL more (lower R)
KF_Q_CODA  = 1e-5; KF_R_CODA  = 5e-3   # trust CoDA more  (lower R)
KF_Q_BL    = 1e-5; KF_R_BL    = 5e-2   # smooth baseline more (higher R)


# Number of random calibration splits to average over (Issue #5)
EARLY_STOP_VAL_SEEDS = 1   # ONLY used for early-stopping checkpoint
                           # selection during meta-training (val_metric
                           # methods). Has NO effect on any final reported
                           # result -- those are all train-seed-paired
                           # (N_TRAIN_SEEDS) now. Renamed from N_CALIB_SEEDS
                           # to make this distinction unambiguous.
N_TRAIN_SEEDS = 5   # final robustness setting (raised from 3): each
                    # meta-learned method is retrained 5x for the combined
                    # training+calibration seed robustness measurement.


# KF — He et al. (2014) random-walk variant (for direct paper comparison)
# State: SOC_k = SOC_{k-1} + noise  (no physics, no current/capacity)
KF_Q_RW = 1e-5; KF_R_RW = 1e-2   # same Q/R as shared KF; state model differs

feature_cols = ["v", "i", "Ts", "delta_v", "delta_i"]
target_col   = "SOC"


CHECKPOINT_DIR = "./checkpoints"; PLOT_DIR = "./plots"
os.makedirs(CHECKPOINT_DIR, exist_ok=True); os.makedirs(PLOT_DIR, exist_ok=True)
print("Config loaded.")
def get_seed_dir(seed):
    path = os.path.join(CHECKPOINT_DIR, f"seed_{seed}")
    os.makedirs(path, exist_ok=True)
    return path



# ── USER: replace with your actual data loader ──────────────────────────────
# Example skeleton:
from pathlib import Path
import pandas as pd

DATA_DIR = Path("./dataset_battery")

# ── USER: map your CSV column names to the notebook's expected names ─────────
COL_MAP = {
    "Voltage":          "v",      # adjust key to match your actual CSV header
    "Current":          "i",
    "Temperature_Cell": "Ts",     # or "Ambient_Temp", "T_cell", etc.
    "SOC":              "SOC",    # likely already "SOC"
    # "Time":           "timestamp"  # uncomment if you have an explicit time column
}

battery_folders = sorted([f for f in DATA_DIR.iterdir() if f.is_dir()])
# MAX_BATTERIES = 2   # ← change this to load more later

dataset_runs = {}
all_data_list = []

for battery_folder in battery_folders:   # ← only first 2
    batt_name = battery_folder.name
    dataset_runs[batt_name] = {}

    for csv_file in sorted(battery_folder.glob("*.csv")):
        file_name = csv_file.stem
        df = pd.read_csv(csv_file)
        # df = df.rename(columns=COL_MAP)

        if "timestamp" not in df.columns:
            df["timestamp"] = np.arange(len(df), dtype=np.float32)

        if " - " in file_name:
            parts = file_name.split(" - ", maxsplit=1)
            profile  = parts[0].strip()
            temp_str = parts[1].strip() if len(parts) > 1 else "Unknown"
        else:
            profile, temp_str = file_name, "Unknown"

        if not temp_str:
            temp_str = "Unknown"

        df["Battery"]     = batt_name
        df["Profile"]     = profile
        df["Temperature"] = temp_str

        dataset_runs[batt_name][file_name] = df
        all_data_list.append(df)

df_all = pd.concat(all_data_list, ignore_index=True)
print(f"Loaded {len(all_data_list)} files across {len(dataset_runs)} batteries.")
print(f"Batteries loaded: {list(dataset_runs.keys())}")

for col in ["v", "i", "Ts", "SOC", "timestamp"]:
    assert col in df_all.columns, f"Missing column: {col}. Check COL_MAP."
print("Column check passed:", ["v", "i", "Ts", "SOC", "timestamp"])
# Add delta features
for b in dataset_runs:
    for run_name in dataset_runs[b]:
        df = dataset_runs[b][run_name]
        df["delta_v"] = df["v"].diff().fillna(0)
        df["delta_i"] = df["i"].diff().fillna(0)

# Flatten all 640 tasks
all_tasks = [
    {"battery": b, "run_name": rn, "df": df}
    for b, runs in dataset_runs.items()
    for rn, df in runs.items()
]
print(f"Total tasks: {len(all_tasks)}")
# assert len(all_tasks) == 639, "Expected 639 tasks"
# ── Battery-level split (50 / 15 / 35) + dedicated sim-to-real holdout ──────
# Each "battery" contributes multiple tasks (runs across temperatures/profiles).
# The split is over BATTERIES so that all runs from one battery stay together.
# This prevents the model from seeing any condition of a test battery during training.
#
# SIM-TO-REAL HOLDOUT: "A123 - simtoreal" is real (not simulated) experimental
# data. It is carved out BEFORE the shuffle/split so it can never land in
# train, val, or test — it is a fourth, fully independent set used only for
# the dedicated sim-to-real evaluation later in the notebook.

# Both real cells are carved out BEFORE the shuffle, so the train/val/test pool
# is exactly the simulated fleet and no real data can land in train, val or test.
# This should reproduce the split the checkpoints were trained with (pool = the
# simulated cells only); the pre-flight check below verifies the number of
# training tasks against the checkpoints.
REAL_BATTERIES = {                          # folder name -> short label (plots, files)
    "A123 - simtoreal":         "A123",
    "INR21700-50E - simtoreal": "INR21700",
}
SIM2REAL_BATTERY_NAMES = list(REAL_BATTERIES.keys())

all_batteries_full = list(dataset_runs.keys())
sim2real_batteries  = [b for b in all_batteries_full if b in SIM2REAL_BATTERY_NAMES]
missing_s2r = [b for b in SIM2REAL_BATTERY_NAMES if b not in all_batteries_full]
if missing_s2r:
    raise FileNotFoundError(
        f"Real battery folder(s) not found in {DATA_DIR}: {missing_s2r}. "
        f"Both real cells are needed for this evaluation.")

all_batteries = [b for b in all_batteries_full if b not in sim2real_batteries]
random.shuffle(all_batteries)   # SEED=42 set globally above

n_batt       = len(all_batteries)
n_train_batt = int(0.50 * n_batt)
n_val_batt   = int(0.15 * n_batt)
n_test_batt  = n_batt - n_train_batt - n_val_batt

train_batteries = all_batteries[:n_train_batt]
val_batteries   = all_batteries[n_train_batt : n_train_batt + n_val_batt]
test_batteries  = all_batteries[n_train_batt + n_val_batt:]

# Expand each battery list into its constituent tasks
train_tasks    = [t for t in all_tasks if t["battery"] in train_batteries]
val_tasks      = [t for t in all_tasks if t["battery"] in val_batteries]
test_tasks     = [t for t in all_tasks if t["battery"] in test_batteries]
sim2real_tasks = [t for t in all_tasks if t["battery"] in sim2real_batteries]

print(f"Batteries — Train: {len(train_batteries)}  Val: {len(val_batteries)}  "
      f"Test: {len(test_batteries)}  Sim2Real: {len(sim2real_batteries)}  "
      f"(total: {len(all_batteries_full)})")
print(f"Tasks     — Train: {len(train_tasks)}  Val: {len(val_tasks)}  "
      f"Test: {len(test_tasks)}  Sim2Real: {len(sim2real_tasks)}")
print(f"Train    batteries: {train_batteries}")
print(f"Val      batteries: {val_batteries}")
print(f"Test     batteries: {test_batteries}")
print(f"Sim2Real batteries: {sim2real_batteries}")
global_scaler = StandardScaler()
global_scaler.fit(np.vstack([t["df"][feature_cols].values.astype(np.float32) for t in train_tasks]))

def make_windows_numpy(df, scaler, window_size=WINDOW_SIZE):
    """Returns contiguous numpy arrays on CPU. Move to device only at batch time."""
    X_raw = scaler.transform(df[feature_cols].values.astype(np.float32))  # (T, d)
    Y_raw = df[target_col].values.astype(np.float32)                       # (T,)
    T, d  = X_raw.shape
    n     = T - window_size
    idx   = np.arange(n)[:, None] + np.arange(window_size)[None, :]       # (n, W)
    X_out = X_raw[idx]                                                     # (n, W, d)
    Y_out = Y_raw[window_size:window_size + n].reshape(n, 1)               # (n, 1)
    t_out = df["timestamp"].values[window_size:window_size + n]
    assert len(X_out) == len(Y_out) == len(t_out), \
        f"Window mismatch: X={len(X_out)}, Y={len(Y_out)}, t={len(t_out)}"
    return np.ascontiguousarray(X_out), np.ascontiguousarray(Y_out), t_out

def build_task_tensors(task_list):
    """Store windows as numpy on CPU — no GPU memory used here."""
    out = []
    for t in task_list:
        X, Y, ts = make_windows_numpy(t["df"], global_scaler)
        out.append({**t, "X": X, "Y": Y, "time": ts})
    return out

def build_loader(prepared_tasks, batch_size=BATCH_SIZE, shuffle=True):
    """
    Concatenate numpy arrays ONCE into CPU tensors, then let PyTorch C++ handle
    all shuffling. Far faster than a Python-level IterableDataset.
    One-time cost: ~30s and ~9 GB RAM for the train split.
    """
    print(f"  Building loader for {len(prepared_tasks)} tasks — concatenating...", end=" ", flush=True)
    all_X    = torch.from_numpy(np.concatenate([t["X"] for t in prepared_tasks], axis=0))
    all_Y    = torch.from_numpy(np.concatenate([t["Y"] for t in prepared_tasks], axis=0))
    all_tidx = torch.cat([
        torch.full((len(t["X"]),), i, dtype=torch.long)
        for i, t in enumerate(prepared_tasks)
    ])
    print(f"done. {all_X.shape[0]:,} windows, X={all_X.element_size()*all_X.nelement()/1e9:.1f} GB")
    ds = TensorDataset(all_X, all_Y, all_tidx)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=(device.type == "cuda"),
        num_workers=2,
        persistent_workers=True,
    )

print("Preprocessing helpers ready (fast concat-once DataLoader).")

def build_task_homogeneous_loader(tasks_prepared, batch_size=BATCH_SIZE, shuffle=True):
    """
    Task-homogeneous DataLoader for CoDA.

    Instead of mixing windows from all tasks in one flat shuffle (which forces
    CoDA's forward() to call functional_call once per unique task per batch),
    this loader yields batches where ALL windows come from the same task.
    This enables the fast path in CoDaModel.forward() — one functional_call
    per batch instead of up to ~320 — reducing CoDA training time from hours
    to minutes while producing identical gradients.

    Structure: for each epoch, tasks are shuffled; within each task, windows
    are shuffled and chunked into batches of `batch_size`. Task boundaries
    never cross a batch boundary.
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    all_loaders = []
    task_order = list(range(len(tasks_prepared)))
    if shuffle:
        import random
        random.shuffle(task_order)

    for tidx in task_order:
        t = tasks_prepared[tidx]
        X = torch.from_numpy(t["X"]).float()
        Y = torch.from_numpy(t["Y"]).float()
        # task index tensor — all same value for this task
        T = torch.full((len(X),), tidx, dtype=torch.long)
        ds = TensorDataset(X, Y, T)
        # Within a task, shuffle windows
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=0, pin_memory=True, drop_last=False)
        all_loaders.append(loader)

    # Wrap as a flat iterable that yields (X_b, Y_b, tidx_b) across all tasks
    class ChainedLoader:
        def __init__(self, loaders):
            self.loaders = loaders
        def __iter__(self):
            for loader in self.loaders:
                yield from loader
        def __len__(self):
            return sum(len(l) for l in self.loaders)

    return ChainedLoader(all_loaders)

print("build_task_homogeneous_loader ready.")

class BasisMLP(nn.Module):
    """Shared windowed MLP: [B, W, d] -> [B, r]. Used by CAMEL and CoDA."""
    def __init__(self, window=WINDOW_SIZE, feat=FEATURE_DIM,
                 hidden=HIDDEN_DIM, r=R_DIM, n_layers=NUM_LAYERS,
                 dropout=0.1):                          # ← add dropout param
        super().__init__()
        in_dim = window * feat
        layers = [nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout)]  # ← after first act
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout)]  # ← after each hidden
        layers.append(nn.Linear(hidden, r))  # ← NO dropout here
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

class BaselineMLP(nn.Module):
    """Plain MLP for supervised baseline: [B, W, d] -> [B, 1].
       Same architecture as BasisMLP except output dim=1 instead of r."""
    def __init__(self, window=WINDOW_SIZE, feat=FEATURE_DIM,
                 hidden=HIDDEN_DIM, n_layers=NUM_LAYERS,
                 dropout=0.1):                          # ← same default as BasisMLP
        super().__init__()
        in_dim = window * feat
        layers = [nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, 1))             # ← NO dropout on output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# Verify param counts match (should be identical except last layer diff: r vs 1)
basis_params    = sum(p.numel() for p in BasisMLP().parameters())
baseline_params = sum(p.numel() for p in BaselineMLP().parameters())
print(f"BasisMLP    params: {basis_params:,}   output → r={R_DIM}")
print(f"BaselineMLP params: {baseline_params:,}   output → 1")
print(f"Difference: {abs(basis_params - baseline_params):,}  "
      f"(= last layer: {HIDDEN_DIM}×{R_DIM}+{R_DIM} vs {HIDDEN_DIM}×1+1)")

class CAMEL(nn.Module):
    def __init__(self, n_tasks, r=R_DIM):
        super().__init__()
        self.v_net  = BasisMLP(r=r)
        self.omegas = nn.Parameter(torch.randn(n_tasks, r, 1) * 0.05)


    def forward(self, x, task_idx):
        V = self.v_net(x)                                       # [B, r]
        omega = self.omegas[task_idx]                           # [B, r, 1]
        return torch.bmm(V.unsqueeze(1), omega).squeeze(1)      # [B, 1]


    def adapt(self, x_calib, y_calib, ridge=RIDGE_LAMBDA):
        """Returns new omega via closed-form ridge regression."""
        with torch.no_grad():
            V = self.v_net(x_calib)
            A = V.T @ V + ridge * torch.eye(V.shape[1], device=V.device)
            return torch.linalg.solve(A, V.T @ y_calib)        # [r, 1]


    @torch.no_grad()
    def predict_with_omega(self, x, omega):
        return (self.v_net(x) @ omega).clamp(0, 1)


    def val_metric(self, val_tasks, ridge=RIDGE_LAMBDA, n_seeds=EARLY_STOP_VAL_SEEDS):
        """
        Adaptation-based val metric evaluated on *validation* tasks only.
        Uses n_seeds different random calibration splits and averages RMSE,
        so early-stopping is not fooled by a lucky calib draw.
        The backbone is frozen during adaptation (no gradient flows to v_net).
        """
        self.eval()
        rmses = []
        with torch.no_grad():
            for t in val_tasks:
                Xs = torch.from_numpy(t["X"]).to(device)
                Ys = torch.from_numpy(t["Y"]).to(device)
                N_c = max(R_DIM * 3, int(CALIB_FRAC * len(Xs)))
                task_rmses = []
                for s in range(n_seeds):
                    g = torch.Generator(device=device); g.manual_seed(s)
                    cidx  = torch.randperm(len(Xs), generator=g, device=device)[:N_c]
                    omega = self.adapt(Xs[cidx], Ys[cidx], ridge=ridge)
                    Yp    = self.predict_with_omega(Xs, omega)
                    task_rmses.append(torch.sqrt(torch.mean((Yp - Ys) ** 2)).item())
                rmses.append(float(np.mean(task_rmses)))
                del Xs, Ys
        return float(np.mean(rmses))


print("CAMEL defined.")

adaptation_step_coda = 40   # L-BFGS typically converges by step 10-15;
                              # lower this once you've confirmed via cell 37's loss-curve plot

# ── CoDA: full-weight hypernetwork (Kirchmeyer et al. 2022) ─────────────────
#
# Architecture (faithful to JAX reference implementation):
#
#   θ_adapted = θ_mlp + lift(e)
#   pred      = MLP(x ; θ_adapted)
#
# The MLP always outputs dimension 1 (SOC prediction).
# The lifting network maps e ∈ R^{emb_size} → Δθ  with the SAME shape as
# every parameter in the MLP backbone.  This is exactly:
#   p_upd = lift.apply(p_lift, e)
#   p_new = tree_map(lambda p, u: p + u, p_mlp, p_upd)
#   preds = mlp.apply(p_new, x)
#
# TRAINING BUG FIX (v4 regression):
#   The previous forward() used  e_unique = e[0:1]  (first sample's embedding).
#   The DataLoader mixes windows from many tasks in one batch.  When task_idx
#   varies within a batch, e[0:1] applies the WRONG task's Δθ to most samples,
#   producing a contradictory gradient signal that collapses training (RMSE=0.05).
#
#   Fix: forward() now groups samples by task_idx and applies the correct Δθ
#   to each group.  This adds one loop over unique tasks per batch but is
#   correct and still fully vectorised within each task group.
# ────────────────────────────────────────────────────────────────────────────

class CoDaModel(nn.Module):
    def __init__(self, n_tasks, r=R_DIM, emb_size=None):
        super().__init__()
        emb_size = emb_size or r
        self.emb_size   = emb_size
        self.backbone   = BaselineMLP(dropout=0.1)   # shared θ_mlp; outputs 1
        self.embeddings = nn.Parameter(torch.randn(n_tasks, emb_size) * 0.01)

        # Lifting: e → flat Δθ of same total size as backbone parameters
        self._backbone_param_shapes = [
            (name, p.shape, p.numel())
            for name, p in self.backbone.named_parameters()
        ]
        total_backbone_params = sum(s for _, _, s in self._backbone_param_shapes)
        self.lifting = nn.Linear(emb_size, total_backbone_params, bias=False)
        nn.init.normal_(self.lifting.weight, std=1e-3)   # start near identity

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _apply_delta(self, delta_flat):
        """Flat Δθ vector → {param_name: delta_tensor} matching backbone."""
        result = {}; offset = 0
        for name, shape, numel in self._backbone_param_shapes:
            result[name] = delta_flat[offset:offset + numel].view(shape)
            offset += numel
        return result

    def _adapted_forward(self, x_flat, e_1d):
        """
        Run backbone with θ_mlp + Δθ(e) on pre-flattened x_flat.
        e_1d: (emb_size,) — single task embedding.
        """
        delta_flat = self.lifting(e_1d)          # (total_params,)
        delta_dict = self._apply_delta(delta_flat)
        adapted_params = {
            name: p + delta_dict[name]
            for name, p in self.backbone.named_parameters()
        }
        try:
            from torch.func import functional_call
            return functional_call(self.backbone, adapted_params, (x_flat,))
        except ImportError:
            # Fallback: temporarily swap weights
            orig = {n: p.data.clone() for n, p in self.backbone.named_parameters()}
            for n, p in self.backbone.named_parameters():
                p.data = adapted_params[n]
            out = self.backbone.net(x_flat)
            for n, p in self.backbone.named_parameters():
                p.data = orig[n]
            return out

    # ── Meta-training forward ─────────────────────────────────────────────────
    def forward(self, x, task_idx):
        """
        Mixed-batch forward: each unique task in task_idx gets its own Δθ.
        Output shape: (B, 1).

        Fast path: if the entire batch belongs to one task (task-homogeneous
        DataLoader, which we use for CoDA to avoid the per-task functional_call
        overhead), we skip the loop and do a single adapted forward pass.
        Fallback: loop over unique tasks — correct but slower for mixed batches.
        """
        x_flat = x.view(x.size(0), -1) if x.dim() == 3 else x
        unique_tids = task_idx.unique()

        if len(unique_tids) == 1:
            # ── Fast path: entire batch is one task ──────────────────────────
            return self._adapted_forward(x_flat, self.embeddings[unique_tids[0]])

        # ── General path: mixed-task batch ───────────────────────────────────
        out = torch.zeros(x_flat.size(0), 1, device=x.device, dtype=x_flat.dtype)
        for tid in unique_tids:
            mask = (task_idx == tid)
            out[mask] = self._adapted_forward(x_flat[mask], self.embeddings[tid])
        return out

    # ── Test-time adaptation ──────────────────────────────────────────────────
    def adapt(self, x_calib, y_calib, n_steps=adaptation_step_coda, lr=1.0,
              optimizer="lbfgs"):
        """
        Optimise a fresh embedding e by gradient descent on x_calib/y_calib.
        backbone + lifting weights are frozen; only e is trainable.

        optimizer="lbfgs" (default): quasi-Newton L-BFGS with a Wolfe line
        search (`strong_wolfe`). e is only `emb_size`-dimensional — exactly
        the small, smooth regime L-BFGS is built for — so it uses curvature
        (history_size=10) instead of a fixed step. The line search re-derives
        a safe step every iteration, so a larger nominal `lr` still converges
        stably; in practice this needs ~10-15 steps instead of ~150-200 for
        Adam (see cell 37's loss-curve plot to confirm on your data).
        optimizer="adam": legacy first-order fallback, kept for comparison /
        debugging.

        Returns (e_detached, loss_history). loss_history always has length
        n_steps so the downstream plotting in cell 37 (np.array(...) over a
        [n_tasks, EARLY_STOP_VAL_SEEDS, n_steps] grid) keeps working unchanged.
        """
        e = nn.Parameter(torch.zeros(self.emb_size, device=x_calib.device))
        x_flat = x_calib.view(x_calib.size(0), -1) if x_calib.dim() == 3 else x_calib
        loss_history = []

        def _loss():
            pred = self._adapted_forward(x_flat, e)
            l1   = self.lifting(e).abs().mean()   # sparsity on Δθ
            return F.mse_loss(pred, y_calib) + L1_WEIGHT * l1

        if optimizer == "lbfgs":
            opt = optim.LBFGS([e], lr=lr, max_iter=1, history_size=10,
                               line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = _loss()
                loss.backward()
                return loss

            for _ in range(n_steps):
                loss = opt.step(closure)
                loss_history.append(loss.item())
        else:
            opt = optim.Adam([e], lr=lr)
            for _ in range(n_steps):
                opt.zero_grad()
                loss = _loss()
                loss.backward()
                opt.step()
                loss_history.append(loss.item())

        return e.detach(), loss_history

    @torch.no_grad()
    def predict_with_emb(self, x, e):
        """
        e can be (emb_size,) or (1, emb_size) — always normalised to 1-D.
        Returns clamped prediction (B, 1).
        """
        e_1d   = e.view(-1)
        x_flat = x.view(x.size(0), -1) if x.dim() == 3 else x
        return self._adapted_forward(x_flat, e_1d).clamp(0, 1)

    # ── Early-stopping validation metric ─────────────────────────────────────
    def val_metric(self, val_tasks, n_steps=adaptation_step_coda,
                   lr_emb=1.0, n_seeds=EARLY_STOP_VAL_SEEDS):
        self.eval()
        rmses = []
        for t in val_tasks:
            Xs  = torch.from_numpy(t["X"]).float().to(device)
            Ys  = torch.from_numpy(t["Y"]).float().to(device)
            N_c = max(16, int(CALIB_FRAC * len(Xs)))
            task_rmses = []
            for s in range(n_seeds):
                g = torch.Generator(device=device); g.manual_seed(s)
                cidx = torch.randperm(len(Xs), generator=g, device=device)[:N_c]
                e, _ = self.adapt(Xs[cidx], Ys[cidx], n_steps=n_steps, lr=lr_emb)
                Yp   = self.predict_with_emb(Xs, e)
                task_rmses.append(torch.sqrt(torch.mean((Yp - Ys) ** 2)).item())
            rmses.append(float(np.mean(task_rmses)))
            del Xs, Ys
        return float(np.mean(rmses))

print("CoDA (full-weight hypernetwork, corrected mixed-batch forward) defined.")
print(f"  Backbone params : {sum(p.numel() for p in BaselineMLP().parameters()):,}")
n_lift = sum(p.numel() for p in nn.Linear(R_DIM,
    sum(p.numel() for p in BaselineMLP().parameters()), bias=False).parameters())
print(f"  Lifting params  : {n_lift:,}  (emb_size={R_DIM} → total_backbone_params)")
class CoDaModelEfficient(nn.Module):
    """
    CoDA with a LoRA-style per-layer low-rank lift. emb_size is UNCHANGED
    (still R_DIM=10 by default) -- the full task embedding is still what
    gets optimised at test time, with the same adaptation richness as the
    original full-rank CoDaModel.

    For each backbone weight matrix W_l in R^{out x in}:
        delta_W_l = U_l @ diag(e) @ V_l^T,   U_l in R^{out x r}, V_l in R^{in x r}
    where r = emb_size and e in R^r is the SAME embedding shared across every
    layer. Parameter count for the weight lift: r * sum_l(out_l + in_l),
    instead of r * P for the flat version -- scales with LAYER WIDTHS summed,
    not with the PRODUCT of widths (matrix size).

    Biases use a small flat Linear(emb_size, total_bias_params) lift, since
    bias vectors are already tiny (no low-rank structure needed there).

    Same external interface as CoDaModel (forward/adapt/predict_with_emb/
    val_metric) -- drop-in compatible with every _sweep_eval_coda call site.
    """
    def __init__(self, n_tasks, r=R_DIM, emb_size=None):
        super().__init__()
        emb_size = emb_size or r
        self.emb_size   = emb_size
        self.backbone   = BaselineMLP(dropout=0.1)
        self.embeddings = nn.Parameter(torch.randn(n_tasks, emb_size) * 0.01)

        self._weight_specs = []   # (name, out_dim, in_dim)
        self._bias_specs   = []   # (name, numel)
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
        """e_1d: (emb_size,) -> {param_name: delta_tensor} matching backbone."""
        result = {}
        for name, out_dim, in_dim in self._weight_specs:
            key = name.replace(".", "_")
            U_l, V_l = self.U[key], self.V[key]        # (out,r), (in,r)
            result[name] = (U_l * e_1d) @ V_l.T          # (out,in)
        if self.bias_lift is not None:
            bias_flat = self.bias_lift(e_1d)
            offset = 0
            for name, numel in self._bias_specs:
                result[name] = bias_flat[offset:offset + numel]
                offset += numel
        return result

    def _adapted_forward(self, x_flat, e_1d):
        delta_dict = self._apply_delta(e_1d)
        adapted_params = {name: p + delta_dict[name]
                          for name, p in self.backbone.named_parameters()}
        from torch.func import functional_call
        return functional_call(self.backbone, adapted_params, (x_flat,))

    def forward(self, x, task_idx):
        x_flat = x.view(x.size(0), -1) if x.dim() == 3 else x
        unique_tids = task_idx.unique()
        if len(unique_tids) == 1:
            return self._adapted_forward(x_flat, self.embeddings[unique_tids[0]])
        out = torch.zeros(x_flat.size(0), 1, device=x.device, dtype=x_flat.dtype)
        for tid in unique_tids:
            mask = (task_idx == tid)
            out[mask] = self._adapted_forward(x_flat[mask], self.embeddings[tid])
        return out

    def adapt(self, x_calib, y_calib, n_steps=adaptation_step_coda, lr=1.0,
              optimizer="lbfgs"):
        e = nn.Parameter(torch.zeros(self.emb_size, device=x_calib.device))
        x_flat = x_calib.view(x_calib.size(0), -1) if x_calib.dim() == 3 else x_calib
        loss_history = []

        def _loss():
            pred = self._adapted_forward(x_flat, e)
            delta_dict = self._apply_delta(e)
            l1 = torch.cat([d.flatten() for d in delta_dict.values()]).abs().mean()
            return F.mse_loss(pred, y_calib) + L1_WEIGHT * l1

        if optimizer == "lbfgs":
            opt = optim.LBFGS([e], lr=lr, max_iter=1, history_size=10,
                               line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad()
                loss = _loss()
                loss.backward()
                return loss
            for _ in range(n_steps):
                loss = opt.step(closure)
                loss_history.append(loss.item())
        else:
            opt = optim.Adam([e], lr=lr)
            for _ in range(n_steps):
                opt.zero_grad()
                loss = _loss()
                loss.backward()
                opt.step()
                loss_history.append(loss.item())

        return e.detach(), loss_history

    @torch.no_grad()
    def predict_with_emb(self, x, e):
        e_1d   = e.view(-1)
        x_flat = x.view(x.size(0), -1) if x.dim() == 3 else x
        return self._adapted_forward(x_flat, e_1d).clamp(0, 1)

    def val_metric(self, val_tasks, n_steps=adaptation_step_coda,
                   lr_emb=1.0, n_seeds=EARLY_STOP_VAL_SEEDS):
        self.eval()
        rmses = []
        for t in val_tasks:
            Xs  = torch.from_numpy(t["X"]).float().to(device)
            Ys  = torch.from_numpy(t["Y"]).float().to(device)
            N_c = max(16, int(CALIB_FRAC * len(Xs)))
            task_rmses = []
            for s in range(n_seeds):
                g = torch.Generator(device=device); g.manual_seed(s)
                cidx = torch.randperm(len(Xs), generator=g, device=device)[:N_c]
                e, _ = self.adapt(Xs[cidx], Ys[cidx], n_steps=n_steps, lr=lr_emb)
                Yp   = self.predict_with_emb(Xs, e)
                task_rmses.append(torch.sqrt(torch.mean((Yp - Ys) ** 2)).item())
            rmses.append(float(np.mean(task_rmses)))
            del Xs, Ys
        return float(np.mean(rmses))

print("CoDA (LoRA-style efficient lift) defined.")

# Parameter count comparison: flat lift vs LoRA-style lift
_backbone_params = sum(p.numel() for p in BaselineMLP().parameters())
_flat_lift_params = _backbone_params * R_DIM
_lora_model_tmp = CoDaModelEfficient(n_tasks=1, r=R_DIM, emb_size=R_DIM)
_lora_lift_params = (sum(p.numel() for p in _lora_model_tmp.U.values())
                     + sum(p.numel() for p in _lora_model_tmp.V.values())
                     + (sum(p.numel() for p in _lora_model_tmp.bias_lift.parameters())
                        if _lora_model_tmp.bias_lift is not None else 0))
del _lora_model_tmp
print(f"  Backbone params:            {_backbone_params:>10,}")
print(f"  Flat lift (original CoDA):  {_flat_lift_params:>10,}  "
      f"-> total ~{_backbone_params+_flat_lift_params:,} "
      f"({(_backbone_params+_flat_lift_params)/_backbone_params:.1f}x backbone)")
print(f"  LoRA-style lift (efficient):{_lora_lift_params:>10,}  "
      f"-> total ~{_backbone_params+_lora_lift_params:,} "
      f"({(_backbone_params+_lora_lift_params)/_backbone_params:.1f}x backbone)  "
      f"[emb_size still {R_DIM}]")
# ── Cell 8  Kalman Filter – Coulomb-counting state model ────────────────────
# Q and R are fixed (shared or ad-hoc per model).
# C_nom is estimated per-task from the CALIB_FRAC window using Coulomb counting.

KF_Q       = 1e-5;  KF_R       = 1e-2
KF_Q_CAMEL = 1e-5;  KF_R_CAMEL = 5e-3
KF_Q_CODA  = 1e-5;  KF_R_CODA  = 5e-3
KF_Q_BL    = 1e-5;  KF_R_BL    = 5e-2




# ── True nominal capacity, from cell datasheets (mAh) ────────────────────────
# Source: collaboration shared spreadsheet (Nominal discharge capacity [Ah]
# column, converted here from mAh as given). Keyed by battery folder name
# (matching t["battery"]).
TRUE_CAPACITY_MAH = {
    "ANR26650M1":    2300,
    "PD3032":         180,
    "INR_21700_P45B":4500,
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
    # Sim-to-real battery: folder is "A123 - simtoreal", confirmed to
    # correspond to part ANR26650M1 (2300 mAh).
    "A123 - simtoreal": 1100,
    # Real INR cell (folder "INR21700-50E - simtoreal"). 4500 mAh is the value the
    # INR trajectory plots have been using so far (the 'INR_21700_P45B' row of the
    # shared spreadsheet). VERIFY it is the right number for the cell used in the
    # experiments: a Samsung INR21700-50E is normally rated around 5000 mAh.
    "INR21700-50E - simtoreal": 4900,
}

def get_true_c_nom(battery_name):
    """
    True nominal capacity [Coulombs], converting mAh -> Ah -> Ampere-seconds
    (C_nom = mAh/1000 * 3600), for consistency with compute_c_nom's units.
    Returns None if the battery is not in the lookup table.
    """
    mah = TRUE_CAPACITY_MAH.get(battery_name)
    if mah is None:
        return None
    return (mah / 1000.0) * 3600.0   # Coulombs


def get_c_nom_for_eval(battery_name, df_task=None, calib_idx=None):
    """
    C_nom used throughout evaluation: TRUE capacity only, no estimation. Raises
    if the battery has no entry in TRUE_CAPACITY_MAH instead of silently
    returning something else. df_task/calib_idx are accepted and ignored so the
    existing call sites keep working.
    """
    c_true = get_true_c_nom(battery_name)
    if c_true is None:
        raise ValueError(
            f"No true capacity for battery '{battery_name}' in TRUE_CAPACITY_MAH. "
            f"Add its capacity there; estimation is disabled.")
    return c_true


def kalman_filter_coulomb(z, current, dt, q=KF_Q, r=KF_R, c_nom=3600.0):
    """
    1D Kalman filter with Coulomb-counting state dynamics.
    State  : SOC_k = SOC_{k-1} - (I_k * Δt_k) / C_nom
    Measure: z_k  = SOC_k + noise  (NN output)

    Usage – shared KF (same Q, R for all models):
      kalman_filter_coulomb(Yraw, I_arr, dt_arr, c_nom=c_nom_task)
    Usage – ad-hoc KF (per-model Q, R):
      kalman_filter_coulomb(Yraw, I_arr, dt_arr, q=KF_Q_CAMEL, r=KF_R_CAMEL, c_nom=c_nom_task)
    """
    n = len(z)

    # Belt-and-suspenders: sanitize inputs before the recursion. The
    # filter is recursive (x[k] depends on x[k-1]), so a single NaN
    # anywhere in z/current would otherwise propagate through every
    # subsequent timestep once it enters the state.
    z       = np.nan_to_num(z, nan=0.0, posinf=1.0, neginf=0.0)
    current = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(c_nom) or c_nom <= 0:
        c_nom = 3600.0

    x      = np.zeros(n);  p = np.zeros(n)
    x[0]   = z[0];         p[0] = 1.0
    dt_arr = np.full(n, dt) if np.isscalar(dt) else np.asarray(dt, dtype=np.float64)
    dt_arr = np.nan_to_num(dt_arr, nan=0.0, posinf=0.0, neginf=0.0)

    for k in range(1, n):
        x_pred = np.clip(x[k-1] - (current[k] * dt_arr[k]) / c_nom, 0.0, 1.0)
        p_pred = p[k-1] + q
        K      = p_pred / (p_pred + r)
        x[k]   = np.clip(x_pred + K * (z[k] - x_pred), 0.0, 1.0)
        p[k]   = (1 - K) * p_pred
    return x


print(f"KF ready  shared Q={KF_Q} R={KF_R}")
print(f"Ad-hoc:  CAMEL Q={KF_Q_CAMEL} R={KF_R_CAMEL} | CoDA Q={KF_Q_CODA} R={KF_R_CODA} | BL Q={KF_Q_BL} R={KF_R_BL}")

def kalman_filter_random_walk(z, q=KF_Q_RW, r=KF_R_RW):
    """
    He et al. (2014) random-walk Kalman filter.
    State: SOC_k = SOC_{k-1} + noise   (no physics, no current/capacity info)
    Measurement: z_k = SOC_k + noise   (NN output = noisy measurement)
    This matches the KF described in He et al. (2014) exactly.
    Use this for direct comparison against that paper's reported numbers.
    """
    n    = len(z)
    x    = np.zeros(n); p = np.zeros(n)
    x[0] = z[0];        p[0] = 1.0
    for k in range(1, n):
        p_pred = p[k-1] + q
        K      = p_pred / (p_pred + r)
        x[k]   = x[k-1] + K * (z[k] - x[k-1])
        p[k]   = (1 - K) * p_pred
    return np.clip(x, 0.0, 1.0)

print("kalman_filter_random_walk ready  (He et al. 2014 state model)")
VAL_EVERY = 5

def training_loop(model, train_loader, val_prepared, epochs=EPOCHS, desc="Model"):
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, min_lr=1e-5)
    loss_fn = nn.MSELoss()
    train_losses, val_metrics = [], []
    best_val = float("inf"); best_state = None

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0; n = 0
        for X_b, Y_b, tidx_b in train_loader:
            X_b  = X_b.to(device, non_blocking=True)
            Y_b  = Y_b.to(device, non_blocking=True)
            tidx_b = tidx_b.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(X_b, tidx_b)
            # L2 regularisation on task-specific params — safe for both CAMEL and CoDA
            if hasattr(model, "omegas"):
                task_reg = RIDGE_LAMBDA * model.omegas.pow(2).mean()
            elif hasattr(model, "embeddings"):
                task_reg = RIDGE_LAMBDA * model.embeddings.pow(2).mean()
            else:
                task_reg = 0.0
            loss = loss_fn(pred, Y_b) + task_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(X_b); n += len(X_b)
        train_losses.append(ep_loss / n)

        if (ep + 1) % VAL_EVERY == 0 or ep == epochs - 1:
            val_rmse = model.val_metric(val_prepared)
            val_metrics.append((ep + 1, val_rmse))
            scheduler.step(val_rmse)
            if val_rmse < best_val:
                best_val = val_rmse
                best_state = copy.deepcopy(model.state_dict())
            if (ep + 1) % 200 == 0 or ep == 0:
                print(f"{desc}  Ep {ep+1:4d}  Train={train_losses[-1]:.5f}  "
                      f"Val RMSE={val_rmse:.5f}  Best={best_val:.5f}")

    return train_losses, val_metrics, best_state

# Only the real batteries are windowed: this script is evaluation-only, so the
# train/val/test windows and the training DataLoaders (several GB of RAM) that
# the training pipeline builds here are not needed. train_tasks is still used,
# to fit the scaler and to size the per-task parameters of the checkpoints.
print("Pre-computing sliding windows for the real batteries (CPU numpy)...")
sim2real_prepared = build_task_tensors(sim2real_tasks)
s2r_by_battery = {b: [t for t in sim2real_prepared if t["battery"] == b] for b in REAL_BATTERIES}
for _b, _tl in s2r_by_battery.items():
    print(f"  {REAL_BATTERIES[_b]:>9s}: {len(_tl)} tasks, {sum(len(t['X']) for t in _tl):,} windows")
assert all(len(v) > 0 for v in s2r_by_battery.values()), (
    "a real battery has no tasks -- check the folder names in REAL_BATTERIES")

# ── Pre-flight: all checkpoints present, and trained on THIS split ────────
_required = ["camel_best.pt", "coda_best.pt", "coda_efficient_best.pt", "maml_windowed_best.pt"]
_missing = [os.path.join(get_seed_dir(s), f) for s in range(N_TRAIN_SEEDS) for f in _required
            if not os.path.exists(os.path.join(get_seed_dir(s), f))]
if _missing:
    raise FileNotFoundError("Missing checkpoint file(s):\n  " + "\n  ".join(_missing))
_n_ck = torch.load(os.path.join(get_seed_dir(0), "camel_best.pt"),
                   map_location="cpu")["state_dict"]["omegas"].shape[0]
if _n_ck != len(train_tasks):
    raise RuntimeError(
        f"The checkpoints were trained on {_n_ck} training tasks but this split gives "
        f"{len(train_tasks)}. The train/val/test split differs from the one used at "
        f"training time (different set of simulated batteries in ./dataset_battery, "
        f"or a different seed). Predictions would not be valid, so stopping here.")
print(f"Pre-flight OK: {len(_required) * N_TRAIN_SEEDS} checkpoints found, "
      f"trained on the same {_n_ck} training tasks as this split.")

def run_inference(X, model_fn, chunk=2048):
    """Run model_fn over X in chunks; returns numpy 1D array."""
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            preds.append(model_fn(X[i:i+chunk]))
    return torch.cat(preds).cpu().numpy().flatten()


_calib_call_counter = [0]   # mutable counter so each call gets a different seed


def get_calib_idx(N, N_c, seed=None):
    """
    Return N_c sorted indices sampled without replacement from [0, N).
    If seed is None (default), a fresh seed is drawn from a global counter
    so that repeated calls across tasks and runs never reuse the same split.
    Pass an explicit integer seed only when reproducibility is required
    (e.g. inside sweep functions that average over multiple seeds themselves).
    Result is always a CPU LongTensor — safe to index both numpy arrays and
    CUDA tensors.
    """
    if seed is None:
        seed = _calib_call_counter[0]
        _calib_call_counter[0] += 1
    g = torch.Generator()   # CPU generator
    g.manual_seed(seed)
    idx, _ = torch.sort(torch.randperm(N, generator=g)[:N_c])
    return idx   # CPU LongTensor


print("Inference helpers ready (get_calib_idx uses rolling seeds by default).")
# ── Cell 22  Jeong & Bae (2022) – Full second-order MAML SOC Meta-Learner ────
#
# Paper: "Estimating battery state-of-charge with a few target training data
#         by meta-learning" (Journal of Power Sources, 2022)
#         Jeong & Bae, Hanyang University
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT THEY DO  vs  WHAT WE DO  vs  OUR EXTENSIONS
# ─────────────────────────────────────────────────────────────────────────────
#
# Jeong & Bae 2022               │  This notebook (baseline impl.)  │ Extensions added here
# ─────────────────────────────── │ ──────────────────────────────── │ ──────────────────────
# Algorithm    MAML (full 2nd-   │  CAMEL (closed-form ridge)       │  MAML kept 2nd-order
#              order, Finn'17)   │  CoDA  (embed. gradient-descent) │  (no FOMAML approx.)
# Architecture DNN (fully-conn.) │  Shared MLP backbone (BasisMLP)  │  Two DNN variants:
#   Inputs:    V, I, T           │  V, I, Ts, ΔV, ΔI (richer)      │  A) Static  – raw
#   Windowing: NONE (1 timestep) │  WINDOW_SIZE = 60 timesteps      │     single-step (as paper)
#                                │                                   │  B) Windowed – 60-step
#                                │                                   │     (same as CAMEL/CoDA)
# Outer optim  SGD               │  AdamW                           │  AdamW (more stable)
# Inner steps  9 gradient steps  │  CAMEL: closed-form              │  same 9 inner steps
#                                │  CoDA:  100 Adam steps           │
# KF           NOT used          │  Coulomb-counting KF             │  Two variants per DNN:
#                                │                                   │   (i)  no KF  (paper-faithful)
#                                │                                   │   (ii) Coulomb-KF (our extension)
# Data         real batteries    │  simulated (20-battery dataset)  │  same simulated dataset
#
# ─────────────────────────────────────────────────────────────────────────────
# WHY WE DEVIATE FROM THE PAPER  (transparency / fairness motivation)
# ─────────────────────────────────────────────────────────────────────────────
#  1. FULL SECOND-ORDER MAML (not FOMAML)
#     Jeong & Bae use the exact Finn et al. 2017 formulation with second-order
#     gradients through the inner loop.  We honour this by setting
#     create_graph=True in the inner update.  Cost: ~2–3× slower than FOMAML;
#     justified because it is what the paper actually proposes.
#
#  2. STATIC vs WINDOWED DNN INPUT
#     The paper feeds a single timestep (V, I, T) to the DNN.  Our CAMEL/CoDA
#     baseline uses WINDOW_SIZE=60 timestep windows plus derived features
#     (ΔV, ΔI, Ts) for richer temporal context.
#     We implement BOTH to isolate the effect of windowing:
#       • MAMLStatic  – 3 raw features, no windowing   (faithful to paper)
#       • MAMLWindowed – same feature set as CAMEL/CoDA (fair comparison)
#     This lets us attribute any performance gap to architecture choice vs
#     the meta-learning algorithm itself.
#
#  3. COULOMB-KF POST-PROCESSING — reported ALONGSIDE a paper-faithful no-KF row
#     The original paper does not apply a Kalman filter; its headline numbers
#     (MSE 0.0176%, MAE 1.0075% on US06, fine-tuned w/ 96 points, 9 steps) are
#     raw-NN numbers. We add the same Coulomb-counting KF used for CAMEL/CoDA
#     as an *extension*, not a substitution. The summary table below reports
#     THREE modes so the paper-faithful number is never conflated with our
#     extension:
#       (a) raw NN, no KF        ← closest to what the paper actually reports
#       (b) + shared KF
#       (c) + ad-hoc KF
#
#  4. OUTER OPTIMISER: AdamW
#     §4 of the paper states "...an Adam optimizer were used in this study"
#     for pre-training; SGD is only stated explicitly for the 9-step
#     fine-tuning (inner loop, §4.2). So AdamW here (vs. their Adam) is a
#     minor variant of what the paper already used for the outer loop, not
#     a departure from SGD as an earlier version of this comment claimed.
#
#  7. LITERAL PAPER-REPLICATION ROW (MAML-PaperExact)
#     MAML-Static/MAML-Windowed above use CALIB_FRAC (10% of the run, ~1.8-2k
#     points on our tasks) and MAML_N_ININER=25 steps at test time — chosen to
#     match CAMEL/CoDA's own test-time adaptation budget for an apples-to-
#     apples comparison BETWEEN meta-learners on this dataset. That is NOT the
#     same as the paper's actual protocol: a FIXED 96 data points (regardless
#     of run length) and 9 gradient steps (§3, §4.1, Fig. 4), with a 64x4
#     ReLU DNN (§4) rather than our SiLU/NUM_LAYERS-deep backbone. Reusing
#     those numbers unmodified would silently give our models ~20x more
#     adaptation data than the paper's headline claim, making any "vs paper"
#     comparison unfair in our favor. MAML-PaperExact below reproduces the
#     paper's exact fine-tuning budget/architecture as an ADDITIONAL row, so
#     the notebook reports both:
#       (a) MAML-Static / MAML-Windowed  → fair vs CAMEL/CoDA on this dataset
#       (b) MAML-PaperExact              → literal vs Jeong & Bae (2022)'s
#                                           own reported numbers (Table 3)
#
#  5. NO CLAMPING INSIDE ANY LOSS USED FOR GRADIENTS (consistency w/ CAMEL/CoDA)
#     CAMEL and CoDA never clamp inside their adapt()/training loss — clamp(0,1)
#     only appears in their no_grad() inference helpers (predict_with_omega /
#     predict_with_emb). clamp() has zero gradient outside [0,1], so clamping
#     inside a loss that drives an optimizer step silently kills learning
#     signal near the boundaries and creates a train/test mismatch. We now
#     apply the SAME rule everywhere in MAML: meta-train inner+outer loss,
#     maml_val_metric, and the test-time fine-tuning loop are all UNCLAMPED.
#     clamp(0,1) is applied ONLY at final inference (predictions returned to
#     the caller / used for RMSE / fed into the KF), exactly like CAMEL/CoDA.
#
#  6. DROPOUT CONSISTENCY BETWEEN META-TRAIN AND TEST-TIME FINE-TUNING
#     functional_forward() always skips Dropout (so the inner-loop dynamics
#     seen during meta-training never include dropout noise). Previously,
#     test-time fine-tuning used the model's real forward() in .train() mode,
#     which DID apply dropout — a mismatch between the adaptation procedure
#     Φ was optimized for and the one it was evaluated with. Test-time
#     fine-tuning now also goes through functional_forward (dropout-free),
#     matching meta-training exactly. (Eval-mode dropout was already a no-op
#     either way, since nn.Dropout is inactive in .eval(); the fix specifically
#     targets the gradient-step phase, which previously ran in .train() mode.)
#
# ─────────────────────────────────────────────────────────────────────────────
# FULL SECOND-ORDER MAML – HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
#  Meta-training (outer loop, over tasks τᵢ from train_tasks):
#    1. Sample task τᵢ with data Dᵢ = (Xᵢ, Yᵢ)
#    2. Split Dᵢ into support Dᵢˢᵘᵖᵖ  and query Dᵢᵠᵘᵉʳʸ
#    3. Inner loop (create_graph=True → track 2nd-order grads):
#         φᵢ = Φ − α ∇_Φ L(f_Φ, Dᵢˢᵘᵖᵖ)   [N_INNER_STEPS_TR SGD steps]
#    4. Accumulate meta-loss:
#         L_meta += L(f_φᵢ, Dᵢᵠᵘᵉʳʸ)
#    5. Outer update with 2nd-order gradients:
#         Φ ← Φ − β ∇_Φ Σ L_meta          [AdamW step; grads flow through φᵢ]
#
#  Fine-tuning at test time (new battery):
#    1. Take N_INNER_STEPS SGD steps on Dᵗᵃʳᵍᵉᵗ starting from meta-init Φ
#    2. Predict SOC → clamp(0,1) at inference → optionally post-process w/ KF
#
#  ITEM #9 — RESOLVED (previously deferred)
#  Two separate problems were tangled together and are now both fixed:
#
#  (a) Support/query were drawn from calib_frac × N windows, not the full task.
#      CAMEL/CoDA's BasisMLP backbone is fit using the FULL DataLoader (every
#      window of every training task swept each epoch); calib_frac is reserved
#      exclusively for test-time/val-time adapt() calls, never for backbone
#      training. MAML's inner loop was incorrectly calib_frac-restricted even
#      during META-TRAINING, starving the outer loop of signal (and then
#      halving that already-tiny set again via support_frac=0.5). Fixed:
#      support/query are now drawn from the FULL task (all N windows) during
#      meta-training. calib_frac is used ONLY in maml_val_metric and
#      maml_test_eval, exactly mirroring CAMEL/CoDA's adapt() usage pattern.
#
#  (b) One outer step per epoch, with tasks_per_batch == all training tasks,
#      meant the task population was swept exactly once total (since
#      MAML_EPOCHS was being kept at 1 during debugging) and never revisited.
#      Fixed: maml_meta_train now takes a SMALL task-batch per outer step and
#      runs MULTIPLE outer steps per epoch — looping until the full task
#      population has been swept once — the same semantic as one epoch of a
#      CAMEL/CoDA DataLoader (which iterates minibatches until every window of
#      every task has been seen once). MAML_TASKS_BATCH is now a per-OUTER-STEP
#      batch size (CAMEL/CoDA's BATCH_SIZE plays an analogous role at the
#      window level), not "all tasks in one shot."
# ─────────────────────────────────────────────────────────────────────────────

import copy, random
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ── Hyperparameters ───────────────────────────────────────────────────────────
# Inner LR is now per-variant because input dimensionality differs dramatically:
# Static:   3 features  → loss landscape is smooth → 2e-2 works fine
# Windowed: 300 features (60×5) → much higher curvature → 2e-2 overshoots,
#           causing the adaptation loss to INCREASE for steps 1-7 (U-shape).
#           Lower LR prevents overshoot while still converging in 15 steps.
MAML_LR_INNER        = 2e-2    # Static inner-loop α
MAML_LR_INNER_WIND   = 1e-2   # lower LR prevents NaN explosion in 300-dim space    # Windowed inner-loop α (lower — 300-dim input)
MAML_LR_OUTER    = 2e-3     # outer-loop β  (paper uses SGD-0.001; we use AdamW)
MAML_N_INNER     = 25       # inner steps at test-time  (paper: 9)
MAML_N_INNER_TR  = 5        # inner steps during meta-training (faster, common)
MAML_EPOCHS      = 600       # meta-training epochs — now matches CoDA (was 1;
# MAML_EPOCHS      = 1       # meta-training epochs — now matches CoDA (was 1;
                             # safe to raise now that each epoch sweeps the
                             # full task population over many outer steps,
                             # see item #9 fix above)
MAML_TASKS_BATCH = 32       # tasks per OUTER STEP (was: all training tasks in
                             # one outer step). One epoch = enough outer steps
                             # to sweep train_prepared once, shuffled — the
                             # task-level analogue of CAMEL/CoDA's BATCH_SIZE
                             # at the window level.
MAML_CALIB_FRAC   = CALIB_FRAC   # 0.10 – used at val/test time AND as the
                                  # support fraction during meta-training.
                                  # This keeps train/test adaptation budget
                                  # consistent: the inner loop always adapts
                                  # on ~10% of windows, which matches the
                                  # test-time scenario the model will face.
                                  # (Previously the inner loop used 70% of
                                  # all windows — a 7× larger budget than
                                  # available at test time, creating a
                                  # train/test adaptation mismatch.)

# Static DNN input size: 3 raw features per timestep (V, I, T)
MAML_STATIC_IN = 3          # matches the paper's input dimensionality
MAML_GRAD_ACCUM = MAML_TASKS_BATCH 

# ── Literal paper-replication protocol (§3, §4.1, Fig. 4) ────────────────────
# Used ONLY by the MAML-PaperExact model/eval below — MAML_CALIB_FRAC/
# MAML_N_INNER above stay as-is for the MAML-Static/MAML-Windowed rows that
# are meant to be comparable to CAMEL/CoDA's own adaptation budget.
MAML_NC_PAPER        = 96   # fixed number of fine-tuning points (NOT a
                             # fraction of run length — the paper's headline
                             # claim is specifically about this small a
                             # FIXED budget working, regardless of task size)
MAML_N_INNER_PAPER   = 9    # gradient steps at fine-tuning (§4.1)

# ─────────────────────────────────────────────────────────────────────────────
# Architecture A – MAMLStatic (faithful to Jeong & Bae 2022)
#   Input: single timestep → 3 raw features  (no window, no derived features)
#   Same hidden width/depth as BaselineMLP for a fair parameter count.
# ─────────────────────────────────────────────────────────────────────────────
class MAMLStaticMLP(nn.Module):
    """DNN matching the paper: single-timestep input (V, I, T)."""
    def __init__(self, in_dim=MAML_STATIC_IN,
                 hidden=HIDDEN_DIM, depth=NUM_LAYERS, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(),
                       nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x may arrive as (B, WINDOW, FEAT) from the shared pipeline;
        # for the static model we only use the last timestep's raw features.
        if x.dim() == 3:
            x = x[:, -1, :MAML_STATIC_IN]   # (B, 3)
        elif x.dim() == 2 and x.size(1) != MAML_STATIC_IN:
            x = x[:, :MAML_STATIC_IN]
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture B – MAMLWindowed (comparable to CAMEL/CoDA)
#   Input: WINDOW_SIZE × FEATURE_DIM  (same as BaselineMLP / BasisMLP)
#   Reuses BaselineMLP verbatim – MAML is a training algorithm, not an arch.
# ─────────────────────────────────────────────────────────────────────────────
# BaselineMLP is already defined in an earlier cell; we alias it here.
MAMLWindowedMLP = BaselineMLP    # identical architecture, different training


# ─────────────────────────────────────────────────────────────────────────────
# Architecture C – MAMLPaperExact (literal reproduction of Jeong & Bae 2022, §4)
#   64 neurons, FOUR hidden layers (paper: "64 neurons and four hidden
#   layers"), ReLU (paper: "a rectified linear unit activation function"),
#   single-timestep (V,I,T) input, no dropout (not mentioned in the paper).
#   Used ONLY for the paper-exact row (see item #7 above) — MAMLStaticMLP
#   stays as the parameter-matched-to-baseline variant used elsewhere.
# ─────────────────────────────────────────────────────────────────────────────
class MAMLPaperExactMLP(nn.Module):
    """Literal reproduction of the DNN in Jeong & Bae (2022), §4."""
    def __init__(self, in_dim=MAML_STATIC_IN, hidden=64, n_hidden_layers=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :MAML_STATIC_IN]
        elif x.dim() == 2 and x.size(1) != MAML_STATIC_IN:
            x = x[:, :MAML_STATIC_IN]
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Functional forward – works for any Sequential of Linear/SiLU/Dropout
# Always skips Dropout (deterministic, no-noise inner loop) — used in BOTH
# meta-training AND test-time fine-tuning for consistency (fix #2 / item 6).
# ─────────────────────────────────────────────────────────────────────────────
def functional_forward(model, x, params):
    """
    Run model.forward using `params` instead of model's own weights.
    Preprocesses input based on model type BEFORE hitting the linear layers.
    Dropout is always skipped — this function is used identically during
    meta-training AND test-time fine-tuning, so the inner-loop dynamics Φ
    was optimized for exactly match the dynamics used at evaluation.
    """
    # ── Step 1: preprocess input identically to model.forward() ──────────────
    if isinstance(model, (MAMLStaticMLP, MAMLPaperExactMLP)):
        # Static / paper-exact: take only last timestep's first 3 features
        if x.dim() == 3:
            x = x[:, -1, :MAML_STATIC_IN]       # (B, WINDOW, FEAT) → (B, 3)
        elif x.dim() == 2 and x.size(1) != MAML_STATIC_IN:
            x = x[:, :MAML_STATIC_IN]            # (B, WINDOW*FEAT) → (B, 3)
        # x is now (B, 3) — matches first Linear(3, hidden)
    else:
        # Windowed (BaselineMLP / MAMLWindowedMLP): flatten window × features
        if x.dim() == 3:
            x = x.view(x.size(0), -1)            # (B, WINDOW, FEAT) → (B, WINDOW*FEAT)
        # x is now (B, WINDOW*FEAT) — matches first Linear(WINDOW*FEAT, hidden)

    # ── Step 2: run the linear stack with fast_weights (dropout always skipped) ─
    param_iter = iter(params.values())
    for layer in list(model.net.children()):
        if isinstance(layer, nn.Linear):
            w, b = next(param_iter), next(param_iter)
            x = F.linear(x, w, b)
        elif isinstance(layer, nn.SiLU):
            x = F.silu(x)
        elif isinstance(layer, nn.ReLU):
            x = F.relu(x)                        # MAMLPaperExactMLP (§4: ReLU)
        elif isinstance(layer, nn.Dropout):
            pass    # always skipped — consistent in meta-train AND fine-tune
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Inner-loop update  (second-order MAML  –  create_graph=True by default)
# Loss is UNCLAMPED (matches CAMEL/CoDA's adapt(); clamp only at inference).
# ─────────────────────────────────────────────────────────────────────────────
def maml_inner_update(model, x_sup, y_sup,
                      lr=MAML_LR_INNER,
                      n_steps=MAML_N_INNER_TR,
                      create_graph=True):          # ← True = full 2nd-order
    """
    Run n_steps of SGD on (x_sup, y_sup) starting from model params.
    Returns fast-weights dict; does NOT mutate model in-place.

    create_graph=True  → second-order MAML (Finn et al. 2017, faithful to paper)
    create_graph=False → FOMAML approximation (faster, not used here)

    Loss is computed on UNCLAMPED predictions (fix #4) — clamp(0,1) has zero
    gradient outside [0,1], so clamping inside a loss used for gradients would
    kill learning signal near the SOC boundaries and create a mismatch with
    how CAMEL/CoDA compute their adaptation loss (also unclamped).
    """
    fast_weights = {n: p.clone()
                    for n, p in model.named_parameters()}

    for _ in range(n_steps):
        pred = functional_forward(model, x_sup, fast_weights)   # unclamped
        loss = F.mse_loss(pred, y_sup)

        if torch.isnan(loss):
            fast_weights = {n: p.clone()
                            for n, p in model.named_parameters()}
            break

        grads = torch.autograd.grad(
            loss, fast_weights.values(),
            create_graph=create_graph,
            allow_unused=True
        )

        _gs   = [g if g is not None else torch.zeros_like(w)
                 for (_, w), g in zip(fast_weights.items(), grads)]
        _norm = torch.sqrt(sum(g.pow(2).sum() for g in _gs))
        _clip = torch.clamp(1.0 / (_norm + 1e-8), max=1.0)
        _gs   = [g * _clip for g in _gs]

        # ── THE ACTUAL FIX ────────────────────────────────────────────────
        # .detach() genuinely severs the autograd graph (unlike calling
        # .requires_grad_(True) alone on an already-True non-leaf tensor,
        # which is a no-op and does NOT cut the graph — verified empirically).
        # When create_graph=True (2nd-order meta-training), detaching at
        # every inner step destroys the path from the final fast weights
        # back through all n_steps to the original model parameters Φ —
        # which is exactly what the outer loop's .backward() needs. This
        # was why Φ.grad was ~0 and the outer loop never moved: clipping
        # was correct, but doing it via detach+reattach silently broke
        # 2nd-order MAML every single inner step.
        # When create_graph=False (test-time fine-tuning / FOMAML), we DO
        # want to detach each step — no 2nd-order signal is needed there,
        # and detaching keeps the graph from growing across many steps.
        if create_graph:
            fast_weights = {
                n: w - lr * g
                for (n, w), g in zip(fast_weights.items(), _gs)
            }
        else:
            fast_weights = {
                n: (w - lr * g).detach().requires_grad_(True)
                for (n, w), g in zip(fast_weights.items(), _gs)
            }

        if any(torch.isnan(w).any() for w in fast_weights.values()):
            fast_weights = {n: p.clone()
                            for n, p in model.named_parameters()}
            break

    return fast_weights


# ─────────────────────────────────────────────────────────────────────────────
# Validation metric  (adaptation-based, multi-seed)
# Adaptation loss now UNCLAMPED (fix #4); RMSE reporting still clamps at
# inference, since that's the number that should reflect physical SOC ∈[0,1].
# Uses functional_forward (fix #2) so dropout is skipped exactly as in meta-training.
# ─────────────────────────────────────────────────────────────────────────────
def maml_val_metric(model, val_tasks_prepared,
                    n_inner=MAML_N_INNER,
                    lr_inner=MAML_LR_INNER,
                    n_seeds=EARLY_STOP_VAL_SEEDS):
    """Fine-tune from Φ on each val task, measure RMSE on the full run."""
    model.eval()
    rmses  = []
    saved  = copy.deepcopy(model.state_dict())
    for t in val_tasks_prepared:
        X = torch.from_numpy(t["X"]).float().to(device)
        Y = torch.from_numpy(t["Y"]).float().to(device)
        N = len(X)
        Nc = max(R_DIM + 3, int(MAML_CALIB_FRAC * N))
        task_rmses = []
        for s in range(n_seeds):
            idx = get_calib_idx(N, Nc, seed=s)

            # ── adaptation: functional_forward, unclamped loss, dropout-free ─
            fast_w = maml_inner_update(
                model, X[idx], Y[idx],
                lr=lr_inner, n_steps=n_inner,
                create_graph=False)   # no need to backprop through val metric

            # ── inference: clamp(0,1) only here, matching CAMEL/CoDA ────────
            with torch.no_grad():
                Yp = functional_forward(model, X, fast_w).clamp(0, 1)
            task_rmses.append(
                torch.sqrt(torch.mean((Yp - Y) ** 2)).item())
        rmses.append(float(np.mean(task_rmses)))
    model.load_state_dict(saved)
    return float(np.mean(rmses))


# ─────────────────────────────────────────────────────────────────────────────
# Meta-training loop  (shared by both DNN variants)
# Outer loss is UNCLAMPED (fix #4) — matches CAMEL/CoDA's training loss.
#
# ITEM #9 FIX: support/query are now drawn from the FULL task (all N windows),
# not calib_frac × N — matching how CAMEL/CoDA's backbone is fit on the full
# DataLoader. Each epoch runs enough outer steps to sweep the entire shuffled
# train_tasks population once, batched MAML_TASKS_BATCH tasks per outer step
# (the task-level analogue of CAMEL/CoDA's window-level BATCH_SIZE).
# ─────────────────────────────────────────────────────────────────────────────
def maml_meta_train(model, train_tasks_prepared, val_tasks_prepared,
                    label="MAML",
                    epochs=MAML_EPOCHS,
                    lr_outer=MAML_LR_OUTER,
                    tasks_per_batch=MAML_TASKS_BATCH,
                    support_frac=0.7,
                    max_support=256,   # cap support windows/task regardless of
                                       # how many windows the task has (avg task
                                       # has ~21.6k windows; 0.7 x that was ~15k
                                       # windows unrolled through 2nd-order
                                       # autograd per task — the main OOM driver)
                    max_query=256,     # cap query windows/task (was: ALL
                                       # remaining windows, ~6.5k/task on avg)
                    n_inner_train=MAML_N_INNER_TR,
                    lr_inner=MAML_LR_INNER):
    """
    Full second-order MAML meta-training.

    One EPOCH = sweep the entire (shuffled) train_tasks_prepared population
    once, `tasks_per_batch` tasks per outer step — the task-level analogue of
    one epoch of CAMEL/CoDA's window-level DataLoader.

    Each outer step:
      1. Take the next `tasks_per_batch` tasks from this epoch's shuffled order.
      2. For each task: split its FULL set of windows into support/query
         (NOT calib_frac-restricted — the backbone should see everything a
         task has to offer during meta-training, exactly as CAMEL/CoDA's
         BasisMLP does via the full DataLoader).
      3. Run `n_inner_train` SGD steps on the support set with
         create_graph=True (2nd-order gradients retained).
      4. Evaluate query loss with fast-weights → accumulate.
      5. Backward through the entire unrolled inner loop → AdamW step on Φ.

    calib_frac is intentionally NOT used anywhere in this function — it is
    reserved for maml_val_metric and maml_test_eval (test-time/val-time
    adaptation budget), matching CAMEL/CoDA's adapt() usage.
    """
    outer_opt = optim.AdamW(model.parameters(), lr=lr_outer,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        outer_opt, mode='min', factor=0.5, patience=100, min_lr=1e-5)

    train_losses, val_metrics = [], []
    best_val      = float('inf')
    best_state    = None
    global_step   = 0   # counts outer steps across all epochs, for VAL_EVERY

    n_tasks = len(train_tasks_prepared)

    for ep in range(epochs):
        model.train()
        epoch_order = list(train_tasks_prepared)
        random.shuffle(epoch_order)

        epoch_losses = []

        # ── sweep the full task population this epoch, tasks_per_batch at a time ─
                # ── sweep the full task population this epoch, tasks_per_batch at a time ─
        for batch_start in range(0, n_tasks, tasks_per_batch):
            # Filter on CPU (t["X"] is still numpy here) before touching the GPU.
            batch = [t for t in epoch_order[batch_start: batch_start + tasks_per_batch]
                     if len(t["X"]) >= 4]
            if not batch:
                continue

            outer_opt.zero_grad()
            batch_loss_sum = 0.0

            for t in batch:
                X = torch.from_numpy(t["X"]).float().to(device)
                Y = torch.from_numpy(t["Y"]).float().to(device)
                N = len(X)

                # ── support/query: small fixed-size subsets, capped regardless
                # of task size (see max_support/max_query above — this is what
                # was causing the OOM: unbounded per-task window counts).
                perm      = torch.randperm(N, device=X.device)
                n_sup     = min(max(R_DIM + 3, int(support_frac * N)), max_support, N - 1)
                n_qry     = min(N - n_sup, max_query)
                idx_sup   = perm[:n_sup]
                idx_query = perm[n_sup:n_sup + n_qry]

                x_sup, y_sup = X[idx_sup],   Y[idx_sup]
                x_qry, y_qry = X[idx_query], Y[idx_query]

                # ── 2nd-order inner loop (unclamped loss, fix #4) ───────────
                fast_w = maml_inner_update(
                    model, x_sup, y_sup,
                    lr=lr_inner, n_steps=n_inner_train,
                    create_graph=True)           # retain graph for outer BP

                pred_qry  = functional_forward(model, x_qry, fast_w)   # unclamped
                task_loss = F.mse_loss(pred_qry, y_qry)

                # ── backward PER TASK, not accumulated across the whole batch ─
                # Frees each task's 2nd-order graph immediately (peak memory
                # ~1 task) instead of holding all len(batch) graphs alive until
                # a single .backward() at the end (peak memory ~tasks_per_batch
                # x task graph — this was the other half of the OOM).
                # Gradients still accumulate correctly in .grad across repeated
                # backward() calls, matching one backward() on the summed loss.
                (task_loss / len(batch)).backward()
                batch_loss_sum += task_loss.item()

                del X, Y, fast_w, pred_qry, task_loss

            # ── Outer-gradient NaN guard ─────────────────────────────────────
            # clip_grad_norm_ computes ONE total_norm across all parameters.
            # If any task's backward produced a NaN/Inf gradient, total_norm
            # becomes NaN, the clip coefficient becomes NaN, and EVERY
            # gradient gets scaled by NaN — permanently corrupting Φ via
            # optimizer.step(). Detect this BEFORE clipping/stepping and
            # skip the update entirely rather than applying a corrupted step.
            _grad_is_nan = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if _grad_is_nan:
                print(f"  [WARN] NaN/Inf outer gradient at epoch {ep+1}, "
                      f"batch_start={batch_start} — skipping this outer step "
                      f"(Φ left unchanged)")
                outer_opt.zero_grad()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                outer_opt.step()
                epoch_losses.append(batch_loss_sum / max(len(batch), 1))
                global_step += 1

        if not epoch_losses:
            continue
        train_losses.append(float(np.mean(epoch_losses)))

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            val_rmse = maml_val_metric(model, val_tasks_prepared,
                                       n_inner=MAML_N_INNER_TR,
                                       lr_inner=lr_inner)
            val_metrics.append((ep + 1, val_rmse))
            scheduler.step(val_rmse)
            _sd_now = model.state_dict()
            _sd_ok  = not any(not torch.isfinite(v).all() for v in _sd_now.values())
            if val_rmse < best_val and val_rmse == val_rmse and _sd_ok:  # not NaN
                best_val   = val_rmse
                best_state = copy.deepcopy(_sd_now)
            elif not _sd_ok:
                print(f"  [WARN] model state_dict contains NaN/Inf at epoch {ep+1} "
                      f"— NOT saving as best_state (keeping previous checkpoint)")
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"{label}  Ep {ep+1:4d}  "
                      f"Train {train_losses[-1]:.5f}  "
                      f"Val RMSE {val_rmse:.5f}  "
                      f"Best {best_val:.5f}  "
                      f"(outer steps so far: {global_step})")

    if best_state is None:
        # Safety net: if VAL_EVERY never triggered (e.g. epochs < VAL_EVERY),
        # fall back to the final model state rather than returning None.
        best_state = copy.deepcopy(model.state_dict())

    return train_losses, val_metrics, best_state


# ─────────────────────────────────────────────────────────────────────────────
# Test-time evaluation  (shared by both variants)
# Tracks adaptation loss at every inner SGD step per task/seed.
#
# FIXES APPLIED:
#  - fix #2 (dropout): fine-tuning now uses maml_inner_update/functional_forward
#    instead of the model's real forward() in .train() mode, so dropout is
#    skipped exactly as during meta-training (previously dropout WAS active
#    here, a mismatch).
#  - fix #4 (clamp): the fine-tuning loss (and the recorded adaptation-curve
#    loss) is now UNCLAMPED, matching meta-training and CAMEL/CoDA. clamp(0,1)
#    is applied only when producing the final predictions used for RMSE/KF.
# ─────────────────────────────────────────────────────────────────────────────
def maml_test_eval(model, best_state, test_prepared, kf_q, kf_r,
                   label="MAML", lr_inner=None, n_inner=None, n_calib_fixed=None,
                   train_seed=None):
    """
    n_inner: overrides MAML_N_INNER (steps of test-time fine-tuning).
    n_calib_fixed: if given, a FIXED number of calibration points is used for
    every task (paper's protocol — a fixed 96 points, §3/Fig. 4) instead of
    CALIB_FRAC × N (fraction of the task's own length, our internal-fairness
    convention shared with CAMEL/CoDA). See item #7 in the header comment.
    train_seed: if given, the internal calibration-seed loop is replaced by a
    SINGLE pass at seed=train_seed (pairs calib_seed to train_seed, matching
    the convention used everywhere else). If None (default), falls back to
    the original EARLY_STOP_VAL_SEEDS internal sweep.
    """
    if lr_inner is None:
        lr_inner = MAML_LR_INNER

    # ── Sanity check: best_state must be NaN/Inf-free before we use it ──────
    _nan_keys = [k for k, v in best_state.items() if not torch.isfinite(v).all()]
    if _nan_keys:
        raise ValueError(
            f"best_state contains NaN/Inf in {_nan_keys}. "
            "Meta-training produced a corrupted checkpoint. "
            "Reduce MAML_LR_OUTER / MAML_LR_INNER, raise support_frac "
            "(too-small support_frac starves the inner loop and makes the "
            "unrolled 2nd-order graph ill-conditioned), or re-run training "
            "now that the outer-gradient NaN guard is in place."
        )
    if n_inner is None:
        n_inner = MAML_N_INNER
    results, all_adapt_curves = [], []

    for t in test_prepared:
        X     = torch.from_numpy(t["X"]).float().to(device)
        Y     = torch.from_numpy(t["Y"]).float().to(device)
        ts    = t["time"]
        df    = t["df"]
        Ytrue = Y.cpu().numpy().flatten()
        N     = len(X)
        Nc    = min(n_calib_fixed, N - 1) if n_calib_fixed is not None \
                else max(R_DIM + 3, int(CALIB_FRAC * N))

        seed_preds, seed_curves = [], []
        c_nom = get_c_nom_for_eval(t["battery"])   # constant per task, no seed dependence

        _seed_range = [train_seed] if train_seed is not None else range(EARLY_STOP_VAL_SEEDS)
        for s in _seed_range:
            # seed s controls both calib draw and any stochastic ops in fine-tuning
            torch.manual_seed(s)
            idx  = get_calib_idx(N, Nc, seed=s)
            xcal = X[idx]
            ycal = Y[idx]

            # ── FIX: tensors from state_dict have no grad → enable it here ──
            fast_weights = {
                n: p.clone().requires_grad_(True)          # ← THE FIX
                for n, p in best_state.items()
            }

            step_losses = []
            for step in range(n_inner):
                with torch.no_grad():
                    loss_before = F.mse_loss(
                        functional_forward(model, xcal, fast_weights), ycal).item()
                step_losses.append(loss_before)

                pred  = functional_forward(model, xcal, fast_weights)
                loss  = F.mse_loss(pred, ycal)
                grads = torch.autograd.grad(
                    loss, fast_weights.values(),
                    create_graph=False, allow_unused=True)

                # ── Gradient clipping (same norm as meta-train outer step) ─
                # Prevents NaN explosion in high-dim (300-feat) input space.
                _MAX_NORM = 1.0
                _gs = [g if g is not None else torch.zeros_like(w)
                        for (_, w), g in zip(fast_weights.items(), grads)]
                _norm = torch.sqrt(sum(g.pow(2).sum() for g in _gs))
                _coef = torch.clamp(_MAX_NORM / (_norm + 1e-8), max=1.0)
                _gs   = [g * _coef for g in _gs]

                fast_weights = {
                    n: (w - lr_inner * g).detach().requires_grad_(True)
                    for (n, w), g in zip(fast_weights.items(), _gs)
                }

                # ── NaN guard: explosion despite clipping → reset to Φ ────
                if any(torch.isnan(w).any() for w in fast_weights.values()):
                    print(f"  [WARN] NaN weights at step {step}, "
                          f"seed {s} — resetting to meta-init Φ")
                    fast_weights = {n: p.clone().detach().requires_grad_(True)
                                    for n, p in best_state.items()}
                    break

            with torch.no_grad():
                loss_after = F.mse_loss(
                    functional_forward(model, xcal, fast_weights), ycal).item()
            step_losses.append(loss_after)
            seed_curves.append(step_losses)

            with torch.no_grad():
                _pred = run_inference(
                    X, lambda x: functional_forward(
                        model, x, fast_weights).clamp(0, 1))
                if np.isnan(_pred).any():
                    print(f"  [WARN] NaN prediction seed {s} — using meta-init Φ")
                    _fw0  = {n: p.clone() for n, p in best_state.items()}
                    _pred = run_inference(
                        X, lambda x: functional_forward(
                            model, x, _fw0).clamp(0, 1))
                seed_preds.append(_pred)


        # mean curve across seeds for this task → (N_INNER+1,)
        task_curve = np.mean(seed_curves, axis=0)
        all_adapt_curves.append(task_curve)

        Yraw   = np.mean(seed_preds, axis=0)
        I_arr  = df["i"].values[WINDOW_SIZE:].astype(np.float64)
        dt_arr = np.diff(ts.astype(np.float64), prepend=ts[0])

        Ykf_shared = kalman_filter_coulomb(Yraw, I_arr, dt_arr, c_nom=c_nom)
        Ykf_adhoc  = kalman_filter_coulomb(Yraw, I_arr, dt_arr,
                                           q=kf_q, r=kf_r, c_nom=c_nom)

        results.append(dict(
            battery=t["battery"], run=t.get("run", t.get("runname", "")),
            true=Ytrue, pred_raw=Yraw,
            pred_kf=Ykf_shared, pred_kf_adhoc=Ykf_adhoc,
            time=ts, c_nom=c_nom,
            adapt_curve=task_curve,           # (N_INNER+1,)
            rmse_raw      =float(np.sqrt(mean_squared_error(Ytrue, Yraw))),
            rmse_kf       =float(np.sqrt(mean_squared_error(Ytrue, Ykf_shared))),
            rmse_kf_adhoc =float(np.sqrt(mean_squared_error(Ytrue, Ykf_adhoc))),
            mae_raw       =float(mean_absolute_error(Ytrue, Yraw)),
            mae_kf        =float(mean_absolute_error(Ytrue, Ykf_shared)),
            mae_kf_adhoc  =float(mean_absolute_error(Ytrue, Ykf_adhoc)),
        ))

    return results, np.array(all_adapt_curves)   # (n_tasks, N_INNER+1)

# ─────────────────────────────────────────────────────────────────────────────
# ── Adaptation-loss plot helper ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def plot_adaptation_loss(adapt_curves_dict, save_path):
    """
    adapt_curves_dict: { label_str : adapt_curves_array (n_tasks, N_INNER+1) }
    Plots:
      Left  – mean ± std across tasks (shaded band)
      Right – per-task curves (light lines) + mean (thick line)
    """
    steps  = np.arange(MAML_N_INNER + 1)
    colors = {"MAML-Static":   "#7a39bb",
              "MAML-Windowed": "#437a22"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: mean ± std ──────────────────────────────────────────────────────
    ax = axes[0]
    for label, curves in adapt_curves_dict.items():
        mu  = curves.mean(axis=0)
        std = curves.std(axis=0)
        c   = colors.get(label, "#006494")
        ax.plot(steps, mu, color=c, lw=2.5, marker='o', ms=5, label=label)
        ax.fill_between(steps, mu - std, mu + std, color=c, alpha=0.18)

    ax.set_title("Adaptation Loss – mean ± std across test tasks")
    ax.set_xlabel(f"Inner SGD step  (0 = meta-init Φ,  {MAML_N_INNER} = adapted)")
    ax.set_ylabel("Calib-set MSE (log scale, unclamped)")
    ax.set_yscale("log")
    ax.set_xticks(steps)
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Right: per-task spaghetti + bold mean ─────────────────────────────────
    ax = axes[1]
    for label, curves in adapt_curves_dict.items():
        c   = colors.get(label, "#006494")
        mu  = curves.mean(axis=0)
        for task_curve in curves:
            ax.plot(steps, task_curve, color=c, lw=0.6, alpha=0.25)
        ax.plot(steps, mu, color=c, lw=2.8, marker='o', ms=5,
                label=f"{label} (mean)")

    ax.set_title("Adaptation Loss – per-task trajectories")
    ax.set_xlabel(f"Inner SGD step  (0 = meta-init Φ,  {MAML_N_INNER} = adapted)")
    ax.set_ylabel("Calib-set MSE (log scale, unclamped)")
    ax.set_yscale("log")
    ax.set_xticks(steps)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(
        "MAML Test-Time Adaptation  –  Calibration-set loss at each inner step\n"
        "Left: mean±std  |  Right: individual task curves",
        fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Adaptation-loss plot saved → {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# ── Train Variant A: MAML-Static ─────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def train_baseline(X_train, Y_train, epochs=1000):
    """
    Train BaselineMLP from scratch on X_train / Y_train.
    Scheduler: ReduceLROnPlateau (same as CAMEL/CoDA meta-training loop).
    Returns (model, list_of_epoch_losses).
    """
    # seed controls BOTH model initialisation and calib draw (set before calling)
    model   = BaselineMLP(dropout=0.1).to(device)
    opt     = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # ReduceLROnPlateau instead of CosineAnnealingLR — matches CAMEL/CoDA (Issue #7)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(
                  opt, mode="min", factor=0.5, patience=50, min_lr=1e-5)
    dl      = DataLoader(TensorDataset(X_train, Y_train),
                         batch_size=min(512, len(X_train)),
                         shuffle=True, pin_memory=False)
    loss_fn = nn.MSELoss()
    losses  = []
    for _ in range(epochs):
        model.train()
        ep_loss = 0.0
        for X_b, Y_b in dl:
            opt.zero_grad()
            l = loss_fn(model(X_b), Y_b)
            l.backward(); opt.step()
            ep_loss += l.item() * len(X_b)
        epoch_loss = ep_loss / len(X_train)
        sched.step(epoch_loss)   # ReduceLROnPlateau needs the metric
        losses.append(epoch_loss)
    return model, losses



# ── Comprehensive Data-Efficiency Sweep: 5 variants, train-seed-paired ──────
calib_fracs = [0.001, 0.005, 0.01, 0.05, 0.10, 0.5]
# calib_fracs = [0.001]

def _align(*arrays):
    n = min(len(a) for a in arrays)
    return tuple(a[:n] for a in arrays)


def _sweep_eval_camel(model, frac, task_list, seed=0):
    model.eval(); rmses = []
    for t in task_list:
        X     = torch.from_numpy(t["X"]).float().to(device)
        Y     = torch.from_numpy(t["Y"]).float().to(device)
        Ytrue = Y.cpu().numpy().flatten()
        N = len(X); Nc = max(R_DIM + 3, int(frac * N))
        idx   = get_calib_idx(N, Nc, seed=seed)
        omega = model.adapt(X[idx], Y[idx])
        Yraw  = run_inference(X, lambda x: model.predict_with_omega(x, omega))
        c_nom = get_c_nom_for_eval(t["battery"], t["df"], idx)
        I_arr  = t["df"]["i"].values[WINDOW_SIZE:].astype(np.float64)
        dt_arr = np.diff(t["time"].astype(np.float64), prepend=t["time"][0])
        Yraw, I_arr, dt_arr = _align(Yraw, I_arr, dt_arr)
        Ytrue_t = Ytrue[:len(Yraw)]
        Ykf = kalman_filter_coulomb(Yraw, I_arr, dt_arr, c_nom=c_nom)
        rmses.append(float(np.sqrt(mean_squared_error(Ytrue_t, Ykf))))
    return float(np.mean(rmses)), float(np.std(rmses))


def _sweep_eval_coda(model, frac, task_list, seed=0):
    model.eval(); rmses = []
    for t in task_list:
        X     = torch.from_numpy(t["X"]).float().to(device)
        Y     = torch.from_numpy(t["Y"]).float().to(device)
        Ytrue = Y.cpu().numpy().flatten()
        N = len(X); Nc = max(16, int(frac * N))
        idx  = get_calib_idx(N, Nc, seed=seed)
        e, _ = model.adapt(X[idx], Y[idx], n_steps=adaptation_step_coda, lr=5e-3)
        Yraw  = run_inference(X, lambda x: model.predict_with_emb(x, e))
        c_nom = get_c_nom_for_eval(t["battery"], t["df"], idx)
        I_arr  = t["df"]["i"].values[WINDOW_SIZE:].astype(np.float64)
        dt_arr = np.diff(t["time"].astype(np.float64), prepend=t["time"][0])
        Yraw, I_arr, dt_arr = _align(Yraw, I_arr, dt_arr)
        Ytrue_t = Ytrue[:len(Yraw)]
        Ykf = kalman_filter_coulomb(Yraw, I_arr, dt_arr, c_nom=c_nom)
        rmses.append(float(np.sqrt(mean_squared_error(Ytrue_t, Ykf))))
    return float(np.mean(rmses)), float(np.std(rmses))


def _sweep_eval_baseline(frac, task_list, seed=0):
    rmses = []
    for t in task_list:
        X     = torch.from_numpy(t["X"]).float().to(device)
        Y     = torch.from_numpy(t["Y"]).float().to(device)
        Ytrue = Y.cpu().numpy().flatten()
        N = len(X); Nc = max(10, int(frac * N))
        idx = get_calib_idx(N, Nc, seed=seed)
        bl, _ = train_baseline(X[idx], Y[idx], epochs=1000)
        bl.eval()
        Yraw  = run_inference(X, lambda x: bl(x).clamp(0, 1))
        c_nom = get_c_nom_for_eval(t["battery"], t["df"], idx)
        I_arr  = t["df"]["i"].values[WINDOW_SIZE:].astype(np.float64)
        dt_arr = np.diff(t["time"].astype(np.float64), prepend=t["time"][0])
        Yraw, I_arr, dt_arr = _align(Yraw, I_arr, dt_arr)
        Ytrue_t = Ytrue[:len(Yraw)]
        Ykf = kalman_filter_coulomb(Yraw, I_arr, dt_arr, c_nom=c_nom)
        rmses.append(float(np.sqrt(mean_squared_error(Ytrue_t, Ykf))))
    return float(np.mean(rmses)), float(np.std(rmses))


def _sweep_eval_maml(model, best_state, frac, task_list, seed=0,
                      n_inner_steps=MAML_N_INNER, inner_lr=MAML_LR_INNER):
    rmses = []
    for t in task_list:
        X = torch.from_numpy(t["X"]).float().to(device)
        Y = torch.from_numpy(t["Y"]).float().to(device)
        Ytrue = Y.cpu().numpy().flatten()
        N = len(X); Nc = max(10, int(frac * N))
        idx = get_calib_idx(N, Nc, seed=seed)
        x_cal, y_cal = X[idx], Y[idx]

        fast_weights = {n: p.clone().requires_grad_(True) for n, p in best_state.items()}
        for _ in range(n_inner_steps):
            pred = functional_forward(model, x_cal, fast_weights)
            loss = F.mse_loss(pred, y_cal)

            # NaN guard: forward pass already broken (e.g. exploded from a
            # previous step, or extreme out-of-distribution input like real
            # sim-to-real data the scaler was never fit on) -- stop adapting
            # and fall back to the meta-initialisation for this task/seed.
            if torch.isnan(loss):
                fast_weights = {n: p.clone().requires_grad_(True) for n, p in best_state.items()}
                break

            grads = torch.autograd.grad(loss, fast_weights.values(), allow_unused=True)

            # # Gradient clipping: same threshold used everywhere else in the
            # # MAML pipeline. Prevents a single large step (common with
            # # out-of-distribution inputs) from exploding the fast weights.
            # _gs = [g if g is not None else torch.zeros_like(w)
            #        for (_, w), g in zip(fast_weights.items(), grads)]
            # _norm = torch.sqrt(sum(g.pow(2).sum() for g in _gs))
            # _clip = torch.clamp(1.0 / (_norm + 1e-8), max=1.0)
            # _gs   = [g * _clip for g in _gs]

            fast_weights = {
                n: (w - inner_lr * g).detach().requires_grad_(True)
                for n, w, g in zip(fast_weights.keys(), fast_weights.values(), grads)
            }

            # If weights exploded despite clipping, reset to meta-init and stop.
            if any(torch.isnan(w).any() for w in fast_weights.values()):
                fast_weights = {n: p.clone().requires_grad_(True) for n, p in best_state.items()}
                break

        with torch.no_grad():
            Yraw = run_inference(X, lambda x: functional_forward(model, x, fast_weights).clamp(0, 1))
            if np.isnan(Yraw).any() or np.isinf(Yraw).any():
                # Belt-and-suspenders: should not happen after the guards above,
                # but fall back to the unmodified meta-init prediction if it does.
                _fw0 = {n: p.clone() for n, p in best_state.items()}
                Yraw = run_inference(X, lambda x: functional_forward(model, x, _fw0).clamp(0, 1))
        c_nom = get_c_nom_for_eval(t["battery"], t["df"], idx)
        I_arr  = t["df"]["i"].values[WINDOW_SIZE:].astype(np.float64)
        dt_arr = np.diff(t["time"].astype(np.float64), prepend=t["time"][0])
        Yraw, I_arr, dt_arr = _align(Yraw, I_arr, dt_arr)
        Ytrue_t = Ytrue[:len(Yraw)]
        Ykf = kalman_filter_coulomb(Yraw, I_arr, dt_arr, c_nom=c_nom)
        rmses.append(float(np.sqrt(mean_squared_error(Ytrue_t, Ykf))))

    return float(np.mean(rmses)), float(np.std(rmses))

# ═════════════════════════════════════════════════════════════════════════
#  EVALUATION: real batteries (sim-to-real) and simulated test batteries
# ═════════════════════════════════════════════════════════════════════════
# All models were trained on simulated cells only. Every method is evaluated
# on each real battery separately and on the simulated test batteries, always
# train-seed-paired (seed k <-> checkpoint k <-> calibration draw k), with the
# Kalman filter always using the TRUE capacity of the cell
# (get_c_nom_for_eval raises if a battery has no entry).

METHOD_NAMES  = ["CAMEL", "CoDA",  "Baseline", "MAML"]
SKIP_METHODS  = set()     # e.g. {"Baseline"} for a quick smoke test: Baseline
                          # retrains an MLP for 1000 epochs per task and seed
METHODS       = [m for m in METHOD_NAMES if m not in SKIP_METHODS]
METHOD_COLORS = {"CAMEL": "#01696f", "CoDA": "#006494", 
                 "Baseline": "#964219", "MAML": "#d62728"}
METHOD_MARKERS = {"CAMEL": "o", "CoDA": "s", 
                  "Baseline": "^", "MAML": "D"}
# These two dicts are still used by the tables, plots and CSV writer (with
# .get(m, m)), so they must exist even though they are empty now. Add entries
# only if a method whose display name differs from its key is re-added, e.g.
#   METHOD_TITLE  = {"CoDA-Efficient": "CoDA-Efficient (25%)"}
#   CSV_KEY       = {"CoDA-Efficient": "CoDA_Eff25"}
METHOD_TITLE  = {}
CSV_KEY       = {}
LABELS        = list(REAL_BATTERIES.values())            # ["A123", "INR21700"]
REAL_DATASETS = {lab: s2r_by_battery[b] for b, lab in REAL_BATTERIES.items()}

# Simulated-test evaluation: the same sweep on the held-out simulated batteries,
# which produces data_efficiency_final.png (+ .pdf) and simulated_test_summary_true.csv.
RUN_SIM_SWEEP   = True     # False -> skip it and only do the real batteries
SIM_LABEL       = "Simulated"
SIM_TASK_STRIDE = 1        # 1 = every test task (the full test set). k > 1 keeps every
                           # k-th task: k times faster but NOT the full test set, so use
                           # it for smoke tests only

PANEL_LAYOUT  = (1, 5)    # (rows, cols) of the per-method figure; (2, 3) is
                          # easier to read when the figure is shrunk to a page
MAKE_PER_BATTERY_EFFICIENCY_PLOTS = True   # mean-RMSE + std-bar figure per battery

TRAJ_SEED     = 0
TRAJ_FRACS    = [0.01, 0.10]
TRAJ_RUN_NAME = {lab: None for lab in LABELS}   # None -> first run of that battery;
                                                # or e.g. {"A123": "DST - 0", ...}

print("\n" + "=" * 78)
print("REAL-BATTERY EVALUATION (sim-to-real) -- true-capacity Kalman filter")
print("=" * 78)
for b, lab in REAL_BATTERIES.items():
    print(f"  {lab:>9s}  ({b}): {len(s2r_by_battery[b])} tasks, "
          f"C_nom = {get_c_nom_for_eval(b)/3600:.2f} Ah")

# Windows of the simulated TEST batteries (the same split as in the notebook:
# battery-level, real cells carved out first). Built here, before any long
# computation, so memory or capacity problems show up in the first minute.
SIM_DATASETS = {}
if RUN_SIM_SWEEP:
    print("\nPre-computing sliding windows for the simulated test batteries (CPU numpy)...")
    test_prepared = build_task_tensors(test_tasks[::SIM_TASK_STRIDE])
    _n_w = sum(len(t["X"]) for t in test_prepared)
    print(f"  Simulated test: {len(set(t['battery'] for t in test_prepared))}/{len(test_batteries)} batteries "
          f"{sorted(test_batteries)}, {len(test_prepared)} tasks (stride {SIM_TASK_STRIDE}), "
          f"{_n_w:,} windows, ~{_n_w * WINDOW_SIZE * FEATURE_DIM * 4 / 1e9:.1f} GB of RAM")
    _no_cap = sorted({t["battery"] for t in test_prepared if get_true_c_nom(t["battery"]) is None})
    if _no_cap:
        raise ValueError(
            f"No true capacity in TRUE_CAPACITY_MAH for the simulated test batteries {_no_cap}. "
            f"Estimation is disabled: add them before starting the (long) sweep.")
    SIM_DATASETS = {SIM_LABEL: test_prepared}

# ---------------------------------------------------------------------------
# Single-task adaptation helper (used by the trajectory plots) -- unchanged
# ---------------------------------------------------------------------------
def _adapt_one_task(method, task, seed=TRAJ_SEED, frac=CALIB_FRAC):
    """Adapt a single method on a single task at a given calib fraction;
    return raw prediction + KF inputs (true-capacity KF applied by the caller)."""
    seed_dir = get_seed_dir(seed)
    X = torch.from_numpy(task["X"]).float().to(device)
    Y = torch.from_numpy(task["Y"]).float().to(device)
    Ytrue = Y.cpu().numpy().flatten()
    N = len(X)

    if method == "CAMEL":
        ck = torch.load(os.path.join(seed_dir, "camel_best.pt"), map_location=device)
        m = CAMEL(n_tasks=len(train_tasks), r=R_DIM).to(device)
        m.load_state_dict(ck["state_dict"]); m.eval()
        Nc = max(R_DIM + 3, int(frac * N)); idx = get_calib_idx(N, Nc, seed=seed)
        omega = m.adapt(X[idx], Y[idx])
        Yraw = run_inference(X, lambda x: m.predict_with_omega(x, omega))
    elif method == "CoDA":
        fname = "coda_best.pt" if method == "CoDA" else "coda_efficient_best.pt"
        ck = torch.load(os.path.join(seed_dir, fname), map_location=device)
        cls = CoDaModel if method == "CoDA" else CoDaModelEfficient
        m = cls(n_tasks=len(train_tasks), r=R_DIM, emb_size=R_DIM).to(device)
        m.load_state_dict(ck["state_dict"]); m.eval()
        Nc = max(16, int(frac * N)); idx = get_calib_idx(N, Nc, seed=seed)
        e, _ = m.adapt(X[idx], Y[idx], n_steps=adaptation_step_coda, lr=5e-3)
        Yraw = run_inference(X, lambda x: m.predict_with_emb(x, e))
    elif method == "Baseline":
        Nc = max(10, int(frac * N)); idx = get_calib_idx(N, Nc, seed=seed)
        bl, _ = train_baseline(X[idx], Y[idx], epochs=1000)
        bl.eval()
        Yraw = run_inference(X, lambda x: bl(x).clamp(0, 1))
    elif method == "MAML":
        maml_state = torch.load(os.path.join(seed_dir, "maml_windowed_best.pt"), map_location=device)
        m = MAMLWindowedMLP().to(device)
        Nc = max(10, int(frac * N)); idx = get_calib_idx(N, Nc, seed=seed)
        fast_weights = {n: p.clone().requires_grad_(True) for n, p in maml_state.items()}
        for _ in range(MAML_N_INNER):
            pred = functional_forward(m, X[idx], fast_weights)
            loss = F.mse_loss(pred, Y[idx])
            if torch.isnan(loss):
                fast_weights = {n: p.clone().requires_grad_(True) for n, p in maml_state.items()}
                break
            grads = torch.autograd.grad(loss, fast_weights.values(), allow_unused=True)
            # Gradient clipping intentionally removed here -- max_norm=1.0
            # is far too tight for a ~23K-parameter model (an average
            # per-parameter gradient of ~0.0065 already exceeds it), so it
            # was clipping essentially every step, not just runaway ones,
            # silently shrinking MAML's adaptation to a fraction of its
            # intended size. Matches the fix already applied in
            # _sweep_eval_maml. The NaN guards below are what actually
            # provide crash-safety; the clip was redundant on top of them.
            grads = [g if g is not None else torch.zeros_like(w)
                     for (_, w), g in zip(fast_weights.items(), grads)]
            fast_weights = {n: (w - MAML_LR_INNER * g).detach().requires_grad_(True)
                            for (n, w), g in zip(fast_weights.items(), grads)}
            if any(torch.isnan(w).any() for w in fast_weights.values()):
                fast_weights = {n: p.clone().requires_grad_(True) for n, p in maml_state.items()}
                break
        with torch.no_grad():
            Yraw = run_inference(X, lambda x: functional_forward(m, x, fast_weights).clamp(0, 1))

    I_arr  = task["df"]["i"].values[WINDOW_SIZE:].astype(np.float64)
    dt_arr = np.diff(task["time"].astype(np.float64), prepend=task["time"][0])
    Yraw, I_arr, dt_arr = _align(Yraw, I_arr, dt_arr)
    Ytrue_t = Ytrue[:len(Yraw)]
    return Yraw, I_arr, dt_arr, Ytrue_t, idx


# ---------------------------------------------------------------------------
# Helpers for the sweep
# ---------------------------------------------------------------------------
def _load_meta_model(method, seed):
    """Load the meta-trained checkpoint of `method` for training seed `seed`.
    Returns (model, extra) where extra is MAML's meta-initialisation state."""
    seed_dir = get_seed_dir(seed)
    if method == "CAMEL":
        ck = torch.load(os.path.join(seed_dir, "camel_best.pt"), map_location=device)
        m = CAMEL(n_tasks=len(train_tasks), r=R_DIM).to(device)
        m.load_state_dict(ck["state_dict"])
        return m, None
    if method == "CoDA":
        fname = "coda_best.pt" if method == "CoDA" else "coda_efficient_best.pt"
        cls   = CoDaModel if method == "CoDA" else CoDaModelEfficient
        ck = torch.load(os.path.join(seed_dir, fname), map_location=device)
        m = cls(n_tasks=len(train_tasks), r=R_DIM, emb_size=R_DIM).to(device)
        m.load_state_dict(ck["state_dict"])
        return m, None
    if method == "MAML":
        st = torch.load(os.path.join(seed_dir, "maml_windowed_best.pt"), map_location=device)
        return MAMLWindowedMLP().to(device), st
    raise ValueError(method)


def _eval_method(method, loaded, frac, tasks, seed):
    """Mean (over tasks) KF-filtered RMSE of one method on one list of tasks."""
    if method == "CAMEL":
        return _sweep_eval_camel(loaded[0], frac, tasks, seed=seed)[0]
    if method == "CoDA":
        return _sweep_eval_coda(loaded[0], frac, tasks, seed=seed)[0]
    if method == "MAML":
        return _sweep_eval_maml(loaded[0], loaded[1], frac, tasks, seed=seed)[0]
    if method == "Baseline":
        return _sweep_eval_baseline(frac, tasks, seed=seed)[0]
    raise ValueError(method)


def _csv_path(lab):
    if lab == SIM_LABEL:
        return f"{PLOT_DIR}/simulated_test_summary_true.csv"
    return f"{PLOT_DIR}/sim2real_summary_true_{lab}.csv"


def _dump_csv(res, n_done):
    """(Re)write one CSV per dataset in `res` with the calibration fractions done
    so far -- so a partial result survives if the long run is interrupted."""
    for lab in res:
        cols = {"calib_frac": calib_fracs[:n_done]}
        for m in METHODS:
            k = CSV_KEY.get(m, m)
            cols[f"{k}_mean"] = res[lab][m]["mean"][:n_done]
            cols[f"{k}_std"]  = res[lab][m]["std"][:n_done]
        pd.DataFrame(cols).to_csv(_csv_path(lab), index=False)


def run_sweep(datasets):
    """Train-seed-paired sweep over the calibration fractions.
    datasets: {label: list of prepared tasks}. For every fraction and seed, each
    checkpoint is loaded ONCE and evaluated on every dataset. Returns
    res[label][method] = {"mean": array, "std": array} (one entry per fraction)."""
    labels = list(datasets)
    res = {lab: {m: {"mean": [], "std": []} for m in METHODS} for lab in labels}
    n_tasks = {lab: len(datasets[lab]) for lab in labels}
    print(f"\nSweep over {', '.join(f'{lab} ({n_tasks[lab]} tasks)' for lab in labels)}: "
          f"{len(calib_fracs)} calibration fractions x {N_TRAIN_SEEDS} train-seed-paired "
          f"evaluations x {len(METHODS)} methods")
    t0 = time.time()
    for fi, f in enumerate(calib_fracs):
        vals = {lab: {m: [] for m in METHODS} for lab in labels}
        for ts in range(N_TRAIN_SEEDS):
            for method in METHODS:
                loaded = None if method == "Baseline" else _load_meta_model(method, ts)
                for lab in labels:
                    vals[lab][method].append(_eval_method(method, loaded, f, datasets[lab], ts))
            print(f"    frac={f*100:g}%: seed {ts+1}/{N_TRAIN_SEEDS} done "
                  f"({time.time()-t0:.0f}s elapsed)")
        for lab in labels:
            for m in METHODS:
                res[lab][m]["mean"].append(float(np.mean(vals[lab][m])))
                res[lab][m]["std"].append(float(np.std(vals[lab][m])))
            print(f"  frac={f*100:6.2f}%  {lab:>9s}: " + "  ".join(
                f"{m}={res[lab][m]['mean'][-1]:.4f}+/-{res[lab][m]['std'][-1]:.4f}" for m in METHODS))
        _dump_csv(res, fi + 1)
        print(f"  [{fi+1}/{len(calib_fracs)} fractions done, {time.time()-t0:.0f}s elapsed]")
    for lab in labels:
        for m in METHODS:
            res[lab][m]["mean"] = np.array(res[lab][m]["mean"])
            res[lab][m]["std"]  = np.array(res[lab][m]["std"])
    return res


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _print_tables(res):
    for lab in res:
        print(f"\n  {lab}: mean RMSE +/- std over {N_TRAIN_SEEDS} train-seed-paired evaluations")
        print(f"  {'Frac':>7}" + "".join(f"{METHOD_TITLE.get(m, m):>22}" for m in METHODS))
        for i, f in enumerate(calib_fracs):
            print(f"  {f*100:>6g}%" + "".join(
                f"{res[lab][m]['mean'][i]:>14.4f}+/-{res[lab][m]['std'][i]:.4f}" for m in METHODS))


def plot_per_method_two_batteries(res):
    """One panel per method, two lines per panel: A123 (solid) vs INR21700 (dashed)."""
    pct = np.array([f * 100 for f in calib_fracs])
    nrows, ncols = PANEL_LAYOUT
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.2 * nrows),
                             sharey=True, squeeze=False)
    axes_flat = axes.ravel()
    line_style = {LABELS[0]: dict(fmt="o-",  alpha=1.0),
                  LABELS[1]: dict(fmt="s--", alpha=0.65)}
    for ax, m in zip(axes_flat, METHODS):
        for lab in LABELS:
            mean, std = res[lab][m]["mean"], res[lab][m]["std"]
            st = line_style[lab]
            ax.plot(pct, mean, st["fmt"], color=METHOD_COLORS[m], lw=2, alpha=st["alpha"], label=lab)
            ax.fill_between(pct, np.maximum(mean - std, 0), mean + std,
                            color=METHOD_COLORS[m], alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.set_title(METHOD_TITLE.get(m, m), fontsize=14)
        ax.set_xlabel("Calibration data used at test (%, log scale)", fontsize=15)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=11)
    for ax in axes_flat[len(METHODS):]:
        ax.set_visible(False)
    for r in range(nrows):
        axes[r, 0].set_ylabel("Mean RMSE", fontsize=15)
    plt.suptitle(f"Sim-to-Real: {LABELS[0]} vs {LABELS[1]}, per Model", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/sim2real_{LABELS[0]}_vs_{LABELS[1]}_per_model_true.pdf",
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved sim2real_{LABELS[0]}_vs_{LABELS[1]}_per_model_true.pdf")

def plot_efficiency_nostd(res, lab, n_tasks, head, desc, out_paths):
    """Mean RMSE vs calibration fraction (all methods), for ONE dataset.
    Top panel of plot_efficiency only: no std bars."""
    pct = np.array([f * 100 for f in calib_fracs])
    fig = plt.figure(figsize=(12, 5))
    ax_top = fig.add_subplot(111)
    for m in METHODS:
        ax_top.plot(pct, res[lab][m]["mean"], f"{METHOD_MARKERS[m]}-", color=METHOD_COLORS[m],
                    lw=2.2, ms=6, label=METHOD_TITLE.get(m, m))
    ax_top.axvline(x=CALIB_FRAC * 100, color="black", lw=1,
                   label=f"Main eval point ({CALIB_FRAC*100:.0f}%)")
    ax_top.set_xscale("log")
    ax_top.set_xlabel("Calibration data used at test (%, log scale)", fontsize=15)
    ax_top.set_ylabel(f"Mean RMSE on {desc}", fontsize=15)
    ax_top.set_title(f"{head} "
                     f"({N_TRAIN_SEEDS} train-seed-paired evaluations, {n_tasks} tasks)", fontsize=16)
    ax_top.legend(fontsize=15, ncol=3, loc="upper right")
    ax_top.grid(alpha=0.3, which="both")
    for path in out_paths:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved " + ", ".join(os.path.basename(p) for p in out_paths))

def plot_efficiency(res, lab, n_tasks, head, desc, out_paths):
    """Mean RMSE vs calibration fraction (all methods) + std bars, for ONE dataset.
    Used for every real battery and for the simulated test set, so all of these
    figures share exactly the same layout and font sizes.
      head      : first part of the top title
      desc      : short dataset description used in the axis labels / bottom title
      out_paths : files to write (extension decides the format)"""
    pct = np.array([f * 100 for f in calib_fracs])
    fig = plt.figure(figsize=(12, 9))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.4)

    ax_top = fig.add_subplot(gs[0])
    for m in METHODS:
        ax_top.plot(pct, res[lab][m]["mean"], f"{METHOD_MARKERS[m]}-", color=METHOD_COLORS[m],
                    lw=2.2, ms=6, label=METHOD_TITLE.get(m, m))
    ax_top.axvline(x=CALIB_FRAC * 100, color="black", lw=1,
                   label=f"Main eval point ({CALIB_FRAC*100:.0f}%)")
    ax_top.set_xscale("log")
    ax_top.tick_params(axis="both", which="both", labelsize=12)   # al posto delle due righe set_xticklabels / set_yticklabels
    ax_top.set_xlabel("Calibration data used at test (%, log scale)", fontsize=12)
    ax_top.set_ylabel(f"Mean RMSE on {desc}", fontsize=12)
    ax_top.set_title(f"{head} "
                     f"({N_TRAIN_SEEDS} train-seed-paired evaluations, {n_tasks} tasks)", fontsize=14)
    ax_top.legend(fontsize=10, ncol=5, loc="upper right")
    ax_top.grid(alpha=0.3, which="both")

    ax_bot = fig.add_subplot(gs[1])
    x_pos, bar_w = np.arange(len(calib_fracs)), 0.8 / len(METHODS)
    for i, m in enumerate(METHODS):
        offset = (i - (len(METHODS) - 1) / 2) * bar_w
        ax_bot.bar(x_pos + offset, res[lab][m]["std"], width=bar_w, color=METHOD_COLORS[m],
                   label=METHOD_TITLE.get(m, m), edgecolor="white", linewidth=0.5)
    if CALIB_FRAC in calib_fracs:
        ei = calib_fracs.index(CALIB_FRAC)
        ax_bot.axvspan(ei - 0.5, ei + 0.5, color="gray", alpha=0.08, zorder=0)
    ax_bot.set_xticks(x_pos)
    ax_bot.set_xticklabels([f"{f*100:g}%" for f in calib_fracs], fontsize=12)   # questa va bene com'è
    ax_bot.tick_params(axis="y", which="both", labelsize=12)                    # al posto di set_yticklabels(fontsize=12)
    ax_bot.set_xlabel("Calibration fraction used at test time", fontsize=12)
    ax_bot.set_ylabel("RMSE std", fontsize=12)
    ax_bot.set_title(f"Combined Std — grouped by calibration fraction ({desc})", fontsize=14)
    ax_bot.legend(fontsize=10, ncol=5, loc="upper right")
    ax_bot.grid(axis="y", alpha=0.3)

    for path in out_paths:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved " + ", ".join(os.path.basename(p) for p in out_paths))


def plot_efficiency_per_battery(res, lab):
    """Real battery `lab`: data-efficiency figure (same file name as before)."""
    plot_efficiency_nostd(res, lab, len(REAL_DATASETS[lab]),
                    head=f"Sim-to-Real Data Efficiency on {lab}", desc=lab,
                    out_paths=[f"{PLOT_DIR}/data_efficiency_sim2real_true_{lab}.pdf"])


def plot_simulated_efficiency(res):
    """Simulated test batteries: data_efficiency_final.png (+ a vector .pdf)."""
    plot_efficiency(res, SIM_LABEL, len(SIM_DATASETS[SIM_LABEL]),
                    head="Data Efficiency on Simulated Test Batteries",
                    desc="simulated test batteries",
                    out_paths=[f"{PLOT_DIR}/data_efficiency_final.png",
                               f"{PLOT_DIR}/data_efficiency_final.pdf"])


def plot_trajectories():
    """True SOC vs KF-filtered estimate of every method, one figure per real battery,
    one row per calibration budget in TRAJ_FRACS."""
    print("\n" + "=" * 78)
    print(f"TRAJECTORY PLOTS -- true-capacity KF, calib {', '.join(f'{f*100:g}%' for f in TRAJ_FRACS)}")
    print("=" * 78)
    for b, lab in REAL_BATTERIES.items():
        tasks = s2r_by_battery[b]
        want  = TRAJ_RUN_NAME.get(lab)
        task  = tasks[0] if want is None else next((t for t in tasks if t.get("run_name") == want), None)
        if task is None:
            raise ValueError(f"{lab}: run '{want}' not found. Available: "
                             f"{[t.get('run_name') for t in tasks]}")
        c_nom = get_c_nom_for_eval(b)
        print(f"  {lab}: run = {task.get('run_name', '')}  |  C_nom = {c_nom/3600:.2f} Ah")
        print(f"      runs available: {[t.get('run_name') for t in tasks]}")

        fig, axes = plt.subplots(len(TRAJ_FRACS), 1, figsize=(11, 4.5 * len(TRAJ_FRACS)), sharex=False)
        axes = np.atleast_1d(axes)
        for row, frac in enumerate(TRAJ_FRACS):
            ax = axes[row]
            ax.plot(task["time"], task["Y"].flatten(), "k-", lw=2.2, label="True SOC", zorder=10)
            for method in METHODS:
                Yraw, I_arr, dt_arr, Ytrue_t, _ = _adapt_one_task(method, task, frac=frac)
                t_axis = task["time"][:len(Yraw)]
                Ykf = kalman_filter_coulomb(Yraw, I_arr, dt_arr, c_nom=c_nom)
                ax.plot(t_axis, Ykf, "-", color=METHOD_COLORS[method], lw=1.8, label=f"{method} +KF")
            ax.set_title(f"{lab} (simtoreal) | {task.get('run_name', '')}  "
                         f"|  calib={frac*100:.0f}%  |  C_nom(true)={c_nom/3600:.2f}Ah", fontsize=16)
            ax.set_xlabel("Time (s)", fontsize=15)
            ax.set_ylabel("SOC", fontsize=15)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=15, loc="lower left", ncol=2)
            ax.grid(alpha=0.3)
        plt.suptitle(f"{lab} Sim-to-Real: True-Capacity KF, Low vs High Calibration Budget",
                     fontsize=16, y=1.01)
        plt.tight_layout()
        fname = f"{PLOT_DIR}/{lab.lower()}_trajectory_true_capacity_by_calibfrac.pdf"
        plt.savefig(fname, bbox_inches="tight")
        plt.show()
        print(f"Saved {fname}")


# ---------------------------------------------------------------------------
# Run. Trajectories first: they take minutes, so any problem (e.g. a wrong
# run name) shows up before the long sweeps. Then the real batteries (their
# results and figures are complete before the much longer simulated pass
# starts), and finally the simulated test batteries. Both sweeps retrain the
# Baseline MLP for every task, seed and calibration fraction.
# ---------------------------------------------------------------------------
plot_trajectories()

# 1) real batteries (A123, INR21700)
real_res = run_sweep(REAL_DATASETS)
_print_tables(real_res)
plot_per_method_two_batteries(real_res)
if MAKE_PER_BATTERY_EFFICIENCY_PLOTS:
    for _lab in LABELS:
        plot_efficiency_per_battery(real_res, _lab)
_dump_csv(real_res, len(calib_fracs))
print("\nSaved: " + ", ".join(f"sim2real_summary_true_{l}.csv" for l in LABELS))

# 2) simulated test batteries -> data_efficiency_final.png
if RUN_SIM_SWEEP:
    sim_res = run_sweep(SIM_DATASETS)
    _print_tables(sim_res)
    plot_simulated_efficiency(sim_res)
    _dump_csv(sim_res, len(calib_fracs))
    print("\nSaved: simulated_test_summary_true.csv")