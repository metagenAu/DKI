"""Adapted copy of the original DKI training loop for head-to-head comparison.

Two integration paths live side by side, selected by ``--batched``:

* **single** (default) — the *original* DKI recipe: each sample in the
  minibatch is integrated on its own with a separate ``odeint`` call, looping
  in Python (``for i in range(batch_size)``). This is what
  ``legacy/DKI_original.py`` did. The ``ODEFunc`` used here is the original
  non-batch-safe form whose mean-field term is only correct for a ``(1, N)``
  state.
* **batched** (``--batched``) — the *same* replicator dynamics and training
  recipe, but the whole minibatch is integrated in one ``odeint`` call over a
  ``(B, N)`` state. This needs the batch-safe ``ODEFuncBatched``, whose
  mean-field term is a per-row reduction and so gives identical dynamics for
  every row independently. The only thing that changes versus the single path
  is *how* the integration is dispatched, isolating the speedup from batching.

Both paths share the same parameterisation (two ``Linear(N, N)`` layers, in the
same construction order) so, under a fixed seed, they start from identical
weights. ``ODEFunc`` and ``ODEFuncBatched`` compute mathematically identical
dynamics for a single sample, so the two paths track each other up to the
adaptive solver coupling its step size across rows in batched mode.

Behavioural changes vs. legacy/DKI_original.py:
  * Reads CSVs directly from ``--data`` (defaults to ``data/``) instead of
    ``../data/<dataset>/``.
  * ``--epochs`` is configurable.
  * Returns timing + final val-loss so we can compare against the new dki/
    package (and the single vs. batched paths against each other) on identical
    splits and seeds.

Everything else (fixed-step Euler grid t=[0,100] with dt=0.01 -> 10,000 steps,
2-Linear fitness, BC loss summed over the batch, deepcopy-best-val) is unchanged.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint


def loss_bc(p_i, q_i):
    return torch.sum(torch.abs(p_i - q_i)) / torch.sum(torch.abs(p_i + q_i))


def loss_bc_rows(p_pred, p_true):
    """Per-row Bray-Curtis dissimilarity. ``(B, N), (B, N) -> (B,)``."""
    num = torch.abs(p_pred - p_true).sum(dim=-1)
    den = torch.abs(p_pred + p_true).sum(dim=-1)
    return num / den


def process_data(P):
    Z = P.copy()
    Z[Z > 0] = 1
    P = P / P.sum(axis=0)[np.newaxis, :]
    Z = Z / Z.sum(axis=0)[np.newaxis, :]
    P = P.astype(np.float32)
    Z = Z.astype(np.float32)
    return torch.from_numpy(P.T), torch.from_numpy(Z.T)


class ODEFunc(torch.nn.Module):
    """Original DKI dynamics. Correct only for a single ``(1, N)`` state.

    The mean-field term ``ones(N,1) @ y @ outᵀ`` contracts over the batch axis,
    so feeding a ``(B, N)`` state mixes samples together. Use it the way the
    original did: one sample at a time inside a Python loop.
    """

    def __init__(self, N):
        super().__init__()
        self.fcc1 = torch.nn.Linear(N, N)
        self.fcc2 = torch.nn.Linear(N, N)

    def forward(self, t, y):  # noqa: ARG002
        out = self.fcc1(y)
        out = self.fcc2(out)
        f = torch.matmul(
            torch.matmul(torch.ones(y.size(dim=1), 1, device=y.device), y),
            torch.transpose(out, 0, 1),
        )
        return torch.mul(y, out - torch.transpose(f, 0, 1))


class ODEFuncBatched(torch.nn.Module):
    """Batch-safe form of :class:`ODEFunc`.

    Identical two-``Linear`` fitness, but the replicator mean-field term is a
    per-row reduction ``<y, out>`` instead of the original's batch-collapsing
    matmul. For a ``(1, N)`` state this is exactly the original's dynamics; for
    a ``(B, N)`` state every row evolves independently, so the whole minibatch
    can be integrated in one ``odeint`` call.
    """

    def __init__(self, N):
        super().__init__()
        self.fcc1 = torch.nn.Linear(N, N)
        self.fcc2 = torch.nn.Linear(N, N)

    def forward(self, t, y):  # noqa: ARG002
        out = self.fcc2(self.fcc1(y))
        mean_fitness = (y * out).sum(dim=-1, keepdim=True)
        return y * (out - mean_fitness)


def integrate_set(func, z, t, batched):
    """Integrate every row of ``z`` to ``t[-1]``. ``(B, N) -> (B, N)``.

    ``batched=False`` reproduces the original per-sample loop; ``batched=True``
    runs a single ``odeint`` over the stacked state.
    """
    if batched:
        traj = odeint(func, z, t)
        return traj[-1]
    rows = []
    for i in range(z.size(dim=0)):
        traj = odeint(func, z[i].unsqueeze(dim=0), t)
        rows.append(traj[-1].reshape(-1))
    return torch.stack(rows, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--out", default="results/legacy")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--batched",
        action="store_true",
        help="Integrate the whole minibatch in one odeint call (batch-safe "
        "dynamics) instead of the original per-sample Python loop.",
    )
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tag = "batched" if args.batched else "single"

    P = np.loadtxt(Path(args.data) / "Ptrain.csv", delimiter=",")
    n_cols = P.shape[1]
    rng = np.random.default_rng(args.seed)
    val_idx = rng.choice(n_cols, size=int(args.val_fraction * n_cols), replace=False)
    train_idx = np.setdiff1d(np.arange(n_cols), val_idx)
    ptrn, ztrn = process_data(P[:, train_idx])
    pval, zval = process_data(P[:, val_idx])
    M, N = ptrn.shape
    ptrn, ztrn = ptrn.to(device), ztrn.to(device)
    pval, zval = pval.to(device), zval.to(device)

    batch_time = 100
    t = torch.arange(0.0, batch_time, 0.01, device=device)

    func_cls = ODEFuncBatched if args.batched else ODEFunc
    func = func_cls(N).to(device)
    optim_ = torch.optim.Adam(func.parameters(), lr=args.lr)

    loss_train_hist = []
    loss_val_hist = []
    epoch_times = []
    Loss_opt = float("inf")
    best_model = copy.deepcopy(func)

    for e in range(args.epochs):
        t0 = time.perf_counter()
        s = torch.from_numpy(
            np.random.choice(np.arange(M, dtype=np.int64), args.batch_size, replace=False)
        )
        batch_p = ztrn[s, :]
        batch_q = ptrn[s, :]

        optim_.zero_grad()
        q_pred = integrate_set(func, batch_p, t, args.batched)
        loss = loss_bc_rows(q_pred, batch_q).sum()

        # Validation
        with torch.no_grad():
            q_val = integrate_set(func, zval, t, args.batched)
            val_mean = loss_bc_rows(q_val, pval).sum().item() / zval.size(dim=0)

        loss_train_hist.append(loss.item() / args.batch_size)
        loss_val_hist.append(val_mean)
        if val_mean <= Loss_opt:
            Loss_opt = val_mean
            best_model = copy.deepcopy(func)

        loss.backward()
        optim_.step()

        epoch_times.append(time.perf_counter() - t0)
        if e % 5 == 0 or e == args.epochs - 1:
            print(
                f"[legacy:{tag}] epoch {e:4d}  train_bc={loss_train_hist[-1]:.4f}  "
                f"val_bc={val_mean:.4f}  best={Loss_opt:.4f}  sec={epoch_times[-1]:.2f}"
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "epoch_times.npy", np.array(epoch_times))
    np.save(out / "val_loss.npy", np.array(loss_val_hist))

    print(f"[legacy:{tag}] best val BC: {Loss_opt:.6f}")
    print(
        f"[legacy:{tag}] mean epoch wall-clock: {np.mean(epoch_times):.3f}s "
        f"(median {np.median(epoch_times):.3f}s)"
    )


if __name__ == "__main__":
    main()
