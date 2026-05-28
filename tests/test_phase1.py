"""Phase-1 tests: simplex preservation, loss correctness, batched-vs-loop equivalence."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import dki.model as model_mod
from dki.losses import bray_curtis
from dki.model import ReplicatorODEFunc, integrate


def test_bray_curtis_known_case():
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.0, 1.0]])
    assert torch.isclose(bray_curtis(a, b), torch.tensor(1.0))

    a = torch.tensor([[0.5, 0.5]])
    b = torch.tensor([[0.5, 0.5]])
    assert torch.isclose(bray_curtis(a, b), torch.tensor(0.0))

    # batch average
    a = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    b = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
    assert torch.isclose(bray_curtis(a, b), torch.tensor(0.5))


def test_simplex_preserved_under_replicator():
    torch.manual_seed(0)
    N = 8
    func = ReplicatorODEFunc(N)
    # Random samples on the simplex
    z = torch.rand(5, N)
    z = z / z.sum(dim=-1, keepdim=True)
    out = integrate(func, z, t_final=10.0)
    # Sum-to-1 preserved to within solver tolerance
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), sums
    # Non-negativity preserved (no species goes negative)
    assert (out >= -1e-6).all()


def test_batched_equals_per_sample():
    """Batched integration should give the same result as per-sample integration."""
    torch.manual_seed(42)
    N = 6
    func = ReplicatorODEFunc(N)
    z = torch.rand(4, N)
    z = z / z.sum(dim=-1, keepdim=True)

    batched = integrate(func, z, t_final=5.0).detach()
    one_at_a_time = torch.stack(
        [integrate(func, z[i : i + 1], t_final=5.0).detach()[0] for i in range(z.shape[0])]
    )
    assert torch.allclose(batched, one_at_a_time, atol=1e-4), (batched - one_at_a_time).abs().max()


def test_integrate_falls_back_on_underflow_assertion(monkeypatch):
    """A stiff field makes dopri5 underflow dt and raise AssertionError; integrate
    must recover via the fixed-step fallback instead of killing the run."""
    N = 5
    z = torch.rand(2, N)
    z = z / z.sum(dim=-1, keepdim=True)
    methods = []

    def fake_odeint(func, y0, t, method=None, rtol=None, atol=None, options=None):
        methods.append(method)
        if method == "dopri5":
            raise AssertionError("underflow in dt 3.1e-15")
        return torch.stack([y0, y0])  # fixed-step fallback returns a finite trajectory

    monkeypatch.setattr(model_mod, "odeint", fake_odeint)
    out = integrate(ReplicatorODEFunc(N), z, t_final=100.0)
    assert methods == ["dopri5", "rk4"]
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_integrate_falls_back_on_nonfinite_result(monkeypatch):
    """A non-finite adaptive result (no exception raised) also triggers the fallback."""
    N = 5
    z = torch.rand(2, N)
    z = z / z.sum(dim=-1, keepdim=True)
    methods = []

    def fake_odeint(func, y0, t, method=None, rtol=None, atol=None, options=None):
        methods.append(method)
        if method == "dopri5":
            return torch.stack([y0, torch.full_like(y0, float("nan"))])
        return torch.stack([y0, y0])

    monkeypatch.setattr(model_mod, "odeint", fake_odeint)
    out = integrate(ReplicatorODEFunc(N), z, t_final=100.0)
    assert methods == ["dopri5", "rk4"]
    assert torch.isfinite(out).all()


def test_integrate_reraises_when_fallback_disabled(monkeypatch):
    N = 5
    z = torch.rand(2, N)
    z = z / z.sum(dim=-1, keepdim=True)

    def fake_odeint(func, y0, t, method=None, rtol=None, atol=None, options=None):
        raise AssertionError("underflow in dt")

    monkeypatch.setattr(model_mod, "odeint", fake_odeint)
    with pytest.raises(AssertionError):
        integrate(ReplicatorODEFunc(N), z, t_final=100.0, fallback_steps=0)
