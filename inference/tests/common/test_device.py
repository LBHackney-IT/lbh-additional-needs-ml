import torch

from src.common.device import resolve_device


def test_resolve_device_returns_explicit_device_without_checking_hardware():
    requested = torch.device("cpu")
    resolved = resolve_device(requested)
    assert resolved == requested


def test_resolve_device_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    resolved = resolve_device()
    assert resolved == torch.device("cuda")


def test_resolve_device_uses_mps_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    resolved = resolve_device()
    assert resolved == torch.device("mps")


def test_resolve_device_falls_back_to_cpu_without_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    resolved = resolve_device()
    assert resolved == torch.device("cpu")
