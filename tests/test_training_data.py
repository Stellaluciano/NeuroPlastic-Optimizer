from unittest.mock import ANY

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch missing")


def test_build_dataloaders_passes_dataset_and_loader_options(monkeypatch):
    from neuroplastic_optimizer.training import data as training_data

    dataset_calls = []
    loader_calls = []

    class FakeDataset:
        pass

    def fake_mnist(root, train, download, transform):
        dataset_calls.append(
            {
                "root": root,
                "train": train,
                "download": download,
                "transform": transform,
            }
        )
        return FakeDataset()

    def fake_loader(dataset, batch_size, shuffle, **kwargs):
        loader_calls.append(
            {
                "dataset": dataset,
                "batch_size": batch_size,
                "shuffle": shuffle,
                "kwargs": kwargs,
            }
        )
        return object()

    monkeypatch.setattr(training_data.datasets, "MNIST", fake_mnist)
    monkeypatch.setattr(training_data, "DataLoader", fake_loader)

    train_loader, test_loader = training_data.build_dataloaders(
        dataset="mnist",
        batch_size=64,
        num_workers=2,
        data_root="custom_data",
        download=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    assert train_loader is not None
    assert test_loader is not None
    assert dataset_calls == [
        {"root": "custom_data", "train": True, "download": False, "transform": ANY},
        {"root": "custom_data", "train": False, "download": False, "transform": ANY},
    ]
    assert loader_calls[0]["batch_size"] == 64
    assert loader_calls[0]["shuffle"] is True
    assert loader_calls[0]["kwargs"] == {
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }
    assert loader_calls[1]["shuffle"] is False


def test_build_dataloaders_forces_persistent_workers_false_when_num_workers_zero(monkeypatch):
    from neuroplastic_optimizer.training import data as training_data

    loader_kwargs = []

    def fake_cifar10(root, train, download, transform):
        return object()

    def fake_loader(dataset, batch_size, shuffle, **kwargs):
        loader_kwargs.append(kwargs)
        return object()

    monkeypatch.setattr(training_data.datasets, "CIFAR10", fake_cifar10)
    monkeypatch.setattr(training_data, "DataLoader", fake_loader)

    training_data.build_dataloaders(
        dataset="cifar10",
        batch_size=32,
        num_workers=0,
        persistent_workers=True,
        prefetch_factor=2,
    )

    assert loader_kwargs[0]["num_workers"] == 0
    assert loader_kwargs[0]["persistent_workers"] is False
    assert "prefetch_factor" not in loader_kwargs[0]


def test_build_dataloaders_download_false_error_contains_actionable_hint(monkeypatch):
    from neuroplastic_optimizer.training import data as training_data

    def fake_fashion_mnist(root, train, download, transform):
        raise RuntimeError("dataset not found")

    monkeypatch.setattr(training_data.datasets, "FashionMNIST", fake_fashion_mnist)

    with pytest.raises(RuntimeError, match="请预先下载到 data_root"):
        training_data.build_dataloaders(
            dataset="fashionmnist",
            batch_size=16,
            num_workers=1,
            data_root="missing_data",
            download=False,
        )
