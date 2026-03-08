from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def _dataloader_kwargs(
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> dict:
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def _build_synthetic_loader(
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
):
    train_x = torch.randn(1024, 1, 28, 28)
    train_y = torch.randint(0, 10, (1024,))
    test_x = torch.randn(256, 1, 28, 28)
    test_y = torch.randint(0, 10, (256,))
    train = TensorDataset(train_x, train_y)
    test = TensorDataset(test_x, test_y)
    loader_kwargs = _dataloader_kwargs(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


def build_dataloaders(
    dataset: str,
    batch_size: int,
    num_workers: int,
    data_root: str = "data",
    download: bool = True,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
):
    dataset = dataset.lower()
    if dataset == "synthetic_mnist":
        return _build_synthetic_loader(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    try:
        if dataset in {"mnist", "fashionmnist"}:
            normalize = transforms.Normalize((0.5,), (0.5,))
            transform = transforms.Compose([transforms.ToTensor(), normalize])
            dset_cls = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
            train = dset_cls(data_root, train=True, download=download, transform=transform)
            test = dset_cls(data_root, train=False, download=download, transform=transform)
        elif dataset == "cifar10":
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
                ]
            )
            train = datasets.CIFAR10(data_root, train=True, download=download, transform=transform_train)
            test = datasets.CIFAR10(data_root, train=False, download=download, transform=transform_test)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
    except RuntimeError as exc:
        if not download:
            raise RuntimeError(
                f"Failed to load dataset '{dataset}' from data_root='{data_root}'. "
                "请预先下载到 data_root，或将 download=True 自动下载。"
            ) from exc
        raise

    loader_kwargs = _dataloader_kwargs(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, test_loader
