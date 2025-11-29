#!/usr/bin/env python3
"""Train a multi-label attribute head from Qwen-derived ground truth.

This script uses:

- `tmp/ground_truth_auto.jsonl` (or similar) as **weak labels** for attributes.
- Precomputed SigLIP embeddings stored in the `image_embedding` table and
  referenced as `.npy` files under the cache root.

It trains a small linear head with BCE-with-logits loss on top of SigLIP
embeddings to predict attribute label keys such as ``"attr.has_person"`` or
``"attr.is_screenshot"``. The resulting weights are saved as a PyTorch state
dict that can later be integrated into the scene classifier.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sqlalchemy import select
from sqlalchemy.orm import Session
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from utils.logging import get_logger
from vibe_photos.cache_helpers import resolve_cache_root
from vibe_photos.config import Settings, load_settings
from vibe_photos.db import ImageEmbedding, open_primary_session
from vibe_photos.labels.scene_schema import ALL_ATTRIBUTE_LABEL_KEYS

LOGGER = get_logger(__name__, extra={"command": "train_attribute_head_from_qwen"})


ATTR_KEYS: list[str] = list(ALL_ATTRIBUTE_LABEL_KEYS)


@dataclass
class TrainConfig:
    gt_path: Path
    db_url: str
    cache_root: Path
    output_path: Path
    epochs: int
    batch_size: int
    learning_rate: float
    val_fraction: float
    max_samples: int | None
    seed: int


@dataclass
class TrainResult:
    num_samples: int
    num_attrs: int
    attr_keys: list[str]
    train_accuracy: float
    val_accuracy: float
    output_path: Path


def _set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_ground_truth_records(path: Path) -> list[dict[str, Any]]:
    """Load ground-truth-style records from JSON or JSONL."""

    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            try:
                obj = json.loads(line_stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
        return records

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [obj for obj in data if isinstance(obj, dict)]
    return []


def _load_attribute_labels_from_gt(path: Path) -> dict[str, list[bool]]:
    """Return mapping image_id -> list[bool] over ATTR_KEYS."""

    records = _load_ground_truth_records(path)
    label_by_id: dict[str, list[bool]] = {}

    for record in records:
        image_id = record.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            continue

        attrs = record.get("attributes") or {}
        if not isinstance(attrs, dict):
            continue

        values: list[bool] = []
        missing = False
        for key in ATTR_KEYS:
            value = attrs.get(key)
            if isinstance(value, bool):
                values.append(bool(value))
            else:
                missing = True
                break

        if missing:
            continue

        label_by_id[image_id] = values

    if not label_by_id:
        raise RuntimeError(f"No usable attribute labels found in {path}")

    return label_by_id


def _load_embedding_rows(session: Session, model_name: str) -> dict[str, str]:
    """Return mapping image_id -> relative embedding path for the given model."""

    rows = session.execute(
        select(ImageEmbedding.image_id, ImageEmbedding.embedding_path).where(
            ImageEmbedding.model_name == model_name
        )
    ).all()
    return {row.image_id: row.embedding_path for row in rows}


def _resolve_embedding_path(cache_root: Path, rel_path: str) -> Path:
    path_obj = Path(rel_path)
    if path_obj.is_absolute():
        return path_obj
    return cache_root / path_obj


def _build_training_arrays(
    *,
    session: Session,
    settings: Settings,
    cache_root: Path,
    gt_path: Path,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load embeddings + attribute labels into dense numpy arrays."""

    label_by_id = _load_attribute_labels_from_gt(gt_path)
    embedding_model_name = settings.models.embedding.resolved_model_name()

    embedding_paths = _load_embedding_rows(session, embedding_model_name)
    common_ids = sorted(set(label_by_id.keys()) & set(embedding_paths.keys()))
    if not common_ids:
        raise RuntimeError("No overlap between ground truth image_ids and ImageEmbedding rows.")

    rng = np.random.default_rng(seed=seed)
    if max_samples is not None and max_samples > 0 and len(common_ids) > max_samples:
        common_ids = list(rng.choice(common_ids, size=max_samples, replace=False))

    vectors: list[np.ndarray] = []
    labels: list[list[float]] = []

    for image_id in common_ids:
        attr_values = label_by_id.get(image_id)
        if attr_values is None:
            continue

        rel_path = embedding_paths.get(image_id)
        if rel_path is None:
            continue

        emb_path = _resolve_embedding_path(cache_root, rel_path)
        try:
            vec = np.load(emb_path)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error(
                "train_attr_head_load_embedding_error",
                extra={"image_id": image_id, "path": str(emb_path), "error": str(exc)},
            )
            continue

        if vec.ndim != 1:
            LOGGER.error(
                "train_attr_head_invalid_embedding_shape",
                extra={"image_id": image_id, "path": str(emb_path), "shape": tuple(vec.shape)},
            )
            continue

        vectors.append(vec.astype(np.float32))
        labels.append([1.0 if v else 0.0 for v in attr_values])

    if not vectors:
        raise RuntimeError("No valid embeddings collected for training.")

    X = np.stack(vectors, axis=0)
    y = np.asarray(labels, dtype=np.float32)

    LOGGER.info(
        "train_attr_head_dataset_summary",
        extra={
            "num_samples": int(X.shape[0]),
            "embedding_dim": int(X.shape[1]),
            "num_attrs": y.shape[1],
            "attr_keys": ATTR_KEYS,
        },
    )

    return X, y, ATTR_KEYS


def _split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random train/val split."""

    num_samples = X.shape[0]
    if num_samples == 0:
        raise ValueError("Cannot split empty dataset.")

    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(indices)

    if val_fraction <= 0.0:
        return X, y, np.empty((0, X.shape[1]), dtype=X.dtype), np.empty((0, y.shape[1]), dtype=y.dtype)

    val_size = int(num_samples * val_fraction)
    if val_size <= 0:
        return X, y, np.empty((0, X.shape[1]), dtype=X.dtype), np.empty((0, y.shape[1]), dtype=y.dtype)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    return X_train, y_train, X_val, y_val


def _train_attribute_head(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[dict[str, Any], float, float]:
    """Train a linear multi-label head with BCE-with-logits loss."""

    input_dim = int(X_train.shape[1])
    num_attrs = int(y_train.shape[1])

    model = torch.nn.Linear(input_dim, num_attrs)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_acc = 0.0
    val_acc = 0.0

    X_val_tensor: Tensor | None = None
    y_val_tensor: Tensor | None = None
    if X_val.shape[0] > 0:
        X_val_tensor = torch.from_numpy(X_val).float().to(device)
        y_val_tensor = torch.from_numpy(y_val).float().to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_elems = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * int(batch_y.size(0))

            probs = torch.sigmoid(logits)
            preds = probs >= 0.5
            correct = (preds == (batch_y >= 0.5))
            total_correct += int(correct.sum().item())
            total_elems += int(correct.numel())

        train_loss = total_loss / max(1, int(len(train_dataset)))
        train_acc = total_correct / max(1, total_elems)

        if X_val_tensor is not None and y_val_tensor is not None and y_val_tensor.numel() > 0:
            model.eval()
            with torch.no_grad():
                logits_val = model(X_val_tensor)
                probs_val = torch.sigmoid(logits_val)
                preds_val = probs_val >= 0.5
                correct_val = (preds_val == (y_val_tensor >= 0.5))
                val_acc = int(correct_val.sum().item()) / max(1, int(correct_val.numel()))
        else:
            val_acc = 0.0

        LOGGER.info(
            "train_attr_head_epoch",
            extra={
                "epoch": epoch,
                "epochs": epochs,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            },
        )

    state: dict[str, Any] = {
        "input_dim": input_dim,
        "num_attrs": num_attrs,
        "state_dict": model.state_dict(),
    }

    final_train_acc = float(train_acc)
    final_val_acc = float(val_acc)
    return state, final_train_acc, final_val_acc


def train_attribute_head(config: TrainConfig) -> TrainResult:
    """Top-level training entrypoint."""

    _set_random_seeds(config.seed)
    settings = load_settings()

    with open_primary_session(config.db_url) as session:
        X, y, attr_keys = _build_training_arrays(
            session=session,
            settings=settings,
            cache_root=config.cache_root,
            gt_path=config.gt_path,
            max_samples=config.max_samples,
            seed=config.seed,
        )

    X_train, y_train, X_val, y_val = _split_train_val(
        X=X,
        y=y,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(
        "train_attr_head_device",
        extra={"device": str(device)},
    )

    state, train_acc, val_acc = _train_attribute_head(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        device=device,
    )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "attr_keys": attr_keys,
            "state": state,
            "seed": config.seed,
        },
        config.output_path,
    )

    LOGGER.info(
        "train_attr_head_complete",
        extra={
            "output_path": str(config.output_path),
            "num_samples": int(X.shape[0]),
            "num_attrs": len(attr_keys),
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        },
    )

    return TrainResult(
        num_samples=int(X.shape[0]),
        num_attrs=len(attr_keys),
        attr_keys=attr_keys,
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        output_path=config.output_path,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-label attribute head from ground_truth_auto.jsonl."
    )
    parser.add_argument(
        "--gt",
        type=Path,
        required=True,
        help="Path to auto ground truth JSON/JSONL (e.g. tmp/ground_truth_auto.jsonl).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Primary PostgreSQL database URL. Defaults to databases.primary_url in settings.yaml.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="Cache root directory where embeddings are stored. Defaults to cache.root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/attribute_head_from_qwen.pt"),
        help="Output path for the trained head (PyTorch .pt file).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size for training (default: 256).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer (default: 1e-3).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of samples to reserve for validation (default: 0.1).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max number of samples to use for training (default: use all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling and training (default: 42).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()

    primary_target = args.db or settings.databases.primary_url
    cache_input = args.cache_root or settings.cache.root
    cache_root = resolve_cache_root(cache_input)

    config = TrainConfig(
        gt_path=args.gt,
        db_url=primary_target,
        cache_root=cache_root,
        output_path=args.output.resolve(),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
        val_fraction=float(args.val_fraction),
        max_samples=int(args.max_samples) if args.max_samples is not None else None,
        seed=int(args.seed),
    )

    result = train_attribute_head(config)
    LOGGER.info(
        "train_attr_head_summary",
        extra={
            "output_path": str(result.output_path),
            "num_samples": result.num_samples,
            "num_attrs": result.num_attrs,
            "train_accuracy": result.train_accuracy,
            "val_accuracy": result.val_accuracy,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


