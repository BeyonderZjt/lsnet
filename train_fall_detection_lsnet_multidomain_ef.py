import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from fall_detection_dataset_lsnet_multibranch_ef import (
    IDX_TO_CLASS_NAME,
    build_multidomain_dataloaders,
)
from model_lsnet_multi_ef import build_lsnet_multidomain_ef, count_flops, count_parameters


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    num_classes = len(IDX_TO_CLASS_NAME)
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for batch in loader:
        doppler = batch["doppler"].to(device, non_blocking=True)
        range_tensor = batch["range"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(doppler, range_tensor)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        for t, p in zip(labels.cpu(), preds.cpu()):
            conf[t.long(), p.long()] += 1

    val_loss = total_loss / max(total, 1)
    val_acc = correct / max(total, 1)

    recalls = []
    precisions = []
    f1s = []
    for c in range(num_classes):
        tp = conf[c, c].item()
        fn = conf[c, :].sum().item() - tp
        fp = conf[:, c].sum().item() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "loss": val_loss,
        "acc": val_acc,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": conf.tolist(),
    }




def parse_epoch_list(epoch_list_text: str) -> Set[int]:
    """Parse comma-separated epoch ids, e.g. "49" or "10,20,49"."""
    if not epoch_list_text.strip():
        return set()

    epochs: Set[int] = set()
    for raw in epoch_list_text.split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid epoch id in --save_epoch_list: '{token}'") from exc
        if value <= 0:
            raise ValueError(f"Epoch id must be positive, got {value}")
        epochs.add(value)
    return epochs

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        doppler = batch["doppler"].to(device, non_blocking=True)
        range_tensor = batch["range"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(doppler, range_tensor)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(correct / max(total,1)):.4f}")

    return running_loss / max(total, 1), correct / max(total, 1)


def main(args):
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader, val_loader = build_multidomain_dataloaders(
        root_dir=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_num_workers=args.val_num_workers,
        train_ratio=args.train_ratio,
        image_size=args.image_size,
        seed=args.seed,
    )

    model = build_lsnet_multidomain_ef(num_classes=args.num_classes, variant=args.variant)
    model.to(device)
    param_stats = count_parameters(model)
    print(
        "Model parameters: "
        f"total={param_stats['total']:,}, "
        f"trainable={param_stats['trainable']:,}, "
        f"non_trainable={param_stats['non_trainable']:,}"
    )
    if not args.no_flops:
        flops = count_flops(model, image_size=args.image_size)
        print(f"Model FLOPs (batch=1, dual-input): {flops:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    use_amp = torch.cuda.is_available() and not args.cpu and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epoch_ckpt_dir = Path(args.output_dir) / "epoch_checkpoints"
    target_epochs = parse_epoch_list(args.save_epoch_list)
    if args.save_epoch_interval > 0 or target_epochs:
        epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    best_epoch = -1
    metrics_log = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        val_metrics = evaluate(model=model, loader=val_loader, criterion=criterion, device=device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_acc": train_acc,
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
        }
        metrics_log.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, Path(args.output_dir) / "last.pth")

        should_save_by_interval = args.save_epoch_interval > 0 and epoch % args.save_epoch_interval == 0
        should_save_by_list = epoch in target_epochs
        if should_save_by_interval or should_save_by_list:
            epoch_ckpt_path = epoch_ckpt_dir / f"epoch_{epoch:03d}.pth"
            torch.save(ckpt, epoch_ckpt_path)
            print(f"Epoch checkpoint saved to: {epoch_ckpt_path}")

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            best_epoch = epoch
            torch.save(ckpt, Path(args.output_dir) / "best.pth")

    with open(Path(args.output_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, ensure_ascii=False, indent=2)

    print(f"Training done. best_val_acc={best_acc:.4f} at epoch={best_epoch}")
    print(f"Best model saved to: {Path(args.output_dir) / 'best.pth'}")


def get_args():
    parser = argparse.ArgumentParser("LSNet multidomain early-fusion fall/action detection")
    parser.add_argument("--dataset_root", type=str, required=True, help="包含 image/ 和 Range_Time/ 的根目录")
    parser.add_argument("--output_dir", type=str, default="runs/lsnet_multidomain_ef")

    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--variant", type=str, default="lsnet_ef_nano", choices=["lsnet_ef_nano", "lsnet_ef_micro", "lsnet_ef_tiny"])

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_num_workers", type=int, default=0, help="验证集 DataLoader worker 数，默认 0 以避免 worker 崩溃")
    parser.add_argument("--train_ratio", type=float, default=0.8)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_flops", action="store_true", help="跳过 FLOPs 统计")
    parser.add_argument(
        "--save_epoch_interval",
        type=int,
        default=0,
        help="按间隔保存 epoch checkpoint，0 表示关闭；例如 5 表示每 5 个 epoch 保存一次",
    )
    parser.add_argument(
        "--save_epoch_list",
        type=str,
        default="",
        help="按指定 epoch 保存 checkpoint，例如 '49' 或 '10,20,49'",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
