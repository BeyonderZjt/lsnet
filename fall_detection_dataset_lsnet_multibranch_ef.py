import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as F


CLASS_MAP: Dict[str, int] = {
    "1": 0,  # walking
    "2": 1,  # sitting down
    "3": 2,  # standing up
    "4": 3,  # pick up an object
    "5": 4,  # drink water
    "6": 5,  # fall
}

IDX_TO_CLASS_NAME: Dict[int, str] = {
    0: "walking",
    1: "sitting down",
    2: "standing up",
    3: "pick up an object",
    4: "drink water",
    5: "fall",
}


@dataclass
class DomainCrop:
    left_ratio: float
    top_ratio: float
    right_ratio: float
    bottom_ratio: float


def _crop_by_ratio(image: Image.Image, crop_cfg: DomainCrop) -> Image.Image:
    w, h = image.size
    left = int(w * crop_cfg.left_ratio)
    top = int(h * crop_cfg.top_ratio)
    right = int(w * crop_cfg.right_ratio)
    bottom = int(h * crop_cfg.bottom_ratio)
    right = max(left + 1, right)
    bottom = max(top + 1, bottom)
    return F.crop(image, top=top, left=left, height=bottom - top, width=right - left)


class MultiDomainSpectrogramDataset(Dataset):
    """
    输入前融合的数据集：
    - 多普勒-时间谱图目录: <root>/image/{1..6}
    - 距离-时间谱图目录: <root>/Range_Time/{1..6}
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        train_ratio: float = 0.8,
        image_size: int = 224,
        doppler_crop: DomainCrop = DomainCrop(0.03, 0.08, 0.97, 0.92),
        range_crop: DomainCrop = DomainCrop(0.05, 0.06, 0.95, 0.94),
        seed: int = 42,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.doppler_crop = doppler_crop
        self.range_crop = range_crop

        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unsupported split={split}")

        self.samples = self._gather_samples(train_ratio=train_ratio, seed=seed)

        tfms: List[transforms.Compose] = [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ]
        if normalize:
            tfms.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        self.to_tensor = transforms.Compose(tfms)

    def _gather_samples(self, train_ratio: float, seed: int) -> List[Tuple[Path, Path, int]]:
        doppler_root = self.root_dir / "image"
        range_root = self.root_dir / "Range_Time"
        if not doppler_root.exists() or not range_root.exists():
            raise FileNotFoundError(
                f"Dataset directory must contain 'image' and 'Range_Time'. Got: {self.root_dir}"
            )

        all_samples: List[Tuple[Path, Path, int]] = []
        rng = random.Random(seed)

        for class_dir_name, label in CLASS_MAP.items():
            d_dir = doppler_root / class_dir_name
            r_dir = range_root / class_dir_name
            if not d_dir.exists() or not r_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {d_dir} or {r_dir}")

            d_imgs = sorted([p for p in d_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
            r_imgs = sorted([p for p in r_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])

            if len(d_imgs) != len(r_imgs):
                raise ValueError(
                    f"Class {class_dir_name} image count mismatch: Doppler={len(d_imgs)} Range={len(r_imgs)}"
                )

            class_pairs: List[Tuple[Path, Path, int]] = []
            for dp, rp in zip(d_imgs, r_imgs):
                if dp.stem != rp.stem:
                    # 如果命名不一致，依然按照排序配对，但给出显式报错，避免错配。
                    raise ValueError(
                        f"Filename mismatch under class {class_dir_name}: {dp.name} vs {rp.name}"
                    )
                class_pairs.append((dp, rp, label))

            rng.shuffle(class_pairs)

            if self.split == "all":
                selected = class_pairs
            else:
                cut = int(len(class_pairs) * train_ratio)
                cut = min(max(cut, 1), len(class_pairs) - 1)
                if self.split == "train":
                    selected = class_pairs[:cut]
                else:  # val / test
                    selected = class_pairs[cut:]

            all_samples.extend(selected)

        rng.shuffle(all_samples)
        return all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        doppler_path, range_path, label = self.samples[index]

        doppler_img = Image.open(doppler_path).convert("RGB")
        range_img = Image.open(range_path).convert("RGB")

        doppler_img = _crop_by_ratio(doppler_img, self.doppler_crop)
        range_img = _crop_by_ratio(range_img, self.range_crop)

        doppler_tensor = self.to_tensor(doppler_img)
        range_tensor = self.to_tensor(range_img)

        return {
            "doppler": doppler_tensor,
            "range": range_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "doppler_path": str(doppler_path),
            "range_path": str(range_path),
        }


def build_multidomain_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    val_num_workers: int | None = None,
    train_ratio: float = 0.8,
    image_size: int = 224,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = MultiDomainSpectrogramDataset(
        root_dir=root_dir,
        split="train",
        train_ratio=train_ratio,
        image_size=image_size,
        seed=seed,
    )
    val_dataset = MultiDomainSpectrogramDataset(
        root_dir=root_dir,
        split="val",
        train_ratio=train_ratio,
        image_size=image_size,
        seed=seed,
    )

    if val_num_workers is None:
        val_num_workers = num_workers

    train_loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    val_loader_kwargs = {
        "dataset": val_dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": val_num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
    if val_num_workers > 0:
        val_loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(**val_loader_kwargs)
    return train_loader, val_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_num_workers", type=int, default=0)
    args = parser.parse_args()

    train_loader, val_loader = build_multidomain_dataloaders(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_num_workers=args.val_num_workers,
    )

    b = next(iter(train_loader))
    print("Train batch doppler:", b["doppler"].shape)
    print("Train batch range:", b["range"].shape)
    print("Train batch label:", b["label"].shape)
    print("Train samples:", len(train_loader.dataset), "Val samples:", len(val_loader.dataset))
