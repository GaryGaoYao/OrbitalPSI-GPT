# -*- coding: utf-8 -*-
import os
import json
from glob import glob
from typing import Dict, List, Tuple, Optional

import math
import random
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import timm  # pip install timm

# =========================
# 配置区：改这里即可 —— Heatmap 版
# =========================
CONFIG = {
    "images_dir": r"D:\Codes\Skull_Landmarks_TL\Models-TIMM\data\images-all",
    "labels_dir": r"D:\Codes\Skull_Landmarks_TL\Models-TIMM\data\labels_xy",

    # 训练与日志
    "img_size": 512,               # 输入网络的正方形尺寸
    "batch_size": 96,
    "lr": 3e-4,
    "epochs": 250,
    "val_ratio": 0.15,
    "seed": 2024,
    "augment": True,
    "save_dir": r"D:\Codes\Skull_Landmarks_TL\Models-TIMM\runs\exp_heatmap",
    "num_workers": 8,      # Windows/Thonny 卡住设 0
    "pin_memory": True,
    "tb_samples": 2,

    # 关键点
    "num_points": 12,

    # Heatmap 参数
    "heatmap_size": 128,   # 通常为 img_size / 4（或 /8），与 decoder 输出匹配
    "sigma": 2.5,          # 高斯半径（单位：heatmap 像素）

    # 单骨干名（当 backbones 列表为空时使用）
    "backbone_name": "convnext_tiny",
    "pretrained": True,

    "backbones": [
        "densenet121"
    ],

    # 数据增强（几何）强度
    "geom_aug": {
        "do_flip": False,
        "do_affine": False,
        "max_rotate_deg": 8,
        "scale_range": (0.95, 1.05),
        "translate_frac": 0.03
    },

    # 评估阈值
    "pck_px": (2, 4, 8),
    "pck_norm": (0.01, 0.02),

    # 学习率调度器
    "scheduler": {
        "use": True,
        "type": "cosine",     # "cosine" 或 None
        "eta_min": 1e-6
    },

    # 早停
    "early_stop": {
        "use": True,
        "patience": 30,
        "min_delta": 0.0,
        "warmup": 250
    },
}

# --------------------
# 实用函数
# --------------------
def set_seed(seed: int = 2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def draw_points_on_image(img: Image.Image, points_xy_norm: np.ndarray, color=(255, 0, 0)):
    """在 PIL 图像上画归一化(0~1)坐标点（仅可视化）。"""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    pts = points_xy_norm.reshape(-1, 2)
    for i, (xn, yn) in enumerate(pts):
        x = float(xn) * w
        y = float(yn) * h
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
        draw.text((x + 4, y + 2), str(i + 1), fill=color)

def sample_affine_params(orig_w: int, orig_h: int, cfg: dict) -> Tuple[float, float, float, float]:
    max_deg = cfg.get("max_rotate_deg", 0)
    min_s, max_s = cfg.get("scale_range", (1.0, 1.0))
    tfrac = cfg.get("translate_frac", 0.0)
    angle = random.uniform(-max_deg, max_deg)
    scale = random.uniform(min_s, max_s)
    max_dx = orig_w * tfrac
    max_dy = orig_h * tfrac
    dx = random.uniform(-max_dx, max_dx)
    dy = random.uniform(-max_dy, max_dy)
    return angle, scale, dx, dy

def apply_affine_to_points(pts_xy: np.ndarray, angle_deg: float, scale: float, dx: float, dy: float, cx: float, cy: float) -> np.ndarray:
    pts = pts_xy.reshape(-1, 2).copy()
    rad = math.radians(angle_deg)
    cosA = math.cos(rad) * scale
    sinA = math.sin(rad) * scale
    pts[:, 0] -= cx
    pts[:, 1] -= cy
    x_new = pts[:, 0] * cosA - pts[:, 1] * sinA
    y_new = pts[:, 0] * sinA + pts[:, 1] * cosA
    pts[:, 0] = x_new + cx + dx
    pts[:, 1] = y_new + cy + dy
    return pts.reshape(-1)

def gaussian_2d(shape, center, sigma):
    """在给定 heatmap shape 上生成一个以 center 为中心的 2D 高斯核。"""
    H, W = shape
    y = np.arange(0, H, 1, np.float32)
    x = np.arange(0, W, 1, np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    cy, cx = center
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    return g

def make_heatmaps(points_xy_norm: np.ndarray, heatmap_size: int, sigma: float) -> np.ndarray:
    """从归一化坐标生成 K 张 heatmap（H=heatmap_size, W=heatmap_size）。"""
    K = points_xy_norm.size // 2
    hm = np.zeros((K, heatmap_size, heatmap_size), dtype=np.float32)
    pts = points_xy_norm.reshape(-1, 2)
    for i, (xn, yn) in enumerate(pts):
        # 边界外的点（如果有）直接跳过
        if xn < 0 or xn > 1 or yn < 0 or yn > 1:
            continue
        cx = xn * (heatmap_size - 1)
        cy = yn * (heatmap_size - 1)
        g = gaussian_2d((heatmap_size, heatmap_size), (cy, cx), sigma=sigma)
        hm[i] = np.clip(np.maximum(hm[i], g), 0.0, 1.0)
    return hm

def decode_heatmaps(hm: torch.Tensor, img_size: int) -> torch.Tensor:
    """
    从预测热力图解码关键点坐标（归一化 0~1）。
    hm: [B, K, Hm, Wm]
    返回: [B, K*2]，归一化坐标
    """
    B, K, Hm, Wm = hm.shape
    # Argmax
    hm_reshaped = hm.view(B, K, -1)
    idx = hm_reshaped.argmax(dim=-1)  # [B, K]
    y = (idx // Wm).float()
    x = (idx %  Wm).float()

    # 可选：二次插值细化（soft-argmax 也可）
    # 这里用一个简单的 3x3 窗口的二次插值近似（可略）
    # 为简洁起见暂省略，已有 Argmax 已足够稳定

    # 归一化到 0~1 空间（相对原图）
    x_norm = x / (Wm - 1)
    y_norm = y / (Hm - 1)
    coords = torch.stack([x_norm, y_norm], dim=-1).view(B, K * 2)
    return coords

# --------------------
# 数据集（先做几何增强 & 缩放，再生成热图）
# --------------------
class HeatmapLandmarkDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, img_size: int,
                 expected_pts: int, augment: bool, geom_cfg: dict,
                 heatmap_size: int, sigma: float):
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.expected_pts = expected_pts
        self.augment = augment
        self.geom_cfg = geom_cfg or {}
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        img_paths = []
        for e in exts:
            img_paths.extend(glob(os.path.join(images_dir, e)))
        img_paths = sorted(img_paths)
        if not img_paths:
            raise RuntimeError(f"images_dir 里未找到图片：{images_dir}")

        # label 索引： filename -> json path
        self.label_index: Dict[str, str] = {}
        json_paths = []
        for pat in ("*.json", "*.JSON"):
            json_paths.extend(glob(os.path.join(labels_dir, pat)))
        if not json_paths:
            raise RuntimeError(f"labels_dir 里未找到任何 JSON：{labels_dir}")

        for jp in json_paths:
            try:
                with open(jp, "r", encoding="utf-8-sig") as f:
                    obj = json.load(f)
                fn = obj.get("filename")
                if isinstance(fn, str) and fn.strip():
                    self.label_index[fn] = jp
            except Exception as e:
                print(f"[WARN] 解析标签失败：{jp} - {e}")

        self.samples = []
        miss = 0
        for ip in img_paths:
            base = os.path.basename(ip)
            if base in self.label_index:
                self.samples.append((ip, self.label_index[base]))
            else:
                stem = os.path.splitext(base)[0]
                candidate = None
                for k in self.label_index.keys():
                    if os.path.splitext(k)[0] == stem:
                        candidate = self.label_index[k]
                        break
                if candidate:
                    self.samples.append((ip, candidate))
                else:
                    miss += 1

        if not self.samples:
            raise RuntimeError("未匹配到任何 (图片, 标签) 对。请检查 labels 中 JSON 的 'filename' 字段是否与图片名对应。")
        if miss > 0:
            print(f"[INFO] 有 {miss} 张图片未在 labels 中找到匹配标签。")

        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        self.color_aug = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)

    def __len__(self):
        return len(self.samples)

    def _resize_points(self, pts_xy: np.ndarray, orig_w: int, orig_h: int, new_w: int, new_h: int) -> np.ndarray:
        pts = pts_xy.copy().reshape(-1, 2)
        pts[:, 0] = pts[:, 0] * (new_w / orig_w)
        pts[:, 1] = pts[:, 1] * (new_h / orig_h)
        return pts.reshape(-1)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        with open(json_path, "r", encoding="utf-8-sig") as f:
            obj = json.load(f)
        pts_list = obj.get("points", None)
        if not isinstance(pts_list, list):
            raise ValueError(f"{json_path} 缺少 'points' 数组")
        if len(pts_list) != self.expected_pts * 2:
            raise ValueError(f"{json_path} 点维度为 {len(pts_list)}，不等于 {self.expected_pts * 2}")

        pts_xy = np.array(pts_list, dtype=np.float32)

        # --- 几何增强：与图同步 ---
        if self.augment:
            if self.geom_cfg.get("do_flip", True) and random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                pts = pts_xy.reshape(-1, 2)
                pts[:, 0] = orig_w - pts[:, 0]
                pts_xy = pts.reshape(-1)

            if self.geom_cfg.get("do_affine", True) and (
                self.geom_cfg.get("max_rotate_deg", 0) > 0
                or self.geom_cfg.get("scale_range", (1,1)) != (1,1)
                or self.geom_cfg.get("translate_frac", 0) > 0
            ):
                angle, scale, dx, dy = sample_affine_params(orig_w, orig_h, self.geom_cfg)
                center = (orig_w * 0.5, orig_h * 0.5)
                img = TF.affine(img, angle=angle, translate=(dx, dy), scale=scale,
                                shear=[0.0, 0.0], center=center)
                pts_xy = apply_affine_to_points(pts_xy, angle, scale, dx, dy, cx=center[0], cy=center[1])

        # --- 缩放到固定尺寸 ---
        new_size = (CONFIG["img_size"], CONFIG["img_size"])
        img = img.resize(new_size, Image.BILINEAR)
        # 先把点缩放到新图尺寸（像素坐标）
        pts_xy_resized = self._resize_points(pts_xy, orig_w, orig_h, new_size[0], new_size[1])

        # --- 颜色增强 ---
        if self.augment:
            img = self.color_aug(img)

        img_t = self.to_tensor(img)
        img_t = self.norm(img_t)

        # --- 归一化坐标（相对新图）→ 生成热力图 ---
        pts_xy_norm = pts_xy_resized.copy().reshape(-1, 2)
        pts_xy_norm[:, 0] /= new_size[0]
        pts_xy_norm[:, 1] /= new_size[1]
        pts_xy_norm = pts_xy_norm.reshape(-1)

        heatmaps = make_heatmaps(pts_xy_norm, self.heatmap_size, self.sigma)  # [K, Hm, Wm]
        hm_t = torch.from_numpy(heatmaps.astype(np.float32))                   # float32

        # 同时返回 GT 归一化坐标（用于 TensorBoard 可视化/评估）
        target_coords_norm = torch.from_numpy(pts_xy_norm.astype(np.float32))  # [K*2]
        return img_t, hm_t, target_coords_norm, os.path.basename(img_path)

# --------------------
# 模型（timm 骨干 + 轻量上采样 decoder → heatmaps）
# --------------------
class HeatmapHead(nn.Module):
    """
    将骨干输出的全局特征映射回空间分辨率。
    这里使用简单的转置卷积堆叠把特征上采样到 heatmap_size。
    也可以换成更强的 U-Net/FPN/HRNet（此处保持轻量通用）。
    """
    def __init__(self, in_ch: int, out_kpts: int, hm_size: int):
        super().__init__()
        self.hm_size = hm_size
        # 假设骨干输出是 [B, C, H/32, W/32] 或 [B, C, 1, 1]（具体看骨干）
        # 这里用一个通用的“自适应上采样到 hm_size”的方案：
        self.proj = nn.Conv2d(in_ch, 256, kernel_size=1, bias=False)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(64,  64,  kernel_size=4, stride=2, padding=1), nn.ReLU(True),
        )
        self.head = nn.Conv2d(64, out_kpts, kernel_size=1)

    def forward(self, feat: torch.Tensor):
        # feat: [B, C] or [B, C, h, w]
        if feat.ndim == 2:
            # [B, C] -> [B, C, 1, 1]
            feat = feat[:, :, None, None]
        x = self.proj(feat)
        # 逐步上采样到接近 hm_size
        x = self.up(x)
        # 最后插值到精确 hm_size
        x = nn.functional.interpolate(x, size=(self.hm_size, self.hm_size), mode="bilinear", align_corners=False)
        out = self.head(x)  # [B, K, Hm, Wm]
        return out

class TimmBackboneHeatmap(nn.Module):
    def __init__(self, kpts: int, name="convnext_tiny", hm_size=128, pretrained=True):
        super().__init__()
        # 使用 timm 骨干，取中间特征（不要 global_pool 输出向量）
        # 简单做法：num_classes=0, global_pool="" 可让很多 timm 返回 [B, C, H', W']
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
        # 估计通道数
        if hasattr(self.backbone, "num_features"):
            in_ch = self.backbone.num_features
        else:
            # 兜底：尝试一次前向来探测通道数（需要一张假图）
            with torch.no_grad():
                dummy = torch.zeros(1, 3, CONFIG["img_size"], CONFIG["img_size"])
                feat = self.backbone(dummy)
                in_ch = feat.shape[1] if feat.ndim == 4 else feat.shape[1]
        self.head = HeatmapHead(in_ch=in_ch, out_kpts=kpts, hm_size=hm_size)

    def forward(self, x):
        feat = self.backbone(x)  # [B, C, h, w] or [B, C]
        hm = self.head(feat)     # [B, K, Hm, Wm]
        return hm

# --------------------
# 训练/验证/指标（Heatmap 版）
# --------------------
def train_one_epoch(model, loader, optimizer, device, scaler: torch.cuda.amp.GradScaler = None):
    model.train()
    running = 0.0
    for imgs, hms, _, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        hms = hms.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                loss = nn.functional.mse_loss(preds, hms, reduction="mean")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(imgs)
            loss = nn.functional.mse_loss(preds, hms, reduction="mean")
            loss.backward()
            optimizer.step()

        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    running = 0.0
    for imgs, hms, _, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        hms = hms.to(device, non_blocking=True)
        preds = model(imgs)
        loss = nn.functional.mse_loss(preds, hms, reduction="mean")
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def eval_metrics_full(model, loader, device, img_size=512, pck_px=(2,4,8), pck_norm=(0.01, 0.02)):
    model.eval()
    mse_running, n_samples = 0.0, 0
    per_point_px_all = []
    per_sample_mean_px = []
    per_point_nrm_all = []

    for imgs, _, gts_norm, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        gts_norm = gts_norm.to(device, non_blocking=True)  # [B, K*2]

        preds_hm = model(imgs)                   # [B, K, Hm, Wm]
        preds_norm = decode_heatmaps(preds_hm, img_size)  # [B, K*2]

        # 评价 MSE 取坐标空间的 MSE（可选：也可记录 heatmap MSE）
        mse = nn.functional.mse_loss(preds_norm, gts_norm, reduction="mean")
        mse_running += mse.item() * imgs.size(0)
        n_samples += imgs.size(0)

        B = preds_norm.size(0)
        pred_xy = preds_norm.view(B, -1, 2) * img_size
        gt_xy   = gts_norm.view(B, -1, 2) * img_size
        d_px = (pred_xy - gt_xy).pow(2).sum(dim=-1).sqrt()  # [B, K]

        per_point_px_all.append(d_px.cpu())
        per_sample_mean_px.extend(d_px.mean(dim=1).cpu().tolist())

        diag = math.sqrt(img_size**2 + img_size**2)
        d_nrm = d_px / diag
        per_point_nrm_all.append(d_nrm.cpu())

    if n_samples == 0:
        return {}

    per_point_px_all = torch.cat(per_point_px_all, dim=0).numpy().ravel()
    per_point_nrm_all = torch.cat(per_point_nrm_all, dim=0).numpy().ravel()

    avg_mse = (mse_running / n_samples)
    mean_px = float(np.mean(per_point_px_all))
    median_px = float(np.median(per_point_px_all))
    mean_nme = float(np.mean(per_point_nrm_all))
    median_nme = float(np.median(per_point_nrm_all))

    pck_px_dict = {f"PCK@{t}px": float(np.mean(per_point_px_all < t)) for t in pck_px}
    pck_norm_dict = {f"PCK@{int(r*100)}%diag": float(np.mean(per_point_nrm_all < r)) for r in pck_norm}

    Tthr = img_size * 0.05
    grid = np.linspace(0, Tthr, 50)
    pcks = [np.mean(per_point_px_all < g) for g in grid]
    auc_pck = float(np.trapz(pcks, grid) / Tthr)  # 0~1

    metrics = {
        "CoordMSE": avg_mse,
        "MeanPx": mean_px,
        "MedianPx": median_px,
        "MeanNME": mean_nme,
        "MedianNME": median_nme,
        "AUC-PCK(0..5% diag)": auc_pck
    }
    metrics.update(pck_px_dict)
    metrics.update(pck_norm_dict)
    return metrics

@torch.no_grad()
def tb_add_images(writer: SummaryWriter, tag: str, imgs_t, preds_norm, gts_norm, epoch, max_n=2):
    """写入可视化：原图+预测/GT 点（归一化 → 图像像素）"""
    to_pil = T.ToPILImage()
    inv_norm = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    n = min(max_n, imgs_t.size(0))
    for i in range(n):
        img_vis = inv_norm(imgs_t[i].cpu()).clamp(0, 1)
        img_vis = to_pil(img_vis)

        pred_img = img_vis.copy()
        draw_points_on_image(pred_img, preds_norm[i].cpu().numpy().reshape(-1), color=(255, 0, 0))
        gt_img = img_vis.copy()
        draw_points_on_image(gt_img, gts_norm[i].cpu().numpy().reshape(-1), color=(0, 200, 0))

        to_t = T.ToTensor()
        writer.add_image(f"{tag}/pred_{i}", to_t(pred_img), epoch)
        writer.add_image(f"{tag}/gt_{i}", to_t(gt_img), epoch)

# =========================
# 单个骨干：完整训练
# =========================
def train_one_backbone(cfg, backbone_name: str):
    set_seed(cfg["seed"] + (hash(backbone_name) % 1000))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    use_amp = torch.cuda.is_available()

    save_dir = os.path.join(cfg["save_dir"], backbone_name)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs"))

    print(f"\n===== Backbone: {backbone_name} (Heatmap) =====")
    print("torch:", torch.__version__, " cuda.available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("save_dir:", save_dir)

    ds = HeatmapLandmarkDataset(
        images_dir=cfg["images_dir"],
        labels_dir=cfg["labels_dir"],
        img_size=cfg["img_size"],
        expected_pts=cfg["num_points"],
        augment=cfg["augment"],
        geom_cfg=cfg["geom_aug"],
        heatmap_size=cfg["heatmap_size"],
        sigma=cfg["sigma"]
    )
    val_len = max(1, int(len(ds) * cfg["val_ratio"]))
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(cfg["seed"]))

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
        persistent_workers=cfg["num_workers"] > 0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
        persistent_workers=cfg["num_workers"] > 0
    )

    model = TimmBackboneHeatmap(
        kpts=cfg["num_points"], name=backbone_name,
        hm_size=cfg["heatmap_size"], pretrained=cfg["pretrained"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    scheduler = None
    if cfg["scheduler"].get("use") and cfg["scheduler"].get("type") == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg["epochs"],
            eta_min=cfg["scheduler"].get("eta_min", 1e-6)
        )
        print(f"[Scheduler] CosineAnnealingLR(T_max={cfg['epochs']}, eta_min={cfg['scheduler'].get('eta_min',1e-6)})")

    # 早停
    best_val = float("inf")
    best_path = os.path.join(save_dir, f"best_{backbone_name}.pth")
    no_improve = 0
    warmup = cfg["early_stop"].get("warmup", 0)
    patience = cfg["early_stop"].get("patience", 20)
    min_delta = cfg["early_stop"].get("min_delta", 0.0)
    use_early = cfg["early_stop"].get("use", False)

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss = eval_one_epoch(model, val_loader, device)

        # 评估（把热力图解码为坐标再评）
        metrics = eval_metrics_full(model, val_loader, device, img_size=cfg["img_size"],
                                    pck_px=cfg["pck_px"], pck_norm=cfg["pck_norm"])

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LR", current_lr, epoch)

        print(f"[{backbone_name}] Epoch {epoch}/{cfg['epochs']} | "
              f"lr={current_lr:.3e}  trainHM={tr_loss:.6f}  valHM={val_loss:.6f}  "
              f"val_px={metrics.get('MeanPx',0):.2f}px  PCK@4px={metrics.get('PCK@4px',0):.3f}")

        # 标量日志
        writer.add_scalar("LossHM/train", tr_loss, epoch)
        writer.add_scalar("LossHM/val", val_loss, epoch)
        if metrics:
            for k, v in metrics.items():
                writer.add_scalar(f"Val/{k}", v, epoch)

        # 图片可视化（pred vs gt）
        with torch.no_grad():
            for imgs_t, _, gts_norm, _ in val_loader:
                imgs_t = imgs_t.to(device, non_blocking=True)
                preds_hm = model(imgs_t)
                preds_norm = decode_heatmaps(preds_hm, cfg["img_size"])
                tb_add_images(writer, "Images", imgs_t, preds_norm, gts_norm.to(device), epoch, max_n=cfg["tb_samples"])
                break

        # 保存最优（按 val heatmap MSE 判定）
        improved = (best_val - val_loss) > min_delta
        if improved:
            best_val = val_loss
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_hm_mse": val_loss,
                "img_size": cfg["img_size"],
                "heatmap_size": cfg["heatmap_size"],
                "sigma": cfg["sigma"],
                "backbone": backbone_name
            }, best_path)
            print(f"  -> ✅ 保存最优权重: {best_path}")
        else:
            no_improve += 1

        if scheduler is not None:
            scheduler.step()

        if use_early and epoch > warmup and no_improve >= patience:
            print(f"⏹️  触发早停：连续 {no_improve} 个 epoch 验证集无提升（min_delta={min_delta}），在第 {epoch} 个 epoch 停止。")
            writer.add_text("EarlyStop",
                            f"Stopped at epoch {epoch}, best_valHM={best_val:.6f}", epoch)
            break

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    writer.close()
    del model
    torch.cuda.empty_cache()
    print(f"[{backbone_name}] Done. Logs: {os.path.join(save_dir, 'logs')}")

# =========================
# main：遍历骨干逐个训练
# =========================
def main():
    cfg = CONFIG
    os.makedirs(cfg["save_dir"], exist_ok=True)

    backbones = cfg.get("backbones", []) or [cfg["backbone_name"]]
    print("将依次训练这些骨干（Heatmap）：", backbones)

    for name in backbones:
        try:
            train_one_backbone(cfg, name)
        except Exception as e:
            import traceback
            print(f"❌ 训练 {name} 失败：{e}")
            traceback.print_exc()
            torch.cuda.empty_cache()

    print("\n全部骨干已跑完。一次查看所有实验日志：")
    print(f'tensorboard --logdir "{cfg["save_dir"]}"')

if __name__ == "__main__":
    main()

# 用法：
# 1) 改 CONFIG 路径；可把 heatmap_size 设为 img_size/4（如 128），sigma=2~4 常用
# 2) 运行本脚本开始训练
# 3) 打开 TensorBoard 查看：tensorboard --logdir "D:\Codes\Skull_Landmarks_TL\Models-TIMM\runs\exp_timm_small_finetune_heatmap"
