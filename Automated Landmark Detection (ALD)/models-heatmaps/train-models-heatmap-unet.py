# -*- coding: utf-8 -*-
"""
train_unet_heatmap.py  —— 仅 U-Net 的关键点热图回归训练脚本
- 热图损失: "mse" 或 "kl"（概率式，推荐）
- 坐标解码: "softargmax" 或 "argmax"
- 写入 TensorBoard: LossHM/*、Val/*、Images/*、Heatmap/*
"""

import os
import json
from glob import glob
from typing import Dict, Tuple

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

# =========================
# 配置区：改这里即可
# =========================
CONFIG = {
    "images_dir": r"D:\Codes\Skull_Landmarks_TL\Models-TIMM\data\images-all",
    "labels_dir": r"D:\Codes\Skull_Landmarks_TL\Models-TIMM\data\labels_xy",

    # 训练与日志
    "img_size": 512,                # 输入网络尺寸（正方形）
    "batch_size": 48,               # U-Net 显存占用较大，先小点
    "lr": 3e-4,
    "epochs": 250,
    "val_ratio": 0.15,
    "seed": 2024,
    "augment": True,
    "save_dir": r"D:\Codes\Skull_Landmarks_TL\Models-TIMM\runs\exp_heatmap",
    "run_name": "unet",             # 日志子目录名
    "num_workers": 8,               # Windows/Thonny 卡住可设 0
    "pin_memory": True,
    "tb_samples": 2,                # 每轮可视化图片数量

    # 关键点
    "num_points": 12,

    # Heatmap 参数
    "heatmap_size": 128,            # 通常为 img_size / 4 或 /8
    "sigma": 3.0,                   # 高斯 σ（heatmap 像素）

    # 损失与解码
    "heat_loss": "kl",              # "mse" 或 "kl"（推荐 "kl"）
    "decode_kind": "softargmax",    # "softargmax" 或 "argmax"
    "softargmax_beta": 10.0,        # softmax 温度，越大峰越尖（5~15 常用）

    # 评估阈值
    "pck_px": (2, 4, 8),
    "pck_norm": (0.01, 0.02),

    # 数据增强（几何）
    "geom_aug": {
        "do_flip": False,
        "do_affine": False,
        "max_rotate_deg": 8,
        "scale_range": (0.95, 1.05),
        "translate_frac": 0.03
    },

    # 学习率调度器
    "scheduler": {
        "use": True,
        "type": "cosine",
        "eta_min": 1e-6
    },

    # 早停
    "early_stop": {
        "use": True,
        "patience": 30,
        "min_delta": 0.0,
        "warmup": 150
    },

    # 可视化热图（第 0 关键点）
    "tb_show_heat": True,
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
        if xn < 0 or xn > 1 or yn < 0 or yn > 1:
            continue
        cx = xn * (heatmap_size - 1)
        cy = yn * (heatmap_size - 1)
        g = gaussian_2d((heatmap_size, heatmap_size), (cy, cx), sigma=sigma)
        hm[i] = np.clip(np.maximum(hm[i], g), 0.0, 1.0)
    return hm

# ---------- 坐标解码 ----------
@torch.no_grad()
def softargmax2d(logits: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    """
    logits: (B,K,H,W)
    返回: (B,K,2) 的归一化坐标 (x_norm, y_norm)
    """
    B, K, H, W = logits.shape
    flat = logits.view(B, K, -1) * beta
    prob = torch.softmax(flat, dim=-1)  # (B,K,HW)

    ys = torch.linspace(0, 1, H, device=logits.device)
    xs = torch.linspace(0, 1, W, device=logits.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1).view(-1, 2)  # (HW,2)
    coords = torch.matmul(prob, grid)                 # (B,K,2)
    return coords

@torch.no_grad()
def argmax_decode(hm: torch.Tensor) -> torch.Tensor:
    """
    hm: (B,K,H,W) —— 概率或 logits 均可
    返回: (B,K,2) 归一化坐标
    """
    B, K, H, W = hm.shape
    idx = hm.view(B, K, -1).argmax(dim=-1)  # [B, K]
    y = (idx // W).float()
    x = (idx %  W).float()
    x_norm = x / (W - 1)
    y_norm = y / (H - 1)
    coords = torch.stack([x_norm, y_norm], dim=-1)    # (B,K,2)
    return coords

def decode_heatmaps(hm: torch.Tensor, kind: str, beta: float) -> torch.Tensor:
    if kind == "softargmax":
        return softargmax2d(hm, beta=beta)
    elif kind == "argmax":
        return argmax_decode(hm)
    else:
        raise ValueError(f"Unknown decode_kind: {kind}")

# --------------------
# 数据集
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
                img = TF.affine(img, angle=angle, translate=(int(dx), int(dy)), scale=scale,
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

        # 同时返回 GT 归一化坐标（用于评估/可视化）
        target_coords_norm = torch.from_numpy(pts_xy_norm.astype(np.float32))  # [K*2]
        return img_t, hm_t, target_coords_norm, os.path.basename(img_path)

# --------------------
# 模型（U-Net）
# --------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNetHeatmap(nn.Module):
    """
    经典 U-Net：输入 (B,3,img_size,img_size)，输出 (B,K,heatmap_size,heatmap_size)
    """
    def __init__(self, num_kpts: int, img_size: int, heatmap_size: int,
                 channels=(64,128,256,512), in_ch: int = 3):
        super().__init__()
        c1, c2, c3, c4 = channels

        # Encoder
        self.enc1 = DoubleConv(in_ch, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c3, c4)

        # Decoder
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)

        self.outc = nn.Conv2d(c1, num_kpts, 1)
        self.heatmap_size = heatmap_size

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)              # (B,c1,H,W)
        e2 = self.enc2(self.pool1(e1)) # (B,c2,H/2,W/2)
        e3 = self.enc3(self.pool2(e2)) # (B,c3,H/4,W/4)
        b  = self.bottleneck(self.pool3(e3))  # (B,c4,H/8,W/8)

        # Decoder + skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        heat = self.outc(d1)  # (B,K,H,W) 与输入同尺寸
        if heat.shape[-1] != self.heatmap_size:
            heat = nn.functional.interpolate(
                heat, size=(self.heatmap_size, self.heatmap_size), mode="bilinear", align_corners=False
            )
        return heat

# --------------------
# 损失
# --------------------
def heatmap_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "mse") -> torch.Tensor:
    """
    pred/target: (B,K,H,W)
    kind: "mse" | "kl"
    """
    if kind == "mse":
        return nn.functional.mse_loss(pred, target, reduction="mean")
    elif kind == "kl":
        # 概率式：pred 做 log_softmax，target 归一化为概率分布
        B, K, H, W = pred.shape
        logP = torch.log_softmax(pred.view(B, K, -1), dim=-1)          # (B,K,HW)
        Q = target.view(B, K, -1)
        Q = Q / (Q.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.nn.functional.kl_div(logP, Q, reduction="batchmean")
    else:
        raise ValueError(f"Unknown heat_loss: {kind}")

# --------------------
# 训练/验证/指标
# --------------------
def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    running = 0.0
    for imgs, hms, _, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        hms = hms.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                loss = heatmap_loss(preds, hms, CONFIG["heat_loss"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(imgs)
            loss = heatmap_loss(preds, hms, CONFIG["heat_loss"])
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
        loss = heatmap_loss(preds, hms, CONFIG["heat_loss"])
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def eval_metrics_full(model, loader, device, img_size=512, pck_px=(2,4,8), pck_norm=(0.01, 0.02)):
    model.eval()
    mse_running, n_samples = 0.0, 0
    per_point_px_all = []
    per_point_nrm_all = []

    for imgs, _, gts_norm, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        gts_norm = gts_norm.to(device, non_blocking=True)  # [B, K*2]

        preds_hm = model(imgs)  # (B,K,Hm,Wm)
        coords = decode_heatmaps(
            preds_hm, kind=CONFIG["decode_kind"], beta=CONFIG["softargmax_beta"]
        )  # (B,K,2) in [0,1]
        preds_norm = coords.view(coords.size(0), -1)       # (B,2K)

        coord_mse = nn.functional.mse_loss(preds_norm, gts_norm, reduction="mean")
        mse_running += coord_mse.item() * imgs.size(0)
        n_samples += imgs.size(0)

        B = preds_norm.size(0)
        pred_xy = preds_norm.view(B, -1, 2) * img_size
        gt_xy   = gts_norm.view(B, -1, 2) * img_size
        d_px = (pred_xy - gt_xy).pow(2).sum(dim=-1).sqrt()  # [B, K]
        per_point_px_all.append(d_px.cpu())

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

# --------------------
# 单次完整训练
# --------------------
def train_unet(cfg):
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    use_amp = torch.cuda.is_available()

    save_dir = os.path.join(cfg["save_dir"], cfg.get("run_name", "unet"))
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs"))

    print(f"\n===== U-Net Heatmap Training =====")
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
        persistent_workers=bool(cfg["num_workers"] > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
        persistent_workers=bool(cfg["num_workers"] > 0)
    )

    model = UNetHeatmap(
        num_kpts=cfg["num_points"],
        img_size=cfg["img_size"],
        heatmap_size=cfg["heatmap_size"],
        channels=(64,128,256,512),
        in_ch=3
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    # scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

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
    best_path = os.path.join(save_dir, f"best_unet.pth")
    no_improve = 0
    warmup = cfg["early_stop"].get("warmup", 0)
    patience = cfg["early_stop"].get("patience", 20)
    min_delta = cfg["early_stop"].get("min_delta", 0.0)
    use_early = cfg["early_stop"].get("use", False)

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss = eval_one_epoch(model, val_loader, device)

        # 评估（把热力图解码为坐标再评）
        metrics = eval_metrics_full(
            model, val_loader, device,
            img_size=cfg["img_size"],
            pck_px=cfg["pck_px"], pck_norm=cfg["pck_norm"]
        )

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LR", current_lr, epoch)

        print(f"[UNet] Epoch {epoch}/{cfg['epochs']} | "
              f"lr={current_lr:.3e}  trainHM={tr_loss:.6f}  valHM={val_loss:.6f}  "
              f"val_px={metrics.get('MeanPx',0):.2f}px  PCK@4px={metrics.get('PCK@4px',0):.3f}")

        # 标量日志
        writer.add_scalar("LossHM/train", tr_loss, epoch)
        writer.add_scalar("LossHM/val", val_loss, epoch)
        if metrics:
            for k, v in metrics.items():
                writer.add_scalar(f"Val/{k}", v, epoch)

        # 图片可视化（pred vs gt + 可选热图）
        with torch.no_grad():
            for imgs_t, hms_t, gts_norm, _ in val_loader:
                imgs_t = imgs_t.to(device, non_blocking=True)
                preds_hm = model(imgs_t)
                preds_norm = decode_heatmaps(
                    preds_hm, kind=CONFIG["decode_kind"], beta=CONFIG["softargmax_beta"]
                ).view(imgs_t.size(0), -1)
                tb_add_images(writer, "Images", imgs_t, preds_norm, gts_norm.to(device), epoch, max_n=cfg["tb_samples"])

                if CONFIG.get("tb_show_heat", True):
                    hm0 = preds_hm[0, 0:1]
                    hm0n = (hm0 - hm0.min()) / (hm0.max() - hm0.min() + 1e-6)
                    writer.add_image("Heatmap/kpt0", hm0n, epoch)
                break

        # 保存最优（按 val heatmap loss 判定）
        improved = (best_val - val_loss) > min_delta
        if improved:
            best_val = val_loss
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_hm_loss": val_loss,
                "img_size": cfg["img_size"],
                "heatmap_size": cfg["heatmap_size"],
                "sigma": cfg["sigma"],
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
    print(f"[UNet] Done. Logs: {os.path.join(save_dir, 'logs')}")

# --------------------
# main
# --------------------
def main():
    cfg = CONFIG
    os.makedirs(cfg["save_dir"], exist_ok=True)
    train_unet(cfg)
    print("\n训练完成。查看 TensorBoard：")
    print(f'tensorboard --logdir "{os.path.join(cfg["save_dir"], cfg.get("run_name","unet"))}"')

if __name__ == "__main__":
    main()

# 用法：
# 1) 修改 CONFIG 中 images_dir/labels_dir 路径。
# 2) 建议 heat_loss="kl" + decode_kind="softargmax"(beta=10)；heatmap_size=128、sigma=2~4。
# 3) 运行脚本；启动 TensorBoard 查看曲线与可视化。
