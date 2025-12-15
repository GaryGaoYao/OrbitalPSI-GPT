# -*- coding: utf-8 -*-
"""
train_unet_heatmap_priors_ablation.py
在你的 U-Net 热图回归 + 结构先验脚本基础上，增加“消融试验”总控：
- 自动遍历多组损失开关（逐项去掉 / 全部去掉 / 仅热图）
- 为每个实验组单独创建日志目录：{save_dir}/{run_name}__{exp_tag}
- 训练完成后载入“最佳权重”在验证集完整评估（CoordMSE、MeanPx、PCK、AUC-PCK）
- 将所有实验组结果汇总保存为 CSV：{save_dir}/{run_name}_ablation_summary.csv

使用方法：
python train_unet_heatmap_priors_ablation.py
（按需修改 CONFIG 顶部路径与参数）
"""

import os
import json
from glob import glob
from typing import Dict, Tuple, List, Optional

import math
import random
import argparse
import csv
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    "img_size": 512,
    "batch_size": 48,
    "lr": 3e-4,
    "epochs": 250,
    "val_ratio": 0.15,
    "seed": 2024,
    "augment": True,
    "save_dir": r"D:\Codes\Skull_Landmarks_TL\Models-TIMM\runs\exp_heatmap_priors-ablation",
    "run_name": "unet_priors",
    "num_workers": 8,         # Windows/Thonny 卡住可设 0
    "pin_memory": True,
    "tb_samples": 2,

    # 关键点
    "num_points": 12,

    # Heatmap 参数
    "heatmap_size": 128,
    "sigma": 3,

    # 损失与解码
    "heat_loss": "mse",              # "mse" 或 "kl"
    "decode_kind": "softargmax",    # "softargmax" 或 "argmax"
    "softargmax_beta": 10.0,

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
        "warmup": 250
    },

    # 可视化热图（第 0 关键点）
    "tb_show_heat": True,

    # ====== 结构先验配置 ======
    "lr_pairs": [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)],
    "rim_left_order":  [0, 2, 4,  1, 3, 5],
    "rim_right_order": [6, 8, 10, 7, 9, 11],
    "ssm": {
        "use": True,
        "mode": "auto",       # "auto" 或 "file"
        "mu_path": r"",
        "w_path": r"",
        "var_ratio": 0.97,
        "max_dim": 12
    },
    "loss_weights": {
        "heatmap": 1.0,
        "sym": 0.5,
        "ssm": 0.1,
        "graph": 0.05,
        "edge": 0.05
    },
}

# ===== 实验设计 =====
# 想精简就删掉对应条目；想加别的也可照样添加
ABLATIONS = [
    # ("all",            {"sym":None, "graph":None, "edge":None, "ssm":None}),  # None=使用原始权重
    ("minus_sym",      {"sym":0.0}),
    ("minus_graph",    {"graph":0.0}),
    ("minus_edge",     {"edge":0.0}),
    ("minus_ssm",      {"ssm":0.0}),
    # ("heatmap_only",   {"sym":0.0, "graph":0.0, "edge":0.0, "ssm":0.0}),
]
REPEATS = 1  # 同一配置重复次数（设 >1 可做均值/方差统计）

# --------------------
# 工具函数（与你原脚本一致或小幅增补）
# --------------------
def set_seed(seed: int = 2024):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def draw_points_on_image(img: Image.Image, points_xy_norm: np.ndarray, color=(255, 0, 0)):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    pts = points_xy_norm.reshape(-1, 2)
    for i, (xn, yn) in enumerate(pts):
        x = float(xn) * w; y = float(yn) * h; r = 3
        draw.ellipse((x-r, y-r, x+r, y+r), outline=color, width=2)
        draw.text((x+4, y+2), str(i+1), fill=color)

def sample_affine_params(orig_w: int, orig_h: int, cfg: dict) -> Tuple[float, float, float, float]:
    max_deg = cfg.get("max_rotate_deg", 0)
    min_s, max_s = cfg.get("scale_range", (1.0, 1.0))
    tfrac = cfg.get("translate_frac", 0.0)
    angle = random.uniform(-max_deg, max_deg)
    scale = random.uniform(min_s, max_s)
    max_dx = orig_w * tfrac; max_dy = orig_h * tfrac
    dx = random.uniform(-max_dx, max_dx); dy = random.uniform(-max_dy, max_dy)
    return angle, scale, dx, dy

def apply_affine_to_points(pts_xy: np.ndarray, angle_deg: float, scale: float, dx: float, dy: float, cx: float, cy: float) -> np.ndarray:
    pts = pts_xy.reshape(-1, 2).copy()
    rad = math.radians(angle_deg)
    cosA = math.cos(rad) * scale; sinA = math.sin(rad) * scale
    pts[:, 0] -= cx; pts[:, 1] -= cy
    x_new = pts[:, 0]*cosA - pts[:, 1]*sinA
    y_new = pts[:, 0]*sinA + pts[:, 1]*cosA
    pts[:, 0] = x_new + cx + dx; pts[:, 1] = y_new + cy + dy
    return pts.reshape(-1)

def gaussian_2d(shape, center, sigma):
    H, W = shape
    y = np.arange(0, H, 1, np.float32)
    x = np.arange(0, W, 1, np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    cy, cx = center
    g = np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*sigma*sigma))
    return g

def make_heatmaps(points_xy_norm: np.ndarray, heatmap_size: int, sigma: float) -> np.ndarray:
    K = points_xy_norm.size // 2
    hm = np.zeros((K, heatmap_size, heatmap_size), dtype=np.float32)
    pts = points_xy_norm.reshape(-1, 2)
    for i, (xn, yn) in enumerate(pts):
        if xn<0 or xn>1 or yn<0 or yn>1: continue
        cx = xn * (heatmap_size - 1); cy = yn * (heatmap_size - 1)
        g = gaussian_2d((heatmap_size, heatmap_size), (cy, cx), sigma=sigma)
        hm[i] = np.clip(np.maximum(hm[i], g), 0.0, 1.0)
    return hm

def softargmax2d(logits: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
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
    B, K, H, W = hm.shape
    idx = hm.view(B, K, -1).argmax(dim=-1)
    y = (idx // W).float(); x = (idx % W).float()
    x_norm = x / (W - 1); y_norm = y / (H - 1)
    return torch.stack([x_norm, y_norm], dim=-1)

def decode_heatmaps(hm: torch.Tensor, kind: str, beta: float) -> torch.Tensor:
    return softargmax2d(hm, beta=beta) if kind=="softargmax" else argmax_decode(hm)

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
        for e in exts: img_paths.extend(glob(os.path.join(images_dir, e)))
        img_paths = sorted(img_paths)
        if not img_paths:
            raise RuntimeError(f"images_dir 里未找到图片：{images_dir}")

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
                        candidate = self.label_index[k]; break
                if candidate: self.samples.append((ip, candidate))
                else: miss += 1

        if not self.samples:
            raise RuntimeError("未匹配到任何 (图片, 标签) 对。请检查 labels 中 JSON 的 'filename' 字段是否与图片名对应。")
        if miss > 0:
            print(f"[INFO] 有 {miss} 张图片未在 labels 中找到匹配标签。")

        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        self.color_aug = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)

    def __len__(self): return len(self.samples)

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

        # 几何增强
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

        # 缩放
        new_size = (CONFIG["img_size"], CONFIG["img_size"])
        img = img.resize(new_size, Image.BILINEAR)
        pts_xy_resized = self._resize_points(pts_xy, orig_w, orig_h, new_size[0], new_size[1])

        # 颜色增强
        if self.augment: img = self.color_aug(img)

        img_t = self.to_tensor(img); img_t = self.norm(img_t)

        # 归一化坐标
        pts_xy_norm = pts_xy_resized.copy().reshape(-1, 2)
        pts_xy_norm[:, 0] /= new_size[0]; pts_xy_norm[:, 1] /= new_size[1]
        pts_xy_norm = pts_xy_norm.reshape(-1)

        heatmaps = make_heatmaps(pts_xy_norm, self.heatmap_size, self.sigma)  # [K,Hm,Wm]
        hm_t = torch.from_numpy(heatmaps.astype(np.float32))
        target_coords_norm = torch.from_numpy(pts_xy_norm.astype(np.float32))  # [K*2]
        return img_t, hm_t, target_coords_norm, os.path.basename(img_path)

# --------------------
# 模型
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
    def __init__(self, num_kpts: int, img_size: int, heatmap_size: int,
                 channels=(64,128,256,512), in_ch: int = 3):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.enc1 = DoubleConv(in_ch, c1); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(c1, c2);    self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(c2, c3);    self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c3, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2); self.dec3 = DoubleConv(c3+c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2); self.dec2 = DoubleConv(c2+c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2); self.dec1 = DoubleConv(c1+c1, c1)
        self.outc = nn.Conv2d(c1, num_kpts, 1)
        self.heatmap_size = heatmap_size

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        heat = self.outc(d1)
        if heat.shape[-1] != self.heatmap_size:
            heat = F.interpolate(heat, size=(self.heatmap_size, self.heatmap_size),
                                 mode="bilinear", align_corners=False)
        return heat

# --------------------
# 损失
# --------------------
def heatmap_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "mse") -> torch.Tensor:
    if kind == "mse":
        return F.mse_loss(pred, target, reduction="mean")
    elif kind == "kl":
        B, K, H, W = pred.shape
        logP = torch.log_softmax(pred.view(B, K, -1), dim=-1)  # (B,K,HW)
        Q = target.view(B, K, -1); Q = Q / (Q.sum(dim=-1, keepdim=True) + 1e-8)
        return F.kl_div(logP, Q, reduction="batchmean")
    else:
        raise ValueError(f"Unknown heat_loss: {kind}")

def symmetry_loss(preds_norm: torch.Tensor, pairs: List[Tuple[int,int]], weight=0.5, x0: float = 0.5):
    if weight <= 0 or not pairs: return preds_norm.new_tensor(0.0)
    B, K2 = preds_norm.shape; K = K2 // 2
    xy = preds_norm.view(B, K, 2)
    loss = preds_norm.new_tensor(0.0)
    for l, r in pairs:
        xl, yl = xy[:, l, 0], xy[:, l, 1]
        xr, yr = xy[:, r, 0], xy[:, r, 1]
        xr_mirror = 2*x0 - xr
        loss = loss + ((xl - xr_mirror)**2 + (yl - yr)**2).mean()
    return weight * loss / max(1, len(pairs))

def graph_smooth_loss(preds_norm: torch.Tensor, ring_order_left: List[int], ring_order_right: List[int], lam=0.05):
    if lam <= 0: return preds_norm.new_tensor(0.0)
    B, K2 = preds_norm.shape; K = K2 // 2
    xy = preds_norm.view(B, K, 2)
    def ring_smooth(order):
        if not order: return xy.new_tensor(0.0)
        diffs = []; L = len(order)
        for i in range(L):
            a, b = order[i], order[(i+1)%L]
            diffs.append((xy[:, a] - xy[:, b])**2)
        return torch.stack(diffs, 0).sum(0).mean()
    return lam * (ring_smooth(ring_order_left) + ring_smooth(ring_order_right))

def _inv_norm_image(imgs: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485,0.456,0.406], device=imgs.device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=imgs.device).view(1,3,1,1)
    return (imgs*std + mean).clamp(0,1)

def _sobel_grad(imgs_01: torch.Tensor) -> torch.Tensor:
    gray = (0.2989*imgs_01[:,0] + 0.5870*imgs_01[:,1] + 0.1140*imgs_01[:,2]).unsqueeze(1)
    kx = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], device=imgs_01.device, dtype=imgs_01.dtype)
    ky = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], device=imgs_01.device, dtype=imgs_01.dtype)
    gx = F.conv2d(gray, kx, padding=1); gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy)

def edge_attraction_loss(imgs_t: torch.Tensor, preds_norm: torch.Tensor, weight=0.05):
    if weight <= 0: return preds_norm.new_tensor(0.0)
    B, C, H, W = imgs_t.shape
    imgs_01 = _inv_norm_image(imgs_t); grad = _sobel_grad(imgs_01)  # [B,1,H,W]
    xy = preds_norm.view(B, -1, 2)
    gx = xy[:,:,0]*2 - 1; gy = xy[:,:,1]*2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # [B,K,1,2]
    sampled = F.grid_sample(grad, grid, mode='bilinear', align_corners=True)
    gval = sampled.squeeze(1).squeeze(-1)  # [B,K]
    return weight * (-gval.mean())

def procrustes_align(X: np.ndarray) -> np.ndarray:
    N, D = X.shape; K = D // 2
    Xc = X.copy().reshape(N, K, 2)
    Xc = Xc - Xc.mean(axis=1, keepdims=True)
    scale = np.sqrt((Xc**2).sum(axis=(1,2), keepdims=True) / (K))
    scale[scale == 0] = 1.0
    Xn = (Xc / scale).reshape(N, D)
    return Xn

def pca_from_data(X: np.ndarray, var_ratio=0.97, max_dim=12) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0); Xm = X - mu
    U, S, Vt = np.linalg.svd(Xm, full_matrices=False)
    exp = (S**2); cum = np.cumsum(exp / exp.sum())
    d = int(np.searchsorted(cum, var_ratio) + 1)
    d = min(max(d, 1), max_dim)
    W = Vt[:d].T  # [D,d]
    return mu, W

def ssm_prior_loss(preds_norm: torch.Tensor, mu: Optional[torch.Tensor], W: Optional[torch.Tensor], weight=0.1):
    if weight <= 0 or (mu is None) or (W is None):
        return preds_norm.new_tensor(0.0)
    y = preds_norm
    z = (y - mu) @ W
    y_proj = mu + z @ W.t()
    return weight * ((y - y_proj) ** 2).mean()

def build_ssm_from_dataset(images_dir, labels_dir, img_size, expected_pts,
                           heatmap_size=32, sigma=2.0, var_ratio=0.97, max_dim=12) -> Tuple[np.ndarray, np.ndarray]:
    tmp_ds = HeatmapLandmarkDataset(images_dir, labels_dir, img_size, expected_pts,
                                    augment=False, geom_cfg={}, heatmap_size=heatmap_size, sigma=sigma)
    coords = []
    for i in range(len(tmp_ds)):
        _, _, target_coords_norm, _ = tmp_ds[i]
        coords.append(target_coords_norm.numpy())
    X = np.stack(coords, axis=0)
    X_aligned = procrustes_align(X)
    mu, W = pca_from_data(X_aligned, var_ratio=var_ratio, max_dim=max_dim)
    print(f"[SSM] PCA 完成：d={W.shape[1]} (var_ratio≥{var_ratio})")
    return mu, W

# --------------------
# 训练/验证/指标（核心未改）
# --------------------
def train_one_epoch(model, loader, optimizer, device, cfg, ssm_mu_t, ssm_W_t, scaler=None):
    model.train()
    lw = cfg["loss_weights"]
    pairs = cfg["lr_pairs"]
    ringL = cfg["rim_left_order"]
    ringR = cfg["rim_right_order"]
    beta  = cfg["softargmax_beta"]

    running = {"hm":0.0, "sym":0.0, "graph":0.0, "edge":0.0, "ssm":0.0, "total":0.0}
    nitems = 0

    for imgs, hms, _, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        hms = hms.to(device, non_blocking=True)
        B = imgs.size(0)
        nitems += B

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(imgs)  # (B,K,Hm,Wm)
                loss_hm = heatmap_loss(preds, hms, cfg["heat_loss"])
                coords_soft = softargmax2d(preds, beta=beta).view(B, -1)

                loss_sym   = symmetry_loss(coords_soft, pairs, weight=lw["sym"])
                loss_graph = graph_smooth_loss(coords_soft, ringL, ringR, lam=lw["graph"])
                loss_edge  = edge_attraction_loss(imgs, coords_soft, weight=lw["edge"])
                loss_ssm   = ssm_prior_loss(coords_soft, ssm_mu_t, ssm_W_t, weight=lw["ssm"])

                loss_total = lw["heatmap"]*loss_hm + loss_sym + loss_graph + loss_edge + loss_ssm
            scaler.scale(loss_total).backward()
            scaler.step(optimizer); scaler.update()
        else:
            preds = model(imgs)
            loss_hm = heatmap_loss(preds, hms, cfg["heat_loss"])
            coords_soft = softargmax2d(preds, beta=beta).view(B, -1)
            loss_sym   = symmetry_loss(coords_soft, pairs, weight=lw["sym"])
            loss_graph = graph_smooth_loss(coords_soft, ringL, ringR, lam=lw["graph"])
            loss_edge  = edge_attraction_loss(imgs, coords_soft, weight=lw["edge"])
            loss_ssm   = ssm_prior_loss(coords_soft, ssm_mu_t, ssm_W_t, weight=lw["ssm"])
            loss_total = lw["heatmap"]*loss_hm + loss_sym + loss_graph + loss_edge + loss_ssm
            loss_total.backward(); optimizer.step()

        running["hm"]    += loss_hm.item()   * B
        running["sym"]   += (loss_sym.item()   if torch.is_tensor(loss_sym)   else float(loss_sym))   * B
        running["graph"] += (loss_graph.item() if torch.is_tensor(loss_graph) else float(loss_graph)) * B
        running["edge"]  += (loss_edge.item()  if torch.is_tensor(loss_edge)  else float(loss_edge))  * B
        running["ssm"]   += (loss_ssm.item()   if torch.is_tensor(loss_ssm)   else float(loss_ssm))   * B
        running["total"] += loss_total.item() * B

    for k in running: running[k] /= nitems
    return running

@torch.no_grad()
def eval_one_epoch(model, loader, device, cfg):
    model.eval()
    total = 0.0; n = 0
    for imgs, hms, _, _ in loader:
        imgs = imgs.to(device, non_blocking=True); hms = hms.to(device, non_blocking=True)
        preds = model(imgs); loss = heatmap_loss(preds, hms, cfg["heat_loss"])
        total += loss.item() * imgs.size(0); n += imgs.size(0)
    return total / n

@torch.no_grad()
def eval_metrics_full(model, loader, device, img_size=512, pck_px=(2,4,8), pck_norm=(0.01, 0.02),
                      decode_kind="softargmax", beta=10.0):
    model.eval()
    mse_running, n_samples = 0.0, 0
    per_point_px_all = []; per_point_nrm_all = []

    for imgs, _, gts_norm, _ in loader:
        imgs = imgs.to(device, non_blocking=True); gts_norm = gts_norm.to(device, non_blocking=True)
        preds_hm = model(imgs)
        coords = softargmax2d(preds_hm, beta=beta) if decode_kind=="softargmax" else argmax_decode(preds_hm)
        preds_norm = coords.view(coords.size(0), -1)

        coord_mse = F.mse_loss(preds_norm, gts_norm, reduction="mean")
        mse_running += coord_mse.item() * imgs.size(0); n_samples += imgs.size(0)

        B = preds_norm.size(0)
        pred_xy = preds_norm.view(B, -1, 2) * img_size
        gt_xy   = gts_norm.view(B, -1, 2) * img_size
        d_px = (pred_xy - gt_xy).pow(2).sum(dim=-1).sqrt()
        per_point_px_all.append(d_px.cpu())

        diag = math.sqrt(img_size**2 + img_size**2)
        d_nrm = d_px / diag
        per_point_nrm_all.append(d_nrm.cpu())

    if n_samples == 0: return {}

    per_point_px_all = torch.cat(per_point_px_all, dim=0).numpy().ravel()
    per_point_nrm_all = torch.cat(per_point_nrm_all, dim=0).numpy().ravel()

    avg_mse   = (mse_running / n_samples)
    mean_px   = float(np.mean(per_point_px_all))
    median_px = float(np.median(per_point_px_all))
    mean_nme  = float(np.mean(per_point_nrm_all))
    median_nme= float(np.median(per_point_nrm_all))

    pck_px_dict   = {f"PCK@{t}px": float(np.mean(per_point_px_all < t)) for t in pck_px}
    pck_norm_dict = {f"PCK@{int(r*100)}%diag": float(np.mean(per_point_nrm_all < r)) for r in pck_norm}

    Tthr = img_size * 0.05
    grid = np.linspace(0, Tthr, 50)
    pcks = [np.mean(per_point_px_all < g) for g in grid]
    auc_pck = float(np.trapz(pcks, grid) / Tthr)

    metrics = {
        "CoordMSE": avg_mse,
        "MeanPx": mean_px,
        "MedianPx": median_px,
        "MeanNME": mean_nme,
        "MedianNME": median_nme,
        "AUC-PCK(0..5% diag)": auc_pck
    }
    metrics.update(pck_px_dict); metrics.update(pck_norm_dict)
    return metrics

@torch.no_grad()
def tb_add_images(writer: SummaryWriter, tag: str, imgs_t, preds_norm, gts_norm, epoch, max_n=2):
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
# 单次完整训练（返回最优模型路径 + 最终评估指标）
# --------------------
def train_unet(cfg, exp_tag: str):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    use_amp = torch.cuda.is_available()

    save_dir = os.path.join(cfg["save_dir"], f"{cfg.get('run_name','unet_priors')}__{exp_tag}")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs"))

    print(f"\n===== [{exp_tag}] U-Net Heatmap + Priors Training =====")
    print("torch:", torch.__version__, " cuda.available:", torch.cuda.is_available())
    if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    scheduler = None
    if cfg["scheduler"].get("use") and cfg["scheduler"].get("type") == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=cfg["scheduler"].get("eta_min", 1e-6)
        )
        print(f"[Scheduler] CosineAnnealingLR(T_max={cfg['epochs']}, eta_min={cfg['scheduler'].get('eta_min',1e-6)})")

    # ------- SSM 先验参数 -------
    ssm_mu_t, ssm_W_t = None, None
    if cfg["ssm"]["use"] and cfg["loss_weights"].get("ssm", 0.0) > 0:
        if cfg["ssm"]["mode"] == "file" and cfg["ssm"]["mu_path"] and cfg["ssm"]["w_path"] \
           and os.path.exists(cfg["ssm"]["mu_path"]) and os.path.exists(cfg["ssm"]["w_path"]):
            mu = np.load(cfg["ssm"]["mu_path"]); W  = np.load(cfg["ssm"]["w_path"])
            ssm_mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
            ssm_W_t  = torch.tensor(W,  dtype=torch.float32, device=device)
            print(f"[SSM] 从文件加载：d={W.shape[1]}")
        else:
            mu, W = build_ssm_from_dataset(cfg["images_dir"], cfg["labels_dir"], cfg["img_size"], cfg["num_points"],
                                           heatmap_size=32, sigma=2.0,
                                           var_ratio=cfg["ssm"]["var_ratio"], max_dim=cfg["ssm"]["max_dim"])
            ssm_mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
            ssm_W_t  = torch.tensor(W,  dtype=torch.float32, device=device)
            np.save(os.path.join(save_dir, "ssm_mu.npy"), mu)
            np.save(os.path.join(save_dir, "ssm_W.npy"),  W)

    # 早停
    best_val = float("inf")
    best_path = os.path.join(save_dir, f"best_unet_priors.pth")
    no_improve = 0
    warmup = cfg["early_stop"].get("warmup", 0)
    patience = cfg["early_stop"].get("patience", 20)
    min_delta = cfg["early_stop"].get("min_delta", 0.0)
    use_early = cfg["early_stop"].get("use", False)

    for epoch in range(1, cfg["epochs"] + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device, cfg, ssm_mu_t, ssm_W_t, scaler)
        val_hm = eval_one_epoch(model, val_loader, device, cfg)

        metrics = eval_metrics_full(
            model, val_loader, device,
            img_size=cfg["img_size"],
            pck_px=cfg["pck_px"], pck_norm=cfg["pck_norm"],
            decode_kind=cfg["decode_kind"], beta=cfg["softargmax_beta"]
        )

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LR", current_lr, epoch)
        print(f"[{exp_tag}] Epoch {epoch}/{cfg['epochs']} | "
              f"lr={current_lr:.3e}  trainHM={tr['hm']:.6f}  total={tr['total']:.6f}  "
              f"valHM={val_hm:.6f}  val_px={metrics.get('MeanPx',0):.2f}px  PCK@4px={metrics.get('PCK@4px',0):.3f}")

        writer.add_scalar("LossHM/train", tr["hm"], epoch)
        writer.add_scalar("LossHM/val", val_hm, epoch)
        writer.add_scalar("LossTotal/train", tr["total"], epoch)
        writer.add_scalar("LossPriors/sym", tr["sym"], epoch)
        writer.add_scalar("LossPriors/graph", tr["graph"], epoch)
        writer.add_scalar("LossPriors/edge", tr["edge"], epoch)
        writer.add_scalar("LossPriors/ssm", tr["ssm"], epoch)
        if metrics:
            for k, v in metrics.items():
                writer.add_scalar(f"Val/{k}", v, epoch)

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

        improved = (best_val - val_hm) > min_delta
        if improved:
            best_val = val_hm; no_improve = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_hm_loss": val_hm,
                "img_size": cfg["img_size"],
                "heatmap_size": cfg["heatmap_size"],
                "sigma": cfg["sigma"],
            }, best_path)
            print(f"  -> ✅ 保存最优权重: {best_path}")
        else:
            no_improve += 1

        if scheduler is not None: scheduler.step()
        if use_early and epoch > warmup and no_improve >= patience:
            print(f"⏹️  触发早停：连续 {no_improve} 个 epoch 验证集无提升（min_delta={min_delta}），在第 {epoch} 个 epoch 停止。")
            writer.add_text("EarlyStop", f"Stopped at epoch {epoch}, best_valHM={best_val:.6f}", epoch)
            break

    # 载入最佳并做最终评估
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    final_metrics = eval_metrics_full(
        model, val_loader, device,
        img_size=cfg["img_size"],
        pck_px=cfg["pck_px"], pck_norm=cfg["pck_norm"],
        decode_kind=cfg["decode_kind"], beta=cfg["softargmax_beta"]
    )

    writer.close()
    del model
    torch.cuda.empty_cache()
    print(f"[{exp_tag}] Done. Logs: {os.path.join(save_dir, 'logs')}")
    return best_path, final_metrics

# --------------------
# 实验总控：遍历消融设置，自动修改 loss 权重并运行
# --------------------
def run_ablation(cfg, ablations, repeats=1):
    base_save_dir = cfg["save_dir"]
    os.makedirs(base_save_dir, exist_ok=True)

    summary_rows = []
    summary_csv = os.path.join(base_save_dir, f"{cfg.get('run_name','unet_priors')}_ablation_summary.csv")

    for tag, knobs in ablations:
        for r in range(1, repeats+1):
            exp_tag = f"{tag}" if repeats==1 else f"{tag}_r{r}"

            # —— 深拷贝 cfg 并打补丁 —— 
            cfg_i = json.loads(json.dumps(cfg))  # 简便深拷贝
            # 覆盖损失权重
            lw = cfg_i["loss_weights"]
            for k, v in (knobs or {}).items():
                if v is None: 
                    continue  # 使用原值
                if k in lw: lw[k] = v
            # 若全部先验都被置零，可直接关闭 SSM 构建加速
            if lw.get("ssm", 0.0) <= 0: 
                # 可以仍保留 ssm.use=True 但不触发构建；这里保持原逻辑，仅在 train 中按权重判断
                pass

            # 运行
            best_ckpt, metrics = train_unet(cfg_i, exp_tag)
            row = {
                "exp_tag": exp_tag,
                "best_ckpt": best_ckpt,
                "heat_loss": cfg_i["heat_loss"],
                "decode_kind": cfg_i["decode_kind"],
                "img_size": cfg_i["img_size"],
                "heatmap_size": cfg_i["heatmap_size"],
                "sigma": cfg_i["sigma"],
                # 当前权重记录
                "w_heatmap": lw.get("heatmap", 0.0),
                "w_sym": lw.get("sym", 0.0),
                "w_graph": lw.get("graph", 0.0),
                "w_edge": lw.get("edge", 0.0),
                "w_ssm": lw.get("ssm", 0.0),
            }
            # 合并评估指标
            for k, v in (metrics or {}).items():
                row[k] = v
            summary_rows.append(row)

            # 即时写 CSV（防止中断丢失）
            fieldnames = list(summary_rows[0].keys())
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader(); writer.writerows(summary_rows)

    print("\n✅ 消融试验完成。汇总结果：", summary_csv)
    print('可在 TensorBoard 对比： tensorboard --logdir "{}"'.format(base_save_dir))

# --------------------
# main
# --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=REPEATS, help="同一配置重复次数")
    # 可选：仅跑某些 tag，用逗号分隔，如: --only all,minus_ssm
    parser.add_argument("--only", type=str, default="", help="只运行指定tag，逗号分隔")
    args = parser.parse_args()

    cfg = CONFIG
    os.makedirs(cfg["save_dir"], exist_ok=True)

    abls = ABLATIONS
    if args.only.strip():
        wanted = set([s.strip() for s in args.only.split(",") if s.strip()])
        abls = [a for a in ABLATIONS if a[0] in wanted]
        if not abls:
            print("[WARN] --only 过滤后为空，仍将运行默认 ABLATIONS。")
            abls = ABLATIONS

    run_ablation(cfg, abls, repeats=max(1, int(args.repeats)))

if __name__ == "__main__":
    main()

# 备注：
# 1) 仍需按你的点序调整 lr_pairs / rim_left_order / rim_right_order。
# 2) SSM:
#    - 若 w_ssm > 0 且 mode="auto"：会对训练集(无增强)估计 PCA；缓存到每个实验目录。
#    - 若 w_ssm=0：SSM 不参与训练，不会触发估计（加速）。
# 3) TensorBoard：
#    tensorboard --logdir "{save_dir}"
#    子目录按 run_name__{exp_tag} 区分，方便横向对比。
# 4) 如显存紧张：减小 batch_size 或 heatmap_size。