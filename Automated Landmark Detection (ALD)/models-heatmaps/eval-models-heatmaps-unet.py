# -*- coding: utf-8 -*-
"""
eval_unet_heatmap_export.py
仅评估：加载已训练 U-Net 热图回归模型，导出：
- 预测坐标（CSV + .npy）
- 每个关键点的热图图像（K 张灰度 PNG）
- 原图叠加可视化（预测:红 / GT:绿，若有 GT）

使用方式：
1) 修改 CONFIG 中的路径、参数（images_dir、labels_dir、ckpt_path、save_dir 等）。
2) 运行：python eval_unet_heatmap_export.py
"""

import os
import csv
import math
import json
from glob import glob
from typing import Dict, Tuple, Optional, List

import numpy as np
import matplotlib.cm as cm
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as T
import torchvision.transforms.functional as TF

# =========================
# 配置（按需修改）
# =========================
CONFIG = {
    # 训练集（可留着不用）
    "images_dir": r"D:\Codes\Skull_Landmarks_TL\Models-TIMM\data\images-all",
    "labels_dir": r"D:\Codes\Skull_Landmarks_TL\Models-TIMM\data\labels_xy",

    # —— 新增：显式验证集路径 ——
    "use_eval_dirs": True,  # True=直接使用下面两个验证集路径；False=按 val_ratio 切分
    "eval_images_dir": r"D:\Codes\Skull_Landmarks_TL\Models-TIMM\data\eval-data\images",  # 指定验证图片目录
    "eval_labels_dir": r"D:\Codes\Skull_Landmarks_TL\Models-TIMM\data\eval-data\labels",  # 指定验证JSON目录（filename+points）

    "img_size": 512,
    "num_points": 12,
    "heatmap_size": 128,
    "sigma": 3.0,

    "decode_kind": "softargmax",
    "softargmax_beta": 10.0,

    # 仅在 use_eval_dirs=False 时生效
    "val_ratio": 0.15,
    "seed": 2024,
    "split": "val",  # "train" | "val" | "all"（use_eval_dirs=True 时将忽略此字段）

    "ckpt_path": r"D:\Codes\Skull_Landmarks_TL\Models-TIMM\runs\exp_heatmap\unet\best_unet.pth",

    "batch_size": 2,
    "num_workers": 1,
    "pin_memory": True,
    "export_dir": r"D:\Codes\Skull_Landmarks_TL\Models-TIMM\data\eval-data\results",
    "max_samples": 999999,
    "save_numpy": True,
    "save_csv": True,
    "save_heatmaps": True,
    "save_overlay": True,
    "normalize_heatmap": True,
}

# --------------------
# 工具函数
# --------------------
def set_seed(seed: int = 2024):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def draw_points_on_image(img: Image.Image, points_xy_norm: np.ndarray, color=(255, 0, 0)):
    """在 PIL 图像上画归一化(0~1)坐标点并标号。"""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    pts = points_xy_norm.reshape(-1, 2)
    for i, (xn, yn) in enumerate(pts):
        x = float(xn) * w
        y = float(yn) * h
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
        draw.text((x + 4, y + 2), str(i + 1), fill=color)

def gaussian_2d(shape, center, sigma):
    H, W = shape
    y = np.arange(0, H, 1, np.float32)
    x = np.arange(0, W, 1, np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    cy, cx = center
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    return g

def make_heatmaps(points_xy_norm: np.ndarray, heatmap_size: int, sigma: float) -> np.ndarray:
    """从归一化坐标生成 K 张 heatmap（H=heatmap_size, W=heatmap_size）。用于需要 GT 热图时。"""
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

def save_overlay_heatmaps(img_t, hm_t, out_dir, base_stem, alpha=0.5):
    """
    在原图上叠加每个关键点的彩色热力图并保存。
    img_t: [3,H,W] (标准化后的tensor)
    hm_t: [K,Hm,Wm] (预测热力图)
    """
    os.makedirs(out_dir, exist_ok=True)

    # 反归一化原图
    inv_norm = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    to_pil = T.ToPILImage()
    img_vis = inv_norm(img_t.cpu()).clamp(0, 1)
    img_pil = to_pil(img_vis)

    K, Hm, Wm = hm_t.shape
    for i in range(K):
        h = hm_t[i].detach().cpu().numpy()
        h = (h - h.min()) / (h.max() - h.min() + 1e-6)

        # colormap 转彩色 (jet)
        h_color = cm.jet(h)[:, :, :3]  # [Hm,Wm,3], 0~1
        h_img = (h_color * 255).astype(np.uint8)
        h_pil = Image.fromarray(h_img).resize(img_pil.size, Image.BILINEAR)

        # 叠加
        blended = Image.blend(img_pil, h_pil, alpha=alpha)
        out_path = os.path.join(out_dir, f"{base_stem}_overlay_kpt{i+1}.png")
        blended.save(out_path)

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
    """
    读取图片与 JSON 标签（格式：{"filename":..., "points":[x1,y1,...,xK,yK]}），
    输出：图像张量、GT 热图(可用于可视化/评估)、GT 归一化坐标、文件名。
    """
    def __init__(self, images_dir: str, labels_dir: str, img_size: int,
                 expected_pts: int, heatmap_size: int, sigma: float):
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.expected_pts = expected_pts
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

        # 缩放到固定尺寸
        new_size = (CONFIG["img_size"], CONFIG["img_size"])
        img = img.resize(new_size, Image.BILINEAR)

        img_t = self.to_tensor(img)
        img_t = self.norm(img_t)

        # 归一化坐标（相对新图）
        pts_xy_resized = self._resize_points(pts_xy, orig_w, orig_h, new_size[0], new_size[1])
        pts_xy_norm = pts_xy_resized.copy().reshape(-1, 2)
        pts_xy_norm[:, 0] /= new_size[0]
        pts_xy_norm[:, 1] /= new_size[1]
        pts_xy_norm = pts_xy_norm.reshape(-1)

        # 可选：如果需要 GT 热图可在此生成（当前评估不强依赖）
        heatmaps = make_heatmaps(pts_xy_norm, CONFIG["heatmap_size"], CONFIG["sigma"])
        hm_t = torch.from_numpy(heatmaps.astype(np.float32))

        return img_t, hm_t, torch.from_numpy(pts_xy_norm.astype(np.float32)), os.path.basename(img_path)

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
# 导出：热图、叠加、CSV/NPY
# --------------------
def _to_uint8_img(x01: np.ndarray) -> np.ndarray:
    x = np.clip(x01, 0.0, 1.0)
    x = (x * 255.0 + 0.5).astype(np.uint8)
    return x

def save_heatmaps_per_kpt(hm_t: torch.Tensor, out_dir: str, base_stem: str, normalize: bool = True):
    """
    hm_t: (K,H,W) 张量；对每个关键点保存一张热图 PNG。
    文件名：{base_stem}_kpt{i+1}.png
    """
    os.makedirs(out_dir, exist_ok=True)
    hm = hm_t.detach().cpu().numpy()  # [K,H,W]
    K = hm.shape[0]
    for i in range(K):
        h = hm[i]
        if normalize:
            h = (h - h.min()) / (h.max() - h.min() + 1e-6)
        img = _to_uint8_img(h)
        Image.fromarray(img, mode="L").save(os.path.join(out_dir, f"{base_stem}_kpt{i+1}.png"))

def save_overlay_image(img_t: torch.Tensor, preds_norm: torch.Tensor, gts_norm: Optional[torch.Tensor],
                       out_path: str):
    inv_norm = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    to_pil = T.ToPILImage()
    img_vis = inv_norm(img_t.cpu()).clamp(0,1)
    img_pil = to_pil(img_vis)

    draw_points_on_image(img_pil, preds_norm.cpu().numpy().reshape(-1), color=(255,0,0))
    if gts_norm is not None:
        draw_points_on_image(img_pil, gts_norm.cpu().numpy().reshape(-1), color=(0,200,0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img_pil.save(out_path)

def write_coords_to_csv(csv_path: str, rows: list, K: int):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    headers = ["filename"] + [v for i in range(K) for v in (f"x{i+1}", f"y{i+1}")]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

# --------------------
# 评估主流程
# --------------------
@torch.no_grad()
def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch:", torch.__version__, " cuda.available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # 数据集（评估时不做增强）
    ds_all = HeatmapLandmarkDataset(
        images_dir=cfg["images_dir"],
        labels_dir=cfg["labels_dir"],
        img_size=cfg["img_size"],
        expected_pts=cfg["num_points"],
        heatmap_size=cfg["heatmap_size"],
        sigma=cfg["sigma"]
    )

    # 数据集（评估时不做增强）
    if cfg.get("use_eval_dirs", False):
        # 直接使用你指定的验证集目录
        eval_ds = HeatmapLandmarkDataset(
            images_dir=cfg["eval_images_dir"],
            labels_dir=cfg["eval_labels_dir"],
            img_size=cfg["img_size"],
            expected_pts=cfg["num_points"],
            heatmap_size=cfg["heatmap_size"],
            sigma=cfg["sigma"]
        )
    else:
        # 兼容老逻辑：从全量按比例切分 train/val
        ds_all = HeatmapLandmarkDataset(
            images_dir=cfg["images_dir"],
            labels_dir=cfg["labels_dir"],
            img_size=cfg["img_size"],
            expected_pts=cfg["num_points"],
            heatmap_size=cfg["heatmap_size"],
            sigma=cfg["sigma"]
        )
        if cfg["split"].lower() == "all":
            eval_ds = ds_all
        else:
            val_len = max(1, int(len(ds_all) * cfg["val_ratio"]))
            train_len = len(ds_all) - val_len
            train_ds, val_ds = random_split(
                ds_all, [train_len, val_len],
                generator=torch.Generator().manual_seed(cfg["seed"])
            )
            eval_ds = train_ds if cfg["split"].lower() == "train" else val_ds

    loader = DataLoader(
        eval_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
        persistent_workers=False
    )

    # 模型
    model = UNetHeatmap(
        num_kpts=cfg["num_points"],
        img_size=cfg["img_size"],
        heatmap_size=cfg["heatmap_size"],
        channels=(64,128,256,512),
        in_ch=3
    ).to(device)

    # 加载权重
    if not os.path.exists(cfg["ckpt_path"]):
        raise FileNotFoundError(f"ckpt_path 不存在：{cfg['ckpt_path']}")
    ckpt = torch.load(cfg["ckpt_path"], map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        # 兼容直接保存 state_dict 的情况
        model.load_state_dict(ckpt)
    model.eval()
    print(f"[Eval] Loaded checkpoint: {cfg['ckpt_path']}")

    # 导出目录
    out_root = cfg["export_dir"]
    overlay_dir = os.path.join(out_root, "overlay")
    heatmap_dir = os.path.join(out_root, "heatmaps")
    npy_dir = os.path.join(out_root, "npy")
    os.makedirs(out_root, exist_ok=True)

    rows_csv = []
    saved = 0
    K = cfg["num_points"]

    for imgs, hms_gt, gts_norm, fnames in loader:
        imgs = imgs.to(device, non_blocking=True)
        preds_hm = model(imgs)  # (B,K,Hm,Wm)

        coords = decode_heatmaps(preds_hm, kind=cfg["decode_kind"], beta=cfg["softargmax_beta"])  # (B,K,2)
        preds_norm = coords.view(coords.size(0), -1)  # (B,2K)

        for b in range(imgs.size(0)):
            fname = fnames[b]
            stem  = os.path.splitext(fname)[0]

            # 叠加图
            if cfg["save_overlay"]:
                out_overlay = os.path.join(overlay_dir, f"{stem}_overlay.png")
                save_overlay_image(imgs[b], preds_norm[b], gts_norm[b], out_overlay)

            # 12 张热图（每个关键点一张）
            if cfg["save_heatmaps"]:
                # save_heatmaps_per_kpt(preds_hm[b], heatmap_dir, stem, normalize=cfg["normalize_heatmap"])
                save_overlay_heatmaps(imgs[b], preds_hm[b], heatmap_dir, stem, alpha=0.5)

            # NPY
            if cfg["save_numpy"]:
                os.makedirs(npy_dir, exist_ok=True)
                np.save(os.path.join(npy_dir, f"{stem}_preds_norm.npy"), preds_norm[b].cpu().numpy())  # [2K]
                np.save(os.path.join(npy_dir, f"{stem}_preds_hm.npy"), preds_hm[b].detach().cpu().numpy())  # [K,H,W]

            # CSV 行（归一化坐标）
            xy = preds_norm[b].cpu().numpy().reshape(-1,2)
            row = [fname] + [v for i in range(K) for v in (float(xy[i,0]), float(xy[i,1]))]
            rows_csv.append(row)

            saved += 1
            if saved >= cfg["max_samples"]:
                break

        if saved >= cfg["max_samples"]:
            break

    # 写 CSV
    if cfg["save_csv"] and len(rows_csv) > 0:
        csv_path = os.path.join(out_root, f"coords_{cfg['split'].lower()}.csv")
        write_coords_to_csv(csv_path, rows_csv, K)
        print(f"[Eval] CSV saved: {csv_path}")

    print(f"[Eval] Done. Saved {saved} samples to: {out_root}")

if __name__ == "__main__":
    main()
