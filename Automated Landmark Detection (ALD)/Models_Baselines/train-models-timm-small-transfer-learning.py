import os
import json
from glob import glob
from typing import Dict, List, Tuple

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
# 配置区：改这里即可  —— 迁移学习(特征提取)版
# =========================
CONFIG = {
    "images_dir": r"D:\Codes\Skull_Landmarks_TL\Models-TIMM\data\images-all",
    "labels_dir": r"D:\Codes\Skull_Landmarks_TL\Models-TIMM\data\labels_xy",

    # 训练与日志
    "img_size": 512,  # CNN 随意; Transformer 类一般用 224
    "batch_size": 100,
    "lr": 1e-3,               # 头部(head)学习率
    "epochs": 300,
    "val_ratio": 0.15,
    "seed": 2024,
    "augment": True,
    "save_dir": r"D:\Codes\Skull_Landmarks_TL\Models-TIMM\runs\exp_timm_small_transfer-learning",
    "num_workers": 8,
    "pin_memory": True,
    "tb_samples": 2,

    # 关键点
    "num_points": 12,

    # 单骨干名（当 backbones 列表为空时使用）
    "backbone_name": "convnext_tiny",
    "pretrained": True,

    "backbones": [
        "resnet18", "regnety_008", "resnet18", "densenet121"
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
        "type": "cosine",
        "eta_min": 1e-6
    },

    # 早停
    "early_stop": {
        "use": True,
        "patience": 20,
        "min_delta": 0.0,
        "warmup": 100
    },
}

# --------------------
# 工具函数
# --------------------
def set_seed(seed: int = 2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def draw_points_on_image(img: Image.Image, points_xy_norm: np.ndarray, color=(255, 0, 0)):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for i in range(0, len(points_xy_norm), 2):
        x = float(points_xy_norm[i]) * w
        y = float(points_xy_norm[i + 1]) * h
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
        draw.text((x + 4, y + 2), str(i // 2 + 1), fill=color)

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

# --------------------
# 数据集
# --------------------
class LandmarkDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, img_size: int = 512,
                 expected_pts: int = 12, augment: bool = True, geom_cfg: dict = None):
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.expected_pts = expected_pts
        self.augment = augment
        self.geom_cfg = geom_cfg or {}

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        img_paths = []
        for e in exts:
            img_paths.extend(glob(os.path.join(images_dir, e)))
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

    def _resize_and_norm_points(self, pts_xy: np.ndarray, orig_w: int, orig_h: int,
                                new_w: int, new_h: int) -> np.ndarray:
        pts = pts_xy.copy().reshape(-1, 2)
        pts[:, 0] = pts[:, 0] * (new_w / orig_w)
        pts[:, 1] = pts[:, 1] * (new_h / orig_h)
        pts[:, 0] /= new_w
        pts[:, 1] /= new_h
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

        if self.augment:
            if self.geom_cfg.get("do_flip", True) and random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                pts = pts_xy.reshape(-1, 2)
                pts[:, 0] = orig_w - pts[:, 0]
                pts_xy = pts.reshape(-1)

            if self.geom_cfg.get("do_affine", True) and (self.geom_cfg.get("max_rotate_deg", 0) > 0
                                                         or self.geom_cfg.get("scale_range", (1,1)) != (1,1)
                                                         or self.geom_cfg.get("translate_frac", 0) > 0):
                angle, scale, dx, dy = sample_affine_params(orig_w, orig_h, self.geom_cfg)
                center = (orig_w * 0.5, orig_h * 0.5)
                img = TF.affine(img, angle=angle, translate=(dx, dy), scale=scale,
                                shear=[0.0, 0.0], center=center)
                pts_xy = apply_affine_to_points(pts_xy, angle, scale, dx, dy, cx=center[0], cy=center[1])

        new_size = (CONFIG["img_size"], CONFIG["img_size"])
        img = img.resize(new_size, Image.BILINEAR)
        pts_xy_norm = self._resize_and_norm_points(pts_xy, orig_w, orig_h, new_size[0], new_size[1])

        if self.augment:
            img = self.color_aug(img)

        img_t = self.to_tensor(img)
        img_t = self.norm(img_t)
        target = torch.from_numpy(pts_xy_norm.astype(np.float32))
        return img_t, target, os.path.basename(img_path)

# --------------------
# 模型（timm 骨干 + MLP 头）
# --------------------
class TimmBackboneRegressor(nn.Module):
    def __init__(self, out_dim=24, name="convnext_tiny", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        in_feats = getattr(self.backbone, "num_features", None)
        if in_feats is None:
            raise ValueError(f"timm backbone {name} 未能获取 num_features")
        self.head = nn.Sequential(
            nn.Linear(in_feats, 512), nn.ReLU(True), nn.Dropout(0.25),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out

# ---------- 冻结工具（迁移学习核心） ----------
def freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False

# --------------------
# 训练/验证/指标
# --------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler: torch.cuda.amp.GradScaler = None):
    model.train()
    running = 0.0
    for imgs, targets, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running = 0.0
    for imgs, targets, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = model(imgs)
        loss = criterion(preds, targets)
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def eval_metrics_full(model, loader, device, img_size=512, pck_px=(2,4,8), pck_norm=(0.01, 0.02)):
    model.eval()
    mse_running, n_samples = 0.0, 0
    per_point_px_all, per_point_nrm_all = [], []
    for imgs, targets, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = model(imgs)

        mse = torch.nn.functional.mse_loss(preds, targets, reduction='mean')
        mse_running += mse.item() * imgs.size(0)
        n_samples += imgs.size(0)

        B = preds.size(0)
        pred_xy = preds.view(B, -1, 2) * img_size
        gt_xy   = targets.view(B, -1, 2) * img_size
        d_px = (pred_xy - gt_xy).pow(2).sum(dim=-1).sqrt()

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
    auc_pck = float(np.trapz(pcks, grid) / Tthr)

    metrics = {
        "MSE": avg_mse,
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
def tb_add_images(writer: SummaryWriter, tag: str, imgs_t, preds, gts, epoch, max_n=2):
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
        draw_points_on_image(pred_img, preds[i].cpu().numpy().reshape(-1), color=(255, 0, 0))
        gt_img = img_vis.copy()
        draw_points_on_image(gt_img, gts[i].cpu().numpy().reshape(-1), color=(0, 200, 0))
        to_t = T.ToTensor()
        writer.add_image(f"{tag}/pred_{i}", to_t(pred_img), epoch)
        writer.add_image(f"{tag}/gt_{i}", to_t(gt_img), epoch)

# =========================
# 单骨干完整训练（冻结 backbone，仅训 head）
# =========================
def train_one_backbone(cfg, backbone_name: str):
    set_seed(cfg["seed"] + (hash(backbone_name) % 1000))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    use_amp = torch.cuda.is_available()

    save_dir = os.path.join(cfg["save_dir"], backbone_name)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs"))

    print(f"\n===== Backbone: {backbone_name} (Transfer Learning: freeze backbone) =====")
    print("torch:", torch.__version__, " cuda.available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("save_dir:", save_dir)

    ds = LandmarkDataset(
        images_dir=cfg["images_dir"],
        labels_dir=cfg["labels_dir"],
        img_size=cfg["img_size"],
        expected_pts=cfg["num_points"],
        augment=cfg["augment"],
        geom_cfg=cfg["geom_aug"]
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

    # 模型
    out_dim = cfg["num_points"] * 2
    model = TimmBackboneRegressor(out_dim=out_dim, name=backbone_name,
                                  pretrained=cfg["pretrained"]).to(device)

    # ====== 迁移学习核心：冻结 backbone，仅训练 head ======
    freeze_module(model.backbone)
    # 打印可训练参数数量
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[Freeze] trainable params: {n_trainable} / {n_all}  (only head)")

    # 只把 head 参数交给优化器
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 学习率调度器：只作用于 head 的优化器
    scheduler = None
    if cfg["scheduler"].get("use") and cfg["scheduler"].get("type") == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg["epochs"],
            eta_min=cfg["scheduler"].get("eta_min", 1e-6)
        )
        print(f"[Scheduler] CosineAnnealingLR(T_max={cfg['epochs']}, eta_min={cfg['scheduler'].get('eta_min',1e-6)})")

    # 早停变量
    best_val = float("inf")
    best_path = os.path.join(save_dir, f"best_{backbone_name}.pth")
    no_improve = 0
    warmup = cfg["early_stop"].get("warmup", 0)
    patience = cfg["early_stop"].get("patience", 20)
    min_delta = cfg["early_stop"].get("min_delta", 0.0)
    use_early = cfg["early_stop"].get("use", False)

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)

        metrics = eval_metrics_full(model, val_loader, device, img_size=cfg["img_size"],
                                    pck_px=cfg["pck_px"], pck_norm=cfg["pck_norm"])

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LR", current_lr, epoch)

        print(f"[{backbone_name}] Epoch {epoch}/{cfg['epochs']} | "
              f"lr={current_lr:.3e}  train={tr_loss:.6f}  val={val_loss:.6f}  "
              f"val_px={metrics.get('MeanPx',0):.2f}px  PCK@4px={metrics.get('PCK@4px',0):.3f}")

        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        if metrics:
            for k, v in metrics.items():
                writer.add_scalar(f"Val/{k}", v, epoch)

        with torch.no_grad():
            for imgs_t, targets, _ in val_loader:
                imgs_t = imgs_t.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                preds = model(imgs_t)
                tb_add_images(writer, "Images", imgs_t, preds, targets, epoch, max_n=cfg["tb_samples"])
                break

        improved = (best_val - val_loss) > min_delta
        if improved:
            best_val = val_loss
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
                "img_size": cfg["img_size"],
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
                            f"Stopped at epoch {epoch}, best_val={best_val:.6f}", epoch)
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
    print("将依次训练这些骨干（迁移学习：冻结 backbone，仅训 head）：", backbones)

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
# 1) 改 CONFIG 路径
# 2) 运行本脚本（已是迁移学习：冻结 backbone）
# 3) tensorboard --logdir "D:\Codes\Skull_Landmarks_TL\Models-TIMM\runs\exp_timm_small_transfer-learning\resnet18"
