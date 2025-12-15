# pip install torch torchvision opencv-python pillow numpy pandas
# pip install git+https://github.com/facebookresearch/segment-anything.git

import os, cv2, numpy as np, pandas as pd, torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ========== 配置（CPU 模式）==========
IMG_DIR   = r"D:\Codes\Skull_Landmarks_TL\skull_models_2d_pics"                                     # 输入图像文件夹
OUT_CSV   = r"D:\Codes\Skull_Landmarks_TL\images_counts.csv"               # 输出统计
SAM_TYPE  = "vit_b"                                                                                 # 轻量模型：vit_b
SAM_CKPT  = r"D:\Codes\Skull_Landmarks_TL\sam_vit_b_01ec64.pth"                                     # vit_b 权重
DEVICE    = "cpu"                                                                                   # 强制 CPU

# 若机器线程很多，可视情况限制/释放 CPU 线程
torch.set_num_threads(8)   # 可按需设置

# 掩膜生成器参数（更适合 CPU）
POINTS_PER_SIDE = 16            # 采样密度（比 32 更快）
PRED_IOU_THRESH = 0.90
STABILITY_THRESH = 0.92
BOX_NMS_THRESH = 0.7
MIN_MASK_AREA = 500             # 去小碎片（像素）
DEDUP_IOU_THRESH = 0.5          # 掩膜 IoU 去重阈值

# 为了加速：把大图短边限制到 1024（只用于计数足够）
MAX_SIDE = 1024

def list_images(folder):
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
    return [os.path.join(folder,f) for f in os.listdir(folder)
            if os.path.splitext(f.lower())[1] in exts]

def iou_bool(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0

def resize_keep_aspect(img_rgb, max_side=MAX_SIDE):
    h, w = img_rgb.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img_rgb, 1.0
    scale = max_side / float(s)
    new_w, new_h = int(round(w*scale)), int(round(h*scale))
    img_small = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_small, scale

def count_unique_regions(img_rgb, mask_gen):
    # 可选降采样，加速
    img_small, scale = resize_keep_aspect(img_rgb, MAX_SIDE)

    # 生成掩膜
    masks = mask_gen.generate(img_small)
    # 去小碎片（按缩放修正阈值）
    min_area_scaled = int(MIN_MASK_AREA * (scale**2 if scale < 1.0 else 1.0))
    masks = [m for m in masks if m["area"] >= min_area_scaled]
    if not masks:
        return 0

    # 面积从大到小，掩膜 IoU 去重
    masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)
    kept = []
    for m in masks_sorted:
        seg = m["segmentation"].astype(bool)
        if all(iou_bool(seg, k) < DEDUP_IOU_THRESH for k in kept):
            kept.append(seg)
    return len(kept)

def main():
    img_paths = list_images(IMG_DIR)
    if not img_paths:
        print("输入目录中未找到图像文件。"); return

    # 加载 SAM（CPU）
    sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT).to(DEVICE)
    mask_gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_THRESH,
        box_nms_thresh=BOX_NMS_THRESH,
        min_mask_region_area=MIN_MASK_AREA,
    )

    rows = []
    for p in img_paths:
        bgr = cv2.imread(p)
        if bgr is None:
            print(f"[WARN] 读取失败：{p}"); continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        n_regions = count_unique_regions(rgb, mask_gen)
        rows.append({"image": os.path.basename(p), "num_regions": n_regions})
        print(f"{os.path.basename(p)} -> 区域数: {n_regions}")

    df = pd.DataFrame(rows).sort_values("image")
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 完成：已输出统计到 {OUT_CSV}")

if __name__ == "__main__":
    main()
