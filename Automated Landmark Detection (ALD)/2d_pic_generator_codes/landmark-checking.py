import json
import os
import cv2
import matplotlib.pyplot as plt

# 路径配置
json_path = "mission-skull-por-landmarks.json"  # 你的 JSON 文件
img_dir = "./mission-skull-por"  # 存放图像的文件夹（修改为实际路径）

# 读取 JSON 文件
with open(json_path, "r") as f:
    data = json.load(f)

# 遍历 JSON 中的图像标注
for file_key, file_info in data.items():
    filename = file_info["filename"]  # 图片文件名
    regions = file_info["regions"]    # 点的标注信息

    img_path = os.path.join(img_dir, filename)
    if not os.path.exists(img_path):
        print(f"⚠️ 找不到图像文件: {img_path}")
        continue

    # 读取图像
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 在图像上绘制标注点
    for idx, region in enumerate(regions):
        cx = region["shape_attributes"]["cx"]
        cy = region["shape_attributes"]["cy"]
        cv2.circle(img_rgb, (cx, cy), 5, (255, 0, 0), -1)  # 红点
        cv2.putText(img_rgb, str(idx), (cx+5, cy-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 显示图像
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.title(filename)
    plt.axis("off")
    plt.show()