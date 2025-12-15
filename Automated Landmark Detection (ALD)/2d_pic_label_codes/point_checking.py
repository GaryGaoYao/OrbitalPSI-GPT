import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ===== 修改为你的图片路径 =====
IMAGE_PATH = r"D:\Codes\Skull_Landmarks_TL\mission-skull-li\A0242_clear_A4.png"

# ===== 输入点坐标 (xyxyxy 格式) =====
coords = [485,363,483,249,404,299,549,321,447,251,530,363,295,373,289,256,368,304,232,313,321,253,244,361]


# 拆分为 x 和 y
x = coords[0::2]
y = coords[1::2]

# 读取图片
img = mpimg.imread(IMAGE_PATH)

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.scatter(x, y, color="red", s=40, marker="o")

# 标注编号
for i, (xi, yi) in enumerate(zip(x, y), start=0):
    plt.text(xi+5, yi+5, str(i), color="yellow", fontsize=12)

plt.title("Points on Image")
plt.axis("off")
plt.show()