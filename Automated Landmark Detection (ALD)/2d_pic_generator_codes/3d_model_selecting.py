import os
import zipfile
from pathlib import Path

# 配置路径
input_folder = r"C:\\skull_models"   # 存放所有 zip 的目录
output_folder = r"D:\\Codes\\Skull_Landmarks_TL\\skull_models"     # clear.stl 输出目录
Path(output_folder).mkdir(parents=True, exist_ok=True)

for fname in os.listdir(input_folder):
    if not fname.lower().endswith(".zip"):
        continue

    zip_path = os.path.join(input_folder, fname)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            if member.lower().endswith("clear.stl") and not member.endswith('/'):
                # 取出 zip 内部原始文件名（不带路径）
                out_name = os.path.basename(member)
                out_path = Path(output_folder) / out_name

                # 如果有重名文件，自动加序号避免覆盖
                if out_path.exists():
                    i = 1
                    while True:
                        new_name = f"{out_path.stem}_{i}{out_path.suffix}"
                        new_path = Path(output_folder) / new_name
                        if not new_path.exists():
                            out_path = new_path
                            break
                        i += 1

                # 提取该文件
                with zf.open(member) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())

                print(f"提取完成: {out_path}")
