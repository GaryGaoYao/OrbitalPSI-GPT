#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIA (VGG Image Annotator) JSON -> CSV (filename, cx, cy) converter

- 遍历指定文件夹中的所有 .json 文件
- 保证点的顺序严格按照 JSON 文件中出现的顺序
- 每个 JSON 输出一个对应的 CSV 文件
"""

import json
import csv
import os
from collections import OrderedDict
from typing import Any, Dict

# ===== 在这里修改输入输出文件夹路径 =====
INPUT_DIR = r"D:\Codes\Skull_Landmarks_TL\Mission-skull-labels\Por"
OUTPUT_DIR = r"D:\Codes\Skull_Landmarks_TL\Mission-skull-labels\All"

def process_json(input_json: str, output_csv: str):
    # 保留 JSON 中的顺序
    with open(input_json, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f, object_pairs_hook=OrderedDict)

    rows = []
    max_points = 0  # 用于对齐列数

    # 按顶层 key 顺序遍历
    for key, entry in data.items():
        filename = entry.get('filename') or key
        regions = entry.get('regions', [])

        coords = []
        for region in regions:
            sa = region.get('shape_attributes', {}) or {}
            if sa.get('name') == 'point' and ('cx' in sa) and ('cy' in sa):
                coords.extend([sa['cx'], sa['cy']])

        max_points = max(max_points, len(coords) // 2)
        rows.append([filename] + coords)

    # 生成表头：filename,cx1,cy1,cx2,cy2, ...
    header = ["filename"]
    for i in range(1, max_points + 1):
        header.append(f"cx{i}")
        header.append(f"cy{i}")

    # 写入 CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            while len(row) < len(header):
                row.append("")
            writer.writerow(row)

    print(f"✅ 成功: {input_json} -> {output_csv} (共 {len(rows)} 行)")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".json"):
            input_path = os.path.join(INPUT_DIR, filename)
            output_name = os.path.splitext(filename)[0] + ".csv"
            output_path = os.path.join(OUTPUT_DIR, output_name)

            try:
                process_json(input_path, output_path)
            except Exception as e:
                print(f"⚠️ 跳过: {input_path}, 错误原因: {e}")


if __name__ == "__main__":
    main()