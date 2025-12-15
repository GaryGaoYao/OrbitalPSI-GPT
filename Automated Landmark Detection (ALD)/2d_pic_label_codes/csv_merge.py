#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并指定文件夹下的所有 CSV 文件，去除有缺失值的行
"""

import os
import pandas as pd

# ===== 在这里修改输入/输出文件夹路径 =====
INPUT_DIR = r"D:\Codes\Skull_Landmarks_TL\Mission-skull-labels\All"
OUTPUT_CSV = r"D:\Codes\Skull_Landmarks_TL\Mission-skull-labels\merged_all.csv"

def main():
    all_dfs = []

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".csv"):
            filepath = os.path.join(INPUT_DIR, filename)
            try:
                df = pd.read_csv(filepath)

                # 去掉有缺失值的行（只保留“全满列”的行）
                df.dropna(how="any", inplace=True)

                all_dfs.append(df)
                print(f"✅ 已加载: {filename}, 保留 {len(df)} 行")

            except Exception as e:
                print(f"⚠️ 跳过 {filename}, 错误: {e}")

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"🎉 合并完成: {OUTPUT_CSV}, 总行数 {len(merged_df)}")
    else:
        print("❌ 没有找到可合并的 CSV 文件。")

if __name__ == "__main__":
    main()
