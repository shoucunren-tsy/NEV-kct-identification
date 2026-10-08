"""01 数据清洗：合并年度原始表、清洗 IPC、字段规范、缺失值处理。

输入：data/raw/<年度文件夹>/*.xlsx（表头在第 2 行）
输出：outputs/01_cleaning/01_合并专利数据.parquet
参数：config.yaml → cleaning
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from utils import in_dir, load_config, out_dir

CFG = load_config()
CLEAN = CFG["cleaning"]

# 需要读取的列（缺失的列会被自动跳过）
USE_COLS = [
    "公开号", "公开年",
    "IPC 小类", "IPC 小类- DWPI", "IPC 现版完整",
    "施引专利计数", "权利要求计数", "DWPI 同族专利成员计数",
    "DWPI 手工代码", "公开专利文献类型识别代码",
    "标题 (英语)", "标题 (原始语言)",
]

# 清洗后的标准 IPC 小类格式：1 字母(A-H) + 2 数字 + 1 字母
IPC_PATTERN = CLEAN["ipc_regex"].strip("^$")


def clean_ipc_string(val) -> str:
    """拆逗号/管道符 → 逐项校验 4 位 IPC 小类格式 → 去重 → 拼回。"""
    if pd.isna(val) or str(val).strip() in ("", "nan"):
        return ""
    parts = str(val).replace("|", ",").replace("，", ",").replace(" ", "").split(",")
    valid, seen = [], set()
    for p in parts:
        p = p.strip().upper()
        if re.fullmatch(IPC_PATTERN, p) and p not in seen:
            valid.append(p)
            seen.add(p)
    return ", ".join(valid)


def read_raw_tables() -> pd.DataFrame:
    """遍历 data/raw 下各年度文件夹，读取其中的 xlsx 并纵向合并。"""
    data_dir = in_dir("data_raw")
    frames = []
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        xlsx = [f for f in folder.iterdir() if f.suffix.lower() == ".xlsx"]
        if not xlsx:
            continue
        fp = xlsx[0]
        existing = pd.read_excel(fp, header=1, nrows=0, engine="calamine").columns
        read_cols = [c for c in USE_COLS if c in existing]
        df = pd.read_excel(fp, header=1, usecols=read_cols, engine="calamine")
        print(f"{folder.name[:6]:6s} | {len(df):>7,} 条 | {len(read_cols)}/{len(USE_COLS)} 列")
        frames.append(df)
    if not frames:
        raise SystemExit(f"data/raw 下未找到 xlsx，请先按 data/DATA_NOTES.md 准备数据：{data_dir}")
    return pd.concat(frames, ignore_index=True)


def merge_ipc(df: pd.DataFrame) -> pd.DataFrame:
    """同族内优先取 DWPI 标引 IPC，缺失时用原始版兜底。"""
    prefer_dwpi = CLEAN["ipc_prefer"] == "DWPI"
    dwpi_col, orig_col = "IPC 小类- DWPI", "IPC 小类"
    has_dwpi, has_orig = dwpi_col in df.columns, orig_col in df.columns

    if has_dwpi and has_orig:
        primary, fallback = (dwpi_col, orig_col) if prefer_dwpi else (orig_col, dwpi_col)
        valid = df[primary].notna() & (df[primary].astype(str).str.strip() != "")
        df["IPC原始"] = df[primary].where(valid, df[fallback])
        print(f"IPC合并: 主用={primary}({valid.sum():,}), 兜底={fallback}({(~valid).sum():,})")
    elif has_dwpi or has_orig:
        col = dwpi_col if has_dwpi else orig_col
        df["IPC原始"] = df[col]
        print(f"IPC: 仅 {col} 可用")
    else:
        df["IPC原始"] = None
        print("警告: 无可用 IPC 字段")
    return df


def main() -> None:
    full = read_raw_tables()
    print(f"\n合并: {len(full):,} 条")

    # 年份过滤
    full["年份"] = pd.to_numeric(full["公开年"], errors="coerce")
    y0, y1 = CLEAN["year_start"], CLEAN["year_end"]
    full = full[(full["年份"] >= y0) & (full["年份"] <= y1)]
    print(f"年份过滤 ({y0}-{y1}): {len(full):,} 条")

    # IPC 合并 + 清洗
    full = merge_ipc(full)
    full["IPC"] = full["IPC原始"].apply(clean_ipc_string)
    before = len(full)
    full = full[full["IPC"] != ""]
    print(f"IPC缺失剔除: {before - len(full):,} 条 → 剩余 {len(full):,} 条")

    # 缺失值处理
    miss = CLEAN["missing"]
    full["施引专利计数"] = pd.to_numeric(full["施引专利计数"], errors="coerce").fillna(miss["citation"]).astype(int)
    claims = pd.to_numeric(full["权利要求计数"], errors="coerce")
    med = claims.median()
    full["权利要求计数"] = claims.fillna(med if not np.isnan(med) else 1)
    full["DWPI 同族专利成员计数"] = pd.to_numeric(
        full["DWPI 同族专利成员计数"], errors="coerce").fillna(miss["family_size"]).astype(int)
    print(f"权利要求中位数: {med:.0f}")

    # 技术覆盖范围：完整 IPC 用 | 分隔的段数
    if "IPC 现版完整" in full.columns:
        full["IPC覆盖数"] = full["IPC 现版完整"].apply(
            lambda x: len(str(x).split("|")) if pd.notna(x) and str(x).strip() not in ("", "nan") else 1)
    else:
        full["IPC覆盖数"] = 1

    final_cols = [
        "公开号", "年份", "IPC",
        "施引专利计数", "权利要求计数", "DWPI 同族专利成员计数",
        "IPC覆盖数", "DWPI 手工代码",
        "公开专利文献类型识别代码", "标题 (英语)", "标题 (原始语言)",
    ]
    full = full[[c for c in final_cols if c in full.columns]]

    out_path = out_dir("01_cleaning") / "01_合并专利数据.parquet"
    full.to_parquet(out_path, index=False)
    print(f"\n保存: {out_path}")
    print(f"记录数: {len(full):,} | 年份: {int(full['年份'].min())}-{int(full['年份'].max())}")
    print(f"列: {list(full.columns)}")


if __name__ == "__main__":
    main()
