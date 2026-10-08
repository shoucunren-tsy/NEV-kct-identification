"""02 指标构建：从原始表补充字段，派生经济性/战略性等指标。

输入：data/raw/<年度文件夹>/*.xlsx + outputs/01_cleaning/01_合并专利数据.parquet
输出：outputs/02_indicators/01_合并专利数据_增强版.parquet
说明：技术原创性、技术创新性、战略可控性三项运行时指标在 07 步计算（依赖候选 IPC）。
"""

from __future__ import annotations

import re

import pandas as pd

from utils import in_dir, out_dir

# 需从原始表补充读取的列（公开号用于对齐）
EXTRA_COLS = [
    "公开号",
    "引用的参考文献计数 - 非专利",   # 科学关联度
    "专利权人代码 - DWPI",          # 市场成熟度
    "PCT 公开号",                   # 市场辐射度
    "发明人计数",                   # 市场效益度
    "指定国/地区",                  # 战略覆盖度
    "优先权国家/地区 - DWPI",       # 战略自主性
]


def read_extra_fields() -> pd.DataFrame:
    data_dir = in_dir("data_raw")
    frames = []
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        xlsx = [f for f in folder.iterdir() if f.suffix.lower() == ".xlsx"]
        if not xlsx:
            continue
        fp = xlsx[0]
        existing = pd.read_excel(fp, header=1, nrows=0, engine="calamine").columns
        read_cols = [c for c in EXTRA_COLS if c in existing]
        frames.append(pd.read_excel(fp, header=1, usecols=read_cols, engine="calamine"))
        print(f"  {folder.name[:6]:6s}: {len(frames[-1]):>7,} 条")
    return pd.concat(frames, ignore_index=True)


def parse_patentee_type(code_str) -> float:
    """从 DWPI 专利权人代码解析类型：企业 1.0 / 政府 0.8 / 大学 0.7 / 个人 0.3。"""
    if pd.isna(code_str) or str(code_str).strip() == "":
        return 0.5
    m = re.search(r"\|([CIUG])\|", str(code_str))
    if not m:
        return 0.5
    return {"C": 1.0, "G": 0.8, "U": 0.7, "I": 0.3}[m.group(1)]


def count_countries(val) -> int:
    """指定国/地区计数（分号/竖线/逗号分隔），至少 1。"""
    if pd.isna(val) or str(val).strip() in ("", "nan"):
        return 1
    parts = [p.strip() for p in re.split(r"[;|,]\s*", str(val).strip()) if p.strip()]
    return max(1, len(parts))


def main() -> None:
    extra = read_extra_fields()
    print(f"补充字段合并: {len(extra):,} 条")

    main_df = pd.read_parquet(out_dir("01_cleaning") / "01_合并专利数据.parquet")
    before = len(main_df)
    main_df = main_df.merge(extra, on="公开号", how="left")
    assert len(main_df) == before, f"合并后行数变化: {before} → {len(main_df)}"

    # ---- 派生指标 ----
    main_df["科学关联度"] = pd.to_numeric(main_df["引用的参考文献计数 - 非专利"], errors="coerce").fillna(0)
    main_df["市场成熟度"] = main_df["专利权人代码 - DWPI"].apply(parse_patentee_type)
    main_df["市场辐射度"] = main_df["PCT 公开号"].notna().astype(float)
    main_df["市场效益度"] = pd.to_numeric(main_df["发明人计数"], errors="coerce").fillna(1)
    main_df["战略覆盖度"] = main_df["指定国/地区"].apply(count_countries)
    main_df["战略自主性"] = main_df["优先权国家/地区 - DWPI"].apply(
        lambda v: 1.0 if pd.notna(v) and "CN" in str(v).upper() else 0.0)

    # ---- 重命名，对齐论文指标名 ----
    main_df = main_df.rename(columns={
        "施引专利计数": "技术关联性",
        "权利要求计数": "技术复杂性",
        "IPC覆盖数": "战略安全性",
        "DWPI 同族专利成员计数": "市场价值度",
    })

    out_path = out_dir("02_indicators") / "01_合并专利数据_增强版.parquet"
    main_df.to_parquet(out_path, index=False)
    print(f"\n保存: {out_path}")
    print(f"记录数: {len(main_df):,}")
    print(f"列: {list(main_df.columns)}")


if __name__ == "__main__":
    main()
