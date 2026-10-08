"""03 热点技术识别：按专利内 IPC 数加权，累计频次得热点技术主题。

单专利内每个 IPC 权重 = 1 / 该专利的 IPC 数；跨专利累加得加权频次。
加权总频次 ≥ min_weighted_freq 的 IPC 记为高频技术。

输入：outputs/01_cleaning/01_合并专利数据.parquet
输出：outputs/03_hotspot/{03_高频IPC列表, 03_IPC年度时间序列, 03_全部IPC年度时间序列}.csv
参数：config.yaml → hotspot
"""

from __future__ import annotations

import pandas as pd

from utils import IPC_MEANING, load_config, out_dir

HOTSPOT = load_config()["hotspot"]


def main() -> None:
    df = pd.read_parquet(out_dir("01_cleaning") / "01_合并专利数据.parquet")
    print(f"加载: {len(df):,} 条专利")

    # 展开：一条专利 → 每个 IPC 一行
    df["IPC列表"] = df["IPC"].apply(lambda x: [v.strip() for v in str(x).split(",") if v.strip()])
    df["IPC个数"] = df["IPC列表"].apply(len)
    exploded = df[["公开号", "年份", "IPC列表", "IPC个数"]].explode("IPC列表").rename(
        columns={"IPC列表": "IPC"})
    print(f"展开后: {len(exploded):,} 条 IPC-专利记录")

    # 单专利内每 IPC 权重（第 1 步已去重，故每个 IPC 在专利内最多出现 1 次）
    if HOTSPOT["ipc_weight"] != "inverse_count":
        raise SystemExit(f"未实现的 ipc_weight: {HOTSPOT['ipc_weight']}")
    exploded["权重"] = 1.0 / exploded["IPC个数"]

    # 年度加权频次 → 总加权频次
    yearly = exploded.groupby(["IPC", "年份"])["权重"].sum().reset_index(name="加权频次")
    total = yearly.groupby("IPC")["加权频次"].sum().reset_index(name="总加权频次")
    total = total.sort_values("总加权频次", ascending=False).reset_index(drop=True)
    total["排名"] = range(1, len(total) + 1)

    min_freq = HOTSPOT["min_weighted_freq"]
    high = total[total["总加权频次"] >= min_freq].copy()
    print(f"全部IPC: {len(total)} | 高频IPC (≥{min_freq}): {len(high)} | 占比: {len(high)/len(total):.1%}")

    # 年度矩阵（列=年份，行=IPC）
    pivot = yearly.pivot(index="IPC", columns="年份", values="加权频次").fillna(0).astype(float)
    pivot.columns = [int(c) for c in pivot.columns]
    pivot = pivot[sorted(pivot.columns)]
    high_pivot = pivot.loc[pivot.index.isin(high["IPC"])]

    d = out_dir("03_hotspot")
    high_pivot.to_csv(d / "03_IPC年度时间序列.csv", encoding="utf-8-sig")
    high.to_csv(d / "03_高频IPC列表.csv", encoding="utf-8-sig", index=False)
    pivot.to_csv(d / "03_全部IPC年度时间序列.csv", encoding="utf-8-sig")
    print(f"保存至 {d}（高频 {high_pivot.shape[0]} IPC × {high_pivot.shape[1]} 年）")

    print(f"\nTOP 30 高频 IPC")
    for _, row in high.head(30).iterrows():
        print(f"{row['排名']:<5} {row['IPC']:<8} {IPC_MEANING.get(row['IPC'], ''):<20} {row['总加权频次']:>10.1f}")


if __name__ == "__main__":
    main()
