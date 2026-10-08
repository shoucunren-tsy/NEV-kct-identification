"""08 关键核心技术识别：统计各候选 IPC 的高分专利数，取 ≥ 均值者入选。

高分线 = 全体专利得分前 (100 − high_score_percentile)%，即 P{percentile}。
对每个候选 IPC 计其高分专利数；高分专利数 ≥ 候选间均值者认定为关键核心技术。

输入：outputs/02_indicators/01_合并专利数据_增强版.parquet
      outputs/07_weighting/07_专利得分.csv
      outputs/06_selection/06_初始关键核心技术_有效.csv
输出：outputs/08_technology/{08_全部候选技术计数, 08_最终关键核心技术}.csv
参数：config.yaml → technology, weighting.high_score_percentile
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import IPC_MEANING, NEV_CORES, load_config, out_dir, split_ipc

TECH = load_config()["technology"]
PCT = load_config()["weighting"]["high_score_percentile"]


def main() -> None:
    patents = pd.read_parquet(out_dir("02_indicators") / "01_合并专利数据_增强版.parquet")
    scores = pd.read_csv(out_dir("07_weighting") / "07_专利得分.csv")
    initial = pd.read_csv(out_dir("06_selection") / "06_初始关键核心技术_有效.csv")
    target_ipcs = set(initial["IPC"])

    patents = patents.merge(scores[["公开号", "专利得分"]], on="公开号", how="left")
    patents["IPC_list"] = patents["IPC"].apply(split_ipc)

    thr = np.percentile(patents["专利得分"].dropna(), PCT)
    print(f"候选技术: {len(initial)} 个 | 高分线 P{PCT}: {thr:.4f}")

    rows = []
    for ipc in sorted(target_ipcs):
        mask = patents["IPC_list"].apply(lambda lst: ipc in lst)
        sub = patents.loc[mask, "专利得分"]
        if len(sub) == 0:
            continue
        rows.append({"IPC": ipc, "专利总数": len(sub),
                     "高分专利数": int((sub >= thr).sum())})
    df = pd.DataFrame(rows)
    df["高分占比"] = df["高分专利数"] / df["专利总数"]
    df = initial[["IPC", "社区", "结构洞得分", "度中心度"]].merge(df, on="IPC", how="left")
    df = df.sort_values("高分专利数", ascending=False).reset_index(drop=True)
    df["排名"] = range(1, len(df) + 1)

    if TECH["selection_rule"] != "mean_count":
        raise SystemExit(f"未实现的筛选规则: {TECH['selection_rule']}")
    thr_mean = df["高分专利数"].mean()
    df["是否关键"] = df["高分专利数"] >= thr_mean
    final = df[df["是否关键"]].reset_index(drop=True)
    print(f"高分专利数均值: {thr_mean:.1f} → 关键核心技术 {len(final)} 个")

    d = out_dir("08_technology")
    df.to_csv(d / "08_全部候选技术计数.csv", encoding="utf-8-sig", index=False)
    final.to_csv(d / "08_最终关键核心技术.csv", encoding="utf-8-sig", index=False)

    print(f"\n===== 最终关键核心技术 =====")
    for _, r in final.iterrows():
        tag = " [NEV]" if r["IPC"] in NEV_CORES else ""
        print(f"{int(r['排名']):<4} {r['IPC']:<8} {IPC_MEANING.get(r['IPC'], ''):<20} "
              f"高分 {int(r['高分专利数']):>6,} / {int(r['专利总数']):>7,}{tag}")
    print(f"\n保存至 {d}")


if __name__ == "__main__":
    main()
