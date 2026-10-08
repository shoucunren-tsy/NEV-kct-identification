"""06 候选筛选：结构洞前 top_ratio ∪ 度中心度前 top_ratio（逐社区）。

各社区内分别取结构洞、度中心度前 top_ratio，取并集；社区规模小于
min_community_size 的剔除。本步是候选技术的唯一产出点。

输入：outputs/05_community/05_全部IPC社区与结构洞.csv
输出：outputs/06_selection/{06_初始关键核心技术_全量, 06_初始关键核心技术_有效}.csv
参数：config.yaml → selection
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import NEV_CORES, load_config, out_dir

SEL = load_config()["selection"]


def main() -> None:
    sh = pd.read_csv(out_dir("05_community") / "05_全部IPC社区与结构洞.csv")
    min_size = SEL["min_community_size"]
    ratio = SEL["top_ratio"]

    selected = []
    for _, grp in sh[sh["社区"] >= 0].groupby("社区"):
        if len(grp) < min_size:
            continue
        n_select = max(1, int(np.ceil(len(grp) * ratio)))
        sh_top = grp.nlargest(n_select, "结构洞得分")
        dc_top = grp.nlargest(n_select, "度中心度")
        selected.append(pd.concat([sh_top, dc_top]).drop_duplicates(subset="IPC"))

    initial = pd.concat(selected, ignore_index=True)
    initial = initial.sort_values("结构洞得分", ascending=False).reset_index(drop=True)
    initial["初始排名"] = range(1, len(initial) + 1)
    valid = initial[initial["结构洞得分"] > 0].copy()

    nev_in = [ipc for ipc in NEV_CORES if ipc in set(valid["IPC"])]
    print(f"双维度筛选: {len(valid)} 个初始关键核心技术（社区规模 ≥{min_size}，前 {ratio:.0%}）")
    print(f"NEV核心入选: {len(nev_in)}/{len(NEV_CORES)} — {nev_in}")

    d = out_dir("06_selection")
    initial.to_csv(d / "06_初始关键核心技术_全量.csv", encoding="utf-8-sig", index=False)
    valid.to_csv(d / "06_初始关键核心技术_有效.csv", encoding="utf-8-sig", index=False)
    print(f"保存至 {d}")


if __name__ == "__main__":
    main()
