"""09 稳健性检验：赋权方法对比 + 阈值方案对比。

- 赋权方法对比：加法组合 / 纯熵权 / 纯 CRITIC，看最终关键核心技术集合是否一致；
- 阈值方案对比：不同高分分位数下入选技术集合的稳定性。

输入：outputs/02_indicators/01_合并专利数据_增强版.parquet
      outputs/06_selection/06_初始关键核心技术_有效.csv
输出：outputs/09_validation/09_稳健性对比.csv
参数：config.yaml → validation
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import (INDICATORS_13, REMOVE_FOR_ADDITIVE, additive_weights,
                   build_patent_indicators, critic_weights, entropy_weights,
                   load_config, minmax, out_dir, split_ipc)

VAL = load_config()["validation"]


def tech_set(patents: pd.DataFrame, ipc_lists: list, target_ipcs: set,
             scores: np.ndarray, pct: float) -> set[str]:
    """按「高分专利数 ≥ 候选均值」选出 IPC 集合。"""
    thr = np.percentile(scores, pct)
    rows = []
    for ipc in sorted(target_ipcs):
        idx = [i for i, lst in enumerate(ipc_lists) if ipc in lst]
        if not idx:
            continue
        n_high = int((scores[idx] >= thr).sum())
        rows.append((ipc, n_high))
    counts = pd.DataFrame(rows, columns=["IPC", "高分专利数"])
    return set(counts.loc[counts["高分专利数"] >= counts["高分专利数"].mean(), "IPC"])


def main() -> None:
    patents = pd.read_parquet(out_dir("02_indicators") / "01_合并专利数据_增强版.parquet")
    initial = pd.read_csv(out_dir("06_selection") / "06_初始关键核心技术_有效.csv")
    target_ipcs = set(initial["IPC"])
    patents = build_patent_indicators(patents, target_ipcs)
    ipc_lists = patents["IPC"].apply(split_ipc).tolist()

    cols13 = [c for c in INDICATORS_13 if c in patents.columns and patents[c].std() > 0]
    X13 = patents[cols13].values.astype(np.float64)
    w_ent, w_crit = entropy_weights(X13), critic_weights(X13)
    drop = cols13.index(REMOVE_FOR_ADDITIVE) if REMOVE_FOR_ADDITIVE in cols13 else None
    w_add = additive_weights(w_ent, w_crit, drop)
    cols12 = [c for c in cols13 if c != REMOVE_FOR_ADDITIVE]

    methods = {
        "additive": (cols12, w_add),
        "entropy": (cols13, w_ent),
        "critic": (cols13, w_crit),
    }
    print(f"赋权方法: {VAL['weight_methods']} | 阈值方案: {VAL['threshold_schemes']}")

    results = {}
    for name in VAL["weight_methods"]:
        cols, w = methods[name]
        score = (minmax(patents[cols].values.astype(np.float64)) + 1e-10) @ w
        for pct in VAL["threshold_schemes"]:
            results[f"{name}@P{pct}"] = tech_set(patents, ipc_lists, target_ipcs, score, pct)

    base_key = f"{VAL['weight_methods'][0]}@P{VAL['threshold_schemes'][0]}"
    base = results[base_key]
    rows = []
    for key, s in results.items():
        union = len(base | s)
        rows.append({"方案": key, "入选数": len(s), "与基线交集": len(base & s),
                     "Jaccard": round(len(base & s) / union, 4) if union else 0.0})
    out = pd.DataFrame(rows)
    print(f"\n基线: {base_key}（{len(base)} 个）")
    print(out.to_string(index=False))

    out.to_csv(out_dir("09_validation") / "09_稳健性对比.csv", encoding="utf-8-sig", index=False)


if __name__ == "__main__":
    main()
