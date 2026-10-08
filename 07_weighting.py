"""07 赋权与打分：熵权 + CRITIC 加法组合，对每件专利逐件打分。

权重口径（论文最终采用）：
  1. 在 13 项指标上分别算熵权、CRITIC；
  2. 删去「技术复杂性」，熵/CRITIC 列沿用 13 指标值不重算；
  3. 加法组合 = (熵权 + CRITIC)/2，在剩余 12 项上重新归一。

输入：outputs/02_indicators/01_合并专利数据_增强版.parquet
      outputs/06_selection/06_初始关键核心技术_有效.csv
输出：outputs/07_weighting/{07_指标权重, 07_专利得分, 07_得分分布}.csv
参数：config.yaml → weighting
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import (INDICATORS_13, REMOVE_FOR_ADDITIVE, additive_weights,
                   build_patent_indicators, critic_weights, entropy_weights,
                   load_config, minmax, out_dir)

WGT = load_config()["weighting"]


def main() -> None:
    patents = pd.read_parquet(out_dir("02_indicators") / "01_合并专利数据_增强版.parquet")
    initial = pd.read_csv(out_dir("06_selection") / "06_初始关键核心技术_有效.csv")
    target_ipcs = set(initial["IPC"])
    print(f"全量专利: {len(patents):,} 件, 候选技术: {len(initial)} 个")

    patents = build_patent_indicators(patents, target_ipcs)

    # ---- 13 指标赋权 ----
    cols13 = [c for c in INDICATORS_13 if c in patents.columns and patents[c].std() > 0]
    X13 = patents[cols13].values.astype(np.float64)
    w_ent = entropy_weights(X13)
    w_crit = critic_weights(X13)

    # ---- 12 指标加法组合（删「技术复杂性」后归一）----
    drop = cols13.index(REMOVE_FOR_ADDITIVE) if REMOVE_FOR_ADDITIVE in cols13 else None
    w_add = additive_weights(w_ent, w_crit, drop)
    cols12 = [c for c in cols13 if c != REMOVE_FOR_ADDITIVE]
    print(f"赋权: {WGT['method']} | 指标 {len(cols12)} 项（已删「{REMOVE_FOR_ADDITIVE}」）")
    print(f"权重和: 熵权={w_ent.sum():.4f} CRITIC={w_crit.sum():.4f} 加法组合={w_add.sum():.4f}")

    # ---- 逐专利打分：12 项 min-max 归一后按加法组合权重加权求和 ----
    P12 = minmax(patents[cols12].values.astype(np.float64)) + 1e-10
    patents["专利得分"] = P12 @ w_add

    pct = WGT["high_score_percentile"]
    thr_high = np.percentile(patents["专利得分"], pct)
    print(f"高分线（前 {100 - pct}%，P{pct}）：{thr_high:.4f}")

    # ---- 保存 ----
    d = out_dir("07_weighting")

    w_rows = []
    add_map = {c: w for c, w in zip(cols12, w_add)}
    for i, c in enumerate(cols13):
        w_rows.append({
            "指标": c,
            "熵权法": round(float(w_ent[i]), 4),
            "CRITIC全量": round(float(w_crit[i]), 4),
            "加法组合_12指标": round(float(add_map[c]), 4) if c in add_map else "",
        })
    pd.DataFrame(w_rows).to_csv(d / "07_指标权重.csv", encoding="utf-8-sig", index=False)

    patents[["公开号", "年份", "专利得分"]].to_csv(
        d / "07_专利得分.csv", encoding="utf-8-sig", index=False)
    pd.DataFrame({
        "分位数": ["1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%"],
        "专利得分": [np.percentile(patents["专利得分"], p) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)],
    }).to_csv(d / "07_得分分布.csv", encoding="utf-8-sig", index=False)

    print(f"保存至 {d}")
    for _, r in pd.DataFrame(w_rows).iterrows():
        print(f"  {r['指标']:<10} 熵={r['熵权法']:<8} CRITIC={r['CRITIC全量']:<8} 加法={r['加法组合_12指标']}")


if __name__ == "__main__":
    main()
