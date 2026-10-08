"""04 技术关联网络：Jaccard 共现系数 + 分位数阈值建网。

J(i,j) = c_ij / (c_i + c_j - c_ij)；取 P{percentile} 分位数为边阈值。

输入：outputs/03_hotspot/03_IPC年度时间序列.csv（高频 IPC 清单）
      outputs/01_cleaning/01_合并专利数据.parquet
输出：outputs/04_network/{04_技术关联边列表_P90, 04_共现分布_全量, 04_IPC度排名_P90}.csv
参数：config.yaml → network
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

from utils import load_config, out_dir

NET = load_config()["network"]


def main() -> None:
    hotspot_dir = out_dir("03_hotspot")
    ts = pd.read_csv(hotspot_dir / "03_IPC年度时间序列.csv", index_col=0)
    ipc_list = sorted(ts.index.tolist())
    ipc_set = set(ipc_list)
    n_ipc = len(ipc_list)
    idx = {ipc: i for i, ipc in enumerate(ipc_list)}
    print(f"高频IPC: {n_ipc} 个")

    patents = pd.read_parquet(out_dir("01_cleaning") / "01_合并专利数据.parquet")
    patent_ipcs = patents["IPC"].apply(
        lambda x: [v.strip().upper() for v in str(x).split(",") if v.strip().upper() in ipc_set]
        if pd.notna(x) else [])
    patent_ipcs = patent_ipcs[patent_ipcs.apply(len) > 0]
    print(f"含目标IPC的专利: {len(patent_ipcs):,} / {len(patents):,}")

    # 共现计数 + 每 IPC 的专利数
    cooccur = lil_matrix((n_ipc, n_ipc), dtype=np.int32)
    count = np.zeros(n_ipc, dtype=np.int32)
    for ipcs in patent_ipcs:
        if len(ipcs) == 1:
            count[idx[ipcs[0]]] += 1
        else:
            for a, b in combinations(ipcs, 2):
                i, j = idx[a], idx[b]
                cooccur[i, j] += 1
                cooccur[j, i] += 1
                count[i] += 1
                count[j] += 1
    cooccur = cooccur.tocsr()
    print(f"非零共现对: {cooccur.nnz:,}")

    # Jaccard
    pairs = []
    for i in range(n_ipc):
        ci = count[i]
        if ci == 0:
            continue
        row = cooccur.getrow(i)
        for k, j in enumerate(row.indices):
            if j <= i:
                continue
            cj = count[j]
            if cj == 0:
                continue
            cij = int(row.data[k])
            pairs.append((ipc_list[i], ipc_list[j], cij, cij / (ci + cj - cij)))
    pairs_df = pd.DataFrame(pairs, columns=["IPC_A", "IPC_B", "共现次数", "Jaccard"])
    print(f"有效IPC对: {len(pairs_df):,}")

    # 阈值建网
    if NET["threshold_method"] != "percentile":
        raise SystemExit(f"未实现的阈值方式: {NET['threshold_method']}")
    pct = NET["threshold_percentile"]
    thresh = pairs_df["Jaccard"].quantile(pct / 100.0)
    edges = pairs_df[pairs_df["Jaccard"] >= thresh].sort_values(
        "Jaccard", ascending=False).reset_index(drop=True)

    G = nx.Graph()
    G.add_nodes_from(ipc_list)
    for _, row in edges.iterrows():
        G.add_edge(row["IPC_A"], row["IPC_B"], weight=row["Jaccard"])
    isolated = sum(1 for n in G.nodes() if G.degree(n) == 0)
    largest_cc = max(nx.connected_components(G), key=len)
    print(f"\nP{pct} 网络 (Jaccard >= {thresh:.6f}): 节点 {G.number_of_nodes()}, "
          f"边 {G.number_of_edges():,}, 孤立 {isolated}, 最大连通分量 {len(largest_cc)}")

    d = out_dir("04_network")
    edges[["IPC_A", "IPC_B", "Jaccard"]].to_csv(
        d / "04_技术关联边列表_P90.csv", encoding="utf-8-sig", index=False)
    pairs_df.to_csv(d / "04_共现分布_全量.csv", encoding="utf-8-sig", index=False)
    deg = pd.DataFrame([(n, deg) for n, deg in G.degree()], columns=["IPC", "度"])
    deg.sort_values("度", ascending=False).to_csv(
        d / "04_IPC度排名_P90.csv", encoding="utf-8-sig", index=False)
    print(f"保存至 {d}")


if __name__ == "__main__":
    main()
