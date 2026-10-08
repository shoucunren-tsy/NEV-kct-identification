"""05 社区划分：Leiden 算法 + 结构洞（Burt）与中心度。

结构洞得分 sh = w_bc·中介中心度 + w_ct·(1 − 约束度)。
本步只产出「IPC—社区—结构洞」全表；候选技术筛选在 06 步完成。

输入：outputs/04_network/04_技术关联边列表_P90.csv
      outputs/03_hotspot/03_IPC年度时间序列.csv
输出：outputs/05_community/{05_全部IPC社区与结构洞, 05_社区划分结果}.csv
参数：config.yaml → community
"""

from __future__ import annotations

import igraph as ig
import leidenalg
import networkx as nx
import numpy as np
import pandas as pd

from utils import NEV_CORES, load_config, out_dir

COM = load_config()["community"]
_SH = COM["structural_hole"]


def main() -> None:
    edges_df = pd.read_csv(out_dir("04_network") / "04_技术关联边列表_P90.csv")
    ts = pd.read_csv(out_dir("03_hotspot") / "03_IPC年度时间序列.csv", index_col=0)
    all_ipcs = ts.index.tolist()
    print(f"加载边列表: {len(edges_df):,} 条边, {len(all_ipcs)} 个IPC")

    # 建 networkx 图（weight 沿用 Jaccard 值，与原实现一致以复现论文结果）
    G = nx.Graph()
    G.add_nodes_from(all_ipcs)
    for _, row in edges_df.iterrows():
        G.add_edge(row["IPC_A"], row["IPC_B"], weight=row["Jaccard"])

    # 转 igraph 跑 Leiden
    ipc_to_id = {ipc: i for i, ipc in enumerate(all_ipcs)}
    id_to_ipc = {i: ipc for i, ipc in enumerate(all_ipcs)}
    g = ig.Graph(
        n=len(all_ipcs),
        edges=[(ipc_to_id[a], ipc_to_id[b]) for a, b in G.edges()],
        directed=False,
    )
    g.es["weight"] = [G[a][b]["weight"] for a, b in G.edges()]

    if COM["algorithm"] != "leiden":
        raise SystemExit(f"未实现的社区算法: {COM['algorithm']}")
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, weights=g.es["weight"], seed=COM["seed"])
    n_comm = len(set(partition.membership))
    # 注意：partition.quality() 返回 RB 配置模型质量，不是模块度；
    # 标准（Newman）模块度须用 igraph 按同一分区计算。
    q = g.modularity(partition.membership, weights=g.es["weight"])
    print(f"社区数: {n_comm}, 标准模块度 Q: {q:.4f}（RB 质量 {partition.quality():.4f}）")

    ipc_community = {id_to_ipc[i]: m for i, m in enumerate(partition.membership)}

    # 结构洞：仅在最大连通分量上计算
    main_nodes = max(nx.connected_components(G), key=len)
    G_main = G.subgraph(main_nodes).copy()
    n_main = G_main.number_of_nodes()
    print(f"主分量: {n_main} 个节点 ({n_main / G.number_of_nodes():.1%})")

    betweenness = nx.betweenness_centrality(G_main, weight="weight", normalized=True)
    constraint = nx.constraint(G_main, weight="weight")
    degree_cent = {n: d / (n_main - 1) if n_main > 1 else 0 for n, d in G_main.degree()}
    pagerank = nx.pagerank(G_main, weight="weight", alpha=0.85)

    records = []
    for ipc in all_ipcs:
        if ipc in main_nodes:
            bc, ct = betweenness.get(ipc, 0), constraint.get(ipc, 1)
            records.append({
                "IPC": ipc, "社区": ipc_community.get(ipc, -1),
                "中介中心度": bc, "约束度": ct, "自由度": 1.0 - ct,
                "结构洞得分": _SH["betweenness_weight"] * bc + _SH["constraint_weight"] * (1.0 - ct),
                "度中心度": degree_cent.get(ipc, 0), "PageRank": pagerank.get(ipc, 0),
                "是否在主分量": True,
            })
        else:
            records.append({
                "IPC": ipc, "社区": -1, "中介中心度": 0, "约束度": 1, "自由度": 0,
                "结构洞得分": 0, "度中心度": 0, "PageRank": 0, "是否在主分量": False,
            })
    sh_df = pd.DataFrame(records).sort_values("结构洞得分", ascending=False).reset_index(drop=True)
    sh_df["总排名"] = range(1, len(sh_df) + 1)

    # NEV 核心 IPC 的社区归属（供核对）
    nev_communities: dict = {}
    for ipc in NEV_CORES:
        nev_communities.setdefault(ipc_community.get(ipc, -1), []).append(ipc)
    print(f"NEV核心IPC分布: {len(nev_communities)} 个社区")

    d = out_dir("05_community")
    sh_df.to_csv(d / "05_全部IPC社区与结构洞.csv", encoding="utf-8-sig", index=False)
    pd.DataFrame([(id_to_ipc[i], m) for i, m in enumerate(partition.membership)],
                 columns=["IPC", "社区"]).to_csv(
        d / "05_社区划分结果.csv", encoding="utf-8-sig", index=False)
    print(f"Leiden Q = {q:.4f}, 社区数 = {n_comm}")
    print(f"保存至 {d}")


if __name__ == "__main__":
    main()
