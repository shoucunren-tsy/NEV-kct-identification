"""KCT_Identification 公共模块。

集中放置：配置读取、路径解析、指标定义、赋权函数、专利级指标计算。
各阶段脚本统一从这里取，避免各自硬编码路径与重复定义。
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Windows 控制台默认 GBK，直接 print 中文可能 UnicodeEncodeError；强制 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent


# ============================================================
# 配置与路径
# ============================================================
def load_config() -> dict:
    """读取 config.yaml。"""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def in_dir(key: str) -> Path:
    """按 config.paths 的键解析输入目录，如 in_dir('data_raw')。"""
    return (ROOT / load_config()["paths"][key]).resolve()


def out_dir(stage: str) -> Path:
    """返回某阶段产出目录并确保存在，如 out_dir('04_network')。"""
    d = (ROOT / load_config()["paths"]["outputs"] / stage).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================
# 指标定义
# ============================================================
# 13 项原始指标。论文最终赋权口径为其中 12 项（去掉「技术复杂性」）
INDICATORS_13 = [
    "技术原创性", "技术关联性", "技术复杂性", "科学关联度", "技术创新性",
    "市场价值度", "市场成熟度", "市场辐射度", "市场效益度",
    "战略覆盖度", "战略自主性", "战略可控性", "战略安全性",
]
REMOVE_FOR_ADDITIVE = "技术复杂性"
INDICATORS_12 = [c for c in INDICATORS_13 if c != REMOVE_FOR_ADDITIVE]

# NEV 领域核心 IPC（仅用于结果展示标注，不参与计算）
NEV_CORES = {
    "H01M", "B60L", "H02J", "B60K", "B60W", "H02K", "H02M", "H02P",
    "B62D", "G01R", "F16H", "B60T", "B60H", "G01S", "H05K", "B60R", "H01L",
}

# IPC 小类 → 中文含义（公开分类知识）
IPC_MEANING = {
    "H01M": "电池/储能", "B60L": "电动车辆推进", "B60K": "动力布置",
    "B60W": "混合动力控制", "H02J": "充放电/供电", "H02K": "电机",
    "H02M": "功率变换/逆变器", "H02P": "电机控制", "B62D": "车身/底盘",
    "G01R": "电测量/检测", "B60R": "车辆配件", "H01R": "电连接器",
    "B60H": "热管理", "F02D": "发动机控制", "F16H": "传动装置",
    "G06F": "数据处理", "G06Q": "管理系统", "H04L": "通信传输",
    "B60T": "制动系统", "G05D": "自动控制", "G08G": "交通控制",
    "G01S": "雷达定位", "H04W": "无线通信", "B25J": "机械手/自动化",
    "G05B": "控制系统", "H05K": "印刷电路", "H01L": "半导体",
    "G01M": "力学测试", "H01F": "电磁元件", "H01H": "开关/继电器",
    "F28D": "热交换", "F28F": "热交换部件", "G06N": "AI/机器学习",
    "G06V": "图像识别", "H01B": "电缆/导体", "B60Q": "车辆信号/照明",
    "B60S": "车辆保养", "C01B": "碳/纳米材料", "B32B": "层状复合材料",
    "B01D": "分离/过滤", "C02F": "水处理", "C12M": "酶/微生物",
    "C23C": "涂层/镀膜", "C22B": "冶金/提取", "A23L": "食品加工",
    "D06M": "纺织品处理", "D06C": "纺织品后整理", "D02G": "纺纱/纤维",
    "F41H": "装甲/防护", "H03H": "阻抗网络", "B25F": "组合工具",
    "B26D": "切割", "G21F": "辐射防护", "G21C": "核反应堆",
    "A41D": "服装", "A61L": "医用材料", "A62C": "消防",
    "B08B": "清洁", "B21D": "金属冲压", "B23K": "焊接",
    "B23Q": "机床", "B28D": "石材加工", "B41J": "打印",
    "B60J": "车窗/密封", "B61B": "轨道运输", "B65D": "包装容器",
    "B65G": "输送", "E01H": "道路清洁", "E02D": "地基/基础",
    "E04H": "建筑", "F01P": "发动机冷却", "F03G": "弹簧/储能",
    "F15B": "液压", "F23G": "焚烧/废热", "G01N": "材料检测",
    "G08B": "报警/信号", "G16H": "医疗信息",
}


# ============================================================
# 专利级指标（运行时计算）
# ============================================================
def split_ipc(value) -> list[str]:
    """把逗号分隔的 IPC 字符串拆成规范小类列表。"""
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [s.strip().upper() for s in str(value).split(",") if len(s.strip()) >= 4]


def build_patent_indicators(patents: pd.DataFrame, target_ipcs: set) -> pd.DataFrame:
    """在增强版专利表上补算 3 项运行时指标：技术创新性、战略可控性、技术原创性。

    技术原创性 = 该专利的 IPC 组合对中，首次出现（按年份增量）的对数占比。
    """
    # 技术创新性：发明（A/B 授权文献）记 1.0，实用新型（U/Y）记 0.5
    type_map = {}
    for raw in patents["公开专利文献类型识别代码"].dropna().unique():
        code = str(raw).strip().upper()
        if code.startswith(("A", "B")):
            type_map[code] = 1.0
        elif code.startswith(("U", "Y")):
            type_map[code] = 0.5
        else:
            type_map[code] = 0.5
    patents["技术创新性"] = patents["公开专利文献类型识别代码"].map(type_map).fillna(0.5)

    # 战略可控性：专利涉及几个 IPC 部（首字母 A-H）
    patents["战略可控性"] = patents["IPC"].apply(
        lambda s: len({c[0] for c in split_ipc(s) if c[0] in "ABCDEFGH"}) or 1
    )

    # 技术原创性：按年份累积「未见过的目标 IPC 对」
    patents["_ipc_list"] = patents["IPC"].apply(split_ipc)
    patents["_pairs"] = patents["_ipc_list"].apply(
        lambda lst: {tuple(sorted(p)) for p in combinations(lst, 2)
                     if p[0] in target_ipcs and p[1] in target_ipcs}
    )
    seen: set = set()
    originality = pd.Series(0.0, index=patents.index)
    for year in sorted(patents["年份"].unique()):
        for idx in patents.index[patents["年份"] == year]:
            pairs = patents.at[idx, "_pairs"]
            if pairs:
                original = float(len(pairs - seen))
                originality.at[idx] = original / len(pairs)
                seen.update(pairs)
    patents["技术原创性"] = originality
    patents.drop(columns=["_ipc_list", "_pairs"], inplace=True)
    return patents


# ============================================================
# 赋权函数
# ============================================================
def minmax(X: np.ndarray, zero_fill: float = 1e-10) -> np.ndarray:
    """按列 min-max 归一化；常数列用 zero_fill 兜底避免除零。"""
    lo, hi = X.min(axis=0), X.max(axis=0)
    rng = hi - lo
    rng = np.where(rng == 0, zero_fill, rng)
    return (X - lo) / rng


def entropy_weights(X: np.ndarray) -> np.ndarray:
    """熵权法：E=-k·Σ p·ln p, k=1/ln(n)；w=(1-E)/Σ(1-E)。"""
    P = minmax(X, zero_fill=1e-10) + 1e-10
    P_norm = P / P.sum(axis=0)
    n = X.shape[0]
    E = -(1.0 / np.log(n)) * np.sum(P_norm * np.log(P_norm), axis=0)
    return (1 - E) / (1 - E).sum()


def critic_weights(X: np.ndarray) -> np.ndarray:
    """CRITIC 法（相关系数取绝对值）：C=σ·Σ(1-|r|)；w=C/ΣC。"""
    Z = minmax(X, zero_fill=1.0)
    std = Z.std(axis=0, ddof=0)
    corr = np.corrcoef(Z.T)
    conflict = np.sum(1.0 - np.abs(corr), axis=0)
    C = std * conflict
    return C / C.sum()


def additive_weights(w_entropy: np.ndarray, w_critic: np.ndarray,
                     drop_index: int | None = None) -> np.ndarray:
    """加法组合：(熵权 + CRITIC)/2，再归一。

    drop_index 用于「先按 13 指标算权重、再删掉某一指标、在剩余项上重新归一」的口径
    —— 论文最终采用 12 指标版（删「技术复杂性」）。此时熵/CRITIC 列沿用 13 指标值不重算。
    """
    w = (w_entropy + w_critic) / 2.0
    if drop_index is not None:
        w = np.delete(w, drop_index)
    return w / w.sum()
