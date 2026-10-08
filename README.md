# 一种基于Jaccard共现系数和Leiden社区发现算法的关键核心技术识别方法——以新能源汽车领域为例

**Identification of Key Core Technologies Based on Jaccard Co-occurrence Coefficients and Leiden Community Detection: A Case Study of the New Energy Vehicle Industry**

本仓库提供上述论文的 Python 实现与可复现研究流程，涵盖 IPC 加权频次分析、Jaccard 技术关联网络构建、Leiden 社区发现、结构洞与度中心度筛选、熵权-CRITIC 专利综合评价，以及关键核心技术识别与验证。

This repository provides the Python implementation and reproducible research pipeline for the paper, using patent data from the new energy vehicle industry.

> **本仓库只提供代码，不含任何数据。** Derwent 为 Clarivate 商业数据库，受订阅许可约束不可再分发；
> 使用者需自行通过机构订阅获取数据，按 `data/DATA_NOTES.md` 放好后运行即可复现。
>
> 方法论框架复现并改进自 **万校基等（2025）**。

## 方法改进点

相对原始方法论框架，本项目做了三处改进：

| # | 原方法 | 本项目 | 理由 |
|---|--------|--------|------|
| 1 | Matrix Profile 相似度衡量技术关联 | **Jaccard 共现系数** | 计算更稳定、可解释性强、无需滑窗对齐 |
| 2 | Louvain 社区发现 | **Leiden 算法** | 社区质量更优，兼容性更好 |
| 3 | IPC 层级均值打分 | **逐专利熵权打分** | 保留专利级异质性，避免均值抹平 |

## 流程（9 步）

| 阶段 | 脚本 | 做什么 | 关键产出 |
|------|------|--------|----------|
| 01 | `01_data_cleaning.py` | 合并原始表、清洗 IPC、字段规范、缺失值处理 | 合并专利表 |
| 02 | `02_indicators.py` | 补充字段、派生 12 项指标 | 指标增强表 |
| 03 | `03_hotspot.py` | IPC 加权频次 → 高频技术识别 | 高频 IPC 列表 |
| 04 | `04_network.py` | Jaccard 共现网络构建 + 阈值筛选 | 技术关联边列表 |
| 05 | `05_community.py` | Leiden 社区划分 + 结构洞/中心度 | IPC 社区与结构洞表 |
| 06 | `06_selection.py` | 结构洞 ∪ 度中心度 双维度筛选 | 初始候选技术 |
| 07 | `07_weighting.py` | 熵权 + CRITIC 加法组合赋权、逐专利打分 | 指标权重、专利得分 |
| 08 | `08_technology.py` | 高分专利计数 → 关键核心技术识别 | 最终技术清单 |
| 09 | `09_validation.py` | 赋权方法与阈值稳健性检验 | 稳健性对比表 |

全部参数集中在 `config.yaml`，按顺序执行 `run_pipeline.py` 即可跑通整条流程。

## 目录结构

```
nev-kct-identification/
├── README.md
├── LICENSE                 # MIT License
├── requirements.txt
├── config.yaml             # 全部可调参数
├── utils.py                # 公共模块：配置/路径/指标定义/赋权函数
├── run_pipeline.py         # 流水线入口（顺序调用 01~09）
├── 01_data_cleaning.py     # 01 数据清洗
├── 02_indicators.py        # 02 指标构建
├── 03_hotspot.py           # 03 热点技术识别
├── 04_network.py           # 04 技术关联网络
├── 05_community.py         # 05 社区划分 + 结构洞
├── 06_selection.py         # 06 双维度候选筛选
├── 07_weighting.py         # 07 赋权与逐专利打分
├── 08_technology.py        # 08 关键核心技术识别
├── 09_validation.py        # 09 稳健性检验
├── data/                   # 仅作投放口，本仓库不含任何数据
│   └── DATA_NOTES.md       # 数据来源与获取说明
├── outputs/                # 各阶段产出（不入库）
│   ├── 01_cleaning/
│   │   ...
│   └── 09_validation/
└── docs/
    └── METHOD.md           # 方法细节
```

## 快速开始

```bash
# 1. 建虚拟环境
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash

# 2. 装依赖（国内建议先配镜像源）
pip install -r requirements.txt

# 3. 准备数据（见 data/DATA_NOTES.md），放到 data/raw/

# 4. 跑全流程
python run_pipeline.py
```

## 引用

若本流程对你的研究有帮助，请引用：

> 一种基于Jaccard共现系数和Leiden社区发现算法的关键核心技术识别方法——以新能源汽车领域为例

（论文链接 / DOI 待补充）

## 许可

本项目以 [MIT License](LICENSE) 开源。

> 依赖说明：路线核心依赖 `leidenalg`（GPL-3.0）与 `python-igraph`（GPL-2.0+）。
> 本项目仅以库形式调用，未修改其源码。
