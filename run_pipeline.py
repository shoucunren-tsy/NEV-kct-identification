"""KCT_Identification 流水线入口。

按顺序执行 01~09 各阶段脚本，参数统一从 config.yaml 读取。

用法：
    python run_pipeline.py            # 跑全流程
    python run_pipeline.py --from 04  # 从第 04 步开始
    python run_pipeline.py --only 07  # 只跑第 07 步
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent

# 阶段编号 → 脚本名
STAGES = [
    "01_data_cleaning.py",
    "02_indicators.py",
    "03_hotspot.py",
    "04_network.py",
    "05_community.py",
    "06_selection.py",
    "07_weighting.py",
    "08_technology.py",
    "09_validation.py",
]


def load_config() -> dict:
    """读取 config.yaml。"""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_stage(script: str) -> None:
    """执行单个阶段脚本，失败即中止。"""
    print(f"\n{'=' * 60}\n▶ {script}\n{'=' * 60}")
    result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"[FAIL] {script} 退出码 {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KCT_Identification 流水线")
    parser.add_argument("--from", dest="start", help="从该编号开始（如 04）")
    parser.add_argument("--only", dest="only", help="只跑该编号（如 07）")
    args = parser.parse_args()

    _ = load_config()  # 启动时先校验配置可读

    if args.only:
        script = next((s for s in STAGES if s.startswith(args.only)), None)
        if script is None:
            raise SystemExit(f"未找到阶段：{args.only}")
        run_stage(script)
        return

    stages = STAGES
    if args.start:
        stages = [s for s in STAGES if s >= args.start + "_"]

    for script in stages:
        run_stage(script)

    print("\n✅ 全流程完成")


if __name__ == "__main__":
    main()
