#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算 Harbor Agent 轨迹的不确定性（Entropy）并评估分类指标

功能：
1. 遍历 ~/jobs/2026-05-16__19-11-56 下所有任务
2. 从 trajectory.json 提取 agent reasoning / logprobs
3. 跳过 timeout 任务
4. 计算每个任务的 entropy
5. 使用 entropy 作为 uncertainty score
6. 基于 reward(0/1) 计算 AUROC / AUPRC / F1 等指标
7. 输出统计结果与可视化

默认 entropy 定义：
    H = - mean(logprob)

原因：
trajectory 中已经给出了 token-level logprobs。
若 logprob 越低（越负），说明模型越不确定。
因此：
    entropy 越高 => 不确定性越强

支持：
- token mean entropy
- step mean entropy
- normalized entropy

依赖：
    pip install numpy pandas scikit-learn matplotlib tqdm

使用：
    python evaluate_gaia_entropy_metrics.py /path/to/harbor_job
    python evaluate_gaia_entropy_metrics.py --root /path/to/harbor_job
"""

import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================
# 配置
# =========================

DEFAULT_ROOT = Path("~/jobs/2026-05-16__19-11-56")

TIMEOUT_EXCEPTION = "AgentTimeoutError"


# =========================
# 工具函数
# =========================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if len(xs) == 0:
        return None
    return float(np.mean(xs))


def best_threshold_by_youden_j(y_true, scores):
    """
    基于 ROC Youden's J 选择最佳阈值
    """

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))

    return {
        "threshold": float(thresholds[best_idx]),
        "youden_j": float(j_scores[best_idx]),
    }


def extract_reward_mapping(result_data):
    """
    提取:
        task_id -> reward (0/1)
    """

    evals = result_data["stats"]["evals"]

    # 默认取第一个 evaluator
    eval_key = list(evals.keys())[0]

    reward_stats = evals[eval_key]["reward_stats"]["reward"]

    mapping = {}

    for reward_str, task_ids in reward_stats.items():
        reward = float(reward_str)

        for tid in task_ids:
            mapping[tid] = reward

    return mapping


def extract_timeout_set(result_data):
    """
    提取 timeout task set
    """

    evals = result_data["stats"]["evals"]
    eval_key = list(evals.keys())[0]

    exc_stats = evals[eval_key].get("exception_stats", {})

    timeout_tasks = set(exc_stats.get(TIMEOUT_EXCEPTION, []))

    return timeout_tasks


# =========================
# Entropy 计算
# =========================

def entropy_from_logprobs(logprobs: List[float]) -> Optional[float]:
    """
    基于 token logprob 计算 entropy proxy

    使用:
        entropy = -mean(logprob)

    logprob 越负，entropy 越高
    """

    if not logprobs:
        return None

    logprobs = np.array(logprobs, dtype=np.float32)

    return float(-np.mean(logprobs))


def collect_logprobs_from_step(step) -> List[float]:
    """
    从单个 step 提取 logprobs
    """

    metrics = step.get("metrics", {})
    lps = metrics.get("logprobs", [])

    if not isinstance(lps, list):
        return []

    cleaned = []

    for x in lps:
        try:
            x = float(x)

            if math.isfinite(x):
                cleaned.append(x)

        except Exception:
            pass

    return cleaned


def extract_entropy_from_trajectory(traj_path: Path):
    """
    从 trajectory.json 计算 entropy
    """

    try:
        data = load_json(traj_path)
    except Exception as e:
        print(f"[ERROR] load {traj_path}: {e}")
        return None

    steps = data.get("steps", [])

    all_logprobs = []

    assistant_steps = 0

    for step in steps:

        if step.get("source") != "agent":
            continue

        assistant_steps += 1

        lps = collect_logprobs_from_step(step)

        all_logprobs.extend(lps)

    if len(all_logprobs) == 0:
        return None

    entropy = entropy_from_logprobs(all_logprobs)

    return {
        "entropy": entropy,
        "n_logprobs": len(all_logprobs),
        "n_agent_steps": assistant_steps,
    }


# =========================
# 主逻辑
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate entropy-based uncertainty metrics on Harbor trajectories."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=None,
        help="Harbor job root containing result.json and trial directories.",
    )
    parser.add_argument(
        "--root",
        dest="root_flag",
        type=Path,
        default=None,
        help="Harbor job root containing result.json and trial directories.",
    )
    return parser.parse_args()


def resolve_root(args) -> Path:
    root = args.root_flag or args.root or DEFAULT_ROOT
    return root.expanduser().resolve()


def main():
    args = parse_args()
    root = resolve_root(args)
    result_json = root / "result.json"
    save_csv = root / "entropy_results.csv"
    save_plot = root / "entropy_roc.png"
    save_pr_plot = root / "entropy_pr.png"

    if not result_json.exists():
        raise FileNotFoundError(f"Missing result.json: {result_json}")

    print("=" * 80)
    print("Loading result.json")
    print("=" * 80)

    print(f"Root: {root}")

    result_data = load_json(result_json)

    reward_mapping = extract_reward_mapping(result_data)

    timeout_tasks = extract_timeout_set(result_data)

    print(f"Total tasks: {len(reward_mapping)}")
    print(f"Timeout tasks: {len(timeout_tasks)}")

    rows = []

    print("\n" + "=" * 80)
    print("Processing trajectories")
    print("=" * 80)

    for task_dir in tqdm(sorted(root.iterdir())):

        if not task_dir.is_dir():
            continue

        task_name = task_dir.name

        # reward_stats 中使用的 task_id
        if task_name not in reward_mapping:
            continue

        # skip timeout
        if task_name in timeout_tasks:
            continue

        traj_path = task_dir / "agent" / "trajectory.json"

        if not traj_path.exists():
            continue

        entropy_info = extract_entropy_from_trajectory(traj_path)

        if entropy_info is None:
            continue

        reward = reward_mapping[task_name]

        rows.append({
            "task": task_name,
            "reward": reward,
            "success": int(reward > 0.5),
            "entropy": entropy_info["entropy"],
            "n_logprobs": entropy_info["n_logprobs"],
            "n_agent_steps": entropy_info["n_agent_steps"],
        })

    df = pd.DataFrame(rows)

    print("\nCollected samples:", len(df))

    if len(df) == 0:
        print("No valid samples found.")
        return

    # =========================
    # 分类指标
    # =========================

    """
    entropy 越大 => 越不确定 => 越可能失败

    AUROC 通常：
        score 越大 => 越可能 positive

    因此这里：
        positive = failure

    即：
        y_true = 1 表示失败
    """

    y_true = 1 - df["success"].values
    scores = df["entropy"].values

    auroc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)

    threshold_info = best_threshold_by_youden_j(y_true, scores)
    threshold = threshold_info["threshold"]

    y_pred = (scores >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    print(f"AUROC (failure detection using entropy): {auroc:.6f}")
    print(f"AUPRC (failure detection using entropy): {auprc:.6f}")
    print(f"Best threshold (Youden's J): {threshold:.6f}")
    print(f"Youden's J: {threshold_info['youden_j']:.6f}")
    print(f"Accuracy @ threshold: {accuracy:.6f}")
    print(f"Precision @ threshold: {precision:.6f}")
    print(f"Recall @ threshold: {recall:.6f}")
    print(f"F1 @ threshold: {f1:.6f}")
    print(f"Confusion matrix [tn fp; fn tp]: [{tn} {fp}; {fn} {tp}]")

    print("\nEntropy statistics:")
    print(df.groupby("success")["entropy"].describe())

    # =========================
    # 保存 CSV
    # =========================

    df.to_csv(save_csv, index=False)

    print(f"\nSaved CSV:")
    print(save_csv)

    # =========================
    # 绘制 ROC
    # =========================

    fpr, tpr, _ = roc_curve(y_true, scores)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Entropy-based Failure Detection")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_plot, dpi=200)
    plt.close()

    print(f"\nSaved ROC plot:")
    print(save_plot)

    # =========================
    # 绘制 PR
    # =========================

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)

    plt.figure(figsize=(6, 6))
    plt.plot(recall_curve, precision_curve, label=f"AUPRC = {auprc:.4f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Entropy-based Failure Detection")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_pr_plot, dpi=200)
    plt.close()

    print(f"\nSaved PR plot:")
    print(save_pr_plot)

    # =========================
    # Top uncertain tasks
    # =========================

    print("\nTop-20 highest entropy tasks:\n")

    topk = df.sort_values("entropy", ascending=False).head(20)

    print(topk[
        ["task", "entropy", "success", "n_agent_steps"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
