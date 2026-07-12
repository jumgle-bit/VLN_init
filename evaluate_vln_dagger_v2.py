"""
闭环比较第一轮 DAgger 与第二轮 DAgger 模型。

模型输入只包括语言指令、RGB、深度、上一动作和循环隐藏状态；
目标坐标与 reference_path 不会传给策略。Success 距离阈值为 3 米。
"""

import json
import os
import sys


os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")


import numpy as np
import torch

import habitat


WORK_ROOT = os.path.expanduser("~/habitat-work")
sys.path.insert(0, WORK_ROOT)


try:
    import evaluate_vln_dagger as base
except ImportError as error:
    raise ImportError(
        "没有找到 ~/habitat-work/evaluate_vln_dagger.py。"
        "请把本文件放在 ~/habitat-work/ 后运行。"
    ) from error


CHECKPOINTS = {
    "v1_best_accuracy": os.path.join(
        WORK_ROOT,
        "checkpoints_dagger",
        "vln_dagger_best_accuracy.pth",
    ),
    "v2_best_loss": os.path.join(
        WORK_ROOT,
        "checkpoints_dagger_v2",
        "vln_dagger_v2_best_loss.pth",
    ),
    "v2_best_accuracy": os.path.join(
        WORK_ROOT,
        "checkpoints_dagger_v2",
        "vln_dagger_v2_best_accuracy.pth",
    ),
    "v2_last": os.path.join(
        WORK_ROOT,
        "checkpoints_dagger_v2",
        "vln_dagger_v2_last.pth",
    ),
}

RESULT_PATH = os.path.join(
    WORK_ROOT,
    "checkpoints_dagger_v2",
    "closed_loop_comparison_v2.json",
)


def check_files():
    missing = [
        path
        for path in CHECKPOINTS.values()
        if not os.path.exists(path)
    ]

    if missing:
        raise FileNotFoundError(
            "缺少以下 checkpoint：\n"
            + "\n".join(missing)
        )


def rank_key(item):
    result = item[1]

    distance = result["average_distance_to_goal"]

    if not np.isfinite(distance):
        distance = float("inf")

    return (
        result["average_success"],
        result["average_spl"],
        -distance,
    )


def main():
    torch.set_num_threads(4)
    check_files()

    config = base.make_config()

    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type,
        config=config.habitat.dataset,
    )

    selected_episodes = [
        episode
        for episode in dataset.episodes
        if base.SCENE_NAME in episode.scene_id
    ]

    selected_episodes = sorted(
        selected_episodes,
        key=lambda episode: int(episode.episode_id),
    )

    if len(selected_episodes) == 0:
        raise RuntimeError(
            "没有找到 val_seen Episode"
        )

    dataset.episodes = selected_episodes

    print("=" * 70)
    print("开始第一轮与第二轮 DAgger 闭环对比")
    print("评估 Episode 数量：", len(selected_episodes))
    print("成功距离阈值：", base.SUCCESS_DISTANCE, "m")
    print("最大动作数：", base.MAX_STEPS)
    print("策略不读取目标坐标或参考路径")

    all_results = {}

    with habitat.Env(
        config=config,
        dataset=dataset,
    ) as env:
        for name, path in CHECKPOINTS.items():
            all_results[name] = base.evaluate_checkpoint(
                checkpoint_name=name,
                checkpoint_path=path,
                env=env,
                episodes=selected_episodes,
            )

    baseline = all_results["v1_best_accuracy"]

    for name, result in all_results.items():
        result["success_change_vs_v1"] = float(
            result["average_success"]
            - baseline["average_success"]
        )

        result["spl_change_vs_v1"] = float(
            result["average_spl"]
            - baseline["average_spl"]
        )

        result["distance_change_vs_v1"] = float(
            result["average_distance_to_goal"]
            - baseline["average_distance_to_goal"]
        )

    winner_name, winner_result = max(
        all_results.items(),
        key=rank_key,
    )

    output = {
        "evaluation_protocol": {
            "split": "val_seen",
            "scene": base.SCENE_NAME,
            "num_episodes": len(selected_episodes),
            "success_distance": base.SUCCESS_DISTANCE,
            "max_steps": base.MAX_STEPS,
            "policy_has_goal_position": False,
            "policy_has_reference_path": False,
        },
        "winner": winner_name,
        "results": all_results,
    }

    os.makedirs(
        os.path.dirname(RESULT_PATH),
        exist_ok=True,
    )

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 70)
    print("四个模型闭环对比完成")

    for name, result in all_results.items():
        print(
            f"{name}: "
            f"Epoch={result['checkpoint_epoch']}, "
            f"Success={result['average_success']:.4f}, "
            f"SPL={result['average_spl']:.4f}, "
            f"Distance="
            f"{result['average_distance_to_goal']:.4f}m"
        )

    print("-" * 70)
    print("当前闭环最佳模型：", winner_name)
    print(
        "最佳 Success：",
        f"{winner_result['average_success']:.4f}",
    )
    print(
        "最佳 SPL：",
        f"{winner_result['average_spl']:.4f}",
    )
    print(
        "相对 V1 Success 变化：",
        f"{winner_result['success_change_vs_v1']:+.4f}",
    )
    print(
        "相对 V1 SPL 变化：",
        f"{winner_result['spl_change_vs_v1']:+.4f}",
    )
    print("结果文件：", RESULT_PATH)


if __name__ == "__main__":
    main()