"""
对 STOP-aware 的全部 Epoch 做闭环 VLN 评估。

比较对象：
    1. 第一轮 DAgger best_accuracy
    2. 第二轮 DAgger last
    3. STOP-aware Epoch 1 到 Epoch 5

排序规则：Success 优先，其次 SPL，最后平均目标距离。
策略不读取目标坐标或 reference_path。
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
        "请把本文件放到 ~/habitat-work/ 后运行。"
    ) from error


CHECKPOINT_DIR = os.path.join(
    WORK_ROOT,
    "checkpoints_stop_aware",
)

CHECKPOINTS = {
    "v1_best_accuracy": os.path.join(
        WORK_ROOT,
        "checkpoints_dagger",
        "vln_dagger_best_accuracy.pth",
    ),
    "v2_last": os.path.join(
        WORK_ROOT,
        "checkpoints_dagger_v2",
        "vln_dagger_v2_last.pth",
    ),
}

for epoch in range(1, 6):
    CHECKPOINTS[f"stop_aware_epoch_{epoch:02d}"] = os.path.join(
        CHECKPOINT_DIR,
        f"vln_stop_aware_epoch_{epoch:02d}.pth",
    )


RESULT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "closed_loop_all_epochs.json",
)


def check_checkpoint_files():
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


def add_closed_loop_statistics(result):
    episodes = result["episodes"]

    result["active_stop_rate"] = float(
        np.mean(
            [
                float(episode["stopped"])
                for episode in episodes
            ]
        )
    )

    result["successful_episode_ids"] = [
        episode["episode_id"]
        for episode in episodes
        if episode["success"] >= 0.5
    ]

    result["mean_num_steps"] = float(
        np.mean(
            [
                episode["steps"]
                for episode in episodes
            ]
        )
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
    check_checkpoint_files()

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
    print("开始 STOP-aware 全 Epoch 闭环评估")
    print("模型数量：", len(CHECKPOINTS))
    print("每个模型 Episode 数量：", len(selected_episodes))
    print(
        "总闭环回合数：",
        len(CHECKPOINTS) * len(selected_episodes),
    )
    print("Success 距离阈值：", base.SUCCESS_DISTANCE, "m")
    print("最大动作数：", base.MAX_STEPS)
    print("策略不读取目标坐标或参考路径")

    all_results = {}

    with habitat.Env(
        config=config,
        dataset=dataset,
    ) as env:
        for name, path in CHECKPOINTS.items():
            result = base.evaluate_checkpoint(
                checkpoint_name=name,
                checkpoint_path=path,
                env=env,
                episodes=selected_episodes,
            )

            add_closed_loop_statistics(result)
            all_results[name] = result

    baseline = all_results["v1_best_accuracy"]

    for result in all_results.values():
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

    ranked_results = sorted(
        all_results.items(),
        key=rank_key,
        reverse=True,
    )

    winner_name, winner_result = ranked_results[0]

    output = {
        "evaluation_protocol": {
            "split": "val_seen",
            "scene": base.SCENE_NAME,
            "num_episodes": len(selected_episodes),
            "num_models": len(CHECKPOINTS),
            "success_distance": base.SUCCESS_DISTANCE,
            "max_steps": base.MAX_STEPS,
            "policy_has_goal_position": False,
            "policy_has_reference_path": False,
        },
        "winner": winner_name,
        "ranking": [
            name
            for name, _ in ranked_results
        ],
        "results": all_results,
    }

    os.makedirs(
        CHECKPOINT_DIR,
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
    print("STOP-aware 全 Epoch 闭环评估完成")

    for rank, (name, result) in enumerate(
        ranked_results,
        start=1,
    ):
        successful_ids = ",".join(
            result["successful_episode_ids"]
        )

        if not successful_ids:
            successful_ids = "none"

        print(
            f"#{rank} {name}: "
            f"Epoch={result['checkpoint_epoch']}, "
            f"Success={result['average_success']:.4f}, "
            f"SPL={result['average_spl']:.4f}, "
            f"Distance="
            f"{result['average_distance_to_goal']:.4f}m, "
            f"StopRate={result['active_stop_rate']:.4f}, "
            f"SuccessIDs={successful_ids}"
        )

    print("-" * 70)
    print("最终闭环最佳模型：", winner_name)
    print(
        "Checkpoint：",
        CHECKPOINTS[winner_name],
    )
    print(
        "Success：",
        f"{winner_result['average_success']:.4f}",
    )
    print(
        "SPL：",
        f"{winner_result['average_spl']:.4f}",
    )
    print(
        "相对 V1 Success：",
        f"{winner_result['success_change_vs_v1']:+.4f}",
    )
    print(
        "相对 V1 SPL：",
        f"{winner_result['spl_change_vs_v1']:+.4f}",
    )
    print("结果文件：", RESULT_PATH)


if __name__ == "__main__":
    main()