"""
第二轮 DAgger 数据采集器。

它复用已经验证可运行的 collect_dagger_data.py 中的 Habitat 配置、
Oracle、观测预处理和轨迹保存逻辑，只替换以下部分：

1. 使用第一轮 DAgger 训练得到的 best_accuracy 模型。
2. Oracle 执行概率 BETA 从 0.5 降为 0.2。
3. 数据保存到 vln_dagger_data_v2/train，不覆盖第一轮数据。
4. 元数据标记为 dagger_iteration_2。
5. R2R Success 距离阈值设为标准的 3.0 米。

模型大约执行 80% 的动作；Oracle 始终为每个访问状态生成监督标签。
如果模型过早预测 STOP，Oracle 会接管，避免轨迹立刻结束。
"""

import json
import os
import sys
import traceback


# 必须在导入 Habitat-Sim 之前设置。
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")


import torch

import habitat
from habitat.config import read_write


WORK_ROOT = os.path.expanduser("~/habitat-work")
sys.path.insert(0, WORK_ROOT)


try:
    import collect_dagger_data as base
except ImportError as error:
    raise ImportError(
        "没有找到 ~/habitat-work/collect_dagger_data.py。"
        "请把本文件放在 ~/habitat-work/ 后再运行。"
    ) from error


from train_vln_dagger import ImprovedVLNPolicy


OUTPUT_ROOT = os.path.join(
    WORK_ROOT,
    "vln_dagger_data_v2",
    "train",
)

CHECKPOINT_PATH = os.path.join(
    WORK_ROOT,
    "checkpoints_dagger",
    "vln_dagger_best_accuracy.pth",
)

# 第二轮让当前策略更多地访问自己的状态分布。
BETA = 0.2

MAX_STEPS = 250
SUCCESS_DISTANCE = 3.0
RANDOM_SEED = 84
DEVICE = torch.device("cpu")


# base 中的函数运行时会读取这些模块全局变量，因此在采集前统一替换。
base.OUTPUT_ROOT = OUTPUT_ROOT
base.CHECKPOINT_PATH = CHECKPOINT_PATH
base.BETA = BETA
base.MAX_STEPS = MAX_STEPS
base.DEVICE = DEVICE


def load_model_v2():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"没有找到第一轮 DAgger 模型：{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model = ImprovedVLNPolicy(
        **checkpoint["model_config"]
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("已加载第二轮采集策略：", CHECKPOINT_PATH)
    print("Checkpoint Epoch：", checkpoint["epoch"])
    print(
        "Checkpoint Val accuracy：",
        checkpoint.get("validation_accuracy", "unknown"),
    )
    print("第二轮 DAgger BETA：", BETA)
    print("预计模型执行比例：", 1.0 - BETA)

    return model, checkpoint["vocab"]


def make_config_v2():
    config = base.make_config()

    with read_write(config):
        config.habitat.environment.max_episode_steps = MAX_STEPS

        config.habitat.task.measurements.success.success_distance = (
            SUCCESS_DISTANCE
        )

        if "oracle_success" in config.habitat.task.measurements:
            config.habitat.task.measurements[
                "oracle_success"
            ].success_distance = SUCCESS_DISTANCE

    return config


def save_episode_v2(episode, collected):
    # 复用第一轮已经验证过的 NPZ 字段与 JSON 写入逻辑。
    metadata = base.save_episode(
        episode,
        collected,
    )

    metadata["source"] = "dagger_iteration_2"
    metadata["dagger_beta"] = BETA
    metadata["collector_checkpoint"] = CHECKPOINT_PATH
    metadata["success_distance"] = SUCCESS_DISTANCE
    metadata["random_seed"] = RANDOM_SEED

    _, json_path = base.episode_paths(episode)

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return metadata


def write_progress_files(manifest, failures):
    """每完成一条轨迹就写入进度，意外中断时也能保留 manifest。"""
    manifest_path = os.path.join(
        OUTPUT_ROOT,
        "manifest.json",
    )

    failures_path = os.path.join(
        OUTPUT_ROOT,
        "failures.json",
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        failures_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            failures,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main():
    base.set_seed(RANDOM_SEED)
    torch.set_num_threads(4)

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True,
    )

    print("=" * 70)
    print("开始第二轮 DAgger 数据采集")
    print("训练 split：train")
    print("输出目录：", OUTPUT_ROOT)
    print("Oracle 执行概率：", BETA)
    print("模型预计执行概率：", 1.0 - BETA)
    print("Success 距离阈值：", SUCCESS_DISTANCE, "m")
    print("验证集 val_seen 不会参与采集")

    model, vocab = load_model_v2()
    config = make_config_v2()

    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type,
        config=config.habitat.dataset,
    )

    selected_episodes = [
        episode
        for episode in dataset.episodes
        if base.SCENE_NAME in episode.scene_id
    ]

    if len(selected_episodes) == 0:
        raise RuntimeError(
            "训练集中没有找到场景 "
            f"{base.SCENE_NAME}"
        )

    dataset.episodes = selected_episodes

    print(
        "第二轮 DAgger Episode 数量：",
        len(selected_episodes),
    )

    manifest = []
    failures = []

    total_expert_executions = 0
    total_model_executions = 0
    total_steps = 0

    with habitat.Env(
        config=config,
        dataset=dataset,
    ) as env:
        for index, episode in enumerate(
            selected_episodes,
            start=1,
        ):
            npz_path, json_path = base.episode_paths(
                episode
            )

            if (
                os.path.exists(npz_path)
                and os.path.exists(json_path)
            ):
                with open(
                    json_path,
                    "r",
                    encoding="utf-8",
                ) as file:
                    metadata = json.load(file)

                manifest.append(metadata)

                total_expert_executions += int(
                    metadata.get("expert_execution_count", 0)
                )
                total_model_executions += int(
                    metadata.get("model_execution_count", 0)
                )
                total_steps += int(
                    metadata.get("num_steps", 0)
                )

                print(
                    f"[{index:03d}/"
                    f"{len(selected_episodes):03d}] "
                    f"Episode {episode.episode_id} "
                    "已经存在，跳过"
                )

                continue

            print("-" * 70)
            print(
                f"[{index:03d}/"
                f"{len(selected_episodes):03d}] "
                f"采集 Episode {episode.episode_id}"
            )
            print(
                "Prompt：",
                base.get_prompt(episode),
            )

            try:
                collected = base.collect_episode(
                    env=env,
                    episode=episode,
                    model=model,
                    vocab=vocab,
                )

                metadata = save_episode_v2(
                    episode,
                    collected,
                )

                manifest.append(metadata)

                total_expert_executions += int(
                    metadata["expert_execution_count"]
                )
                total_model_executions += int(
                    metadata["model_execution_count"]
                )
                total_steps += int(
                    metadata["num_steps"]
                )

                write_progress_files(
                    manifest,
                    failures,
                )

                print(
                    f"完成："
                    f"{metadata['num_steps']}步，"
                    f"Success="
                    f"{metadata['success']:.0f}，"
                    f"Final distance="
                    f"{metadata['final_distance']:.3f}m，"
                    f"Oracle执行="
                    f"{metadata['expert_execution_count']}，"
                    f"模型执行="
                    f"{metadata['model_execution_count']}"
                )

            except KeyboardInterrupt:
                write_progress_files(
                    manifest,
                    failures,
                )

                print(
                    "\n用户中断；已保存当前进度，"
                    "重新运行会从未完成的 Episode 继续。"
                )
                raise

            except Exception as error:
                print(
                    f"Episode {episode.episode_id} "
                    f"失败：{error}"
                )

                failures.append(
                    {
                        "episode_id": str(
                            episode.episode_id
                        ),
                        "error": str(error),
                    }
                )

                write_progress_files(
                    manifest,
                    failures,
                )

                traceback.print_exc()

    write_progress_files(
        manifest,
        failures,
    )

    execution_total = (
        total_expert_executions
        + total_model_executions
    )

    if execution_total > 0:
        actual_expert_ratio = (
            total_expert_executions
            / execution_total
        )
        actual_model_ratio = (
            total_model_executions
            / execution_total
        )
    else:
        actual_expert_ratio = 0.0
        actual_model_ratio = 0.0

    print("=" * 70)
    print("第二轮 DAgger 采集完成")
    print("成功保存数量：", len(manifest))
    print("失败数量：", len(failures))
    print("总监督状态数：", total_steps)
    print("Oracle 实际执行比例：", f"{actual_expert_ratio:.3f}")
    print("模型实际执行比例：", f"{actual_model_ratio:.3f}")
    print("输出目录：", OUTPUT_ROOT)
    print(
        "Manifest：",
        os.path.join(OUTPUT_ROOT, "manifest.json"),
    )


if __name__ == "__main__":
    main()