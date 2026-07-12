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
from habitat.config import read_write
from habitat.config.default import get_agent_config


WORK_ROOT = os.path.expanduser("~/habitat-work")

sys.path.insert(0, WORK_ROOT)

from train_vln import (
    ACTION_NAMES,
    START_ACTION,
    encode_instruction,
)

from train_vln_dagger import (
    ImprovedVLNPolicy,
)


SCENE_NAME = "17DRP5sb8fy"

DATA_ROOT = os.path.join(
    WORK_ROOT,
    "data",
)

R2R_DATA_PATH = os.path.join(
    DATA_ROOT,
    "datasets/vln/mp3d/r2r/v1/{split}/{split}.json.gz",
)

SCENES_DIR = os.path.join(
    DATA_ROOT,
    "scene_datasets",
)

CHECKPOINTS = {
    "best_loss": os.path.join(
        WORK_ROOT,
        "checkpoints_dagger",
        "vln_dagger_best_loss.pth",
    ),
    "best_accuracy": os.path.join(
        WORK_ROOT,
        "checkpoints_dagger",
        "vln_dagger_best_accuracy.pth",
    ),
}

RESULT_PATH = os.path.join(
    WORK_ROOT,
    "checkpoints_dagger",
    "closed_loop_comparison.json",
)

IMAGE_SIZE = 96
MAX_STEPS = 200
SUCCESS_DISTANCE = 3.0

DEVICE = torch.device("cpu")


def get_prompt(episode):
    instruction = episode.instruction

    if hasattr(instruction, "instruction_text"):
        return instruction.instruction_text.strip()

    if hasattr(instruction, "text"):
        return instruction.text.strip()

    return str(instruction).strip()


def make_config():
    config = habitat.get_config(
        "benchmark/nav/vln_r2r.yaml"
    )

    with read_write(config):
        config.habitat.dataset.split = "val_seen"
        config.habitat.dataset.data_path = R2R_DATA_PATH
        config.habitat.dataset.scenes_dir = SCENES_DIR

        config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
        config.habitat.environment.max_episode_steps = MAX_STEPS

        config.habitat.task.measurements.success.success_distance = (
            SUCCESS_DISTANCE
        )

        if (
            "oracle_success"
            in config.habitat.task.measurements
        ):
            config.habitat.task.measurements[
                "oracle_success"
            ].success_distance = SUCCESS_DISTANCE

        agent_config = get_agent_config(
            config.habitat.simulator
        )

        agent_config.sim_sensors["rgb_sensor"].height = IMAGE_SIZE
        agent_config.sim_sensors["rgb_sensor"].width = IMAGE_SIZE

        agent_config.sim_sensors["depth_sensor"].height = IMAGE_SIZE
        agent_config.sim_sensors["depth_sensor"].width = IMAGE_SIZE

    return config


def observation_to_tensor(observations):
    rgb = observations["rgb"]

    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    rgb_tensor = torch.from_numpy(
        rgb.copy()
    ).permute(
        2,
        0,
        1,
    ).float()

    rgb_tensor = rgb_tensor / 255.0

    depth = observations["depth"]

    if depth.ndim == 2:
        depth = depth[:, :, None]

    depth_tensor = torch.from_numpy(
        depth.copy()
    ).permute(
        2,
        0,
        1,
    ).float()

    observation = torch.cat(
        [rgb_tensor, depth_tensor],
        dim=0,
    ).unsqueeze(0)

    return observation.to(DEVICE)


def metric_to_float(
    metrics,
    name,
    default=0.0,
):
    value = metrics.get(
        name,
        default,
    )

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model = ImprovedVLNPolicy(
        **checkpoint["model_config"]
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Checkpoint：",
        checkpoint_path,
    )
    print(
        "Epoch：",
        checkpoint["epoch"],
    )
    print(
        "Val loss：",
        checkpoint["validation_loss"],
    )
    print(
        "Val accuracy：",
        checkpoint["validation_accuracy"],
    )

    return (
        model,
        checkpoint["vocab"],
        checkpoint,
    )


def evaluate_episode(
    env,
    episode,
    model,
    vocab,
):
    env.episode_iterator = iter(
        [episode]
    )

    observations = env.reset()
    prompt = get_prompt(episode)

    instruction_tokens = encode_instruction(
        prompt,
        vocab,
    )

    with torch.no_grad():
        language_context = model.encode_language(
            instruction_tokens
        )

    hidden = model.initial_hidden(
        DEVICE
    )

    previous_action = START_ACTION
    stopped = False

    action_histogram = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }

    step_count = 0

    while (
        not env.episode_over
        and step_count < MAX_STEPS
    ):
        observation_tensor = observation_to_tensor(
            observations
        )

        with torch.no_grad():
            visual_feature = model.visual_encoder(
                observation_tensor
            )

            logits, hidden = model.navigation_step(
                visual_feature=visual_feature,
                language_context=language_context,
                previous_action=previous_action,
                hidden=hidden,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            action = int(
                torch.argmax(
                    probabilities,
                    dim=1,
                ).item()
            )

        action_histogram[action] += 1
        step_count += 1

        observations = env.step(
            action
        )

        previous_action = action

        if action == 0:
            stopped = True
            break

    metrics = env.get_metrics()

    result = {
        "episode_id": str(
            episode.episode_id
        ),
        "trajectory_id": str(
            getattr(
                episode,
                "trajectory_id",
                "unknown",
            )
        ),
        "prompt": prompt,
        "steps": step_count,
        "stopped": stopped,
        "success": metric_to_float(
            metrics,
            "success",
        ),
        "spl": metric_to_float(
            metrics,
            "spl",
        ),
        "oracle_success": metric_to_float(
            metrics,
            "oracle_success",
        ),
        "distance_to_goal": metric_to_float(
            metrics,
            "distance_to_goal",
            default=float("nan"),
        ),
        "ndtw": metric_to_float(
            metrics,
            "ndtw",
        ),
        "sdtw": metric_to_float(
            metrics,
            "sdtw",
        ),
        "action_histogram": {
            ACTION_NAMES[action]: count
            for action, count
            in action_histogram.items()
        },
    }

    return result


def evaluate_checkpoint(
    checkpoint_name,
    checkpoint_path,
    env,
    episodes,
):
    print("=" * 70)
    print(
        "开始评估模型：",
        checkpoint_name,
    )

    model, vocab, checkpoint = load_model(
        checkpoint_path
    )

    episode_results = []

    for index, episode in enumerate(
        episodes,
        start=1,
    ):
        result = evaluate_episode(
            env=env,
            episode=episode,
            model=model,
            vocab=vocab,
        )

        episode_results.append(
            result
        )

        print("-" * 70)
        print(
            f"[{index}/"
            f"{len(episodes)}] "
            f"Episode "
            f"{result['episode_id']}"
        )
        print(
            "Prompt：",
            result["prompt"],
        )
        print(
            "动作数：",
            result["steps"],
        )
        print(
            "主动STOP：",
            result["stopped"],
        )
        print(
            "Success：",
            result["success"],
        )
        print(
            "SPL：",
            result["spl"],
        )
        print(
            "Distance：",
            result["distance_to_goal"],
        )
        print(
            "动作分布：",
            result["action_histogram"],
        )

    average_success = float(
        np.mean(
            [
                result["success"]
                for result
                in episode_results
            ]
        )
    )

    average_spl = float(
        np.mean(
            [
                result["spl"]
                for result
                in episode_results
            ]
        )
    )

    finite_distances = [
        result["distance_to_goal"]
        for result in episode_results
        if np.isfinite(
            result["distance_to_goal"]
        )
    ]

    if finite_distances:
        average_distance = float(
            np.mean(finite_distances)
        )
    else:
        average_distance = float(
            "nan"
        )

    summary = {
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "validation_loss": float(
            checkpoint["validation_loss"]
        ),
        "validation_accuracy": float(
            checkpoint[
                "validation_accuracy"
            ]
        ),
        "num_episodes": len(
            episode_results
        ),
        "average_success": (
            average_success
        ),
        "average_spl": average_spl,
        "average_distance_to_goal": (
            average_distance
        ),
        "episodes": episode_results,
    }

    print("=" * 70)
    print(
        f"{checkpoint_name}评估完成"
    )
    print(
        f"平均Success："
        f"{average_success:.4f}"
    )
    print(
        f"平均SPL："
        f"{average_spl:.4f}"
    )
    print(
        f"平均目标距离："
        f"{average_distance:.4f} m"
    )

    return summary


def main():
    torch.set_num_threads(4)

    config = make_config()

    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type,
        config=config.habitat.dataset,
    )

    selected_episodes = [
        episode
        for episode in dataset.episodes
        if SCENE_NAME in episode.scene_id
    ]

    selected_episodes = sorted(
        selected_episodes,
        key=lambda episode: int(
            episode.episode_id
        ),
    )

    if len(selected_episodes) == 0:
        raise RuntimeError(
            "没有找到val_seen episode"
        )

    dataset.episodes = selected_episodes

    print(
        "评估Episode数量：",
        len(selected_episodes),
    )
    print(
        "成功距离阈值：",
        SUCCESS_DISTANCE,
        "m",
    )

    all_results = {}

    with habitat.Env(
        config=config,
        dataset=dataset,
    ) as env:
        for name, path in CHECKPOINTS.items():
            all_results[name] = (
                evaluate_checkpoint(
                    checkpoint_name=name,
                    checkpoint_path=path,
                    env=env,
                    episodes=selected_episodes,
                )
            )

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 70)
    print("两个模型对比完成")

    for name, result in all_results.items():
        print(
            f"{name}: "
            f"Epoch="
            f"{result['checkpoint_epoch']}, "
            f"Success="
            f"{result['average_success']:.4f}, "
            f"SPL="
            f"{result['average_spl']:.4f}, "
            f"Distance="
            f"{result['average_distance_to_goal']:.4f}m"
        )

    print(
        "结果文件：",
        RESULT_PATH,
    )


if __name__ == "__main__":
    main()
