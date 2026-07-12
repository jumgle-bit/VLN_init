import json
import os
import random
import sys
import traceback

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import numpy as np
import torch

import habitat
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


WORK_ROOT = os.path.expanduser("~/habitat-work")

sys.path.insert(0, WORK_ROOT)

from train_vln import (
    START_ACTION,
    VLNPolicy,
    encode_instruction,
)


SCENE_NAME = "17DRP5sb8fy"

DATA_ROOT = os.path.join(
    WORK_ROOT,
    "data",
)

OUTPUT_ROOT = os.path.join(
    WORK_ROOT,
    "vln_dagger_data",
    "train",
)

CHECKPOINT_PATH = os.path.join(
    WORK_ROOT,
    "checkpoints",
    "vln_seq2seq_last.pth",
)

R2R_DATA_PATH = os.path.join(
    DATA_ROOT,
    "datasets/vln/mp3d/r2r/v1/{split}/{split}.json.gz",
)

SCENES_DIR = os.path.join(
    DATA_ROOT,
    "scene_datasets",
)

IMAGE_SIZE = 96
MAX_STEPS = 250

# 执行Oracle动作的概率
BETA = 0.5

DEVICE = torch.device("cpu")

ACTION_NAMES = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_prompt(episode):
    instruction = episode.instruction

    if hasattr(instruction, "instruction_text"):
        return instruction.instruction_text.strip()

    if hasattr(instruction, "text"):
        return instruction.text.strip()

    return str(instruction).strip()


def get_instruction_tokens(episode):
    instruction = episode.instruction

    if hasattr(instruction, "instruction_tokens"):
        tokens = instruction.instruction_tokens

        if tokens is not None:
            return np.asarray(
                tokens,
                dtype=np.int64,
            )

    return np.asarray(
        [],
        dtype=np.int64,
    )


def make_config():
    config = habitat.get_config(
        "benchmark/nav/vln_r2r.yaml"
    )

    with read_write(config):
        config.habitat.dataset.split = "train"
        config.habitat.dataset.data_path = R2R_DATA_PATH
        config.habitat.dataset.scenes_dir = SCENES_DIR

        config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
        config.habitat.environment.max_episode_steps = MAX_STEPS

        agent_config = get_agent_config(
            config.habitat.simulator
        )

        agent_config.sim_sensors["rgb_sensor"].height = IMAGE_SIZE
        agent_config.sim_sensors["rgb_sensor"].width = IMAGE_SIZE

        agent_config.sim_sensors["depth_sensor"].height = IMAGE_SIZE
        agent_config.sim_sensors["depth_sensor"].width = IMAGE_SIZE

    return config


def load_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model = VLNPolicy(
        **checkpoint["model_config"]
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("已加载模型：", CHECKPOINT_PATH)
    print("Checkpoint Epoch：", checkpoint["epoch"])

    return model, checkpoint["vocab"]


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

    return torch.cat(
        [rgb_tensor, depth_tensor],
        dim=0,
    ).unsqueeze(0).to(DEVICE)


def prepare_rgb(observations):
    rgb = observations["rgb"]

    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    return rgb.astype(np.uint8)


def prepare_depth(observations):
    depth = observations["depth"]

    if depth.ndim == 2:
        depth = depth[:, :, None]

    return depth.astype(np.float16)


def get_expert_action(
    follower,
    waypoints,
    waypoint_index,
):
    while waypoint_index < len(waypoints):
        action = follower.get_next_action(
            waypoints[waypoint_index]
        )

        if action is None:
            raise RuntimeError(
                "Oracle无法生成纠正动作"
            )

        action = int(action)

        if action == int(HabitatSimActions.stop):
            waypoint_index += 1
            continue

        return action, waypoint_index

    return int(HabitatSimActions.stop), waypoint_index


def episode_paths(episode):
    episode_id = str(episode.episode_id)

    trajectory_id = str(
        getattr(
            episode,
            "trajectory_id",
            "unknown",
        )
    )

    stem = (
        f"episode_{episode_id}_"
        f"trajectory_{trajectory_id}"
    )

    npz_path = os.path.join(
        OUTPUT_ROOT,
        stem + ".npz",
    )

    json_path = os.path.join(
        OUTPUT_ROOT,
        stem + ".json",
    )

    return npz_path, json_path


def collect_episode(
    env,
    episode,
    model,
    vocab,
):
    env.episode_iterator = iter([episode])
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

    hidden = model.initial_hidden(DEVICE)
    previous_action = START_ACTION

    goal = np.asarray(
        episode.goals[0].position,
        dtype=np.float32,
    )

    waypoints = [
        np.asarray(point, dtype=np.float32)
        for point in episode.reference_path
    ]

    if len(waypoints) == 0:
        waypoints = [goal]

    if not np.allclose(waypoints[-1], goal):
        waypoints.append(goal)

    follower = ShortestPathFollower(
        sim=env.sim,
        goal_radius=0.4,
        return_one_hot=False,
    )

    waypoint_index = 0

    rgb_frames = []
    depth_frames = []
    expert_labels = []
    executed_actions = []
    model_actions = []

    expert_execution_count = 0
    model_execution_count = 0

    for step in range(MAX_STEPS):
        expert_action, waypoint_index = get_expert_action(
            follower,
            waypoints,
            waypoint_index,
        )

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

            model_action = int(
                torch.argmax(
                    logits,
                    dim=1,
                ).item()
            )

        rgb_frames.append(
            prepare_rgb(observations)
        )

        depth_frames.append(
            prepare_depth(observations)
        )

        expert_labels.append(
            expert_action
        )

        model_actions.append(
            model_action
        )

        if expert_action == int(
            HabitatSimActions.stop
        ):
            executed_action = expert_action
            expert_execution_count += 1

        elif model_action == int(
            HabitatSimActions.stop
        ):
            # 模型过早停止时由Oracle接管
            executed_action = expert_action
            expert_execution_count += 1

        elif random.random() < BETA:
            executed_action = expert_action
            expert_execution_count += 1

        else:
            executed_action = model_action
            model_execution_count += 1

        executed_actions.append(
            executed_action
        )

        observations = env.step(
            executed_action
        )

        previous_action = executed_action

        if executed_action == int(
            HabitatSimActions.stop
        ):
            break

        if env.episode_over:
            break

    final_position = np.asarray(
        env.sim.get_agent_state().position,
        dtype=np.float32,
    )

    final_distance = float(
        env.sim.geodesic_distance(
            final_position,
            goal,
        )
    )

    metrics = env.get_metrics()

    return {
        "rgb": np.stack(
            rgb_frames,
            axis=0,
        ),
        "depth": np.stack(
            depth_frames,
            axis=0,
        ),
        "expert_actions": np.asarray(
            expert_labels,
            dtype=np.int64,
        ),
        "executed_actions": np.asarray(
            executed_actions,
            dtype=np.int64,
        ),
        "model_actions": np.asarray(
            model_actions,
            dtype=np.int64,
        ),
        "final_distance": final_distance,
        "success": float(
            metrics.get(
                "success",
                final_distance <= 3.0,
            )
        ),
        "spl": float(
            metrics.get("spl", 0.0)
        ),
        "expert_execution_count": (
            expert_execution_count
        ),
        "model_execution_count": (
            model_execution_count
        ),
    }


def save_episode(
    episode,
    collected,
):
    npz_path, json_path = episode_paths(
        episode
    )

    np.savez_compressed(
        npz_path,
        rgb=collected["rgb"],
        depth=collected["depth"],
        actions=collected["expert_actions"],
        executed_actions=collected[
            "executed_actions"
        ],
        model_actions=collected[
            "model_actions"
        ],
        instruction_tokens=get_instruction_tokens(
            episode
        ),
    )

    metadata = {
        "source": "dagger_iteration_1",
        "split": "train",
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
        "scene_id": str(
            episode.scene_id
        ),
        "prompt": get_prompt(episode),
        "num_steps": int(
            len(collected["expert_actions"])
        ),
        "final_distance": float(
            collected["final_distance"]
        ),
        "success": float(
            collected["success"]
        ),
        "spl": float(
            collected["spl"]
        ),
        "expert_execution_count": int(
            collected[
                "expert_execution_count"
            ]
        ),
        "model_execution_count": int(
            collected[
                "model_execution_count"
            ]
        ),
        "npz_path": npz_path,
    }

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


def main():
    set_seed(42)
    torch.set_num_threads(4)

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True,
    )

    model, vocab = load_model()
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

    if len(selected_episodes) == 0:
        raise RuntimeError(
            "训练集中没有找到目标场景"
        )

    dataset.episodes = selected_episodes

    print(
        "DAgger采集Episode数量：",
        len(selected_episodes),
    )

    manifest = []
    failures = []

    with habitat.Env(
        config=config,
        dataset=dataset,
    ) as env:
        for index, episode in enumerate(
            selected_episodes,
            start=1,
        ):
            npz_path, json_path = episode_paths(
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
                    manifest.append(
                        json.load(file)
                    )

                print(
                    f"[{index:03d}/"
                    f"{len(selected_episodes):03d}] "
                    f"Episode {episode.episode_id}"
                    f"已经存在，跳过"
                )

                continue

            print("-" * 70)
            print(
                f"[{index:03d}/"
                f"{len(selected_episodes):03d}] "
                f"采集Episode "
                f"{episode.episode_id}"
            )
            print(
                "Prompt：",
                get_prompt(episode),
            )

            try:
                collected = collect_episode(
                    env=env,
                    episode=episode,
                    model=model,
                    vocab=vocab,
                )

                metadata = save_episode(
                    episode,
                    collected,
                )

                manifest.append(metadata)

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
                print(
                    "\n用户中断，重新运行可继续"
                )
                raise

            except Exception as error:
                print(
                    f"Episode "
                    f"{episode.episode_id}"
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

                traceback.print_exc()

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

    print("=" * 70)
    print("第一轮DAgger采集完成")
    print("成功数量：", len(manifest))
    print("失败数量：", len(failures))
    print("输出目录：", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
