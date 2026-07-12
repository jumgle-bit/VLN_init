import json
import os
import traceback

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import numpy as np

import habitat
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


SCENE_NAME = "17DRP5sb8fy"

DATA_ROOT = os.path.expanduser(
    "~/habitat-work/data"
)

OUTPUT_ROOT = os.path.expanduser(
    "~/habitat-work/vln_expert_data"
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
MAX_STEPS = 500


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


def make_config(split):
    config = habitat.get_config(
        "benchmark/nav/vln_r2r.yaml"
    )

    with read_write(config):
        config.habitat.dataset.split = split

        config.habitat.dataset.data_path = (
            R2R_DATA_PATH
        )

        config.habitat.dataset.scenes_dir = (
            SCENES_DIR
        )

        config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1

        config.habitat.environment.max_episode_steps = (
            MAX_STEPS
        )

        agent_config = get_agent_config(
            config.habitat.simulator
        )

        agent_config.sim_sensors[
            "rgb_sensor"
        ].height = IMAGE_SIZE

        agent_config.sim_sensors[
            "rgb_sensor"
        ].width = IMAGE_SIZE

        agent_config.sim_sensors[
            "depth_sensor"
        ].height = IMAGE_SIZE

        agent_config.sim_sensors[
            "depth_sensor"
        ].width = IMAGE_SIZE

    return config


def prepare_rgb(observations):
    rgb = observations["rgb"]

    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    return rgb.astype(
        np.uint8
    )


def prepare_depth(observations):
    depth = observations["depth"]

    if depth.ndim == 2:
        depth = depth[:, :, None]

    return depth.astype(
        np.float16
    )


def append_training_sample(
    observations,
    action,
    rgb_frames,
    depth_frames,
    actions,
):
    rgb_frames.append(
        prepare_rgb(observations)
    )

    depth_frames.append(
        prepare_depth(observations)
    )

    actions.append(
        int(action)
    )


def collect_one_episode(
    env,
    episode,
):
    env.episode_iterator = iter(
        [episode]
    )

    observations = env.reset()

    goal = np.asarray(
        episode.goals[0].position,
        dtype=np.float32,
    )

    reference_path = [
        np.asarray(
            point,
            dtype=np.float32,
        )
        for point in episode.reference_path
    ]

    if len(reference_path) == 0:
        reference_path = [goal]

    if not np.allclose(
        reference_path[-1],
        goal,
    ):
        reference_path.append(
            goal
        )

    follower = ShortestPathFollower(
        sim=env.sim,
        goal_radius=0.4,
        return_one_hot=False,
    )

    rgb_frames = []
    depth_frames = []
    actions = []

    step_count = 0

    for waypoint in reference_path:
        while not env.episode_over:
            if step_count >= MAX_STEPS - 1:
                raise RuntimeError(
                    "专家轨迹超过最大步数"
                )

            action = follower.get_next_action(
                waypoint
            )

            if action is None:
                raise RuntimeError(
                    "ShortestPathFollower无法生成动作"
                )

            if int(action) == int(
                HabitatSimActions.stop
            ):
                break

            append_training_sample(
                observations=observations,
                action=action,
                rgb_frames=rgb_frames,
                depth_frames=depth_frames,
                actions=actions,
            )

            observations = env.step(
                action
            )

            step_count += 1

        if env.episode_over:
            raise RuntimeError(
                "到达目标前episode已经结束"
            )

    append_training_sample(
        observations=observations,
        action=HabitatSimActions.stop,
        rgb_frames=rgb_frames,
        depth_frames=depth_frames,
        actions=actions,
    )

    observations = env.step(
        HabitatSimActions.stop
    )

    step_count += 1

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

    success = float(
        metrics.get(
            "success",
            final_distance <= 3.0,
        )
    )

    spl = float(
        metrics.get(
            "spl",
            0.0,
        )
    )

    oracle_success = float(
        metrics.get(
            "oracle_success",
            success,
        )
    )

    rgb_array = np.stack(
        rgb_frames,
        axis=0,
    )

    depth_array = np.stack(
        depth_frames,
        axis=0,
    )

    action_array = np.asarray(
        actions,
        dtype=np.int64,
    )

    return {
        "rgb": rgb_array,
        "depth": depth_array,
        "actions": action_array,
        "final_distance": final_distance,
        "success": success,
        "spl": spl,
        "oracle_success": oracle_success,
    }


def get_episode_paths(
    split,
    episode,
):
    split_directory = os.path.join(
        OUTPUT_ROOT,
        split,
    )

    os.makedirs(
        split_directory,
        exist_ok=True,
    )

    episode_id = str(
        episode.episode_id
    )

    trajectory_id = str(
        getattr(
            episode,
            "trajectory_id",
            "unknown",
        )
    )

    file_stem = (
        f"episode_{episode_id}_"
        f"trajectory_{trajectory_id}"
    )

    npz_path = os.path.join(
        split_directory,
        file_stem + ".npz",
    )

    json_path = os.path.join(
        split_directory,
        file_stem + ".json",
    )

    return npz_path, json_path


def load_existing_metadata(
    split,
    episode,
):
    npz_path, json_path = (
        get_episode_paths(
            split,
            episode,
        )
    )

    if not os.path.exists(npz_path):
        return None

    if not os.path.exists(json_path):
        return None

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_episode(
    split,
    episode,
    collected,
):
    npz_path, json_path = (
        get_episode_paths(
            split,
            episode,
        )
    )

    instruction_tokens = (
        get_instruction_tokens(
            episode
        )
    )

    np.savez_compressed(
        npz_path,
        rgb=collected["rgb"],
        depth=collected["depth"],
        actions=collected["actions"],
        instruction_tokens=instruction_tokens,
    )

    metadata = {
        "split": split,
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
        "prompt": get_prompt(
            episode
        ),
        "start_position": [
            float(value)
            for value in episode.start_position
        ],
        "goal_position": [
            float(value)
            for value
            in episode.goals[0].position
        ],
        "reference_path_points": int(
            len(episode.reference_path)
        ),
        "num_steps": int(
            len(collected["actions"])
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
        "oracle_success": float(
            collected["oracle_success"]
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


def collect_split(split):
    print("=" * 70)
    print(
        f"开始处理split：{split}"
    )

    config = make_config(
        split
    )

    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type,
        config=config.habitat.dataset,
    )

    selected_episodes = [
        episode
        for episode in dataset.episodes
        if SCENE_NAME in episode.scene_id
    ]

    print(
        f"{split}中找到"
        f"{len(selected_episodes)}个episode"
    )

    if len(selected_episodes) == 0:
        raise RuntimeError(
            f"{split}中没有找到场景"
            f"{SCENE_NAME}"
        )

    # 关键：让环境只加载已经下载的场景
    dataset.episodes = selected_episodes

    manifest = []
    failures = []

    with habitat.Env(
        config=config,
        dataset=dataset,
    ) as env:

        total_episodes = len(
            selected_episodes
        )

        for index, episode in enumerate(
            selected_episodes,
            start=1,
        ):
            existing_metadata = (
                load_existing_metadata(
                    split,
                    episode,
                )
            )

            if existing_metadata is not None:
                manifest.append(
                    existing_metadata
                )

                print(
                    f"[{index:03d}/"
                    f"{total_episodes:03d}] "
                    f"Episode "
                    f"{episode.episode_id}"
                    f"已经存在，跳过"
                )

                continue

            prompt = get_prompt(
                episode
            )

            print("-" * 70)

            print(
                f"[{index:03d}/"
                f"{total_episodes:03d}] "
                f"正在采集Episode "
                f"{episode.episode_id}"
            )

            print(
                f"Prompt：{prompt}"
            )

            try:
                collected = (
                    collect_one_episode(
                        env,
                        episode,
                    )
                )

                metadata = save_episode(
                    split,
                    episode,
                    collected,
                )

                manifest.append(
                    metadata
                )

                print(
                    "采集完成："
                    f"{metadata['num_steps']}步，"
                    f"Success="
                    f"{metadata['success']:.0f}，"
                    f"SPL="
                    f"{metadata['spl']:.3f}，"
                    f"Final distance="
                    f"{metadata['final_distance']:.3f}m"
                )

            except KeyboardInterrupt:
                print(
                    "\n用户中断采集"
                )

                print(
                    "重新运行程序可以继续"
                )

                raise

            except Exception as error:
                failure = {
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
                    "error": str(error),
                }

                failures.append(
                    failure
                )

                print(
                    f"Episode "
                    f"{episode.episode_id}"
                    f"采集失败：{error}"
                )

                traceback.print_exc()

    split_directory = os.path.join(
        OUTPUT_ROOT,
        split,
    )

    manifest_path = os.path.join(
        split_directory,
        "manifest.json",
    )

    failures_path = os.path.join(
        split_directory,
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

    print(
        f"{split}处理完成"
    )

    print(
        f"成功数量：{len(manifest)}"
    )

    print(
        f"失败数量：{len(failures)}"
    )


def main():
    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True,
    )

    collect_split(
        "train"
    )

    collect_split(
        "val_seen"
    )

    print("=" * 70)
    print(
        "全部专家轨迹采集完成"
    )
    print(
        f"输出目录：{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
