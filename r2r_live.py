import os
import textwrap

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import cv2
import numpy as np

import habitat
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps


SCENE_NAME = "17DRP5sb8fy"
EPISODE_ID = "326"

DATA_ROOT = os.path.expanduser("~/habitat-work/data")

R2R_DATA_PATH = os.path.join(
    DATA_ROOT,
    "datasets/vln/mp3d/r2r/v1/{split}/{split}.json.gz",
)

SCENES_DIR = os.path.join(
    DATA_ROOT,
    "scene_datasets",
)

VIEW_SIZE = 256
FRAME_DELAY_MS = 350

ACTION_NAMES = {
    0: "STOP",
    1: "MOVE FORWARD",
    2: "TURN LEFT",
    3: "TURN RIGHT",
}


def get_prompt(episode):
    instruction = episode.instruction

    if hasattr(instruction, "instruction_text"):
        return instruction.instruction_text

    if hasattr(instruction, "text"):
        return instruction.text

    return str(instruction)


def make_rgb(observations):
    rgb = observations["rgb"]

    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    rgb = rgb.astype(np.uint8)

    rgb_bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    return cv2.resize(
        rgb_bgr,
        (VIEW_SIZE, VIEW_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )


def make_depth(observations):
    depth = observations["depth"].squeeze()

    if depth.max() <= 1.01:
        depth_uint8 = np.clip(
            depth * 255.0,
            0,
            255,
        ).astype(np.uint8)
    else:
        depth_uint8 = np.clip(
            depth / 10.0 * 255.0,
            0,
            255,
        ).astype(np.uint8)

    depth_color = cv2.applyColorMap(
        depth_uint8,
        cv2.COLORMAP_TURBO,
    )

    return cv2.resize(
        depth_color,
        (VIEW_SIZE, VIEW_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )


def make_topdown(metrics):
    topdown_info = metrics.get("top_down_map")

    if topdown_info is None:
        return np.zeros(
            (VIEW_SIZE, VIEW_SIZE, 3),
            dtype=np.uint8,
        )

    map_image = (
        maps.colorize_draw_agent_and_fit_to_height(
            topdown_info,
            VIEW_SIZE,
        )
    )

    if map_image.shape[-1] == 4:
        map_image = map_image[:, :, :3]

    map_image = map_image.astype(np.uint8)

    return cv2.cvtColor(
        map_image,
        cv2.COLOR_RGB2BGR,
    )


def show_frame(
    observations,
    metrics,
    prompt,
    step,
    action_name,
    remaining_distance,
    waypoint_index,
    waypoint_count,
):
    rgb = make_rgb(observations)
    depth = make_depth(observations)
    topdown = make_topdown(metrics)

    views = np.hstack(
        [rgb, depth, topdown]
    )

    prompt_lines = textwrap.wrap(
        "PROMPT: " + prompt,
        width=92,
    )

    header_height = 65 + 25 * len(prompt_lines)

    # 俯视地图会保持场景长宽比，因此宽度不一定等于VIEW_SIZE
    frame = np.full(
    (
        header_height + views.shape[0],
        views.shape[1],
        3,
    ),
    20,
    dtype=np.uint8,
	)

    y = 28

    for line in prompt_lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 25

    status = (
        f"Step: {step:03d} | "
        f"Action: {action_name} | "
        f"Remaining: {remaining_distance:.2f} m | "
        f"Waypoint: {waypoint_index}/{waypoint_count}"
    )

    cv2.putText(
        frame,
        status,
        (12, header_height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    frame[
        header_height:header_height + VIEW_SIZE,
        :,
    ] = views

    cv2.putText(
        frame,
        "RGB",
        (10, header_height + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "DEPTH",
        (VIEW_SIZE + 10, header_height + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "TOP-DOWN MAP",
        (VIEW_SIZE * 2 + 10, header_height + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.imshow(
        "R2R Vision-Language Navigation",
        frame,
    )

    key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF

    return key != ord("q")


def main():
    config = habitat.get_config(
        "benchmark/nav/vln_r2r.yaml"
    )

    with read_write(config):
        config.habitat.dataset.split = "val_seen"
        config.habitat.dataset.data_path = R2R_DATA_PATH
        config.habitat.dataset.scenes_dir = SCENES_DIR

        config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
        config.habitat.environment.max_episode_steps = 500

        agent_config = get_agent_config(
            config.habitat.simulator
        )

        agent_config.sim_sensors["rgb_sensor"].height = 160
        agent_config.sim_sensors["rgb_sensor"].width = 160

        agent_config.sim_sensors["depth_sensor"].height = 160
        agent_config.sim_sensors["depth_sensor"].width = 160

        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_resolution=512,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_goal_positions=True,
                )
            }
        )

    print("正在加载R2R val_seen数据集……")

    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type,
        config=config.habitat.dataset,
    )

    selected = [
        episode
        for episode in dataset.episodes
        if (
            SCENE_NAME in episode.scene_id
            and str(episode.episode_id) == EPISODE_ID
        )
    ]

    if not selected:
        raise RuntimeError(
            f"没有找到Episode {EPISODE_ID}"
        )

    dataset.episodes = selected

    with habitat.Env(
        config=config,
        dataset=dataset,
    ) as env:
        observations = env.reset()
        episode = env.current_episode

        prompt = get_prompt(episode)
        goal = np.asarray(
            episode.goals[0].position,
            dtype=np.float32,
        )

        reference_path = [
            np.asarray(point, dtype=np.float32)
            for point in episode.reference_path
        ]

        if not np.allclose(reference_path[-1], goal):
            reference_path.append(goal)

        start_position = np.asarray(
            episode.start_position,
            dtype=np.float32,
        )

        initial_distance = env.sim.geodesic_distance(
            start_position,
            goal,
        )

        print("=" * 70)
        print("真实R2R-VLN任务加载成功")
        print("Episode ID：", episode.episode_id)
        print("Prompt：", prompt)
        print("起点：", np.round(start_position, 3))
        print("终点：", np.round(goal, 3))
        print("参考路径点数：", len(reference_path))
        print(f"最短路径距离：{initial_distance:.3f} m")
        print("窗口中按Q可以提前退出")
        print("=" * 70)

        follower = ShortestPathFollower(
            sim=env.sim,
            goal_radius=0.4,
            return_one_hot=False,
        )

        metrics = env.get_metrics()

        show_frame(
            observations=observations,
            metrics=metrics,
            prompt=prompt,
            step=0,
            action_name="START",
            remaining_distance=initial_distance,
            waypoint_index=0,
            waypoint_count=len(reference_path),
        )

        step_count = 0
        user_quit = False

        for waypoint_index, waypoint in enumerate(
            reference_path,
            start=1,
        ):
            while not env.episode_over:
                action = follower.get_next_action(
                    waypoint
                )

                if (
                    action is None
                    or int(action)
                    == int(HabitatSimActions.stop)
                ):
                    break

                observations = env.step(action)
                step_count += 1

                position = np.asarray(
                    env.sim.get_agent_state().position,
                    dtype=np.float32,
                )

                remaining_distance = (
                    env.sim.geodesic_distance(
                        position,
                        goal,
                    )
                )

                action_name = ACTION_NAMES.get(
                    int(action),
                    str(action),
                )

                print(
                    f"step={step_count:03d}, "
                    f"action={action_name}, "
                    f"remaining={remaining_distance:.3f} m, "
                    f"waypoint={waypoint_index}/"
                    f"{len(reference_path)}"
                )

                metrics = env.get_metrics()

                keep_running = show_frame(
                    observations=observations,
                    metrics=metrics,
                    prompt=prompt,
                    step=step_count,
                    action_name=action_name,
                    remaining_distance=remaining_distance,
                    waypoint_index=waypoint_index,
                    waypoint_count=len(reference_path),
                )

                if not keep_running:
                    user_quit = True
                    break

            if user_quit or env.episode_over:
                break

        if not user_quit and not env.episode_over:
            observations = env.step(
                HabitatSimActions.stop
            )
            step_count += 1

        final_position = np.asarray(
            env.sim.get_agent_state().position,
            dtype=np.float32,
        )

        final_distance = env.sim.geodesic_distance(
            final_position,
            goal,
        )

        metrics = env.get_metrics()

        print("=" * 70)
        print("R2R导航结束")
        print(f"动作数：{step_count}")
        print(f"最终目标距离：{final_distance:.3f} m")
        print("Success：", metrics.get("success"))
        print("SPL：", metrics.get("spl"))
        print("Oracle Success：", metrics.get("oracle_success"))
        print("按任意键关闭窗口")

        show_frame(
            observations=observations,
            metrics=metrics,
            prompt=prompt,
            step=step_count,
            action_name="FINISHED",
            remaining_distance=final_distance,
            waypoint_index=len(reference_path),
            waypoint_count=len(reference_path),
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
