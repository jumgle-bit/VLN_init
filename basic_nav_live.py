import os

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import cv2
import numpy as np
from PIL import Image

import habitat
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


SCENE = os.path.expanduser(
    "~/habitat-work/data/scene_datasets/"
    "mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
)

OUTPUT_DIR = os.path.expanduser("~/habitat-work/output_live")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 每一步显示时间，单位为毫秒
FRAME_DELAY_MS = 400

ACTION_NAMES = {
    0: "STOP",
    1: "MOVE FORWARD",
    2: "TURN LEFT",
    3: "TURN RIGHT",
}


def sample_episode(sim):
    for _ in range(5000):
        start = np.asarray(
            sim.pathfinder.get_random_navigable_point(),
            dtype=np.float32,
        )

        goal = np.asarray(
            sim.pathfinder.get_random_navigable_point(),
            dtype=np.float32,
        )

        distance = sim.geodesic_distance(start, goal)

        if np.isfinite(distance) and 12.0 <= distance <= 20.0:
            episode = NavigationEpisode(
                episode_id="0",
                scene_id=SCENE,
                start_position=start.tolist(),
                start_rotation=[0.0, 0.0, 0.0, 1.0],
                goals=[
                    NavigationGoal(
                        position=goal.tolist(),
                        radius=0.5,
                    )
                ],
                info={
                    "geodesic_distance": float(distance),
                },
            )

            return episode, goal, float(distance)

    raise RuntimeError("无法采样合适的起点和终点")


def get_rgb(observations):
    rgb = observations["rgb"]

    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    return rgb.astype(np.uint8)


def make_depth_image(observations):
    depth = observations["depth"].squeeze()

    # Habitat通常输出0～1的归一化深度
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

    return cv2.applyColorMap(
        depth_uint8,
        cv2.COLORMAP_TURBO,
    )


def show_live(
    observations,
    step,
    action_name,
    remaining_distance,
):
    rgb = get_rgb(observations)

    # OpenCV使用BGR通道顺序
    rgb_bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    depth_bgr = make_depth_image(observations)

    frame = np.hstack(
        [rgb_bgr, depth_bgr]
    )

    # 放大显示
    frame = cv2.resize(
        frame,
        (768, 384),
        interpolation=cv2.INTER_NEAREST,
    )

    text = (
        f"Step: {step:03d} | "
        f"Action: {action_name} | "
        f"Remaining: {remaining_distance:.2f} m"
    )

    cv2.rectangle(
        frame,
        (0, 0),
        (768, 40),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        frame,
        text,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "RGB",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "DEPTH",
        (395, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.imshow(
        "Habitat Live Navigation",
        frame,
    )

    key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF

    # 按Q可以提前退出
    return key != ord("q")


def save_rgb(observations, filename):
    rgb = get_rgb(observations)

    Image.fromarray(rgb).save(
        os.path.join(
            OUTPUT_DIR,
            filename,
        )
    )


def main():
    if not os.path.exists(SCENE):
        raise FileNotFoundError(
            f"没有找到场景：{SCENE}"
        )

    config = habitat.get_config(
        "benchmark/nav/pointnav/"
        "pointnav_habitat_test.yaml"
    )

    with read_write(config):
        config.habitat.simulator.scene = SCENE

        # 无CUDA设备
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1

        # 使用程序生成的内存任务
        config.habitat.dataset.type = ""

        config.habitat.environment.max_episode_steps = 200
        config.habitat.task.measurements = {}

        agent_config = get_agent_config(
            config.habitat.simulator
        )

        agent_config.sim_sensors["rgb_sensor"].height = 128
        agent_config.sim_sensors["rgb_sensor"].width = 128

        agent_config.sim_sensors["depth_sensor"].height = 128
        agent_config.sim_sensors["depth_sensor"].width = 128

    print("正在创建Habitat环境……")

    with habitat.Env(
        config=config,
        dataset=None,
    ) as env:
        episode, goal, initial_distance = sample_episode(
            env.sim
        )

        env.episode_iterator = iter([episode])
        observations = env.reset()

        print("=" * 60)
        print("实时导航环境创建成功")
        print("起点：", np.round(episode.start_position, 3))
        print("终点：", np.round(goal, 3))
        print(f"初始距离：{initial_distance:.3f} m")
        print("实时窗口中按Q可以提前退出")
        print("=" * 60)

        save_rgb(
            observations,
            "start_rgb.png",
        )

        # 先显示起点画面
        show_live(
            observations,
            step=0,
            action_name="START",
            remaining_distance=initial_distance,
        )

        follower = ShortestPathFollower(
            sim=env.sim,
            goal_radius=0.5,
            return_one_hot=False,
        )

        step_count = 0

        while not env.episode_over:
            action = follower.get_next_action(goal)

            if action is None:
                print("无法继续计算路径")
                break

            observations = env.step(action)
            step_count += 1

            position = env.sim.get_agent_state().position

            remaining_distance = env.sim.geodesic_distance(
                position,
                goal,
            )

            action_name = ACTION_NAMES.get(
                int(action),
                str(action),
            )

            print(
                f"step={step_count:03d}, "
                f"action={action_name}, "
                f"remaining={remaining_distance:.3f} m"
            )

            keep_running = show_live(
                observations,
                step=step_count,
                action_name=action_name,
                remaining_distance=remaining_distance,
            )

            if not keep_running:
                print("用户按Q提前退出")
                break

        final_position = env.sim.get_agent_state().position

        final_distance = env.sim.geodesic_distance(
            final_position,
            goal,
        )

        save_rgb(
            observations,
            "final_rgb.png",
        )

        print("=" * 60)
        print(f"导航动作数：{step_count}")
        print(f"最终距离：{final_distance:.3f} m")
        print(f"导航成功：{final_distance <= 0.5}")
        print("在实时窗口中按任意键关闭")

        # 导航结束后保持最终画面
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
