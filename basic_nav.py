import os

# 纯CPU、无NVIDIA显卡环境
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import numpy as np
from PIL import Image

import habitat
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


# Matterport3D示例场景
SCENE = os.path.expanduser(
    "~/habitat-work/data/scene_datasets/"
    "mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
)

# 输出图片保存位置
OUTPUT_DIR = os.path.expanduser("~/habitat-work/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sample_episode(sim):
    """
    在导航网格中随机生成一个起点和终点。
    要求两点之间的最短可通行距离为3～8米。
    """
    for _ in range(500):
        start = sim.pathfinder.get_random_navigable_point()
        goal = sim.pathfinder.get_random_navigable_point()

        distance = sim.geodesic_distance(start, goal)

        if np.isfinite(distance) and 3.0 <= distance <= 8.0:
            episode = NavigationEpisode(
                episode_id="0",
                scene_id=SCENE,
                start_position=np.asarray(start, dtype=np.float32).tolist(),
                start_rotation=[0.0, 0.0, 0.0, 1.0],
                goals=[
                    NavigationGoal(
                        position=np.asarray(goal, dtype=np.float32).tolist(),
                        radius=0.5,
                    )
                ],
                info={
                    "geodesic_distance": float(distance),
                },
            )

            return episode, goal, float(distance)

    raise RuntimeError("无法采样到合适的起点和终点")


def save_rgb(observations, filename):
    """保存智能体看到的RGB画面。"""
    if "rgb" not in observations:
        return

    rgb = observations["rgb"]

    # 某些版本可能返回RGBA图像，只保留前三个RGB通道
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    Image.fromarray(rgb.astype(np.uint8)).save(
        os.path.join(OUTPUT_DIR, filename)
    )


def main():
    if not os.path.exists(SCENE):
        raise FileNotFoundError(
            f"没有找到场景文件：{SCENE}"
        )

    print("=" * 60)
    print("正在读取Habitat配置")

    config = habitat.get_config(
        "benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )

    with read_write(config):
        # 使用已经下载的Matterport3D示例场景
        config.habitat.simulator.scene = SCENE

        # 关键设置：
        # -1表示不绑定CUDA/NVIDIA设备
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1

        # 不读取缺失的官方测试数据集
        # 程序会在内存中自行生成一个导航任务
        config.habitat.dataset.type = ""

        # 最多执行200个动作
        config.habitat.environment.max_episode_steps = 200

        # 暂时关闭正式评价指标
        config.habitat.task.measurements = {}

        # 获取智能体配置
        agent_config = get_agent_config(
            config.habitat.simulator
        )

        # 降低RGB分辨率，减轻CPU渲染压力
        agent_config.sim_sensors["rgb_sensor"].height = 128
        agent_config.sim_sensors["rgb_sensor"].width = 128

        # 降低深度图分辨率
        agent_config.sim_sensors["depth_sensor"].height = 128
        agent_config.sim_sensors["depth_sensor"].width = 128

    print("正在创建Habitat-Lab环境")

    with habitat.Env(config=config, dataset=None) as env:
        # 在当前场景中随机生成一次导航任务
        episode, goal, initial_distance = sample_episode(
            env.sim
        )

        # 把生成的导航任务加入环境
        env.episode_iterator = iter([episode])

        # 重置环境，使智能体进入起点
        observations = env.reset()

        print("=" * 60)
        print("Habitat-Lab环境创建成功")
        print("观测数据：", list(observations.keys()))
        print(
            "起点：",
            np.round(
                np.asarray(episode.start_position),
                3,
            ),
        )
        print(
            "终点：",
            np.round(goal, 3),
        )
        print(
            f"初始最短路径距离："
            f"{initial_distance:.3f} m"
        )

        # 输出所有观测的形状
        for name, value in observations.items():
            if hasattr(value, "shape"):
                print(
                    f"{name}: "
                    f"shape={value.shape}, "
                    f"dtype={value.dtype}"
                )

        # 保存起点处看到的画面
        save_rgb(
            observations,
            "start_rgb.png",
        )

        # 创建最短路径Oracle
        follower = ShortestPathFollower(
            sim=env.sim,
            goal_radius=0.5,
            return_one_hot=False,
        )

        step_count = 0

        print("=" * 60)
        print("开始自动导航")

        while not env.episode_over:
            # 根据当前位姿和目标位置计算下一个动作
            action = follower.get_next_action(goal)

            if action is None:
                print("最短路径跟随器返回None，停止导航")
                break

            # 执行动作并获得新的RGB-D观测
            observations = env.step(action)
            step_count += 1

            agent_position = (
                env.sim.get_agent_state().position
            )

            remaining_distance = (
                env.sim.geodesic_distance(
                    agent_position,
                    goal,
                )
            )

            print(
                f"step={step_count:03d}, "
                f"action={action}, "
                f"remaining="
                f"{remaining_distance:.3f} m"
            )

            # 每10步保存一张RGB图像
            if step_count % 10 == 0:
                save_rgb(
                    observations,
                    f"step_{step_count:03d}.png",
                )

        # 获取最终位置和距离
        final_position = (
            env.sim.get_agent_state().position
        )

        final_distance = env.sim.geodesic_distance(
            final_position,
            goal,
        )

        # 保存终点处看到的画面
        save_rgb(
            observations,
            "final_rgb.png",
        )

        print("=" * 60)
        print(f"导航动作数：{step_count}")
        print(f"最终距离：{final_distance:.3f} m")
        print(
            f"导航是否成功："
            f"{final_distance <= 0.5}"
        )
        print(
            f"结果图片目录：{OUTPUT_DIR}"
        )


if __name__ == "__main__":
    main()
