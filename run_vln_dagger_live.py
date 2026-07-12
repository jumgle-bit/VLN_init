import argparse
import os
import sys


# 必须在导入 Habitat-Sim 之前设置。
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")


import cv2
import numpy as np
import torch

import habitat
from habitat.config import read_write
from habitat.config.default import get_agent_config


WORK_ROOT = os.path.expanduser("~/habitat-work")
sys.path.insert(0, WORK_ROOT)


from train_vln import (  # noqa: E402
    ACTION_NAMES,
    START_ACTION,
    encode_instruction,
)

from train_vln_dagger import ImprovedVLNPolicy  # noqa: E402


SCENE_NAME = "17DRP5sb8fy"
DATA_ROOT = os.path.join(WORK_ROOT, "data")

R2R_DATA_PATH = os.path.join(
    DATA_ROOT,
    "datasets/vln/mp3d/r2r/v1/{split}/{split}.json.gz",
)

SCENES_DIR = os.path.join(DATA_ROOT, "scene_datasets")

CHECKPOINT_PATHS = {
    "best_accuracy": os.path.join(
        WORK_ROOT,
        "checkpoints_dagger",
        "vln_dagger_best_accuracy.pth",
    ),
    "best_loss": os.path.join(
        WORK_ROOT,
        "checkpoints_dagger",
        "vln_dagger_best_loss.pth",
    ),
    "last": os.path.join(
        WORK_ROOT,
        "checkpoints_dagger",
        "vln_dagger_last.pth",
    ),
}

OUTPUT_DIR = os.path.join(WORK_ROOT, "output_dagger_live")

IMAGE_SIZE = 96
MAX_STEPS = 200
SUCCESS_DISTANCE = 3.0
DEVICE = torch.device("cpu")

WINDOW_NAME = "Learned VLN Policy - Q or ESC to quit"
VIEW_SIZE = 384
PANEL_WIDTH = 360
CANVAS_WIDTH = VIEW_SIZE * 2 + PANEL_WIDTH
CANVAS_HEIGHT = 720


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="实时观看训练后的 DAgger VLN 模型闭环导航。"
    )

    parser.add_argument(
        "--episode-id",
        default="326",
        help="val_seen Episode ID，默认 326。",
    )

    parser.add_argument(
        "--checkpoint",
        choices=tuple(CHECKPOINT_PATHS.keys()),
        default="best_accuracy",
        help="要运行的 checkpoint，默认 best_accuracy。",
    )

    parser.add_argument(
        "--delay-ms",
        type=int,
        default=400,
        help="每个动作显示多少毫秒，默认 400。",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="最大动作数，默认 200。",
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
        help="不保存 AVI 录像。",
    )

    return parser.parse_args()


def get_prompt(episode):
    instruction = episode.instruction

    if hasattr(instruction, "instruction_text"):
        return instruction.instruction_text.strip()

    if hasattr(instruction, "text"):
        return instruction.text.strip()

    return str(instruction).strip()


def make_config(max_steps):
    config = habitat.get_config("benchmark/nav/vln_r2r.yaml")

    with read_write(config):
        config.habitat.dataset.split = "val_seen"
        config.habitat.dataset.data_path = R2R_DATA_PATH
        config.habitat.dataset.scenes_dir = SCENES_DIR

        # -1 表示不选择 CUDA GPU；在你的 VMware CPU 环境中必须这样设置。
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
        config.habitat.environment.max_episode_steps = max_steps

        config.habitat.task.measurements.success.success_distance = (
            SUCCESS_DISTANCE
        )

        if "oracle_success" in config.habitat.task.measurements:
            config.habitat.task.measurements[
                "oracle_success"
            ].success_distance = SUCCESS_DISTANCE

        agent_config = get_agent_config(config.habitat.simulator)

        agent_config.sim_sensors["rgb_sensor"].height = IMAGE_SIZE
        agent_config.sim_sensors["rgb_sensor"].width = IMAGE_SIZE

        agent_config.sim_sensors["depth_sensor"].height = IMAGE_SIZE
        agent_config.sim_sensors["depth_sensor"].width = IMAGE_SIZE

    return config


def load_selected_episode(config, episode_id):
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type,
        config=config.habitat.dataset,
    )

    matches = [
        episode
        for episode in dataset.episodes
        if SCENE_NAME in episode.scene_id
        and str(episode.episode_id) == str(episode_id)
    ]

    if not matches:
        available_ids = sorted(
            {
                str(episode.episode_id)
                for episode in dataset.episodes
                if SCENE_NAME in episode.scene_id
            }
        )

        raise ValueError(
            f"没有找到 Episode {episode_id}。"
            f"当前场景可用的 val_seen ID：{', '.join(available_ids)}"
        )

    episode = matches[0]

    # 只把所选 Episode 交给环境，保证 reset 后不会跳到别的任务。
    dataset.episodes = [episode]

    return dataset, episode


def load_model(checkpoint_name):
    checkpoint_path = CHECKPOINT_PATHS[checkpoint_name]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"没有找到模型：{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model = ImprovedVLNPolicy(
        **checkpoint["model_config"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["vocab"], checkpoint, checkpoint_path


def observation_to_tensor(observations):
    rgb = observations["rgb"]

    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    rgb_tensor = (
        torch.from_numpy(rgb.copy())
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    depth = observations["depth"]

    if depth.ndim == 2:
        depth = depth[:, :, None]

    depth_tensor = (
        torch.from_numpy(depth.copy())
        .permute(2, 0, 1)
        .float()
    )

    observation = torch.cat(
        [rgb_tensor, depth_tensor],
        dim=0,
    ).unsqueeze(0)

    return observation.to(DEVICE)


def rgb_for_display(observations):
    rgb = observations["rgb"]

    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    # Habitat 输出 RGB，OpenCV 显示需要 BGR。
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return cv2.resize(
        bgr,
        (VIEW_SIZE, VIEW_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )


def depth_for_display(observations):
    depth = np.asarray(observations["depth"], dtype=np.float32)

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    valid = depth[depth > 0]

    if valid.size == 0:
        normalized = np.zeros_like(depth, dtype=np.uint8)
    else:
        low = float(np.percentile(valid, 2))
        high = float(np.percentile(valid, 98))

        if high <= low:
            high = low + 1.0

        clipped = np.clip(depth, low, high)
        normalized = ((clipped - low) / (high - low) * 255.0).astype(
            np.uint8
        )

    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)

    return cv2.resize(
        colored,
        (VIEW_SIZE, VIEW_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )


def draw_text(
    image,
    text,
    x,
    y,
    scale=0.52,
    color=(235, 235, 235),
    thickness=1,
):
    cv2.putText(
        image,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def wrap_text_by_width(text, max_width, scale=0.48, thickness=1):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        candidate = word if not current else f"{current} {word}"
        width = cv2.getTextSize(
            candidate,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )[0][0]

        if width <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def make_live_frame(
    observations,
    prompt,
    episode_id,
    checkpoint_name,
    step_number,
    action,
    probabilities,
    status,
):
    canvas = np.full(
        (CANVAS_HEIGHT, CANVAS_WIDTH, 3),
        24,
        dtype=np.uint8,
    )

    rgb = rgb_for_display(observations)
    depth = depth_for_display(observations)

    canvas[52 : 52 + VIEW_SIZE, 0:VIEW_SIZE] = rgb
    canvas[52 : 52 + VIEW_SIZE, VIEW_SIZE : VIEW_SIZE * 2] = depth

    draw_text(canvas, "RGB observation", 12, 34, scale=0.72)
    draw_text(canvas, "Depth observation", VIEW_SIZE + 12, 34, scale=0.72)

    panel_x = VIEW_SIZE * 2 + 18
    panel_right = CANVAS_WIDTH - 18

    draw_text(
        canvas,
        f"Episode {episode_id}",
        panel_x,
        34,
        scale=0.70,
        color=(100, 220, 255),
        thickness=2,
    )

    draw_text(canvas, f"Checkpoint: {checkpoint_name}", panel_x, 66)
    draw_text(canvas, f"Step: {step_number}", panel_x, 94)

    action_name = "START" if action is None else ACTION_NAMES[action]

    draw_text(
        canvas,
        f"Action: {action_name}",
        panel_x,
        126,
        scale=0.63,
        color=(80, 255, 150),
        thickness=2,
    )

    draw_text(canvas, "Action probabilities", panel_x, 162, scale=0.55)

    if probabilities is None:
        probabilities = np.zeros(4, dtype=np.float32)

    bar_left = panel_x + 108
    bar_width = max(80, panel_right - bar_left - 40)

    for index in range(4):
        y = 194 + index * 35
        probability = float(probabilities[index])
        name = ACTION_NAMES[index]

        draw_text(canvas, name[:10], panel_x, y, scale=0.43)

        cv2.rectangle(
            canvas,
            (bar_left, y - 14),
            (bar_left + bar_width, y + 1),
            (65, 65, 65),
            -1,
        )

        cv2.rectangle(
            canvas,
            (bar_left, y - 14),
            (bar_left + int(bar_width * probability), y + 1),
            (40, 190, 245),
            -1,
        )

        draw_text(
            canvas,
            f"{probability:.2f}",
            bar_left + bar_width + 5,
            y,
            scale=0.42,
        )

    draw_text(canvas, "Instruction", panel_x, 350, scale=0.58)

    prompt_lines = wrap_text_by_width(
        prompt,
        max_width=PANEL_WIDTH - 38,
        scale=0.45,
    )

    y = 376
    for line in prompt_lines[:9]:
        draw_text(canvas, line, panel_x, y, scale=0.45)
        y += 22

    cv2.rectangle(
        canvas,
        (0, 580),
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        (18, 18, 18),
        -1,
    )

    draw_text(
        canvas,
        f"Status: {status}",
        18,
        614,
        scale=0.68,
        color=(120, 230, 255),
        thickness=2,
    )

    draw_text(
        canvas,
        "Policy input: instruction + RGB + depth + previous action",
        18,
        650,
        scale=0.58,
    )

    draw_text(
        canvas,
        "The policy does NOT receive the goal position or reference path.",
        18,
        680,
        scale=0.58,
        color=(180, 210, 255),
    )

    draw_text(
        canvas,
        "Press Q or ESC to stop. Final screen waits for any key.",
        18,
        710,
        scale=0.58,
        color=(200, 200, 200),
    )

    return canvas


def metric_to_float(metrics, name, default=0.0):
    value = metrics.get(name, default)

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def create_video_writer(path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(
        path,
        fourcc,
        fps,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
    )

    if not writer.isOpened():
        print("警告：OpenCV无法创建录像，将只显示实时窗口。")
        return None

    return writer


def show_frame(frame, delay_ms, writer=None):
    if writer is not None:
        writer.write(frame)

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(max(1, int(delay_ms))) & 0xFF

    return key in (ord("q"), ord("Q"), 27)


def run_live_episode(
    env,
    episode,
    model,
    vocab,
    checkpoint_name,
    delay_ms,
    max_steps,
    save_video,
):
    env.episode_iterator = iter([episode])
    observations = env.reset()
    prompt = get_prompt(episode)

    instruction_tokens = encode_instruction(prompt, vocab)

    with torch.no_grad():
        language_context = model.encode_language(instruction_tokens)

    hidden = model.initial_hidden(DEVICE)
    previous_action = START_ACTION

    action_histogram = {index: 0 for index in range(4)}
    stopped = False
    user_aborted = False
    step_count = 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_path = os.path.join(
        OUTPUT_DIR,
        f"episode_{episode.episode_id}_{checkpoint_name}.avi",
    )

    fps = max(1.0, min(20.0, 1000.0 / max(1, delay_ms)))
    writer = None

    if save_video:
        writer = create_video_writer(video_path, fps)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_WIDTH, CANVAS_HEIGHT)

    initial_frame = make_live_frame(
        observations=observations,
        prompt=prompt,
        episode_id=episode.episode_id,
        checkpoint_name=checkpoint_name,
        step_number=0,
        action=None,
        probabilities=None,
        status="READY",
    )

    if show_frame(
        initial_frame,
        max(800, delay_ms),
        writer,
    ):
        user_aborted = True

    while (
        not user_aborted
        and not env.episode_over
        and step_count < max_steps
    ):
        observation_tensor = observation_to_tensor(observations)

        with torch.no_grad():
            visual_feature = model.visual_encoder(observation_tensor)

            logits, hidden = model.navigation_step(
                visual_feature=visual_feature,
                language_context=language_context,
                previous_action=previous_action,
                hidden=hidden,
            )

            probability_tensor = torch.softmax(logits, dim=1)
            action = int(torch.argmax(probability_tensor, dim=1).item())
            probabilities = probability_tensor[0].cpu().numpy()

        step_count += 1
        action_histogram[action] += 1

        print(
            f"step={step_count:03d}  "
            f"action={ACTION_NAMES[action]:>12s}  "
            f"prob="
            + ", ".join(
                f"{ACTION_NAMES[index]}:{probabilities[index]:.3f}"
                for index in range(4)
            )
        )

        decision_frame = make_live_frame(
            observations=observations,
            prompt=prompt,
            episode_id=episode.episode_id,
            checkpoint_name=checkpoint_name,
            step_number=step_count,
            action=action,
            probabilities=probabilities,
            status="RUNNING",
        )

        if show_frame(decision_frame, delay_ms, writer):
            user_aborted = True
            break

        observations = env.step(action)
        previous_action = action

        if action == 0:
            stopped = True
            break

    metrics = env.get_metrics()

    success = metric_to_float(metrics, "success")
    spl = metric_to_float(metrics, "spl")
    distance = metric_to_float(
        metrics,
        "distance_to_goal",
        default=float("nan"),
    )

    if user_aborted:
        final_status = "USER ABORTED"
    elif success >= 0.5:
        final_status = (
            f"SUCCESS  SPL={spl:.3f}  Distance={distance:.3f}m"
        )
    else:
        final_status = (
            f"FAILED  SPL={spl:.3f}  Distance={distance:.3f}m"
        )

    final_frame = make_live_frame(
        observations=observations,
        prompt=prompt,
        episode_id=episode.episode_id,
        checkpoint_name=checkpoint_name,
        step_number=step_count,
        action=0 if stopped else None,
        probabilities=None,
        status=final_status,
    )

    # 多写几帧，让录像结尾可以看清最终结果。
    if writer is not None:
        for _ in range(max(1, int(fps * 2))):
            writer.write(final_frame)

    cv2.imshow(WINDOW_NAME, final_frame)

    if not user_aborted:
        cv2.waitKey(0)
    else:
        cv2.waitKey(300)

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()

    result = {
        "episode_id": str(episode.episode_id),
        "prompt": prompt,
        "steps": step_count,
        "stopped": stopped,
        "user_aborted": user_aborted,
        "success": success,
        "spl": spl,
        "distance_to_goal": distance,
        "action_histogram": {
            ACTION_NAMES[action]: count
            for action, count in action_histogram.items()
        },
        "video_path": video_path if writer is not None else None,
    }

    return result


def main():
    args = parse_arguments()

    print("=" * 70)
    print("正在加载 DAgger VLN 实时导航")
    print("Checkpoint：", args.checkpoint)
    print("Episode ID：", args.episode_id)
    print("运行设备：", DEVICE)
    print("成功距离阈值：", SUCCESS_DISTANCE, "m")
    print("模型不会读取目标坐标或参考路径")

    model, vocab, checkpoint, checkpoint_path = load_model(
        args.checkpoint
    )

    print("模型文件：", checkpoint_path)
    print("模型 Epoch：", checkpoint["epoch"])
    print("保存时 Val accuracy：", checkpoint["validation_accuracy"])

    config = make_config(args.max_steps)
    dataset, episode = load_selected_episode(config, args.episode_id)
    prompt = get_prompt(episode)

    print("-" * 70)
    print("Prompt：", prompt)
    print("场景：", episode.scene_id)
    print("窗口内按 Q 或 Esc 可以提前结束")
    print("=" * 70)

    with habitat.Env(config=config, dataset=dataset) as env:
        result = run_live_episode(
            env=env,
            episode=episode,
            model=model,
            vocab=vocab,
            checkpoint_name=args.checkpoint,
            delay_ms=args.delay_ms,
            max_steps=args.max_steps,
            save_video=not args.no_video,
        )

    print("=" * 70)
    print("实时闭环运行结束")
    print("Episode：", result["episode_id"])
    print("动作数：", result["steps"])
    print("主动 STOP：", result["stopped"])
    print("Success：", result["success"])
    print("SPL：", result["spl"])
    print("Distance to goal：", result["distance_to_goal"])
    print("动作分布：", result["action_histogram"])

    if result["video_path"] is not None:
        print("录像文件：", result["video_path"])


if __name__ == "__main__":
    main()