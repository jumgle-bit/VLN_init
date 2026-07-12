import json
import os
import sys
import textwrap

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

from train_vln import (
    ACTION_NAMES,
    START_ACTION,
    VLNPolicy,
    encode_instruction,
)


SCENE_NAME = "17DRP5sb8fy"

DATA_ROOT = os.path.join(
    WORK_ROOT,
    "data",
)

CHECKPOINT_PATH = os.path.join(
    WORK_ROOT,
    "checkpoints",
    "vln_seq2seq_best.pth",
)

RESULT_PATH = os.path.join(
    WORK_ROOT,
    "checkpoints",
    "closed_loop_results.json",
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
DISPLAY_SIZE = 320
MAX_STEPS = 200

SHOW_LIVE = True
FRAME_DELAY_MS = 120

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


def make_rgb_image(observations):
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
        (DISPLAY_SIZE, DISPLAY_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )


def make_depth_image(observations):
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
        (DISPLAY_SIZE, DISPLAY_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )


def show_live_frame(
    observations,
    prompt,
    episode_id,
    step,
    action,
    probabilities,
):
    rgb = make_rgb_image(observations)
    depth = make_depth_image(observations)

    views = np.hstack([rgb, depth])

    prompt_lines = textwrap.wrap(
        "PROMPT: " + prompt,
        width=75,
    )

    header_height = 100 + 25 * len(prompt_lines)

    frame = np.full(
        (
            header_height + DISPLAY_SIZE,
            views.shape[1],
            3,
        ),
        20,
        dtype=np.uint8,
    )

    y = 27

    for line in prompt_lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 24

    action_name = ACTION_NAMES.get(
        action,
        str(action),
    )

    status_text = (
        f"Episode: {episode_id} | "
        f"Step: {step:03d} | "
        f"Model action: {action_name}"
    )

    cv2.putText(
        frame,
        status_text,
        (12, header_height - 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    probability_text = (
        f"STOP={probabilities[0]:.2f}  "
        f"FORWARD={probabilities[1]:.2f}  "
        f"LEFT={probabilities[2]:.2f}  "
        f"RIGHT={probabilities[3]:.2f}"
    )

    cv2.putText(
        frame,
        probability_text,
        (12, header_height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.51,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )

    frame[
        header_height:
        header_height + DISPLAY_SIZE,
        :,
    ] = views

    cv2.putText(
        frame,
        "RGB",
        (10, header_height + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "DEPTH",
        (
            DISPLAY_SIZE + 10,
            header_height + 27,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.imshow(
        "Learned VLN Closed-Loop Evaluation",
        frame,
    )

    key = cv2.waitKey(
        FRAME_DELAY_MS
    ) & 0xFF

    return key != ord("q")


def metric_to_float(
    metrics,
    key,
    default=0.0,
):
    value = metrics.get(
        key,
        default,
    )

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"没有找到模型：{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    vocab = checkpoint["vocab"]
    model_config = checkpoint["model_config"]

    model = VLNPolicy(
        **model_config
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("=" * 70)
    print("加载模型：", CHECKPOINT_PATH)
    print("模型来自Epoch：", checkpoint["epoch"])
    print(
        "保存时验证损失：",
        checkpoint["validation_loss"],
    )
    print(
        "保存时验证准确率：",
        checkpoint["validation_accuracy"],
    )

    return model, vocab


def evaluate_one_episode(
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

    step_count = 0
    stopped = False
    user_quit = False

    action_histogram = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }

    print("-" * 70)
    print("Episode：", episode.episode_id)
    print("Prompt：", prompt)

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

            probabilities_tensor = torch.softmax(
                logits,
                dim=1,
            )

        probabilities = probabilities_tensor[
            0
        ].cpu().numpy()

        action = int(
            torch.argmax(
                probabilities_tensor,
                dim=1,
            ).item()
        )

        action_histogram[action] += 1
        step_count += 1

        if SHOW_LIVE:
            keep_running = show_live_frame(
                observations=observations,
                prompt=prompt,
                episode_id=episode.episode_id,
                step=step_count,
                action=action,
                probabilities=probabilities,
            )

            if not keep_running:
                user_quit = True
                break

        observations = env.step(action)
        previous_action = action

        if action == 0:
            stopped = True
            break

    metrics = env.get_metrics()

    result = {
        "episode_id": str(episode.episode_id),
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
        "user_quit": user_quit,
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

    print("动作数：", result["steps"])
    print("主动STOP：", result["stopped"])
    print("Success：", result["success"])
    print("SPL：", result["spl"])
    print(
        "Distance to goal：",
        result["distance_to_goal"],
    )
    print(
        "动作分布：",
        result["action_histogram"],
    )

    return result


def main():
    torch.set_num_threads(4)

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

    selected_episodes = sorted(
        selected_episodes,
        key=lambda episode: int(
            episode.episode_id
        ),
    )

    if len(selected_episodes) == 0:
        raise RuntimeError(
            "没有找到可评估的val_seen episode"
        )

    dataset.episodes = selected_episodes

    print(
        "闭环评估Episode数量：",
        len(selected_episodes),
    )

    results = []

    with habitat.Env(
        config=config,
        dataset=dataset,
    ) as env:
        for episode in selected_episodes:
            result = evaluate_one_episode(
                env=env,
                episode=episode,
                model=model,
                vocab=vocab,
            )

            results.append(result)

            if result["user_quit"]:
                break

    if SHOW_LIVE:
        cv2.destroyAllWindows()

    if len(results) == 0:
        raise RuntimeError(
            "没有完成任何episode"
        )

    average_success = float(
        np.mean(
            [
                result["success"]
                for result in results
            ]
        )
    )

    average_spl = float(
        np.mean(
            [
                result["spl"]
                for result in results
            ]
        )
    )

    finite_distances = [
        result["distance_to_goal"]
        for result in results
        if np.isfinite(
            result["distance_to_goal"]
        )
    ]

    if finite_distances:
        average_distance = float(
            np.mean(finite_distances)
        )
    else:
        average_distance = float("nan")

    summary = {
        "checkpoint": CHECKPOINT_PATH,
        "num_episodes": len(results),
        "average_success": average_success,
        "average_spl": average_spl,
        "average_distance_to_goal": average_distance,
        "results": results,
    }

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 70)
    print("闭环VLN评估完成")
    print("评估Episode数量：", len(results))
    print(
        f"平均Success：{average_success:.4f}"
    )
    print(
        f"平均SPL：{average_spl:.4f}"
    )
    print(
        f"平均目标距离："
        f"{average_distance:.4f} m"
    )
    print("结果文件：", RESULT_PATH)


if __name__ == "__main__":
    main()
