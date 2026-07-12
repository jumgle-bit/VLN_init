"""实时运行当前 CPU 单场景实验的最终 VLN 模型。"""

import argparse
import os
import sys


WORK_ROOT = os.path.expanduser("~/habitat-work")
sys.path.insert(0, WORK_ROOT)


try:
    import run_vln_dagger_live as live
except ImportError as error:
    raise ImportError(
        "没有找到 ~/habitat-work/run_vln_dagger_live.py。"
        "请把本文件放到 ~/habitat-work/ 后运行。"
    ) from error


FINAL_CHECKPOINT = os.path.join(
    WORK_ROOT,
    "checkpoints_stop_aware",
    "vln_stop_aware_epoch_02.pth",
)


live.CHECKPOINT_PATHS = {
    "final_stop_aware_epoch_02": FINAL_CHECKPOINT,
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

live.OUTPUT_DIR = os.path.join(
    WORK_ROOT,
    "output_final_live",
)

live.WINDOW_NAME = (
    "Final STOP-aware VLN - Q or ESC to quit"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="实时观看最终 STOP-aware VLN 模型。"
    )

    parser.add_argument(
        "--episode-id",
        default="328",
        help="val_seen Episode ID，默认 328。",
    )

    parser.add_argument(
        "--checkpoint",
        choices=tuple(live.CHECKPOINT_PATHS.keys()),
        default="final_stop_aware_epoch_02",
        help="默认运行最终 STOP-aware Epoch 2。",
    )

    parser.add_argument(
        "--delay-ms",
        type=int,
        default=400,
        help="每个动作显示时间，默认 400 毫秒。",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=live.MAX_STEPS,
        help="最大动作数，默认 200。",
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
        help="不保存 AVI 录像。",
    )

    return parser.parse_args()


def main():
    if not os.path.exists(FINAL_CHECKPOINT):
        raise FileNotFoundError(
            f"没有找到最终模型：{FINAL_CHECKPOINT}"
        )

    # 替换原实时脚本的命令行解析器，其余推理和显示逻辑保持完全一致。
    live.parse_arguments = parse_arguments
    live.main()


if __name__ == "__main__":
    main()