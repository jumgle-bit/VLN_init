"""
STOP-aware VLN 训练。

目标：解决第二轮 DAgger 模型能够接近目标、但不主动 STOP 的问题。

方法：
1. 使用 75 Expert + 75 DAgger-v1 + 75 DAgger-v2。
2. 从稳定的第一轮 DAgger best_accuracy 模型继续训练。
3. 将 STOP 类别权重从 3.0 提升到 6.0。
4. 使用较小学习率 2e-4。
5. 每个 Epoch 都保存 checkpoint，之后逐个做闭环评估。

本脚本不会使用 val_seen 进行梯度更新，也不会覆盖旧 checkpoint。
"""

import json
import os


import numpy as np
import torch
import torch.nn as nn


from train_vln import (
    ACTION_NAMES,
    NUM_ACTIONS,
    run_epoch,
    set_random_seed,
)

from train_vln_dagger import (
    ImprovedVLNPolicy,
    load_manifest,
)


WORK_ROOT = os.path.expanduser("~/habitat-work")

EXPERT_MANIFEST = os.path.join(
    WORK_ROOT,
    "vln_expert_data",
    "train",
    "manifest.json",
)

DAGGER_V1_MANIFEST = os.path.join(
    WORK_ROOT,
    "vln_dagger_data",
    "train",
    "manifest.json",
)

DAGGER_V2_MANIFEST = os.path.join(
    WORK_ROOT,
    "vln_dagger_data_v2",
    "train",
    "manifest.json",
)

VAL_MANIFEST = os.path.join(
    WORK_ROOT,
    "vln_expert_data",
    "val_seen",
    "manifest.json",
)

PARENT_CHECKPOINT = os.path.join(
    WORK_ROOT,
    "checkpoints_dagger",
    "vln_dagger_best_accuracy.pth",
)

CHECKPOINT_DIR = os.path.join(
    WORK_ROOT,
    "checkpoints_stop_aware",
)

VOCAB_PATH = os.path.join(
    CHECKPOINT_DIR,
    "vocab.json",
)

HISTORY_PATH = os.path.join(
    CHECKPOINT_DIR,
    "training_history.json",
)

EXPERIMENT_INFO_PATH = os.path.join(
    CHECKPOINT_DIR,
    "experiment_info.json",
)

DEVICE = torch.device("cpu")

NUM_EPOCHS = 5
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
STOP_WEIGHT = 6.0
RANDOM_SEED = 168


def epoch_checkpoint_path(epoch):
    return os.path.join(
        CHECKPOINT_DIR,
        f"vln_stop_aware_epoch_{epoch:02d}.pth",
    )


def count_steps(metadata_list):
    return sum(
        int(item.get("num_steps", 0))
        for item in metadata_list
    )


def count_actions(metadata_list):
    counts = np.zeros(
        NUM_ACTIONS,
        dtype=np.int64,
    )

    unreadable_files = []

    for item in metadata_list:
        path = item["npz_path"]

        try:
            with np.load(path) as data:
                actions = np.asarray(
                    data["actions"],
                    dtype=np.int64,
                )

            counts += np.bincount(
                actions,
                minlength=NUM_ACTIONS,
            )[:NUM_ACTIONS]

        except Exception as error:
            unreadable_files.append(
                {
                    "path": path,
                    "error": str(error),
                }
            )

    if unreadable_files:
        raise RuntimeError(
            "以下训练文件无法读取：\n"
            + "\n".join(
                item["path"]
                for item in unreadable_files
            )
        )

    return counts


def load_parent_model():
    if not os.path.exists(PARENT_CHECKPOINT):
        raise FileNotFoundError(
            f"没有找到父模型：{PARENT_CHECKPOINT}"
        )

    checkpoint = torch.load(
        PARENT_CHECKPOINT,
        map_location=DEVICE,
    )

    model = ImprovedVLNPolicy(
        **checkpoint["model_config"]
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model, checkpoint["vocab"], checkpoint


def serializable_result(result):
    return {
        "loss": float(result["loss"]),
        "accuracy": float(result["accuracy"]),
        "per_class_accuracy": {
            str(action_name): float(accuracy)
            for action_name, accuracy in result[
                "per_class_accuracy"
            ].items()
        },
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    vocab,
    epoch,
    train_result,
    validation_result,
    action_counts,
    data_counts,
):
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab": vocab,
        "architecture": "ImprovedVLNPolicy",
        "model_config": {
            "vocabulary_size": len(vocab),
            "embedding_size": 64,
            "language_size": 128,
            "visual_size": 192,
            "action_embedding_size": 24,
            "hidden_size": 192,
            "dropout": 0.2,
        },
        "validation_loss": float(
            validation_result["loss"]
        ),
        "validation_accuracy": float(
            validation_result["accuracy"]
        ),
        "train_loss": float(train_result["loss"]),
        "train_accuracy": float(
            train_result["accuracy"]
        ),
        "action_names": ACTION_NAMES,
        "training_method": "stop_aware_dagger",
        "parent_checkpoint": PARENT_CHECKPOINT,
        "stop_weight": STOP_WEIGHT,
        "class_weights": [
            STOP_WEIGHT,
            1.0,
            1.0,
            1.0,
        ],
        "action_counts": {
            ACTION_NAMES[index]: int(action_counts[index])
            for index in range(NUM_ACTIONS)
        },
        "data_counts": data_counts,
        "learning_rate": float(
            optimizer.param_groups[0]["lr"]
        ),
    }

    torch.save(checkpoint, path)


def save_json(path, content):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            content,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main():
    set_random_seed(RANDOM_SEED)
    torch.set_num_threads(4)

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    expert_metadata = load_manifest(
        EXPERT_MANIFEST
    )

    dagger_v1_metadata = load_manifest(
        DAGGER_V1_MANIFEST
    )

    dagger_v2_metadata = load_manifest(
        DAGGER_V2_MANIFEST
    )

    validation_metadata = load_manifest(
        VAL_MANIFEST
    )

    training_metadata = (
        expert_metadata
        + dagger_v1_metadata
        + dagger_v2_metadata
    )

    data_counts = {
        "expert_trajectories": len(expert_metadata),
        "dagger_v1_trajectories": len(dagger_v1_metadata),
        "dagger_v2_trajectories": len(dagger_v2_metadata),
        "training_trajectories": len(training_metadata),
        "validation_trajectories": len(validation_metadata),
        "expert_steps": count_steps(expert_metadata),
        "dagger_v1_steps": count_steps(dagger_v1_metadata),
        "dagger_v2_steps": count_steps(dagger_v2_metadata),
        "training_steps": count_steps(training_metadata),
    }

    if len(training_metadata) != 225:
        raise RuntimeError(
            "聚合训练轨迹不是预期的 225 条，"
            f"实际为 {len(training_metadata)} 条。"
        )

    if len(validation_metadata) != 6:
        raise RuntimeError(
            "验证轨迹不是预期的 6 条，"
            f"实际为 {len(validation_metadata)} 条。"
        )

    action_counts = count_actions(
        training_metadata
    )

    model, vocab, parent = load_parent_model()

    with open(
        VOCAB_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            vocab,
            file,
            ensure_ascii=False,
            indent=2,
        )

    class_weights = torch.tensor(
        [STOP_WEIGHT, 1.0, 1.0, 1.0],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-5,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("=" * 70)
    print("开始 STOP-aware VLN 训练")
    print("原始专家轨迹：", len(expert_metadata))
    print("第一轮 DAgger：", len(dagger_v1_metadata))
    print("第二轮 DAgger：", len(dagger_v2_metadata))
    print("合并训练轨迹：", len(training_metadata))
    print("验证轨迹：", len(validation_metadata))
    print("合并监督状态数：", data_counts["training_steps"])
    print("训练动作数量：")

    for index in range(NUM_ACTIONS):
        print(
            f"  {index} {ACTION_NAMES[index]}: "
            f"{int(action_counts[index])}"
        )

    print("动作权重：", class_weights.cpu().numpy())
    print("父模型：", PARENT_CHECKPOINT)
    print("父模型 Epoch：", parent["epoch"])
    print(
        "父模型 Val accuracy：",
        parent.get("validation_accuracy", "unknown"),
    )
    print("词表大小：", len(vocab))
    print("模型参数量：", parameter_count)
    print("训练设备：", DEVICE)
    print("初始学习率：", LEARNING_RATE)
    print("训练 Epoch：", NUM_EPOCHS)
    print("每个 Epoch 都会单独保存")

    initial_validation = run_epoch(
        model=model,
        metadata_list=validation_metadata,
        vocab=vocab,
        criterion=criterion,
        optimizer=None,
    )

    print("-" * 70)
    print(
        "训练前 Val loss：",
        f"{initial_validation['loss']:.4f}",
    )
    print(
        "训练前 Val accuracy：",
        f"{initial_validation['accuracy']:.4f}",
    )

    experiment_info = {
        "training_method": "stop_aware_dagger",
        "parent_checkpoint": PARENT_CHECKPOINT,
        "num_epochs": NUM_EPOCHS,
        "initial_learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "stop_weight": STOP_WEIGHT,
        "class_weights": [
            STOP_WEIGHT,
            1.0,
            1.0,
            1.0,
        ],
        "random_seed": RANDOM_SEED,
        "data_counts": data_counts,
        "action_counts": {
            ACTION_NAMES[index]: int(action_counts[index])
            for index in range(NUM_ACTIONS)
        },
        "initial_validation": serializable_result(
            initial_validation
        ),
        "validation_used_for_gradients": False,
        "checkpoints_saved_every_epoch": True,
    }

    save_json(
        EXPERIMENT_INFO_PATH,
        experiment_info,
    )

    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        print("=" * 70)
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        train_result = run_epoch(
            model=model,
            metadata_list=training_metadata,
            vocab=vocab,
            criterion=criterion,
            optimizer=optimizer,
        )

        validation_result = run_epoch(
            model=model,
            metadata_list=validation_metadata,
            vocab=vocab,
            criterion=criterion,
            optimizer=None,
        )

        scheduler.step(
            validation_result["loss"]
        )

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Train loss：{train_result['loss']:.4f}"
        )
        print(
            f"Train accuracy：{train_result['accuracy']:.4f}"
        )
        print(
            f"Val loss：{validation_result['loss']:.4f}"
        )
        print(
            f"Val accuracy：{validation_result['accuracy']:.4f}"
        )
        print(f"Learning rate：{current_lr:.7f}")
        print("验证集各动作准确率：")

        for action_name, accuracy in (
            validation_result["per_class_accuracy"].items()
        ):
            print(
                f"  {action_name}: {accuracy:.4f}"
            )

        checkpoint_path = epoch_checkpoint_path(
            epoch
        )

        save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            vocab=vocab,
            epoch=epoch,
            train_result=train_result,
            validation_result=validation_result,
            action_counts=action_counts,
            data_counts=data_counts,
        )

        epoch_record = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "checkpoint_path": checkpoint_path,
            "train": serializable_result(train_result),
            "validation": serializable_result(
                validation_result
            ),
        }

        history.append(epoch_record)

        save_json(
            HISTORY_PATH,
            history,
        )

        print("已保存 Epoch checkpoint：", checkpoint_path)

    print("=" * 70)
    print("STOP-aware 训练完成")
    print("Checkpoint 目录：", CHECKPOINT_DIR)
    print("共保存 Epoch checkpoint：", len(history))
    print("训练历史：", HISTORY_PATH)
    print("下一步对全部 Epoch 做闭环评估后再选模型。")


if __name__ == "__main__":
    main()