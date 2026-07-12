"""
在聚合的三组数据上训练第二轮 DAgger VLN 策略。

训练数据：
    75 条原始专家轨迹
    75 条第一轮 DAgger 纠错轨迹
    75 条第二轮 DAgger 纠错轨迹

模型从第一轮 vln_dagger_best_accuracy.pth 继续训练。网络结构和词表
保持不变，使用新的优化器和较小学习率，所有输出保存到新的目录，
不会覆盖第一轮 checkpoint。
"""

import json
import os


import torch
import torch.nn as nn


from train_vln import (
    ACTION_NAMES,
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
    "checkpoints_dagger_v2",
)

BEST_LOSS_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "vln_dagger_v2_best_loss.pth",
)

BEST_ACCURACY_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "vln_dagger_v2_best_accuracy.pth",
)

LAST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "vln_dagger_v2_last.pth",
)

VOCAB_PATH = os.path.join(
    CHECKPOINT_DIR,
    "vocab.json",
)

TRAINING_INFO_PATH = os.path.join(
    CHECKPOINT_DIR,
    "training_info.json",
)

DEVICE = torch.device("cpu")

NUM_EPOCHS = 6
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 126


def count_steps(metadata_list):
    return sum(
        int(item.get("num_steps", 0))
        for item in metadata_list
    )


def load_parent_model():
    if not os.path.exists(PARENT_CHECKPOINT):
        raise FileNotFoundError(
            f"没有找到第一轮最佳模型：{PARENT_CHECKPOINT}"
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

    vocab = checkpoint["vocab"]

    print("父模型：", PARENT_CHECKPOINT)
    print("父模型 Epoch：", checkpoint["epoch"])
    print(
        "父模型 Val loss：",
        checkpoint.get("validation_loss", "unknown"),
    )
    print(
        "父模型 Val accuracy：",
        checkpoint.get("validation_accuracy", "unknown"),
    )

    return model, vocab, checkpoint


def save_checkpoint_v2(
    path,
    model,
    optimizer,
    vocab,
    epoch,
    train_result,
    validation_result,
    data_counts,
):
    checkpoint = {
        "epoch": epoch,
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
        "validation_loss": validation_result["loss"],
        "validation_accuracy": validation_result["accuracy"],
        "train_loss": train_result["loss"],
        "train_accuracy": train_result["accuracy"],
        "action_names": ACTION_NAMES,
        "dagger_iteration": 2,
        "parent_checkpoint": PARENT_CHECKPOINT,
        "data_counts": data_counts,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }

    torch.save(checkpoint, path)


def save_training_info(
    data_counts,
    initial_validation,
):
    information = {
        "dagger_iteration": 2,
        "parent_checkpoint": PARENT_CHECKPOINT,
        "expert_manifest": EXPERT_MANIFEST,
        "dagger_v1_manifest": DAGGER_V1_MANIFEST,
        "dagger_v2_manifest": DAGGER_V2_MANIFEST,
        "validation_manifest": VAL_MANIFEST,
        "data_counts": data_counts,
        "initial_validation": {
            "loss": float(initial_validation["loss"]),
            "accuracy": float(initial_validation["accuracy"]),
            "per_class_accuracy": {
                str(action_name): float(accuracy)
                for action_name, accuracy in initial_validation[
                    "per_class_accuracy"
                ].items()
            },
        },
        "num_epochs": NUM_EPOCHS,
        "initial_learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "class_weights": [3.0, 1.0, 1.0, 1.0],
        "random_seed": RANDOM_SEED,
    }

    with open(
        TRAINING_INFO_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            information,
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

    print("=" * 70)
    print("开始训练第二轮 DAgger VLN 模型")
    print("原始专家轨迹：", len(expert_metadata))
    print("第一轮 DAgger 轨迹：", len(dagger_v1_metadata))
    print("第二轮 DAgger 轨迹：", len(dagger_v2_metadata))
    print("合并训练轨迹：", len(training_metadata))
    print("验证轨迹：", len(validation_metadata))
    print("第二轮纠错状态数：", data_counts["dagger_v2_steps"])
    print("合并监督状态数：", data_counts["training_steps"])

    if len(expert_metadata) != 75:
        print("警告：专家轨迹数量不是预期的 75")

    if len(dagger_v1_metadata) != 75:
        print("警告：第一轮 DAgger 轨迹数量不是预期的 75")

    if len(dagger_v2_metadata) != 75:
        print("警告：第二轮 DAgger 轨迹数量不是预期的 75")

    if len(validation_metadata) != 6:
        print("警告：验证轨迹数量不是预期的 6")

    model, vocab, _ = load_parent_model()

    print("词表大小：", len(vocab))

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
        [3.0, 1.0, 1.0, 1.0],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # 使用新的优化器，不继承第一轮已经衰减或积累的 Adam 状态。
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

    print("动作权重：", class_weights.cpu().numpy())
    print("模型参数量：", parameter_count)
    print("训练设备：", DEVICE)
    print("初始学习率：", LEARNING_RATE)
    print("训练 Epoch：", NUM_EPOCHS)

    # 训练前先测一次，确认父模型权重和数据接口正常。
    initial_validation = run_epoch(
        model=model,
        metadata_list=validation_metadata,
        vocab=vocab,
        criterion=criterion,
        optimizer=None,
    )

    print("-" * 70)
    print(
        "继续训练前 Val loss：",
        f"{initial_validation['loss']:.4f}",
    )
    print(
        "继续训练前 Val accuracy：",
        f"{initial_validation['accuracy']:.4f}",
    )

    save_training_info(
        data_counts=data_counts,
        initial_validation=initial_validation,
    )

    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0

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

        current_lr = optimizer.param_groups[0]["lr"]

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

        save_checkpoint_v2(
            path=LAST_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            vocab=vocab,
            epoch=epoch,
            train_result=train_result,
            validation_result=validation_result,
            data_counts=data_counts,
        )

        if validation_result["loss"] < best_validation_loss:
            best_validation_loss = validation_result["loss"]

            save_checkpoint_v2(
                path=BEST_LOSS_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                vocab=vocab,
                epoch=epoch,
                train_result=train_result,
                validation_result=validation_result,
                data_counts=data_counts,
            )

            print(
                "保存第二轮最佳损失模型：",
                BEST_LOSS_CHECKPOINT,
            )

        if (
            validation_result["accuracy"]
            > best_validation_accuracy
        ):
            best_validation_accuracy = validation_result[
                "accuracy"
            ]

            save_checkpoint_v2(
                path=BEST_ACCURACY_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                vocab=vocab,
                epoch=epoch,
                train_result=train_result,
                validation_result=validation_result,
                data_counts=data_counts,
            )

            print(
                "保存第二轮最佳准确率模型：",
                BEST_ACCURACY_CHECKPOINT,
            )

    print("=" * 70)
    print("第二轮 DAgger 模型训练完成")
    print("最佳损失模型：", BEST_LOSS_CHECKPOINT)
    print("最佳准确率模型：", BEST_ACCURACY_CHECKPOINT)
    print("最后模型：", LAST_CHECKPOINT)
    print("下一步必须进行闭环评估，不能只看 Val accuracy。")


if __name__ == "__main__":
    main()