import json
import os
import random

import numpy as np
import torch
import torch.nn as nn

from train_vln import (
    ACTION_NAMES,
    NUM_ACTIONS,
    START_ACTION,
    build_vocabulary,
    load_episode,
    run_epoch,
    set_random_seed,
)


WORK_ROOT = os.path.expanduser("~/habitat-work")

EXPERT_MANIFEST = os.path.join(
    WORK_ROOT,
    "vln_expert_data",
    "train",
    "manifest.json",
)

DAGGER_MANIFEST = os.path.join(
    WORK_ROOT,
    "vln_dagger_data",
    "train",
    "manifest.json",
)

VAL_MANIFEST = os.path.join(
    WORK_ROOT,
    "vln_expert_data",
    "val_seen",
    "manifest.json",
)

CHECKPOINT_DIR = os.path.join(
    WORK_ROOT,
    "checkpoints_dagger",
)

BEST_LOSS_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "vln_dagger_best_loss.pth",
)

BEST_ACCURACY_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "vln_dagger_best_accuracy.pth",
)

LAST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "vln_dagger_last.pth",
)

VOCAB_PATH = os.path.join(
    CHECKPOINT_DIR,
    "vocab.json",
)

DEVICE = torch.device("cpu")

NUM_EPOCHS = 10
LEARNING_RATE = 7e-4


def load_manifest(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"没有找到manifest：{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    valid_metadata = []

    for item in metadata:
        if os.path.exists(
            item["npz_path"]
        ):
            valid_metadata.append(item)

    return valid_metadata


class SpatialVisualEncoder(nn.Module):

    def __init__(
        self,
        output_size=192,
        dropout=0.2,
    ):
        super().__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(
                4,
                16,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.GroupNorm(4, 16),
            nn.ReLU(),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, 32),
            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            # 保留6×6空间布局，不再压缩为1×1
            nn.AdaptiveAvgPool2d(
                (6, 6)
            ),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                64 * 6 * 6,
                output_size,
            ),

            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, observations):
        feature_map = self.convolution(
            observations
        )

        return self.projection(
            feature_map
        )


class ImprovedVLNPolicy(nn.Module):

    def __init__(
        self,
        vocabulary_size,
        embedding_size=64,
        language_size=128,
        visual_size=192,
        action_embedding_size=24,
        hidden_size=192,
        dropout=0.2,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.word_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=0,
        )

        self.language_gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=language_size,
            batch_first=True,
        )

        self.visual_encoder = (
            SpatialVisualEncoder(
                output_size=visual_size,
                dropout=dropout,
            )
        )

        self.action_embedding = nn.Embedding(
            num_embeddings=NUM_ACTIONS + 1,
            embedding_dim=action_embedding_size,
        )

        recurrent_input_size = (
            visual_size
            + language_size
            + action_embedding_size
        )

        self.navigation_gru = nn.GRUCell(
            input_size=recurrent_input_size,
            hidden_size=hidden_size,
        )

        self.recurrent_dropout = nn.Dropout(
            dropout
        )

        self.action_head = nn.Linear(
            hidden_size,
            NUM_ACTIONS,
        )

    def encode_language(
        self,
        instruction_tokens,
    ):
        embedded = self.word_embedding(
            instruction_tokens.unsqueeze(0)
        )

        _, hidden = self.language_gru(
            embedded
        )

        return hidden[-1]

    def initial_hidden(
        self,
        device,
    ):
        return torch.zeros(
            1,
            self.hidden_size,
            device=device,
        )

    def navigation_step(
        self,
        visual_feature,
        language_context,
        previous_action,
        hidden,
    ):
        previous_action_tensor = torch.tensor(
            [previous_action],
            dtype=torch.long,
            device=visual_feature.device,
        )

        action_feature = self.action_embedding(
            previous_action_tensor
        )

        recurrent_input = torch.cat(
            [
                visual_feature,
                language_context,
                action_feature,
            ],
            dim=1,
        )

        hidden = self.navigation_gru(
            recurrent_input,
            hidden,
        )

        output_hidden = (
            self.recurrent_dropout(
                hidden
            )
        )

        logits = self.action_head(
            output_hidden
        )

        return logits, hidden

    def forward_teacher_forcing(
        self,
        observations,
        instruction_tokens,
        expert_actions,
    ):
        visual_features = self.visual_encoder(
            observations
        )

        language_context = self.encode_language(
            instruction_tokens
        )

        hidden = self.initial_hidden(
            observations.device
        )

        previous_action = START_ACTION
        logits_sequence = []

        for step in range(
            len(expert_actions)
        ):
            visual_feature = (
                visual_features[
                    step
                ].unsqueeze(0)
            )

            logits, hidden = (
                self.navigation_step(
                    visual_feature=visual_feature,
                    language_context=language_context,
                    previous_action=previous_action,
                    hidden=hidden,
                )
            )

            logits_sequence.append(
                logits
            )

            previous_action = int(
                expert_actions[
                    step
                ].item()
            )

        return torch.cat(
            logits_sequence,
            dim=0,
        )


def save_checkpoint(
    path,
    model,
    optimizer,
    vocab,
    epoch,
    validation_result,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "vocab": vocab,
        "architecture": (
            "ImprovedVLNPolicy"
        ),
        "model_config": {
            "vocabulary_size": len(vocab),
            "embedding_size": 64,
            "language_size": 128,
            "visual_size": 192,
            "action_embedding_size": 24,
            "hidden_size": 192,
            "dropout": 0.2,
        },
        "validation_loss": (
            validation_result["loss"]
        ),
        "validation_accuracy": (
            validation_result[
                "accuracy"
            ]
        ),
        "action_names": ACTION_NAMES,
    }

    torch.save(
        checkpoint,
        path,
    )


def main():
    set_random_seed(42)
    torch.set_num_threads(4)

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    expert_metadata = load_manifest(
        EXPERT_MANIFEST
    )

    dagger_metadata = load_manifest(
        DAGGER_MANIFEST
    )

    validation_metadata = load_manifest(
        VAL_MANIFEST
    )

    training_metadata = (
        expert_metadata
        + dagger_metadata
    )

    print("=" * 70)
    print(
        "原始专家轨迹：",
        len(expert_metadata),
    )
    print(
        "DAgger纠错轨迹：",
        len(dagger_metadata),
    )
    print(
        "合并训练轨迹：",
        len(training_metadata),
    )
    print(
        "验证轨迹：",
        len(validation_metadata),
    )

    vocab = build_vocabulary(
        training_metadata
    )

    print(
        "词表大小：",
        len(vocab),
    )

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

    model = ImprovedVLNPolicy(
        vocabulary_size=len(vocab),
    ).to(DEVICE)

    # 使用更温和的动作类别权重
    class_weights = torch.tensor(
        [3.0, 1.0, 1.0, 1.0],
        dtype=torch.float32,
        device=DEVICE,
    )

    print(
        "动作权重：",
        class_weights.cpu().numpy(),
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        )
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "模型参数量：",
        parameter_count,
    )
    print(
        "训练设备：",
        DEVICE,
    )

    best_validation_loss = float(
        "inf"
    )

    best_validation_accuracy = 0.0

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):
        print("=" * 70)
        print(
            f"Epoch {epoch}/"
            f"{NUM_EPOCHS}"
        )

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

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        print(
            f"Train loss："
            f"{train_result['loss']:.4f}"
        )
        print(
            f"Train accuracy："
            f"{train_result['accuracy']:.4f}"
        )
        print(
            f"Val loss："
            f"{validation_result['loss']:.4f}"
        )
        print(
            f"Val accuracy："
            f"{validation_result['accuracy']:.4f}"
        )
        print(
            f"Learning rate："
            f"{current_lr:.7f}"
        )

        print(
            "验证集各动作准确率："
        )

        for action_name, accuracy in (
            validation_result[
                "per_class_accuracy"
            ].items()
        ):
            print(
                f"  {action_name}: "
                f"{accuracy:.4f}"
            )

        save_checkpoint(
            path=LAST_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            vocab=vocab,
            epoch=epoch,
            validation_result=validation_result,
        )

        if (
            validation_result["loss"]
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_result["loss"]
            )

            save_checkpoint(
                path=BEST_LOSS_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                vocab=vocab,
                epoch=epoch,
                validation_result=validation_result,
            )

            print(
                "保存最佳损失模型：",
                BEST_LOSS_CHECKPOINT,
            )

        if (
            validation_result["accuracy"]
            > best_validation_accuracy
        ):
            best_validation_accuracy = (
                validation_result[
                    "accuracy"
                ]
            )

            save_checkpoint(
                path=BEST_ACCURACY_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                vocab=vocab,
                epoch=epoch,
                validation_result=validation_result,
            )

            print(
                "保存最佳准确率模型：",
                BEST_ACCURACY_CHECKPOINT,
            )

    print("=" * 70)
    print("DAgger改进模型训练完成")
    print(
        "最佳损失模型：",
        BEST_LOSS_CHECKPOINT,
    )
    print(
        "最佳准确率模型：",
        BEST_ACCURACY_CHECKPOINT,
    )
    print(
        "最后模型：",
        LAST_CHECKPOINT,
    )


if __name__ == "__main__":
    main()
