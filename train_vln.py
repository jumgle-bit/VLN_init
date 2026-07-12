import json
import os
import random
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DATA_ROOT = os.path.expanduser(
    "~/habitat-work/vln_expert_data"
)

CHECKPOINT_DIR = os.path.expanduser(
    "~/habitat-work/checkpoints"
)

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "vln_seq2seq_best.pth",
)

LAST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "vln_seq2seq_last.pth",
)

VOCAB_PATH = os.path.join(
    CHECKPOINT_DIR,
    "vocab.json",
)

DEVICE = torch.device("cpu")

NUM_ACTIONS = 4
START_ACTION = 4

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

MAX_INSTRUCTION_LENGTH = 80
NUM_EPOCHS = 8
LEARNING_RATE = 1e-3

ACTION_NAMES = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tokenize(text):
    return re.findall(
        r"[a-z0-9']+",
        text.lower(),
    )


def build_vocabulary(metadata_list):
    counter = Counter()

    for metadata in metadata_list:
        counter.update(
            tokenize(
                metadata["prompt"]
            )
        )

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    sorted_words = sorted(
        counter.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    for word, count in sorted_words:
        if count >= 1:
            vocab[word] = len(vocab)

    return vocab


def encode_instruction(
    text,
    vocab,
):
    words = tokenize(text)

    if len(words) == 0:
        words = [UNK_TOKEN]

    words = words[
        :MAX_INSTRUCTION_LENGTH
    ]

    token_ids = [
        vocab.get(
            word,
            vocab[UNK_TOKEN],
        )
        for word in words
    ]

    return torch.tensor(
        token_ids,
        dtype=torch.long,
        device=DEVICE,
    )


def load_manifest(split):
    manifest_path = os.path.join(
        DATA_ROOT,
        split,
        "manifest.json",
    )

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"没有找到：{manifest_path}"
        )

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as file:
        metadata_list = json.load(file)

    valid_metadata = []

    for metadata in metadata_list:
        npz_path = metadata["npz_path"]

        if os.path.exists(npz_path):
            valid_metadata.append(
                metadata
            )

    return valid_metadata


def load_episode(
    metadata,
    vocab,
):
    npz_path = metadata["npz_path"]

    with np.load(npz_path) as data:
        rgb = data["rgb"].copy()
        depth = data["depth"].copy()
        actions = data["actions"].copy()

    rgb_tensor = torch.from_numpy(
        rgb
    ).permute(
        0,
        3,
        1,
        2,
    ).float()

    rgb_tensor = rgb_tensor / 255.0

    depth_tensor = torch.from_numpy(
        depth
    ).permute(
        0,
        3,
        1,
        2,
    ).float()

    observation_tensor = torch.cat(
        [
            rgb_tensor,
            depth_tensor,
        ],
        dim=1,
    ).to(DEVICE)

    action_tensor = torch.from_numpy(
        actions
    ).long().to(DEVICE)

    instruction_tensor = encode_instruction(
        metadata["prompt"],
        vocab,
    )

    return (
        observation_tensor,
        instruction_tensor,
        action_tensor,
    )


class VisualEncoder(nn.Module):

    def __init__(
        self,
        output_size=128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=16,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                output_size=(1, 1)
            ),

            nn.Flatten(),

            nn.Linear(
                in_features=64,
                out_features=output_size,
            ),

            nn.ReLU(),
        )

    def forward(self, observations):
        return self.network(
            observations
        )


class VLNPolicy(nn.Module):

    def __init__(
        self,
        vocabulary_size,
        embedding_size=64,
        language_size=128,
        visual_size=128,
        action_embedding_size=16,
        hidden_size=128,
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

        self.visual_encoder = VisualEncoder(
            output_size=visual_size
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

        self.action_head = nn.Linear(
            in_features=hidden_size,
            out_features=NUM_ACTIONS,
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

        logits = self.action_head(
            hidden
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
                expert_actions[step].item()
            )

        return torch.cat(
            logits_sequence,
            dim=0,
        )


def calculate_class_weights(
    metadata_list,
):
    action_counts = np.zeros(
        NUM_ACTIONS,
        dtype=np.int64,
    )

    for metadata in metadata_list:
        with np.load(
            metadata["npz_path"]
        ) as data:
            actions = data["actions"]

        for action in range(
            NUM_ACTIONS
        ):
            action_counts[action] += int(
                np.sum(actions == action)
            )

    safe_counts = np.maximum(
        action_counts,
        1,
    )

    total = safe_counts.sum()

    weights = (
        total
        / (
            NUM_ACTIONS
            * safe_counts
        )
    )

    weights = np.clip(
        weights,
        0.5,
        8.0,
    )

    print(
        "训练动作数量："
    )

    for action, name in ACTION_NAMES.items():
        print(
            f"  {action} {name}: "
            f"{action_counts[action]}"
        )

    print(
        "损失函数类别权重：",
        np.round(weights, 3),
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )


def run_epoch(
    model,
    metadata_list,
    vocab,
    criterion,
    optimizer=None,
):
    training = optimizer is not None

    if training:
        model.train()
        random.shuffle(
            metadata_list
        )
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_steps = 0

    class_correct = np.zeros(
        NUM_ACTIONS,
        dtype=np.int64,
    )

    class_total = np.zeros(
        NUM_ACTIONS,
        dtype=np.int64,
    )

    for index, metadata in enumerate(
        metadata_list,
        start=1,
    ):
        (
            observations,
            instruction,
            expert_actions,
        ) = load_episode(
            metadata,
            vocab,
        )

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(
            training
        ):
            logits = (
                model.forward_teacher_forcing(
                    observations=observations,
                    instruction_tokens=instruction,
                    expert_actions=expert_actions,
                )
            )

            loss = criterion(
                logits,
                expert_actions,
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        num_steps = len(
            expert_actions
        )

        total_loss += (
            float(loss.item())
            * num_steps
        )

        total_correct += int(
            (
                predictions
                == expert_actions
            ).sum().item()
        )

        total_steps += num_steps

        predictions_cpu = (
            predictions.detach().cpu().numpy()
        )

        actions_cpu = (
            expert_actions.detach().cpu().numpy()
        )

        for action in range(
            NUM_ACTIONS
        ):
            mask = (
                actions_cpu == action
            )

            class_total[action] += int(
                mask.sum()
            )

            class_correct[action] += int(
                (
                    predictions_cpu[mask]
                    == action
                ).sum()
            )

        if (
            training
            and index % 10 == 0
        ):
            print(
                f"  已训练 "
                f"{index}/"
                f"{len(metadata_list)} "
                f"条轨迹"
            )

        del observations
        del instruction
        del expert_actions
        del logits
        del loss

    average_loss = (
        total_loss
        / max(total_steps, 1)
    )

    accuracy = (
        total_correct
        / max(total_steps, 1)
    )

    per_class_accuracy = {}

    for action, name in ACTION_NAMES.items():
        per_class_accuracy[name] = (
            class_correct[action]
            / max(
                class_total[action],
                1,
            )
        )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "per_class_accuracy": (
            per_class_accuracy
        ),
    }


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
        "model_config": {
            "vocabulary_size": len(
                vocab
            ),
            "embedding_size": 64,
            "language_size": 128,
            "visual_size": 128,
            "action_embedding_size": 16,
            "hidden_size": 128,
        },
        "validation_loss": (
            validation_result["loss"]
        ),
        "validation_accuracy": (
            validation_result["accuracy"]
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

    train_metadata = load_manifest(
        "train"
    )

    validation_metadata = load_manifest(
        "val_seen"
    )

    print("=" * 70)

    print(
        "训练轨迹数量：",
        len(train_metadata),
    )

    print(
        "验证轨迹数量：",
        len(validation_metadata),
    )

    vocab = build_vocabulary(
        train_metadata
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

    class_weights = (
        calculate_class_weights(
            train_metadata
        )
    )

    model = VLNPolicy(
        vocabulary_size=len(vocab),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
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

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):
        print("=" * 70)

        print(
            f"Epoch "
            f"{epoch}/"
            f"{NUM_EPOCHS}"
        )

        train_result = run_epoch(
            model=model,
            metadata_list=train_metadata,
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

        print(
            f"Train loss："
            f"{train_result['loss']:.4f}"
        )

        print(
            f"Train action accuracy："
            f"{train_result['accuracy']:.4f}"
        )

        print(
            f"Val loss："
            f"{validation_result['loss']:.4f}"
        )

        print(
            f"Val action accuracy："
            f"{validation_result['accuracy']:.4f}"
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
                path=BEST_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                vocab=vocab,
                epoch=epoch,
                validation_result=validation_result,
            )

            print(
                "已保存新的最佳模型：",
                BEST_CHECKPOINT,
            )

    print("=" * 70)
    print(
        "模型训练完成"
    )
    print(
        "最佳模型：",
        BEST_CHECKPOINT,
    )
    print(
        "最后模型：",
        LAST_CHECKPOINT,
    )


if __name__ == "__main__":
    main()
