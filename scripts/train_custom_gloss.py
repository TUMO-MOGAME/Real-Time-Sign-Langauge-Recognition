import argparse
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.config import CHANNELS, SEQ_LEN  # noqa: E402
from scripts.gloss.backbone import get_model  # noqa: E402
from scripts.gloss.gloss_utils import Preprocess  # noqa: E402
from strip_out_4th_visibility import convert_sequence_file  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom gloss model from MP_Data.")
    parser.add_argument("--data-root", default="MP_Data")
    parser.add_argument("--model-weights", default="models/gloss/custom_gloss_model.weights.h5")
    parser.add_argument("--label-map", default="data/custom_sign_to_prediction_index_map.json")
    parser.add_argument("--report-dir", default="artifacts/custom_gloss")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-sequences-per-sign", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_sequences(data_root, max_sequences_per_sign=None):
    sign_to_sequences = {}
    skipped = []
    data_root = Path(data_root)
    for week_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for sign_dir in sorted(p for p in week_dir.iterdir() if p.is_dir()):
            sign_name = sign_dir.name.lower()
            sequences = sorted(p for p in sign_dir.iterdir() if p.is_dir())
            if max_sequences_per_sign is not None:
                sequences = sequences[:max_sequences_per_sign]
            sign_to_sequences.setdefault(sign_name, [])
            for sequence_dir in sequences:
                expected = [sequence_dir / f"{i}.npy" for i in range(SEQ_LEN)]
                missing = [str(path.name) for path in expected if not path.exists()]
                if missing:
                    skipped.append({"sequence": str(sequence_dir), "missing_frames": missing[:5]})
                    continue
                sign_to_sequences[sign_name].append(str(sequence_dir))
    sign_to_sequences = {k: v for k, v in sign_to_sequences.items() if v}
    return sign_to_sequences, skipped


def build_label_map(sign_to_sequences):
    return {sign_name: idx for idx, sign_name in enumerate(sorted(sign_to_sequences))}


def flatten_examples(sign_to_sequences, label_map):
    paths, labels = [], []
    for sign_name in sorted(sign_to_sequences):
        for sequence_path in sign_to_sequences[sign_name]:
            paths.append(sequence_path)
            labels.append(label_map[sign_name])
    return paths, np.asarray(labels, dtype=np.int32)


def stratified_split(indices, labels, test_size, random_state):
    rng = np.random.default_rng(random_state)
    train_indices, test_indices = [], []
    labels = np.asarray(labels)
    for label in sorted(np.unique(labels)):
        label_indices = indices[labels[indices] == label].copy()
        rng.shuffle(label_indices)
        split_count = int(round(len(label_indices) * test_size))
        split_count = max(1, split_count)
        split_count = min(len(label_indices) - 1, split_count)
        test_indices.extend(label_indices[:split_count].tolist())
        train_indices.extend(label_indices[split_count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return np.asarray(train_indices, dtype=np.int32), np.asarray(test_indices, dtype=np.int32)


def split_examples(paths, labels, val_size, test_size, random_state):
    indices = np.arange(len(paths))
    train_val_idx, test_idx = stratified_split(indices, labels, test_size, random_state)
    adjusted_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = stratified_split(
        train_val_idx,
        labels,
        adjusted_val_size,
        random_state + 1,
    )
    return train_idx, val_idx, test_idx


def sequence_generator(paths, labels):
    for sequence_path, label in zip(paths, labels):
        yield convert_sequence_file(sequence_path).astype(np.float32), np.int32(label)


def build_dataset(paths, labels, batch_size, preprocess_layer, shuffle=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: sequence_generator(paths, labels),
        output_signature=(
            tf.TensorSpec(shape=(SEQ_LEN, 543, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda sequence, label: (tf.squeeze(preprocess_layer(sequence), axis=0), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def save_label_map(label_map, label_map_path):
    os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
    with open(label_map_path, "w", encoding="utf-8") as file:
        json.dump(label_map, file, indent=2)


def save_history_plot(history, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history_dict = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_dict.get("loss", []), label="train")
    axes[0].plot(history_dict.get("val_loss", []), label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history_dict.get("sparse_categorical_accuracy", []), label="train")
    axes[1].plot(history_dict.get("val_sparse_categorical_accuracy", []), label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def compute_confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, predicted_label in zip(y_true, y_pred):
        matrix[int(true_label), int(predicted_label)] += 1
    return matrix


def build_classification_report(y_true, y_pred, class_names):
    matrix = compute_confusion_matrix(y_true, y_pred, len(class_names))
    total = matrix.sum()
    accuracy = np.trace(matrix) / total if total else 0.0
    header = f"{'class':<20}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}"
    lines = [header, ""]
    precisions, recalls, f1_scores, supports = [], [], [], []

    for index, class_name in enumerate(class_names):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        support = matrix[index, :].sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_score = (
            (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        )
        lines.append(
            f"{class_name:<20}{precision:>10.4f}{recall:>10.4f}{f1_score:>10.4f}{int(support):>10}"
        )
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        supports.append(support)

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    weighted_f1 = (
        float(np.average(f1_scores, weights=supports)) if np.sum(supports) else 0.0
    )
    lines.extend(
        [
            "",
            f"{'accuracy':<20}{accuracy:>30.4f}{int(np.sum(supports)):>10}",
            f"{'macro avg':<20}{macro_precision:>10.4f}{macro_recall:>10.4f}{macro_f1:>10.4f}{int(np.sum(supports)):>10}",
            f"{'weighted avg':<20}{'':>10}{'':>10}{weighted_f1:>10.4f}{int(np.sum(supports)):>10}",
        ]
    )
    return "\n".join(lines), accuracy


def save_confusion_matrix(y_true, y_pred, class_names, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = compute_confusion_matrix(y_true, y_pred, len(class_names))
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Custom Gloss Confusion Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def ensure_parent_dir(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def main():
    args = parse_args()
    tf.random.set_seed(args.random_state)
    np.random.seed(args.random_state)
    random.seed(args.random_state)

    sign_to_sequences, skipped = discover_sequences(
        args.data_root, max_sequences_per_sign=args.max_sequences_per_sign
    )
    if not sign_to_sequences:
        raise ValueError(f"No valid sequences found under {args.data_root}")

    label_map = build_label_map(sign_to_sequences)
    paths, labels = flatten_examples(sign_to_sequences, label_map)
    train_idx, val_idx, test_idx = split_examples(
        paths, labels, args.val_size, args.test_size, args.random_state
    )

    train_paths = [paths[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    test_paths = [paths[i] for i in test_idx]
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]

    if not train_paths or not val_paths or not test_paths:
        raise ValueError(
            "One of the splits is empty. Increase available sequences or reduce val/test split sizes."
        )

    os.makedirs(args.report_dir, exist_ok=True)
    save_label_map(label_map, args.label_map)
    with open(os.path.join(args.report_dir, "dataset_summary.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "num_signs": len(label_map),
                "num_sequences": len(paths),
                "train_sequences": len(train_paths),
                "val_sequences": len(val_paths),
                "test_sequences": len(test_paths),
                "skipped_sequences": skipped,
            },
            file,
            indent=2,
        )

    print(f"Found {len(label_map)} signs and {len(paths)} valid sequences.")
    print(f"Split sizes -> train: {len(train_paths)}, val: {len(val_paths)}, test: {len(test_paths)}")
    print(f"Label map saved to: {args.label_map}")

    preprocess_layer = Preprocess(max_len=SEQ_LEN)
    train_ds = build_dataset(train_paths, y_train, args.batch_size, preprocess_layer, shuffle=True)
    val_ds = build_dataset(val_paths, y_val, args.batch_size, preprocess_layer)
    test_ds = build_dataset(test_paths, y_test, args.batch_size, preprocess_layer)

    model = get_model(max_len=None, num_classes=len(label_map))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    first_batch = next(iter(train_ds.take(1)))
    print(f"Training feature batch shape: {first_batch[0].shape}")
    print(f"Expected channel count: {CHANNELS}")

    if args.dry_run:
        logits = model(first_batch[0], training=False)
        print(f"Dry run successful. Model output shape: {logits.shape}")
        return

    ensure_parent_dir(args.model_weights)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.model_weights,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    predictions = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
    report, test_accuracy = build_classification_report(y_test, y_pred, class_names)

    print(f"Test accuracy: {test_accuracy:.2%}")
    print("Classification report:")
    print(report)

    save_history_plot(history, os.path.join(args.report_dir, "training_history.png"))
    save_confusion_matrix(
        y_test,
        y_pred,
        class_names,
        os.path.join(args.report_dir, "confusion_matrix.png"),
    )
    with open(os.path.join(args.report_dir, "classification_report.txt"), "w", encoding="utf-8") as file:
        file.write(report)
    with open(os.path.join(args.report_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump({"test_accuracy": test_accuracy}, file, indent=2)


if __name__ == "__main__":
    main()