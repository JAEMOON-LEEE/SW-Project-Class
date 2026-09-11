"""
MusicLens - Day 5: CNN 구조 설계 + 배치 데이터로더
=========================================================
- generate_melspec.py로 만든 outputs/melspec_index.csv를 읽어서
  tf.data.Dataset 파이프라인을 만든다.
- train/test 곡 목록은 outputs/features.csv를 기준으로,
  train_baseline.py의 song_level_split()과 "완전히 동일한 방식"
  (random_state=42, test_size=0.2, stratify=genre)으로 재현한다.
  -> Day 7에서 SVM/RF(72% 등)와 CNN 정확도를 같은 test set 기준으로
     공정하게 비교하기 위함. 이 부분이 어긋나면 비교 자체가 무의미해진다.
- CNN 학습(fit)은 Day 6에서 진행. 여기서는 구조 설계 + 데이터로더까지.

실행 방법 (프로젝트 루트에서, 동작 확인용):
    python scripts/cnn_pipeline.py
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 설정 (generate_melspec.py / train_baseline.py와 통일)
# ---------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
FEATURES_CSV = os.path.join(OUTPUT_DIR, "features.csv")       # Day4와 같은 곡 목록/split 기준
INDEX_CSV = os.path.join(OUTPUT_DIR, "melspec_index.csv")     # Day5 멜스펙 인덱스

RANDOM_STATE = 42   # train_baseline.py와 동일 -> 동일 split 재현
TEST_SIZE = 0.2      # train_baseline.py와 동일

N_MELS = 128          # generate_melspec.py와 동일
FRAMES = 1249         # generate_melspec.py의 FRAMES와 동일 (29초, hop_length=512 기준)
INPUT_SHAPE = (N_MELS, FRAMES, 1)

BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# 1) Day4와 동일한 train/test 곡(track_id) split 재현
# ---------------------------------------------------------------------------
def get_train_test_track_ids():
    """
    features.csv의 track_id/genre로 Day4의 song_level_split()과 동일한
    방식(같은 random_state, test_size, stratify)으로 train/test 곡을 나눈다.
    -> SVM/RF와 CNN이 정확히 같은 test set으로 평가되도록 보장.
    """
    df = pd.read_csv(FEATURES_CSV, usecols=["track_id", "genre"])
    track_info = df[["track_id", "genre"]].drop_duplicates()

    train_tracks, test_tracks = train_test_split(
        track_info,
        test_size=TEST_SIZE,
        stratify=track_info["genre"],
        random_state=RANDOM_STATE,
    )
    train_ids = set(train_tracks["track_id"])
    test_ids = set(test_tracks["track_id"])

    overlap = train_ids & test_ids
    assert len(overlap) == 0, "데이터 누수 발생! train/test 곡이 겹칩니다."

    print(f"[split 재현] train 곡: {len(train_ids)}, test 곡: {len(test_ids)}")
    return train_ids, test_ids


# ---------------------------------------------------------------------------
# 2) melspec_index.csv를 train/test로 나누고, 라벨 인코딩
# ---------------------------------------------------------------------------
def load_melspec_splits():
    if not os.path.isfile(INDEX_CSV):
        raise FileNotFoundError(
            f"{INDEX_CSV}가 없습니다. 먼저 scripts/generate_melspec.py를 실행해주세요."
        )

    index_df = pd.read_csv(INDEX_CSV)
    train_ids, test_ids = get_train_test_track_ids()

    train_df = index_df[index_df["track_id"].isin(train_ids)].reset_index(drop=True)
    test_df = index_df[index_df["track_id"].isin(test_ids)].reset_index(drop=True)

    # melspec 생성이 아직 덜 끝났다면(예: --limit 테스트 모드) 여기서 바로 알아챌 수 있음
    missing = len(train_ids) + len(test_ids) - len(train_df) - len(test_df)
    if missing > 0:
        print(
            f"[경고] features.csv 기준 {missing}곡이 melspec_index.csv에 아직 없습니다. "
            f"generate_melspec.py를 전체 실행했는지 확인해주세요."
        )

    # labels는 train_baseline.py와 동일하게 정렬된 장르 목록 사용 -> 인덱스 순서 일치
    labels = sorted(index_df["genre"].unique())
    label_to_idx = {g: i for i, g in enumerate(labels)}

    print(f"[melspec split] train {len(train_df)}곡 / test {len(test_df)}곡")
    print(f"[라벨] {labels}")

    return train_df, test_df, labels, label_to_idx


# ---------------------------------------------------------------------------
# 3) 배치 데이터로더 (tf.data.Dataset) - npy를 그때그때 읽어 메모리 절약
# ---------------------------------------------------------------------------
def _make_generator(df: pd.DataFrame, label_to_idx: dict):
    def gen():
        for _, row in df.iterrows():
            mel_db = np.load(row["npy_path"])  # (N_MELS, FRAMES)
            # -80~0dB 범위를 0~1로 정규화 (CNN 학습 안정성을 위해)
            mel_norm = (mel_db + 80.0) / 80.0
            mel_norm = np.clip(mel_norm, 0.0, 1.0)
            mel_norm = mel_norm[..., np.newaxis].astype(np.float32)  # (N_MELS, FRAMES, 1)
            label = label_to_idx[row["genre"]]
            yield mel_norm, label

    return gen


def make_dataset(df: pd.DataFrame, label_to_idx: dict, batch_size=BATCH_SIZE, shuffle=True):
    output_signature = (
        tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(
        _make_generator(df, label_to_idx), output_signature=output_signature
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_STATE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# 4) CNN 구조 설계
# ---------------------------------------------------------------------------
def build_cnn(input_shape=INPUT_SHAPE, num_classes=10):
    """
    Conv2D + BatchNorm + MaxPool 블록 4개 -> GlobalAveragePooling -> Dense -> Dropout -> Softmax
    - GlobalAveragePooling2D를 사용해 Flatten 대비 파라미터 수를 크게 줄임
      (멜스펙 프레임 수가 많아서(1249) Flatten 시 파라미터가 폭증하는 것을 방지)
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="musiclens_cnn")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 동작 확인용 (Day6에서 실제 model.fit()으로 이어짐)
# ---------------------------------------------------------------------------
def main():
    train_df, test_df, labels, label_to_idx = load_melspec_splits()

    train_ds = make_dataset(train_df, label_to_idx, shuffle=True)
    test_ds = make_dataset(test_df, label_to_idx, shuffle=False)

    model = build_cnn(input_shape=INPUT_SHAPE, num_classes=len(labels))
    model.summary()

    # 파이프라인이 실제로 배치를 만들어내는지 확인 (학습은 Day6에서)
    for batch_x, batch_y in train_ds.take(1):
        print(f"[확인] 배치 shape: X={batch_x.shape}, y={batch_y.shape}")

    print("\n[Day5 완료] CNN 구조 + 데이터로더 준비 완료. 학습은 Day6에서 진행합니다.")


if __name__ == "__main__":
    main()