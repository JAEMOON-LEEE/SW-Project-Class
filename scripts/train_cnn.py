"""
train_cnn.py
Day 6 - CNN 실제 학습 및 1차 검증

역할:
  1) outputs/melspec_index.csv 를 읽어서 멜스펙트로그램(.npy) 목록 + 장르 라벨 로드
  2) 곡 단위(track_id)로 train/val/test 분리 (Day4 SVM/RF와 동일하게 random_state=42)
  3) tf.data 파이프라인으로 배치 단위 로딩 (한 번에 메모리에 다 안 올림)
  4) Conv2D + Pooling 반복 -> GlobalAveragePooling -> Dense -> Softmax(10장르) CNN 학습
  5) train/val loss, accuracy 곡선 저장 (과적합 점검용)
  6) test set 성능 평가 + 혼동행렬 저장
  7) 모델/메타데이터 저장 (Day7 최종 비교 및 Day8 웹앱 연결용)

실행 전 확인할 것 (환경: conda activate nlp):
  - outputs/melspec_index.csv 파일의 실제 컬럼명이 아래 CONFIG 섹션과 다르면 맞게 수정하세요.
  - outputs/melspecs/ 폴더 안에 .npy 파일들이 있어야 합니다 (Day5에서 생성됨).
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# =========================================================
# CONFIG - 실제 파일 구조에 맞게 여기만 확인/수정하면 됩니다
# =========================================================
INDEX_CSV = "outputs/melspec_index.csv"   # Day5에서 만든 인덱스 파일
MELSPEC_DIR = "outputs/melspecs"          # .npy 파일들이 있는 폴더

COL_TRACK_ID = "track_id"   # 인덱스 csv의 곡 고유 ID 컬럼명
COL_GENRE = "genre"         # 인덱스 csv의 장르 컬럼명
COL_PATH = "npy_path"       # 인덱스 csv의 .npy 파일 경로 컬럼명 (없으면 아래 fallback 사용)

RANDOM_STATE = 42           # Day4 SVM/RF와 동일 -> Day7에서 공정 비교용
TEST_SIZE = 0.20            # 곡 단위 20% -> 약 200곡
VAL_SIZE_OF_TRAIN = 0.15    # train 중 15%를 validation으로 다시 분리

BATCH_SIZE = 16             # Day5에서 배치 shape 검증한 값과 동일
EPOCHS = 50                 # EarlyStopping이 있어서 대부분 이 전에 멈춤
MELSPEC_SHAPE = (128, 1249)  # (n_mels, time_frames) - Day5 생성 결과와 동일해야 함

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 0. 한글 폰트 설정 (Day4와 동일하게 Malgun Gothic)
# =========================================================
try:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    print("[경고] Malgun Gothic 폰트를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")


# =========================================================
# 1. 인덱스 로드 + 파일 존재 확인
# =========================================================
def load_index():
    df = pd.read_csv(INDEX_CSV)
    print(f"[로드] {INDEX_CSV} -> {len(df)}개 행")

    if COL_PATH not in df.columns:
        # npy_path 컬럼이 따로 없는 경우: track_id.npy 형태로 추정해서 만든다
        print(f"[안내] '{COL_PATH}' 컬럼이 없어 '{MELSPEC_DIR}/<track_id>.npy' 경로로 추정합니다.")
        df[COL_PATH] = df[COL_TRACK_ID].astype(str).apply(
            lambda tid: os.path.join(MELSPEC_DIR, f"{tid}.npy")
        )

    # 실제로 파일이 존재하는 행만 남기기 (혹시 빠진 파일 있으면 스킵)
    exists_mask = df[COL_PATH].apply(os.path.exists)
    missing = (~exists_mask).sum()
    if missing > 0:
        print(f"[경고] .npy 파일을 찾을 수 없는 행 {missing}개는 제외합니다.")
    df = df[exists_mask].reset_index(drop=True)

    print(f"[확인] 최종 사용 가능한 곡 수: {len(df)}")
    return df


# =========================================================
# 2. 곡 단위 train / val / test 분리
#    -> 같은 곡의 조각이 train과 test에 동시에 들어가면 "데이터 누수"가 생김
#       (시험 답안지를 미리 본 것과 같은 효과라 성능이 부풀려짐)
# =========================================================
def split_by_track(df):
    groups = df[COL_TRACK_ID]

    # 1차: train+val(80%) vs test(20%)
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(splitter1.split(df, groups=groups))

    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # 2차: trainval을 다시 train / val로 분리
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE_OF_TRAIN, random_state=RANDOM_STATE)
    train_idx, val_idx = next(splitter2.split(df_trainval, groups=df_trainval[COL_TRACK_ID]))

    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

    print(f"[분리 완료] train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    return df_train, df_val, df_test


# =========================================================
# 3. 정규화 통계 계산 (train 데이터만 사용 -> val/test에는 그대로 적용)
#    멜스펙트로그램은 보통 dB 스케일이라 대략 -80~0 범위 -> 표준화(평균0, 표준편차1)
# =========================================================
def compute_normalization_stats(df_train):
    print("[정규화 통계 계산 중] train 데이터 기준으로 mean/std 계산...")
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for path in df_train[COL_PATH]:
        arr = np.load(path).astype(np.float32)
        total_sum += arr.sum()
        total_sq_sum += (arr ** 2).sum()
        total_count += arr.size

    mean = total_sum / total_count
    std = np.sqrt(total_sq_sum / total_count - mean ** 2)
    print(f"[정규화 통계] mean={mean:.4f}, std={std:.4f}")
    return float(mean), float(std)


# =========================================================
# 4. tf.data 파이프라인 (한 번에 메모리에 안 올리고 배치씩 읽음)
# =========================================================
def make_dataset(df, label_encoder, mean, std, shuffle=False):
    paths = df[COL_PATH].values
    labels = label_encoder.transform(df[COL_GENRE].values)
    labels = labels.astype(np.int32)

    def _load(path, label):
        def _np_load(p):
            arr = np.load(p.numpy().decode("utf-8")).astype(np.float32)
            arr = (arr - mean) / (std + 1e-8)  # 표준화
            arr = arr[..., np.newaxis]  # (128, 1249) -> (128, 1249, 1)
            return arr

        arr = tf.py_function(func=_np_load, inp=[path], Tout=tf.float32)
        arr.set_shape([MELSPEC_SHAPE[0], MELSPEC_SHAPE[1], 1])
        return arr, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_STATE)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# =========================================================
# 5. CNN 모델 정의
#    Conv2D+Pool을 4번 반복하며 점점 추상적인 패턴을 학습 ->
#    GlobalAveragePooling으로 압축 -> Dense로 최종 판단 -> Softmax로 10장르 확률
# =========================================================
def build_model(num_classes):
    inputs = keras.Input(shape=(MELSPEC_SHAPE[0], MELSPEC_SHAPE[1], 1))

    x = inputs
    for filters in [16, 32, 64, 64]:
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)  # 과적합 방지: 학습 중 일부 뉴런 랜덤하게 끄기
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="genre_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =========================================================
# 6. 학습 곡선 그리기 (과적합 점검용)
# =========================================================
def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["loss"], label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_title("Loss 곡선")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train accuracy")
    axes[1].plot(history.history["val_accuracy"], label="val accuracy")
    axes[1].set_title("Accuracy 곡선")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.suptitle("CNN 학습 곡선 (train vs val)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[저장] 학습 곡선 -> {save_path}")


# =========================================================
# 7. 혼동행렬 그리기
# =========================================================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("예측 장르")
    ax.set_ylabel("실제 장르")
    ax.set_title("CNN 혼동행렬 (Test set)")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[저장] 혼동행렬 -> {save_path}")


# =========================================================
# MAIN
# =========================================================
def main():
    df = load_index()
    df_train, df_val, df_test = split_by_track(df)

    label_encoder = LabelEncoder()
    label_encoder.fit(df[COL_GENRE].values)
    class_names = list(label_encoder.classes_)
    print(f"[장르 목록] {class_names}")

    mean, std = compute_normalization_stats(df_train)

    train_ds = make_dataset(df_train, label_encoder, mean, std, shuffle=True)
    val_ds = make_dataset(df_val, label_encoder, mean, std, shuffle=False)
    test_ds = make_dataset(df_test, label_encoder, mean, std, shuffle=False)

    model = build_model(num_classes=len(class_names))
    model.summary()

    checkpoint_path = os.path.join(MODEL_DIR, "cnn_model.h5")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
    ]

    print("\n[학습 시작]\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # 학습 곡선 저장 (과적합 점검)
    plot_training_curves(history, os.path.join(OUTPUT_DIR, "cnn_training_curves.png"))

    # -----------------------------------------------------
    # Test set 평가
    # -----------------------------------------------------
    print("\n[Test set 평가]\n")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f} / Test Accuracy: {test_acc:.4f}")

    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = label_encoder.transform(df_test[COL_GENRE].values)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)

    with open(os.path.join(OUTPUT_DIR, "cnn_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)
    print(f"[저장] 분류 리포트 -> outputs/cnn_classification_report.txt")

    plot_confusion_matrix(
        y_true, y_pred, class_names,
        os.path.join(OUTPUT_DIR, "cnn_confusion_matrix.png"),
    )

    # -----------------------------------------------------
    # 모델 & 메타데이터 저장 (Day7 비교, Day8 웹앱 연결용)
    # -----------------------------------------------------
    model.save(checkpoint_path)  # ModelCheckpoint가 이미 최적 가중치로 저장했지만 최종본도 한 번 더 저장
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "cnn_label_encoder.pkl"))

    metadata = {
        "input_shape": [MELSPEC_SHAPE[0], MELSPEC_SHAPE[1], 1],
        "class_names": class_names,
        "normalization_mean": mean,
        "normalization_std": std,
        "test_accuracy": float(test_acc),
        "random_state": RANDOM_STATE,
        "batch_size": BATCH_SIZE,
    }
    with open(os.path.join(MODEL_DIR, "cnn_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] 모델 -> {checkpoint_path}")
    print(f"[완료] 메타데이터 -> {os.path.join(MODEL_DIR, 'cnn_metadata.json')}")
    print("\nDay 6 CNN 학습 및 1차 검증 완료.")


if __name__ == "__main__":
    main()