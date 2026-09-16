"""
compare_models.py
Day 7 - SVM / RF / CNN 최종 성능 비교 및 모델 선정

역할:
  1) features.csv 기준으로 Day4와 동일한 곡 단위 test set 재현 (SVM/RF용)
  2) SVM(linear), SVM(rbf), RandomForest를 동일 train/test로 재학습 + 평가
  3) 저장된 CNN(cnn_model.h5)을 같은 곡들(test set)로 평가
     -> GroupShuffleSplit은 track_id를 정렬 후 셔플하므로, features.csv와
        melspec_index.csv의 행 순서가 달라도 random_state/test_size가 같으면
        "같은 곡들"이 test set으로 뽑힘 (Day4/Day6와 동일 조건)
  4) 4개 모델(SVM linear/rbf, RF, CNN) 성능을 한 표로 정리 + 막대그래프 저장
  5) 최종 모델 추천 (accuracy 기준 / macro-F1 기준 둘 다 표시)

실행 전 확인:
  - conda activate nlp 상태에서 scripts/ 안에 두고 프로젝트 루트에서 실행
  - outputs/features.csv, outputs/melspec_index.csv, outputs/melspecs/,
    models/cnn_model.h5, models/cnn_metadata.json, models/cnn_label_encoder.pkl
    가 이미 존재해야 함 (Day4, Day5/6 산출물)

[수정 사항]
  - SVC(kernel="linear"/"rbf") 생성 시 probability=True 추가.
    app.py가 predict_proba()로 Top3를 뽑기 때문에 필수. accuracy/predict()
    결과에는 영향 없음(Platt scaling으로 확률 추정치만 추가됨).
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

# =========================================================
# CONFIG
# =========================================================
FEATURES_CSV = "outputs/features.csv"
MELSPEC_INDEX_CSV = "outputs/melspec_index.csv"
MELSPEC_DIR = "outputs/melspecs"

COL_TRACK_ID = "track_id"
COL_GENRE = "genre"
COL_PATH = "npy_path"

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

RANDOM_STATE = 42     # Day4/Day6와 동일 -> 같은 test set 재현용
TEST_SIZE = 0.20

MELSPEC_SHAPE = (128, 1249)
BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


# =========================================================
# 1. features.csv 로드 + 곡 단위 train/test 분리 (Day4와 동일 조건)
# =========================================================
def load_and_split_features():
    df = pd.read_csv(FEATURES_CSV)
    print(f"[로드] {FEATURES_CSV} -> {len(df)}행")

    groups = df[COL_TRACK_ID]
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=groups))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    print(f"[분리] train={len(df_train)}, test={len(df_test)}")

    feature_cols = [c for c in df.columns if c not in (COL_TRACK_ID, COL_GENRE)]
    X_train, y_train = df_train[feature_cols].values, df_train[COL_GENRE].values
    X_test, y_test = df_test[feature_cols].values, df_test[COL_GENRE].values

    return X_train, y_train, X_test, y_test, df_test, feature_cols


# =========================================================
# 2. SVM(linear) / SVM(rbf) / RandomForest 학습 + 평가
# =========================================================
def train_and_eval_baselines(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "SVM (linear)": SVC(kernel="linear", C=1.0, probability=True, random_state=RANDOM_STATE),
        "SVM (rbf)": SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        results[name] = {
            "y_true": y_test,
            "y_pred": y_pred,
        }
        fitted[name] = model
        acc = accuracy_score(y_test, y_pred)
        print(f"[{name}] test accuracy = {acc:.4f}")

    return results, fitted, scaler


# =========================================================
# 3. CNN 평가 (저장된 모델 재사용, 재학습 없음)
# =========================================================
def eval_cnn(test_track_ids):
    with open(os.path.join(MODEL_DIR, "cnn_metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    mean, std = meta["normalization_mean"], meta["normalization_std"]
    class_names = meta["class_names"]

    label_encoder = joblib.load(os.path.join(MODEL_DIR, "cnn_label_encoder.pkl"))
    model = keras.models.load_model(os.path.join(MODEL_DIR, "cnn_model.h5"))

    df_idx = pd.read_csv(MELSPEC_INDEX_CSV)
    if COL_PATH not in df_idx.columns:
        df_idx[COL_PATH] = df_idx[COL_TRACK_ID].astype(str).apply(
            lambda tid: os.path.join(MELSPEC_DIR, f"{tid}.npy")
        )

    # Day4 test set과 동일한 track_id만 사용 -> 진짜 공정 비교
    df_cnn_test = df_idx[df_idx[COL_TRACK_ID].isin(test_track_ids)].reset_index(drop=True)
    matched = len(df_cnn_test)
    print(f"[CNN 평가용 test 매칭] {matched} / {len(test_track_ids)}곡 매칭됨"
          + ("" if matched == len(test_track_ids) else " [경고] 일부 곡이 melspec_index에 없음"))

    paths = df_cnn_test[COL_PATH].values
    labels = label_encoder.transform(df_cnn_test[COL_GENRE].values).astype(np.int32)

    def _load(path, label):
        def _np_load(p):
            arr = np.load(p.numpy().decode("utf-8")).astype(np.float32)
            arr = (arr - mean) / (std + 1e-8)
            return arr[..., np.newaxis]
        arr = tf.py_function(func=_np_load, inp=[path], Tout=tf.float32)
        arr.set_shape([MELSPEC_SHAPE[0], MELSPEC_SHAPE[1], 1])
        return arr, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    y_pred_probs = model.predict(ds)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_idx)
    y_true = df_cnn_test[COL_GENRE].values

    acc = accuracy_score(y_true, y_pred)
    print(f"[CNN] test accuracy (재현된 동일 test set) = {acc:.4f}")

    return {"y_true": y_true, "y_pred": y_pred}


# =========================================================
# 4. 결과 통합 표 + 그래프
# =========================================================
def summarize(all_results):
    rows = []
    for name, res in all_results.items():
        y_true, y_pred = res["y_true"], res["y_pred"]
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")
        rows.append({
            "model": name,
            "test_accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
        })

        report = classification_report(y_true, y_pred, digits=3)
        report_path = os.path.join(OUTPUT_DIR, f"report_{name.replace(' ', '_').replace('(', '').replace(')', '')}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"{name} - Test Accuracy: {acc:.4f}\n\n")
            f.write(report)
        print(f"[저장] {report_path}")

    summary_df = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    summary_path = os.path.join(OUTPUT_DIR, "day7_model_comparison.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[저장] 종합 비교표 -> {summary_path}")
    print(summary_df.to_string(index=False))

    # 막대그래프
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary_df))
    width = 0.25
    ax.bar(x - width, summary_df["test_accuracy"], width, label="Accuracy")
    ax.bar(x, summary_df["macro_f1"], width, label="Macro F1")
    ax.bar(x + width, summary_df["weighted_f1"], width, label="Weighted F1")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"], rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Day 7 - 모델별 최종 성능 비교")
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "day7_model_comparison.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[저장] 비교 그래프 -> {fig_path}")

    return summary_df


# =========================================================
# MAIN
# =========================================================
def main():
    X_train, y_train, X_test, y_test, df_test, feature_cols = load_and_split_features()

    baseline_results, fitted_models, scaler = train_and_eval_baselines(X_train, y_train, X_test, y_test)

    test_track_ids = set(df_test[COL_TRACK_ID].values)
    cnn_result = eval_cnn(test_track_ids)

    all_results = dict(baseline_results)
    all_results["CNN"] = cnn_result

    summary_df = summarize(all_results)

    best_acc = summary_df.iloc[0]
    best_f1_row = summary_df.sort_values("macro_f1", ascending=False).iloc[0]

    print("\n" + "=" * 50)
    print(f"[추천] Accuracy 기준 최종 모델   : {best_acc['model']} ({best_acc['test_accuracy']:.4f})")
    print(f"[추천] Macro-F1 기준 최종 모델   : {best_f1_row['model']} ({best_f1_row['macro_f1']:.4f})")
    print("=" * 50)
    print("\n두 기준이 같은 모델을 가리키면 그걸로 최종 선정하면 됩니다.")
    print("다르면: 클래스 불균형/소수 장르(rock 등) 성능을 더 중시할지에 따라 결정하세요.")
    print("\n[다음 단계] 최종 모델이 baseline(SVM/RF) 계열이면:")
    print("  선택한 모델을 models/baseline_model.pkl 로, scaler를 models/scaler.pkl 로 저장하세요.")
    print("  (아래 코드는 예시 -- 실제로 최종 선택한 모델명으로 바꿔서 실행)")
    print("  예) joblib.dump(fitted_models['SVM (rbf)'], 'models/baseline_model.pkl')")
    print("      joblib.dump(scaler, 'models/scaler.pkl')")
    print("\n[다음 단계] 최종 모델이 CNN이면 models/cnn_model.h5를 그대로 쓰되,")
    print("  app.py가 SVM용(features.csv)이 아니라 CNN용(멜스펙트로그램) 전처리를 타도록")
    print("  분기 로직이 필요합니다 -- Day8에서 app.py 점검 시 같이 확인하세요.")

    # -----------------------------------------------------
    # 최종 모델 저장 (accuracy/macro-F1 둘 다 동의한 모델을 baseline으로 확정)
    # 다른 모델을 최종으로 쓰고 싶으면 FINAL_MODEL_NAME만 바꾸면 됨
    # -----------------------------------------------------
    FINAL_MODEL_NAME = best_acc["model"]  # 예: "SVM (linear)"

    if FINAL_MODEL_NAME in fitted_models:
        joblib.dump(fitted_models[FINAL_MODEL_NAME], os.path.join(MODEL_DIR, "baseline_model.pkl"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
        with open(os.path.join(MODEL_DIR, "final_model_meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "final_model": FINAL_MODEL_NAME,
                "test_accuracy": float(best_acc["test_accuracy"]),
                "macro_f1": float(best_acc["macro_f1"]),
                "feature_cols": feature_cols,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n[완료] 최종 모델 '{FINAL_MODEL_NAME}' -> models/baseline_model.pkl 저장")
        print(f"[완료] 스케일러 -> models/scaler.pkl 저장")
        print(f"[완료] 메타데이터 -> models/final_model_meta.json 저장")
    else:
        print(f"\n[안내] 최종 모델이 CNN이라 baseline_model.pkl 교체는 하지 않았습니다.")
        print("  models/cnn_model.h5를 웹앱에서 그대로 사용하세요.")


if __name__ == "__main__":
    main()