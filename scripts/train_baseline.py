"""
Day 4 — 베이스라인 모델 학습 (SVM, Random Forest)
MusicLens 프로젝트

- 곡 단위(track_id) train/test split (프레임 단위 데이터 누수 방지)
- StandardScaler로 특징 스케일링
- SVM (linear / rbf 커널 비교)
- Random Forest
- Cross-validation으로 1차 정확도 비교
- 결과표를 outputs/baseline_results.csv 로 저장

실행 방법 (프로젝트 루트에서):
    python scripts/train_baseline.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
FEATURES_CSV = os.path.join("outputs", "features.csv")
RESULTS_CSV = os.path.join("outputs", "baseline_results.csv")
CONFUSION_MATRIX_PNG = os.path.join("outputs", "baseline_confusion_matrices.png")

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "baseline_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.pkl")

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5

# features.csv에서 특징이 아닌(메타데이터) 컬럼
NON_FEATURE_COLS = {"track_id", "genre", "file_path", "filename"}


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[에러] {path} 파일이 없습니다.")
        print("먼저 Day 3의 'python scripts/extract_features.py'를 실행해서")
        print("outputs/features.csv를 생성해주세요.")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"[로드 완료] {path} — {len(df)}행, {len(df.columns)}열")

    # 결측치 확인
    n_na = df.isna().sum().sum()
    if n_na > 0:
        print(f"[경고] 결측치 {n_na}개 발견. 해당 행을 제거합니다.")
        before = len(df)
        df = df.dropna()
        print(f"  {before} -> {len(df)}행")

    if "track_id" not in df.columns or "genre" not in df.columns:
        print("[에러] features.csv에 track_id 또는 genre 컬럼이 없습니다.")
        print("Day 2 설계(extract_all_features)를 다시 확인해주세요.")
        sys.exit(1)

    return df


def song_level_split(df: pd.DataFrame):
    """
    곡(track_id) 단위로 먼저 train/test를 나눈 뒤,
    각 곡에 속한 모든 행(프레임 등)을 같은 세트에 배치한다.
    -> 같은 곡의 조각이 train/test 양쪽에 걸쳐 들어가는 데이터 누수를 방지.
    """
    track_info = df[["track_id", "genre"]].drop_duplicates()

    train_tracks, test_tracks = train_test_split(
        track_info,
        test_size=TEST_SIZE,
        stratify=track_info["genre"],
        random_state=RANDOM_STATE,
    )

    train_ids = set(train_tracks["track_id"])
    test_ids = set(test_tracks["track_id"])

    train_df = df[df["track_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["track_id"].isin(test_ids)].reset_index(drop=True)

    print(f"[곡 단위 split] train 곡 수: {len(train_ids)}, test 곡 수: {len(test_ids)}")
    print(f"[곡 단위 split] train 행 수: {len(train_df)}, test 행 수: {len(test_df)}")

    # 누수 검증: train/test 곡이 겹치지 않는지 확인
    overlap = train_ids & test_ids
    assert len(overlap) == 0, f"데이터 누수 발생! 겹치는 track_id: {overlap}"
    print("[검증 완료] train/test 곡 겹침 없음 (데이터 누수 없음)")

    return train_df, test_df


def get_feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def evaluate_model(name, model, X_train, y_train, X_test, y_test, cv_folds):
    print(f"\n{'=' * 60}")
    print(f"모델: {name}")
    print("=" * 60)

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")

    print(f"학습 시간: {train_time:.2f}초")
    print(f"Test 정확도: {test_acc:.4f}")
    print(f"{cv_folds}-Fold CV 정확도: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("\n분류 리포트:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "model": name,
        "test_accuracy": test_acc,
        "cv_mean_accuracy": cv_scores.mean(),
        "cv_std_accuracy": cv_scores.std(),
        "train_time_sec": train_time,
    }, confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))


def plot_confusion_matrices(confusion_matrices: dict, labels, save_path: str):
    """세 모델의 혼동행렬을 한 이미지에 나란히 그려서 저장한다."""
    n_models = len(confusion_matrices)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        # 행(실제 라벨) 기준으로 정규화 -> 각 장르가 어디로 오분류되는지 비율로 비교
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            vmin=0,
            vmax=1,
            ax=ax,
            cbar=ax is axes[-1],
        )
        ax.set_title(name)
        ax.set_xlabel("예측 장르")
        ax.set_ylabel("실제 장르")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[저장 완료] {save_path}")


def summarize_confusions(confusion_matrices: dict, labels, top_n: int = 3):
    """모델별로 가장 많이 헷갈리는 장르 쌍을 텍스트로 요약."""
    print(f"\n{'=' * 60}")
    print("장르 혼동 TOP 쌍 (모델별)")
    print("=" * 60)
    for name, cm in confusion_matrices.items():
        pairs = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and cm[i, j] > 0:
                    pairs.append((labels[i], labels[j], cm[i, j]))
        pairs.sort(key=lambda x: x[2], reverse=True)
        print(f"\n[{name}]")
        for true_g, pred_g, count in pairs[:top_n]:
            print(f"  실제 '{true_g}' -> 예측 '{pred_g}': {count}곡")


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_data(FEATURES_CSV)
    feature_cols = get_feature_columns(df)
    print(f"[특징 컬럼] {len(feature_cols)}개: {feature_cols[:6]}... (일부만 표시)")

    train_df, test_df = song_level_split(df)

    X_train_raw = train_df[feature_cols].values
    y_train = train_df["genre"]
    X_test_raw = test_df[feature_cols].values
    y_test = test_df["genre"]

    # 스케일링 (SVM은 스케일에 민감하므로 필수)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    results = []
    confusion_matrices = {}
    fitted_models = {}  # 모델 이름 -> 학습 완료된 모델 객체 (저장용)

    # --- SVM: linear vs rbf 커널 비교 ---
    for kernel in ["linear", "rbf"]:
        svm_model = SVC(kernel=kernel, C=1.0, random_state=RANDOM_STATE, probability=True)
        res, cm = evaluate_model(
            f"SVM ({kernel})", svm_model, X_train, y_train, X_test, y_test, N_CV_FOLDS
        )
        results.append(res)
        confusion_matrices[f"SVM ({kernel})"] = cm
        fitted_models[f"SVM ({kernel})"] = svm_model

    # --- Random Forest ---
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    res, cm = evaluate_model(
        "Random Forest", rf_model, X_train, y_train, X_test, y_test, N_CV_FOLDS
    )
    results.append(res)
    confusion_matrices["Random Forest"] = cm
    fitted_models["Random Forest"] = rf_model

    # --- 결과 정리 및 저장 ---
    results_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
    results_df.to_csv(RESULTS_CSV, index=False)

    print(f"\n{'=' * 60}")
    print("최종 비교 결과 (정확도 내림차순)")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\n[저장 완료] {RESULTS_CSV}")

    best_model_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    print(f"\n[1차 비교] 가장 높은 test 정확도 모델: {best_model_name}")
    print("(Day 7에서 CNN까지 포함해 최종 모델을 선정합니다.)")

    # --- 혼동행렬 시각화 + 요약 ---
    labels = sorted(y_test.unique())
    plot_confusion_matrices(confusion_matrices, labels, CONFUSION_MATRIX_PNG)
    summarize_confusions(confusion_matrices, labels)

    # --- 모델 / 스케일러 / 메타데이터 저장 (웹앱 연결용) ---
    # 지금은 베이스라인 중 최고 성능 모델을 저장해 웹앱이 즉시 쓸 수 있게 한다.
    # Day 7에서 CNN까지 비교 후 최종 모델로 이 파일을 덮어쓸 예정.
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(
        {
            "feature_cols": feature_cols,   # 예측 시 입력 특징 순서를 맞추기 위해 필요
            "labels": labels,               # 장르 라벨 목록 (정렬된 순서)
            "best_model_name": best_model_name,
            "stage": "baseline",            # Day 7에서 "final"로 교체 예정
        },
        METADATA_PATH,
    )
    print(f"\n[모델 저장 완료]")
    print(f"  모델: {MODEL_PATH} ({best_model_name})")
    print(f"  스케일러: {SCALER_PATH}")
    print(f"  메타데이터: {METADATA_PATH}")
    print("  -> app.py에서 이 세 파일을 로드해 예측에 사용하면 됩니다.")


if __name__ == "__main__":
    main()