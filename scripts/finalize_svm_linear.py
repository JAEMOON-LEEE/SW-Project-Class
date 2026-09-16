"""
scripts/finalize_svm_linear.py

목적:
  웹앱이 쓰는 models/baseline_model.pkl이 아직 SVM(rbf)로 저장되어 있음.
  Day7 compare_models.py에서 최종 선정된 SVM(linear, acc 0.74)을 다시 학습해
  baseline_model.pkl / scaler.pkl / model_metadata.pkl을 갱신한다.

  안전을 위해 기존 baseline_model.pkl, scaler.pkl은 .bak로 백업한 뒤 덮어쓴다.

실행 방법 (프로젝트 루트에서, (nlp) conda 환경):
  python scripts/finalize_svm_linear.py

주의:
  - 재현한 test accuracy가 Day7 기록(0.74)과 많이 다르면 split 로직이
    compare_models.py와 다르다는 뜻이니, 그 경우 저장하지 않고 경고만 출력한다.
"""

import shutil
import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

FEATURES_CSV = Path("outputs/features.csv")
MODELS_DIR = Path("models")
RANDOM_STATE = 42
TEST_SIZE = 0.20
EXPECTED_ACC = 0.74      # Day7 compare_models.py 기록값
ACC_TOLERANCE = 0.03     # 이 이상 차이나면 split이 다른 것으로 간주


def backup(path: Path):
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        print(f"  백업: {path} -> {bak}")


def main():
    metadata_path = MODELS_DIR / "model_metadata.pkl"
    baseline_path = MODELS_DIR / "baseline_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"

    if not FEATURES_CSV.exists():
        print(f"[!] {FEATURES_CSV} 없음. 프로젝트 루트에서 실행 중인지 확인하세요.")
        return
    if not metadata_path.exists():
        print(f"[!] {metadata_path} 없음.")
        return

    metadata = joblib.load(metadata_path)
    feature_cols = metadata["feature_cols"]

    df = pd.read_csv(FEATURES_CSV)
    X = df[feature_cols]
    y = df["genre"]

    print("=" * 60)
    print("1) 곡 단위 train/test split 재현")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  train {len(X_train)}곡 / test {len(X_test)}곡")

    print()
    print("=" * 60)
    print("2) SVM(linear) 학습")
    print("=" * 60)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # app.py에서 predict_proba를 쓰므로 probability=True 필수
    model = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"  test accuracy = {acc:.4f}  (Day7 기록값: {EXPECTED_ACC})")

    print()
    print("=" * 60)
    print("3) 검증 및 저장")
    print("=" * 60)
    if abs(acc - EXPECTED_ACC) > ACC_TOLERANCE:
        print(f"  [!] 재현된 정확도({acc:.4f})가 Day7 기록({EXPECTED_ACC})과 {ACC_TOLERANCE} 이상 차이납니다.")
        print("      compare_models.py의 split 방식이 이 스크립트와 다를 수 있습니다.")
        print("      저장을 건너뜁니다. compare_models.py 코드를 확인 후 다시 시도하세요.")
        return

    print("  [OK] 정확도가 Day7 기록과 근접합니다. 파일을 갱신합니다.")
    backup(baseline_path)
    backup(scaler_path)

    joblib.dump(model, baseline_path)
    joblib.dump(scaler, scaler_path)

    metadata["best_model_name"] = "SVM (linear)"
    joblib.dump(metadata, metadata_path)

    print(f"  저장 완료: {baseline_path}, {scaler_path}, {metadata_path}(best_model_name 갱신)")
    print("  이제 app.py를 재시작해서 웹앱이 SVM(linear)로 예측하는지 확인하세요.")


if __name__ == "__main__":
    main()