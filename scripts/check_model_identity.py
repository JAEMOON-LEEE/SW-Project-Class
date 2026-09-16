"""
scripts/check_model_identity.py

목적:
  웹앱(app.py)이 로드하는 models/baseline_model.pkl이
  실제로 어떤 모델(SVM kernel 종류 등)인지 확인하고,
  Day7에서 최종 선정된 모델(models/final_model_meta.json)과
  일치하는지 검증한다.

실행 방법 (프로젝트 루트에서, (nlp) conda 환경):
  python scripts/check_model_identity.py
"""

import json
import joblib
from pathlib import Path

MODELS_DIR = Path("models")


def describe_model(model):
    """모델 객체를 보고 사람이 읽을 수 있는 설명 문자열을 만든다."""
    cls_name = type(model).__name__

    if cls_name == "SVC":
        kernel = getattr(model, "kernel", "?")
        return f"SVC (kernel={kernel})"

    if cls_name == "RandomForestClassifier":
        n_estimators = getattr(model, "n_estimators", "?")
        return f"RandomForestClassifier (n_estimators={n_estimators})"

    return cls_name


def main():
    baseline_path = MODELS_DIR / "baseline_model.pkl"
    metadata_path = MODELS_DIR / "model_metadata.pkl"
    final_meta_path = MODELS_DIR / "final_model_meta.json"

    print("=" * 60)
    print("1) baseline_model.pkl 실제 내용 확인")
    print("=" * 60)

    if not baseline_path.exists():
        print(f"[!] {baseline_path} 파일이 없습니다.")
        return

    model = joblib.load(baseline_path)
    model_desc = describe_model(model)
    print(f"  로드된 객체 타입: {model_desc}")

    if metadata_path.exists():
        metadata = joblib.load(metadata_path)
        print(f"  model_metadata.pkl 내용: {metadata}")
    else:
        print(f"  [!] {metadata_path} 없음 (FEATURE_COLS 순서 확인 불가)")

    print()
    print("=" * 60)
    print("2) Day7 최종 선정 모델과 비교")
    print("=" * 60)

    if not final_meta_path.exists():
        print(f"  [!] {final_meta_path} 없음 — Day7 compare_models.py 결과를 먼저 확인하세요.")
        return

    with open(final_meta_path, "r", encoding="utf-8") as f:
        final_meta = json.load(f)

    final_model_name = final_meta.get("final_model", "?")
    print(f"  final_model_meta.json 기록: {final_model_name}")

    # 간단한 일치 여부 판정 (SVM linear/rbf 케이스만 우선 체크)
    is_svm = type(model).__name__ == "SVC"
    kernel = getattr(model, "kernel", None)

    match = False
    if "linear" in final_model_name.lower() and is_svm and kernel == "linear":
        match = True
    elif "rbf" in final_model_name.lower() and is_svm and kernel == "rbf":
        match = True
    elif "random forest" in final_model_name.lower() and type(model).__name__ == "RandomForestClassifier":
        match = True

    print()
    if match:
        print(f"  [OK] baseline_model.pkl({model_desc})이 최종 선정 모델({final_model_name})과 일치합니다.")
        print("       웹앱은 이미 올바른 모델을 쓰고 있습니다.")
    else:
        print(f"  [MISMATCH] baseline_model.pkl({model_desc})이 최종 선정 모델({final_model_name})과 다릅니다!")
        print("       -> train_baseline.py를 다시 실행해 SVM linear로 baseline_model.pkl을 재저장하거나,")
        print("          별도 파일로 저장한 뒤 app.py의 로드 경로를 그쪽으로 바꿔야 합니다.")


if __name__ == "__main__":
    main()