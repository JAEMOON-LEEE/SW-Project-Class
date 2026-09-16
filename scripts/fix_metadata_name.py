"""
scripts/fix_metadata_name.py

목적:
  model_metadata.pkl의 best_model_name이 아직 'SVM (rbf)'로 남아있어서
  final_model_meta.json('SVM (linear)')과 불일치함. 실제 동작에는 영향 없지만
  보고서/디버깅 혼란 방지를 위해 맞춰준다.

실행 방법 (프로젝트 루트에서, (nlp) conda 환경):
  python scripts/fix_metadata_name.py
"""

import joblib
from pathlib import Path

MODEL_DIR = Path("models")


def main():
    metadata_path = MODEL_DIR / "model_metadata.pkl"
    final_meta_path = MODEL_DIR / "final_model_meta.json"

    if not metadata_path.exists():
        print(f"[!] {metadata_path} 없음")
        return
    if not final_meta_path.exists():
        print(f"[!] {final_meta_path} 없음")
        return

    metadata = joblib.load(metadata_path)
    old_name = metadata.get("best_model_name")

    import json
    with open(final_meta_path, "r", encoding="utf-8") as f:
        final_meta = json.load(f)
    final_name = final_meta.get("final_model")

    if old_name == final_name:
        print(f"[안내] 이미 일치함: {old_name}")
        return

    metadata["best_model_name"] = final_name
    joblib.dump(metadata, metadata_path)
    print(f"[완료] best_model_name 갱신: '{old_name}' -> '{final_name}'")


if __name__ == "__main__":
    main()
