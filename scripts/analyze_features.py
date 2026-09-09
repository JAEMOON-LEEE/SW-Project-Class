"""
Day 3 - 특징 추출 결과 검증 및 기술통계 요약
====================================================
사용법:
    project root(MusicLens_..._starter/)에서 실행
    python scripts/analyze_features.py

산출물:
    outputs/feature_summary.csv        - 각 특징의 평균/표준편차/최소/최대 등 describe() 결과
    outputs/outlier_report.csv         - 특징별 IQR 기준 이상치 개수/비율
    outputs/mfcc_boxplot_by_genre.png  - (선택) 장르별 MFCC 대표 계수 분포 비교
    콘솔 출력: 행 수, 결측치 여부, 이상치 상위 특징
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 0. 경로 설정
# ---------------------------------------------------------
FEATURES_CSV = "outputs/features.csv"
SUMMARY_CSV = "outputs/feature_summary.csv"
OUTLIER_CSV = "outputs/outlier_report.csv"
BOXPLOT_PNG = "outputs/mfcc_boxplot_by_genre.png"

NON_FEATURE_COLS = {"track_id", "genre", "filename", "file_path"}


def load_features(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} 를 찾을 수 없습니다. 먼저 scripts/extract_features.py 를 실행하세요."
        )
    df = pd.read_csv(path)
    print(f"[1] features.csv 로드 완료 -> 총 {len(df)}행, {len(df.columns)}열")
    return df


def check_missing(df: pd.DataFrame) -> pd.Series:
    print("\n[2] 결측치(NaN) 확인")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("    -> 결측치 없음 (문제 없음)")
    else:
        print("    -> 결측치가 발견된 컬럼:")
        print(missing.to_string())
    return missing


def check_genre_counts(df: pd.DataFrame):
    if "genre" not in df.columns:
        return
    print("\n[3] 장르별 곡 수 확인 (손상 파일 제외 후 100곡에서 얼마나 줄었는지)")
    counts = df["genre"].value_counts().sort_index()
    print(counts.to_string())


def summarize_statistics(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    print("\n[4] 기술통계 요약 (describe) 생성 중...")
    summary = df[feature_cols].describe().T  # 특징을 행으로, 통계량을 열로
    summary = summary.rename(columns={"50%": "median"})
    summary.to_csv(SUMMARY_CSV, encoding="utf-8-sig")
    print(f"    -> 저장 완료: {SUMMARY_CSV}")
    return summary


def detect_outliers_iqr(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    print("\n[5] IQR 기준 이상치 탐지 중...")
    rows = []
    for col in feature_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        is_outlier = (df[col] < lower) | (df[col] > upper)
        n_out = int(is_outlier.sum())
        rows.append(
            {
                "feature": col,
                "n_outliers": n_out,
                "pct_outliers": round(100 * n_out / len(df), 2),
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
            }
        )
    report = pd.DataFrame(rows).sort_values("n_outliers", ascending=False)
    report.to_csv(OUTLIER_CSV, index=False, encoding="utf-8-sig")
    print(f"    -> 저장 완료: {OUTLIER_CSV}")
    print("    -> 이상치가 가장 많은 상위 5개 특징:")
    print(report.head(5).to_string(index=False))
    return report


def plot_mfcc_by_genre(df: pd.DataFrame):
    """장르별 MFCC 1번 계수 평균값 분포를 박스플롯으로 비교 (선택 항목)."""
    candidates = [c for c in df.columns if "mfcc" in c.lower() and "mean" in c.lower()]
    if not candidates or "genre" not in df.columns:
        print("\n[6] MFCC 컬럼을 찾지 못해 박스플롯을 건너뜁니다. (컬럼명을 확인해주세요)")
        return
    target_col = candidates[0]  # 첫 번째 MFCC mean 계수 사용 (예: mfcc1_mean)
    print(f"\n[6] 장르별 '{target_col}' 분포 박스플롯 생성 중...")

    plt.figure(figsize=(12, 6))
    df.boxplot(column=target_col, by="genre", rot=45)
    plt.title(f"Genre-wise distribution of {target_col}")
    plt.suptitle("")
    plt.xlabel("Genre")
    plt.ylabel(target_col)
    plt.tight_layout()
    plt.savefig(BOXPLOT_PNG, dpi=150)
    plt.close()
    print(f"    -> 저장 완료: {BOXPLOT_PNG}")


def main():
    df = load_features(FEATURES_CSV)
    check_missing(df)
    check_genre_counts(df)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n(통계/이상치 계산 대상 특징 수: {len(feature_cols)}개)")

    summarize_statistics(df, feature_cols)
    detect_outliers_iqr(df, feature_cols)
    plot_mfcc_by_genre(df)

    print("\n[완료] Day 3 체크리스트용 산출물이 모두 생성되었습니다.")
    print("  - outputs/feature_summary.csv  (기술통계 요약표)")
    print("  - outputs/outlier_report.csv   (이상치 리포트)")
    print("  - outputs/mfcc_boxplot_by_genre.png (선택, 장르별 분포 비교)")


if __name__ == "__main__":
    main()