"""
장르별 샘플 하나씩을 골라 파형(waveform)과 멜 스펙트로그램을 그려주는 EDA 스크립트.

사용법:
    python scripts/eda_visualize.py

결과:
    outputs/eda_overview.png 에 10개 장르 x 2행(파형/스펙트로그램) 그리드 저장
"""

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "genres_original")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "outputs", "eda_overview.png")
CORRUPTED_FILE = os.path.join(os.path.dirname(__file__), "..", "outputs", "corrupted_files.txt")


def load_skip_files():
    # check_corrupted.py 결과에서 건너뛸 파일명 집합을 만든다
    skip_files = set()
    if not os.path.isfile(CORRUPTED_FILE):
        print("손상 파일 목록이 없습니다. check_corrupted.py를 먼저 실행하는 것을 권장합니다")
        return skip_files
    with open(CORRUPTED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            skip_files.add(os.path.basename(line))
    return skip_files


def pick_one_file_per_genre(data_dir, skip_files):
    samples = {}
    for genre in sorted(os.listdir(data_dir)):
        genre_dir = os.path.join(data_dir, genre)
        if not os.path.isdir(genre_dir):
            continue
        for fname in sorted(os.listdir(genre_dir)):
            if fname.lower().endswith(".wav") and fname not in skip_files:
                samples[genre] = os.path.join(genre_dir, fname)
                break
    return samples


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"[오류] 데이터 폴더를 찾을 수 없습니다: {DATA_DIR}")
        print("README.md의 2단계(GTZAN 다운로드)를 먼저 진행해주세요.")
        return

    SKIP_FILES = load_skip_files()
    samples = pick_one_file_per_genre(DATA_DIR, SKIP_FILES)
    genres = list(samples.keys())
    n = len(genres)

    if n == 0:
        print("샘플 파일을 찾지 못했습니다. 데이터 경로를 확인해주세요.")
        return

    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6))

    for i, genre in enumerate(genres):
        path = samples[genre]
        y, sr = librosa.load(path, sr=22050, duration=30)

        # 1행: 파형
        librosa.display.waveshow(y, sr=sr, ax=axes[0, i], color="steelblue")
        axes[0, i].set_title(genre, fontsize=10)
        axes[0, i].set_xlabel("")
        if i == 0:
            axes[0, i].set_ylabel("Waveform")

        # 2행: 멜 스펙트로그램
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, ax=axes[1, i], cmap="magma")
        if i == 0:
            axes[1, i].set_ylabel("Mel-Spec")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150)
    print(f"저장 완료: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
