"""
GTZAN 데이터셋 내 손상된 오디오 파일을 찾아내는 스크립트.

사용법:
    python scripts/check_corrupted.py

결과:
    - 콘솔에 진행 상황과 손상 파일 목록 출력
    - outputs/corrupted_files.txt 에 손상 파일 경로 저장
"""

import os
import librosa
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "genres_original")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "outputs", "corrupted_files.txt")


def find_wav_files(data_dir):
    wav_files = []
    for genre in sorted(os.listdir(data_dir)):
        genre_dir = os.path.join(data_dir, genre)
        if not os.path.isdir(genre_dir):
            continue
        for fname in sorted(os.listdir(genre_dir)):
            if fname.lower().endswith(".wav"):
                wav_files.append(os.path.join(genre_dir, fname))
    return wav_files


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"[오류] 데이터 폴더를 찾을 수 없습니다: {DATA_DIR}")
        print("README.md의 2단계(GTZAN 다운로드)를 먼저 진행해주세요.")
        return

    wav_files = find_wav_files(DATA_DIR)
    print(f"총 {len(wav_files)}개 파일 검사 시작...\n")

    corrupted = []
    for path in tqdm(wav_files, desc="검사 중"):
        try:
            y, sr = librosa.load(path, sr=None, duration=1.0)
            if y is None or len(y) == 0:
                corrupted.append(path)
        except Exception as e:
            corrupted.append(path)
            print(f"  손상 발견: {path} ({e})")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for path in corrupted:
            f.write(path + "\n")

    print(f"\n검사 완료: 정상 {len(wav_files) - len(corrupted)}개 / 손상 {len(corrupted)}개")
    if corrupted:
        print(f"손상 파일 목록 저장됨: {OUTPUT_FILE}")
        print("특징 추출 단계에서 이 파일들을 제외하고 진행하세요.")
    else:
        print("손상된 파일이 없습니다.")


if __name__ == "__main__":
    main()
