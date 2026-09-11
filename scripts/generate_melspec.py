"""
MusicLens - Day 5: 멜 스펙트로그램 생성 (CNN 입력용)
=========================================================
- extract_features.py와 동일한 규칙 사용:
  * DATA_DIR, CORRUPTED_FILE, SAMPLE_RATE, track_id 형식(genre.filename) 동일
  * 손상 파일은 corrupted_files.txt 기준으로 스킵 (load_skip_files 동일 로직)
  * 이미 처리된 track_id는 재처리하지 않음 (outputs/melspec_index.csv로 캐싱)
- 곡마다 길이가 조금씩 달라도 CNN 입력 shape을 통일하기 위해
  DURATION(초) 기준으로 고정 길이 오디오로 자르거나 0-padding.
- 결과:
    outputs/melspecs/<track_id>.npy      개별 멜 스펙트로그램 (n_mels, frames) float32
    outputs/melspec_index.csv            track_id, genre, npy_path 인덱스 (재개/로딩용)

실행 방법 (프로젝트 루트에서):
    python scripts/generate_melspec.py            # 전체 999곡
    python scripts/generate_melspec.py --limit 5   # 장르별 5곡만 (빠른 테스트용)
"""

import os
import sys
import csv
import argparse

import numpy as np
import librosa

# ---------------------------------------------------------------------------
# 설정 (extract_features.py와 통일)
# ---------------------------------------------------------------------------
DATA_DIR = "data/genres_original"
OUTPUT_DIR = "outputs"
CORRUPTED_FILE = os.path.join(OUTPUT_DIR, "corrupted_files.txt")

MELSPEC_DIR = os.path.join(OUTPUT_DIR, "melspecs")
INDEX_CSV = os.path.join(OUTPUT_DIR, "melspec_index.csv")

SAMPLE_RATE = 22050  # extract_features.py와 동일
DURATION = 29.0      # GTZAN은 30초 내외지만, 파일마다 미세하게 길이가 달라 29초로 고정
N_SAMPLES = int(SAMPLE_RATE * DURATION)

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# 참고용: 위 설정으로 나오는 고정 shape
# frames = 1 + N_SAMPLES // HOP_LENGTH
FRAMES = 1 + N_SAMPLES // HOP_LENGTH
MELSPEC_SHAPE = (N_MELS, FRAMES)


# ---------------------------------------------------------------------------
# extract_features.py에서 그대로 가져온 함수 (동일 로직 유지)
# ---------------------------------------------------------------------------
def load_skip_files():
    """corrupted_files.txt를 읽어 스킵할 파일명(basename) 집합을 반환"""
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


def load_existing_track_ids():
    """이미 melspec_index.csv에 있는 track_id를 읽어와 중복 처리를 방지"""
    existing = set()
    if os.path.isfile(INDEX_CSV):
        with open(INDEX_CSV, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row["track_id"])
    return existing


# ---------------------------------------------------------------------------
# 멜 스펙트로그램 추출
# ---------------------------------------------------------------------------
def fix_length(y: np.ndarray, n_samples: int) -> np.ndarray:
    """오디오를 고정 길이로 자르거나(긴 경우) 0-padding(짧은 경우)한다."""
    return librosa.util.fix_length(y, size=n_samples)


def extract_melspec(file_path: str) -> np.ndarray:
    """
    파일 하나를 로드해 고정 크기의 log-mel spectrogram(dB)으로 변환.
    반환 shape: (N_MELS, FRAMES), dtype float32
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    y = fix_length(y, N_SAMPLES)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # -80~0 dB 근방 (정규화는 로딩 시점에서 수행)

    # 혹시 shape이 예상과 다르면(마지막 프레임 반올림 차이 등) 강제로 맞춰준다
    if mel_db.shape != MELSPEC_SHAPE:
        fixed = np.zeros(MELSPEC_SHAPE, dtype=np.float32)
        n_frames = min(mel_db.shape[1], MELSPEC_SHAPE[1])
        fixed[:, :n_frames] = mel_db[:, :n_frames]
        mel_db = fixed

    return mel_db.astype(np.float32)


# ---------------------------------------------------------------------------
# 메인 파이프라인 (extract_features.py의 캐싱/재개 구조를 그대로 따름)
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GTZAN 멜 스펙트로그램 생성")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="장르별로 처리할 최대 곡 수 (빠른 테스트용, 기본값: 전체)",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MELSPEC_DIR, exist_ok=True)

    skip_files = load_skip_files()
    existing_ids = load_existing_track_ids()

    if not os.path.isdir(DATA_DIR):
        print(f"데이터 폴더를 찾을 수 없습니다: {DATA_DIR}")
        return

    genres = sorted(
        d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
    )
    print(f"[설정] 고정 shape: {MELSPEC_SHAPE} (n_mels={N_MELS}, frames={FRAMES})")
    print(f"[설정] 장르 {len(genres)}개: {genres}")
    if args.limit:
        print(f"[테스트 모드] 장르별 최대 {args.limit}곡만 처리")

    index_exists = os.path.isfile(INDEX_CSV)
    processed_count = 0
    error_count = 0
    skipped_existing = 0

    with open(INDEX_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "genre", "npy_path"])
        if not index_exists:
            writer.writeheader()

        for genre in genres:
            genre_dir = os.path.join(DATA_DIR, genre)
            wav_files = sorted(
                fn for fn in os.listdir(genre_dir) if fn.endswith(".wav")
            )
            if args.limit:
                wav_files = wav_files[: args.limit]

            for fn in wav_files:
                if fn in skip_files:
                    continue

                track_id = f"{genre}.{fn}"  # features.csv와 동일한 형식
                if track_id in existing_ids:
                    skipped_existing += 1
                    continue

                file_path = os.path.join(genre_dir, fn)
                try:
                    mel_db = extract_melspec(file_path)
                except Exception as e:
                    print(f"[에러] {track_id} 처리 실패: {e}")
                    error_count += 1
                    continue

                npy_path = os.path.join(MELSPEC_DIR, f"{track_id}.npy")
                np.save(npy_path, mel_db)

                writer.writerow(
                    {"track_id": track_id, "genre": genre, "npy_path": npy_path}
                )
                f.flush()  # 중간에 중단돼도 지금까지 결과는 남도록

                processed_count += 1
                if processed_count % 20 == 0:
                    print(f"  진행 중... {processed_count}곡 처리 완료 (최근: {track_id})")

    print(f"\n완료: {processed_count}곡 새로 처리, {skipped_existing}곡 이미 존재(스킵), {error_count}곡 에러")
    print(f"인덱스 저장 위치: {INDEX_CSV}")
    print(f"멜 스펙트로그램 저장 위치: {MELSPEC_DIR}/")


if __name__ == "__main__":
    main()