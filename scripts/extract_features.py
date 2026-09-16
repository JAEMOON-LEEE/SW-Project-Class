"""
MusicLens - Day 2: 오디오 특징 추출 함수
=========================================

역할 분담:
- 이재문: MFCC, 스펙트럴 센트로이드/롤오프, 제로크로싱율
- 김승현: 템포(BPM), 크로마, 리듬 관련 특징

설계 원칙:
1. 손상 파일은 Day 1의 corrupted_files.txt를 읽어 자동 제외 (load_skip_files 재사용)
2. 곡 단위 식별자(track_id, genre)를 특징 테이블에 함께 저장 →
   Day 4에서 train/test split을 곡 ID 기준으로 수행하기 위함 (프레임 단위 누수 방지)
3. 추출 결과는 features.csv로 캐싱하여 재실행 시 중복 처리 방지 (이미 처리된 track_id는 스킵)
4. 파일 하나당 특징 벡터는 "평균값(mean) + 표준편차(std)"로 요약하여 고정 길이 벡터화
   (프레임 수는 곡마다 다르므로 통계 요약이 표준적인 접근)

사용법:
    python extract_features.py
    (data/genres_original/ 하위 전체 wav 파일 처리 -> outputs/features.csv 생성)
"""

import os
import csv
import numpy as np
import librosa

# ------------------------------------------------------------------
# 경로 설정 (Day 1 폴더 구조 기준)
# ------------------------------------------------------------------
DATA_DIR = "data/genres_original"
OUTPUT_DIR = "outputs"
CORRUPTED_FILE = os.path.join(OUTPUT_DIR, "corrupted_files.txt")
FEATURES_CSV = os.path.join(OUTPUT_DIR, "features.csv")

SAMPLE_RATE = 22050  # librosa 기본값, 전체 파이프라인에서 통일해서 사용


# ------------------------------------------------------------------
# Day 1에서 이어받는 함수 (그대로 재사용)
# ------------------------------------------------------------------
def load_skip_files():
    """corrupted_files.txt를 읽어 스킵할 파일명 집합을 반환"""
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


# ------------------------------------------------------------------
# [이재문 담당] MFCC / 스펙트럴 계열
# ------------------------------------------------------------------
def extract_mfcc_features(y, sr, n_mfcc=20):
    """
    MFCC 20차 계수를 추출하고 각 계수별 mean/std로 요약.
    반환: {"mfcc_1_mean": ..., "mfcc_1_std": ..., ..., "mfcc_20_std": ...}
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features = {}
    for i in range(n_mfcc):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
    return features


def extract_spectral_features(y, sr):
    """
    스펙트럴 센트로이드, 스펙트럴 롤오프, 스펙트럴 대역폭, 제로크로싱율 추출.
    각 특징은 프레임 시계열의 mean/std로 요약.
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    return {
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_rolloff_std": float(np.std(rolloff)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_bandwidth_std": float(np.std(bandwidth)),
        "zero_crossing_rate_mean": float(np.mean(zcr)),
        "zero_crossing_rate_std": float(np.std(zcr)),
    }


# ------------------------------------------------------------------
# [김승현 담당] 템포 / 리듬 / 크로마 계열
# ------------------------------------------------------------------
def extract_tempo_features(y, sr):
    """
    템포(BPM) 및 리듬 관련 특징 추출.
    - tempo: 곡의 대표 BPM 추정치
    - onset_strength_mean/std: 리듬의 강약 변화 정도 (비트 명확성과 관련)
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
    # librosa 버전에 따라 tempo가 array로 반환될 수 있어 스칼라로 변환
    tempo_value = float(tempo) if np.isscalar(tempo) else float(np.atleast_1d(tempo)[0])

    return {
        "tempo": tempo_value,
        "onset_strength_mean": float(np.mean(onset_env)),
        "onset_strength_std": float(np.std(onset_env)),
    }


def extract_chroma_features(y, sr):
    """
    크로마 특징(12개 음계 클래스별 에너지 분포) 추출.
    장르별 화성적 특징(예: 재즈의 코드 복잡도, 클래식의 조성 등) 구분에 유용.
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features = {}
    for i in range(chroma.shape[0]):
        features[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
        features[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))
    return features


# ------------------------------------------------------------------
# 전체 특징 통합 (곡 1개당 1행)
# ------------------------------------------------------------------
def extract_all_features(file_path, genre, track_id):
    """
    파일 하나에 대해 모든 특징을 추출하고 하나의 딕셔너리로 통합.
    track_id, genre는 이후 곡 단위 train/test split에 사용되므로 반드시 포함.
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    row = {"track_id": track_id, "genre": genre}
    row.update(extract_mfcc_features(y, sr))
    row.update(extract_spectral_features(y, sr))
    row.update(extract_tempo_features(y, sr))
    row.update(extract_chroma_features(y, sr))
    return row


# ------------------------------------------------------------------
# CSV 캐싱을 포함한 전체 파이프라인 실행
# ------------------------------------------------------------------
def load_existing_track_ids():
    """이미 features.csv에 존재하는 track_id를 읽어와 중복 처리를 방지"""
    existing = set()
    if os.path.isfile(FEATURES_CSV):
        with open(FEATURES_CSV, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row["track_id"])
    return existing


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    skip_files = load_skip_files()
    existing_ids = load_existing_track_ids()

    if not os.path.isdir(DATA_DIR):
        print(f"데이터 폴더를 찾을 수 없습니다: {DATA_DIR}")
        return

    genres = sorted(
        d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
    )

    fieldnames = None
    file_exists = os.path.isfile(FEATURES_CSV)
    processed_count = 0
    error_count = 0

    with open(FEATURES_CSV, "a", encoding="utf-8", newline="") as f:
        writer = None

        for genre in genres:
            genre_dir = os.path.join(DATA_DIR, genre)
            wav_files = sorted(
                fn for fn in os.listdir(genre_dir) if fn.endswith(".wav")
            )

            for fn in wav_files:
                if fn in skip_files:
                    continue

                track_id = f"{genre}.{fn}"
                if track_id in existing_ids:
                    continue  # 이미 캐싱된 곡은 재처리하지 않음

                file_path = os.path.join(genre_dir, fn)
                try:
                    row = extract_all_features(file_path, genre, track_id)
                except Exception as e:
                    print(f"[에러] {track_id} 처리 실패: {e}")
                    error_count += 1
                    continue

                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()

                writer.writerow(row)
                processed_count += 1

                if processed_count % 20 == 0:
                    print(f"  진행 중... {processed_count}곡 처리 완료 (최근: {track_id})")

    print(f"\n완료: {processed_count}곡 처리, {error_count}곡 에러")
    print(f"결과 저장 위치: {FEATURES_CSV}")


if __name__ == "__main__":
    main()

    #드디어끝