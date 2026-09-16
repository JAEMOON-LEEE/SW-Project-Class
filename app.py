"""
MusicLens - Day 8: Flask 웹앱
=================================

유튜브 링크 -> 오디오 다운로드 -> 특징 추출(extract_features.py와 동일 로직)
-> 저장된 베이스라인 모델(models/baseline_model.pkl)로 장르 예측

주의:
- 특징 추출 함수는 scripts/extract_features.py 의 로직을 그대로 가져왔습니다.
  extract_features.py를 수정했다면(예: 특징 추가) 아래 함수들도 반드시 같이 수정해야
  train_baseline.py에서 학습한 스케일러/모델과 입력이 어긋나지 않습니다.
- Day 7에서 최종 모델을 CNN 등으로 교체하더라도, models/baseline_model.pkl 자리를
  덮어쓰기만 하면 이 app.py는 그대로 동작하도록 설계했습니다(모델 종류에 따라
  predict_proba 부분만 아래 NOTE 참고해서 손보면 됩니다).

실행 방법 (프로젝트 루트에서, models/ outputs/ 와 같은 위치):
    python app.py
    -> http://127.0.0.1:5000 접속

필수 조건:
    - models/baseline_model.pkl, models/scaler.pkl, models/model_metadata.pkl 존재
      (scripts/train_baseline.py 실행 결과물)
    - 시스템에 ffmpeg 설치 (yt-dlp가 오디오 추출 시 사용)
"""

import os
import glob
import tempfile
import subprocess

import numpy as np
import librosa
import joblib
from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# yt-dlp 실행 파일 경로
# ---------------------------------------------------------------------------
# (nlp) conda 환경의 Python 3.8은 최신 yt-dlp(3.9+ 요구)를 설치할 수 없어서,
# 별도의 conda 환경(예: `conda create -n ytdlp python=3.11`)에 최신 yt-dlp를
# 설치하고, 그 실행 파일을 subprocess로 직접 호출하는 방식으로 우회합니다.
#
# 아래 환경변수로 경로를 지정하세요 (PowerShell 예시):
#   $env:YTDLP_PATH = "C:\Users\사용자명\anaconda3\envs\ytdlp\Scripts\yt-dlp.exe"
#   python app.py
#
# 환경변수를 안 정할 경우, PATH에 등록된 "yt-dlp" 명령어를 그대로 사용합니다.
YTDLP_PATH = os.environ.get("YTDLP_PATH", "yt-dlp")

# conda-forge의 ffmpeg 패키지는 Windows에서 설치가 자주 깨지므로(librsvg/gdk-pixbuf
# post-link 오류), 대신 정적 빌드(zip)를 받아 폴더에 풀어두고 아래 환경변수로
# 경로를 직접 지정하는 방식을 씁니다.
#   $env:FFMPEG_LOCATION = "C:\ffmpeg\bin"
# 지정하지 않으면 시스템 PATH에서 ffmpeg를 찾습니다.
FFMPEG_LOCATION = os.environ.get("FFMPEG_LOCATION")

# ---------------------------------------------------------------------------
# 경로 설정 (train_baseline.py / extract_features.py와 동일한 규칙)
# ---------------------------------------------------------------------------
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "baseline_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.pkl")

SAMPLE_RATE = 22050    # extract_features.py의 SAMPLE_RATE와 반드시 동일해야 함
CLIP_OFFSET_SEC = 30    # 인트로/광고 구간을 스킵
CLIP_DURATION_SEC = 30  # GTZAN 원곡 길이(30초)와 맞춰서 특징 분포를 비슷하게 유지

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 모델 로드 (서버 시작 시 1회만 로드)
# ---------------------------------------------------------------------------
print("[app.py] 모델 로드 중...")
for path in (MODEL_PATH, SCALER_PATH, METADATA_PATH):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} 를 찾을 수 없습니다. 먼저 scripts/train_baseline.py 를 "
            "프로젝트 루트에서 실행해서 모델을 생성해주세요."
        )

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
metadata = joblib.load(METADATA_PATH)

FEATURE_COLS = metadata["feature_cols"]
MODEL_NAME = metadata.get("best_model_name", "Baseline Model")
STAGE = metadata.get("stage", "baseline")

# NOTE: SVC(probability=True) / RandomForestClassifier는 predict_proba를 지원하고,
# model.classes_ 순서대로 확률을 반환합니다. 만약 Day 7에서 CNN(.h5)으로 교체한다면
# 이 부분과 아래 predict() 안의 model.predict_proba(...) 호출부를 CNN 추론 코드로
# 바꿔주면 됩니다 (keras model.predict()는 이미 클래스 순서대로 확률을 반환하므로
# CLASS_ORDER를 metadata["labels"]로 대체하면 됩니다).
CLASS_ORDER = list(model.classes_)

print(f"[app.py] 모델: {MODEL_NAME} (stage={STAGE}) / 특징 {len(FEATURE_COLS)}개 / 장르: {CLASS_ORDER}")


# ---------------------------------------------------------------------------
# 특징 추출 함수 (scripts/extract_features.py 와 동일한 로직 — 반드시 동기화 유지)
# ---------------------------------------------------------------------------
def extract_mfcc_features(y, sr, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features = {}
    for i in range(n_mfcc):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
    return features


def extract_spectral_features(y, sr):
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


def extract_tempo_features(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
    tempo_value = float(tempo) if np.isscalar(tempo) else float(np.atleast_1d(tempo)[0])
    return {
        "tempo": tempo_value,
        "onset_strength_mean": float(np.mean(onset_env)),
        "onset_strength_std": float(np.std(onset_env)),
    }


def extract_chroma_features(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features = {}
    for i in range(chroma.shape[0]):
        features[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
        features[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))
    return features


def extract_all_features_from_audio(y, sr):
    row = {}
    row.update(extract_mfcc_features(y, sr))
    row.update(extract_spectral_features(y, sr))
    row.update(extract_tempo_features(y, sr))
    row.update(extract_chroma_features(y, sr))
    return row


# ---------------------------------------------------------------------------
# 유튜브 오디오 다운로드
# ---------------------------------------------------------------------------
def download_audio(youtube_url, out_dir):
    outtmpl = os.path.join(out_dir, "audio.%(ext)s")
    cmd = [
        YTDLP_PATH,
        "-x",                          # 오디오만 추출
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--no-playlist",
        "-o", outtmpl,
    ]
    if FFMPEG_LOCATION:
        cmd += ["--ffmpeg-location", FFMPEG_LOCATION]
    cmd.append(youtube_url)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"yt-dlp 실행 파일을 찾을 수 없습니다 ('{YTDLP_PATH}'). "
            "YTDLP_PATH 환경변수가 올바른 경로를 가리키는지 확인해주세요."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("오디오 다운로드가 시간 초과되었습니다(120초).")

    if result.returncode != 0:
        # yt-dlp가 stderr에 에러 메시지를 남기므로 그대로 보여줌
        raise RuntimeError(result.stderr.strip()[-500:] or "알 수 없는 yt-dlp 오류")

    wav_files = glob.glob(os.path.join(out_dir, "audio.*"))
    if not wav_files:
        raise RuntimeError("오디오 파일 생성에 실패했습니다.")
    return wav_files[0]


# ---------------------------------------------------------------------------
# 라우트
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    youtube_url = (data.get("url") or "").strip()
    if not youtube_url:
        return jsonify({"error": "url이 필요합니다."}), 400

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1) 오디오 다운로드
        try:
            audio_path = download_audio(youtube_url, tmp_dir)
        except Exception as e:
            return jsonify({"error": f"오디오 다운로드 실패: {e}"}), 502

        # 2) 오디오 로드 (30초 구간, 너무 짧은 영상이면 처음부터)
        try:
            y, sr = librosa.load(
                audio_path, sr=SAMPLE_RATE, offset=CLIP_OFFSET_SEC, duration=CLIP_DURATION_SEC
            )
            if librosa.get_duration(y=y, sr=sr) < 5:
                y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=CLIP_DURATION_SEC)
        except Exception as e:
            return jsonify({"error": f"오디오 로드 실패: {e}"}), 500

        # 3) 특징 추출 (학습 때와 동일한 컬럼 순서로 정렬)
        try:
            feat_row = extract_all_features_from_audio(y, sr)
            X = np.array([[feat_row[col] for col in FEATURE_COLS]])
        except KeyError as e:
            return jsonify({
                "error": f"특징 컬럼이 학습 시점과 다릅니다({e}). "
                         "extract_features.py가 변경되었다면 app.py의 특징 추출 함수도 갱신해주세요."
            }), 500
        except Exception as e:
            return jsonify({"error": f"특징 추출 실패: {e}"}), 500

        # 4) 스케일링 + 예측
        try:
            X_scaled = scaler.transform(X)
            probs = model.predict_proba(X_scaled)[0]
        except Exception as e:
            return jsonify({"error": f"예측 실패: {e}"}), 500

    ranked = sorted(zip(CLASS_ORDER, probs), key=lambda x: x[1], reverse=True)
    top3 = ranked[:3]
    genres = [{"name": name, "confidence": round(float(p) * 100)} for name, p in top3]

    top_name, top_p = top3[0]
    reasoning = (
        f"{MODEL_NAME} 모델이 MFCC·스펙트럴·템포·크로마 특징을 분석한 결과, "
        f"'{top_name}' 장르일 확률을 {round(float(top_p) * 100)}%로 가장 높게 추정했어요. "
        f"(영상 중 30초 구간의 실제 오디오를, GTZAN 데이터셋으로 학습한 모델에 넣어 얻은 결과예요.)"
    )

    return jsonify({
        "genres": genres,
        "reasoning": reasoning,
        "model": MODEL_NAME,
        "stage": STAGE,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)