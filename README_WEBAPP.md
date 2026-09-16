# MusicLens 웹앱 — 설치·실행 가이드

프로젝트 소개는 [README.md](README.md)를 먼저 보세요.  
이 문서는 **웹앱을 내 컴퓨터에서 켜는 방법**만 자세히 적습니다.

## 1. 파일 배치

프로젝트 루트(`MusicLens_Day1_starter`) 기준으로 아래처럼 배치하세요.
`models/`, `outputs/` 폴더는 이미 갖고 계신 것과 같은 위치에 두면 됩니다.

```
MusicLens_Day1_starter/
├── app.py                  <- 이번에 받은 파일
├── templates/
│   └── index.html          <- 이번에 받은 파일
├── requirements.txt        <- 이번에 받은 파일
├── models/
│   ├── baseline_model.pkl  <- train_baseline.py 실행 결과물 (이미 있음)
│   ├── scaler.pkl
│   └── model_metadata.pkl
├── outputs/
├── scripts/
│   ├── extract_features.py
│   └── train_baseline.py
└── data/
```

`(nlp)` conda 환경 기준으로 실행하시면 됩니다 (VS Code ▶ 버튼 말고 터미널에서 직접).

## 2. 패키지 설치

```bash
conda activate nlp
pip install -r requirements.txt
```

librosa, scikit-learn, joblib, numpy는 이미 설치되어 있을 가능성이 높습니다.
새로 필요한 건 `flask`, `yt-dlp`, `soundfile` 정도예요.

## 3. yt-dlp 별도 환경 설치 (중요)

`(nlp)` 환경이 Python 3.8이라 최신 yt-dlp(Python 3.9+ 요구)를 설치할 수 없습니다.
librosa/TensorFlow가 깔린 `(nlp)` 환경은 건드리지 않고, yt-dlp 전용 환경을 새로 만드세요.

```powershell
conda create -n ytdlp python=3.11 -y
conda activate ytdlp
pip install -U yt-dlp
yt-dlp --version          # 2025~2026년대 버전인지 확인
where yt-dlp               # 실행 파일 경로 확인 (예: ...\envs\ytdlp\Scripts\yt-dlp.exe)
```

확인한 경로를 환경변수로 지정한 뒤 `(nlp)` 환경에서 `app.py`를 실행합니다.

```powershell
conda activate nlp
$env:YTDLP_PATH = "C:\Users\사용자명\anaconda3\envs\ytdlp\Scripts\yt-dlp.exe"
$env:FFMPEG_LOCATION = "C:\ffmpeg\bin"
python app.py
```

(매번 새 터미널을 열 때마다 `$env:YTDLP_PATH` 를 다시 지정해야 합니다. 매번 치기 귀찮으면
PowerShell 프로필에 추가하거나, `[Environment]::SetEnvironmentVariable(...)`로 영구 등록하세요.)

`YTDLP_PATH`를 지정하지 않으면 `app.py`는 그냥 `yt-dlp`라는 이름으로 PATH에서 찾으려고 시도합니다.

## 3-1. ffmpeg 설치 (conda 대신 정적 빌드 권장)

conda-forge의 ffmpeg 패키지는 Windows에서 설치가 자주 깨집니다(librsvg 오류 등).
아래처럼 미리 빌드된 실행 파일을 직접 받는 걸 권장합니다.

1. https://www.gyan.dev/ffmpeg/builds/ 에서 **release essentials** zip 다운로드
2. `C:\ffmpeg\bin\ffmpeg.exe` 경로가 되도록 압축 해제
3. `(nlp)` 환경에서 `app.py` 실행하기 전에 환경변수 지정:

```powershell
$env:FFMPEG_LOCATION = "C:\ffmpeg\bin"
```

(지정하지 않으면 시스템 PATH에서 ffmpeg를 찾으려고 시도합니다.)

## 4. 실행

프로젝트 루트에서:

```bash
python app.py
```

터미널에 아래처럼 모델 로드 로그가 뜨면 정상입니다.

```
[app.py] 모델: SVM (linear) (stage=final) / 특징 77개 / 장르: ['blues', 'classical', ...]
```

브라우저에서 http://127.0.0.1:5000 접속하면 카세트 UI가 뜹니다.

## 5. 동작 방식 요약

1. 유튜브 링크 입력 → 프론트에서 oEmbed로 제목/채널/썸네일만 가져옴 (분류에는 안 씀)
2. `/predict`로 링크 전송 → 서버에서:
   - yt-dlp로 오디오 다운로드
   - librosa로 30초 구간 로드 (SAMPLE_RATE=22050, 앞 30초는 스킵)
   - `extract_features.py`와 동일한 로직으로 특징 77개 추출
   - `scaler.pkl`로 스케일링 → `baseline_model.pkl`(최종 모델: SVM linear)로 `predict_proba`
   - 상위 3개 장르 + 확률을 JSON으로 응답
3. 프론트에서 VU미터 스타일로 렌더링

## 6. Day 7에서 CNN으로 최종 모델을 교체하는 경우

`models/baseline_model.pkl` 자리를 최종 모델로 덮어쓰는 방식이라면, 모델이 `predict_proba`를
지원하는 sklearn 계열(SVM, RF 등)일 때는 `app.py` 수정 없이 그대로 동작합니다.

CNN(.h5, Keras)으로 교체한다면 `app.py`에서 아래 두 부분만 손보면 됩니다.

1. 모델 로드부: `joblib.load(MODEL_PATH)` → `tensorflow.keras.models.load_model(...)`
2. `/predict` 안의 입력 전처리: 77개 수치 특징 대신 멜스펙트로그램 이미지를 만들어야 하므로,
   `scripts/generate_melspec.py`의 멜스펙 생성 로직을 가져와야 합니다. 이 경우 알려주시면
   해당 버전도 만들어드릴게요.

## 7. 주의사항

- 저작권 보호(비공개, 지역 제한 등)된 영상은 다운로드가 막혀 있을 수 있어요.
  그럴 경우 `/predict`가 502 에러를 반환하고, 프론트에 에러 메시지가 표시됩니다.
- 예측은 영상 전체가 아니라 **30초 구간**만 사용합니다 (GTZAN 원본 곡 길이와 맞추고,
  처리 시간을 줄이기 위함). 보고서에 이 부분을 방법론 한계로 언급하면 좋습니다.