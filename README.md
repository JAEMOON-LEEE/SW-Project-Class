# MusicLens

GTZAN 음악 장르 데이터로 **오디오에서 특징을 뽑고, 장르를 분류**하는 팀 프로젝트입니다.

처음 보는 사람은 이 순서만 기억하면 됩니다.

1. 음원 파일을 받아서 깨진 곡을 걸러낸다. (Day 1)
2. 각 곡을 숫자 벡터(특징)로 바꾼다. (Day 2)
3. 그 숫자가 괜찮은지 통계로 확인한다. (Day 3)
4. 앞으로 그 숫자로 장르 분류 모델을 학습한다. (예정)

저장소: https://github.com/JAEMOON-LEEE/SW-Project-Class

역할:

- **이재문:** MFCC, 스펙트럴 센트로이드/롤오프, 제로크로싱율
- **김승현:** 템포(BPM), 리듬, 크로마

---

## 데이터셋 (GTZAN)

장르 10개 × 각 100곡 = 원래 1000곡입니다.

blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

Kaggle에서 받습니다 (약 1.2GB):

https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

압축을 풀면 나오는 `genres_original` 폴더를 프로젝트의 `data/` 안에 그대로 넣습니다.

```
MusicLens_Day1_starter/
├── data/
│   └── genres_original/
│       ├── blues/
│       ├── classical/
│       └── ...
├── scripts/
├── outputs/
├── requirements.txt
└── README.md
```

음원(`.wav`)은 용량이 커서 Git에 올리지 않습니다. 각자 위 주소에서 받아 `data/genres_original/`에 두면 됩니다.

잘 알려진 손상 파일: `jazz.00054.wav`  
검사 결과 jazz만 99곡이고, 나머지 장르는 100곡입니다. **전체 999곡**을 사용합니다.

---

## 지금까지 한 일

### Day 1 — 환경, 데이터, EDA (완료)

가상환경과 라이브러리(`librosa`, `pandas`, `scikit-learn` 등)를 맞추고 GTZAN을 받았습니다.

- `scripts/check_corrupted.py`  
  wav를 열어보고 손상 파일을 `outputs/corrupted_files.txt`에 저장합니다.
- `scripts/eda_visualize.py`  
  장르마다 샘플 1곡의 파형과 멜 스펙트로그램을 그립니다.  
  결과는 `outputs/eda_overview.png`입니다.  
  손상 파일 목록이 있으면 그 곡은 자동으로 건너뜁니다.

### Day 2 — 특징 추출 (완료)

곡 하나당 **고정 길이 특징 벡터 1행**을 만듭니다.  
프레임 길이가 곡마다 달라서, 각 특징의 **평균(mean) + 표준편차(std)** 로 요약합니다.

추출하는 것:

| 담당 | 특징 |
|---|---|
| 이재문 | MFCC 20차, 스펙트럴 센트로이드/롤오프/대역폭, 제로크로싱율 |
| 김승현 | 템포, onset(리듬 강도), 크로마 12음 |

- `scripts/extract_features.py`
- 결과: `outputs/features.csv` (999행, 손상 곡 제외)
- 이미 뽑힌 `track_id`는 다시 계산하지 않습니다. (이어서 실행 가능)
- `track_id`와 `genre`를 같이 저장해 두었습니다. 나중에 학습/평가를 **곡 단위**로 나누려고 합니다. 프레임을 섞으면 같은 곡이 train과 test에 들어가 점수가 부풀 수 있습니다.

프로젝트 **루트 폴더**에서 실행해야 합니다.

```bash
python scripts/extract_features.py
```

### Day 3 — 특징 검증, 기술통계, 이상치 (완료)

뽑은 표가 비었는지, 장르별로 몇 곡인지, 값이 얼마나 퍼져 있는지 확인합니다.

- `scripts/analyze_features.py`
- `outputs/feature_summary.csv` — 특징별 평균, 표준편차, 최소/최대 등
- `outputs/outlier_report.csv` — IQR(1.5배) 기준 이상치 개수
- `outputs/mfcc_boxplot_by_genre.png` — 장르별 MFCC 분포

실행 결과 요약:

- 결측치 없음
- jazz 99곡, 나머지 100곡
- 이상치가 많았던 쪽은 주로 **std(곡 안에서의 변화량)**  
  예: `zero_crossing_rate_std`, 고차 `mfcc_*_std`, `chroma_*_std`
- `tempo` 같은 일부 특징은 IQR 이상치 0개

이상치는 “당장 지워야 할 잘못된 파일”이 아닙니다.  
힙합과 클래식처럼 장르 자체가 소리가 달라서, 전체 999곡 기준으로는 바깥값이 나올 수 있습니다.

```bash
python scripts/analyze_features.py
```

---

## 앞으로 할 일

1. **곡 단위 train/test 분할**  
   `track_id` 기준으로 나눕니다. 같은 곡의 정보가 양쪽에 들어가지 않게 합니다.
2. **분류 모델 학습**  
   `features.csv`의 숫자로 10개 장르를 맞춥니다. (예: scikit-learn)
3. **평가와 해석**  
   정확도뿐 아니라 어떤 장르를 헷갈리는지, 어떤 특징이 유용한지 봅니다.
4. **문서/발표 정리**  
   실험 설정, 한계(손상 파일, IQR 이상치의 의미 등)를 적습니다.

---

## 처음 실행하는 사람

Windows 기준입니다. 프로젝트 폴더로 이동한 뒤:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux는 `source venv/bin/activate`를 씁니다.

그다음 GTZAN을 `data/genres_original/`에 넣고, **항상 프로젝트 루트**에서 아래 순서로 실행합니다.

```bash
python scripts/check_corrupted.py
python scripts/eda_visualize.py
python scripts/extract_features.py      # 전체 곡이면 시간이 꽤 걸림
python scripts/analyze_features.py
```

`extract_features.py` / `analyze_features.py`는 경로가 `data/`, `outputs/`처럼 **현재 폴더 기준**입니다. `scripts` 폴더 안에서 실행하면 파일을 못 찾습니다.

---

## 스크립트와 산출물

| 파일 | 하는 일 |
|---|---|
| `scripts/check_corrupted.py` | 손상 wav 찾기 |
| `scripts/eda_visualize.py` | 장르별 파형/스펙트로그램 |
| `scripts/extract_features.py` | 곡 → 특징 표 |
| `scripts/analyze_features.py` | 통계·이상치·박스플롯 |

| 산출물 | 설명 |
|---|---|
| `outputs/corrupted_files.txt` | 손상 파일 경로 |
| `outputs/eda_overview.png` | Day 1 시각화 |
| `outputs/features.csv` | Day 2 특징 표 (로컬 생성, Git 제외) |
| `outputs/feature_summary.csv` | Day 3 기술통계 (로컬 생성, Git 제외) |
| `outputs/outlier_report.csv` | Day 3 이상치 리포트 (로컬 생성, Git 제외) |
| `outputs/mfcc_boxplot_by_genre.png` | 장르별 MFCC 박스플롯 |

`.gitignore` 때문에 `*.csv`와 `data/genres_original/`은 커밋되지 않습니다.  
팀원이 코드를 받은 뒤에는 음원을 넣고 `extract_features.py`를 한 번 돌리면 `features.csv`가 다시 만들어집니다.

---

## 의존성

`requirements.txt`에 있습니다. 핵심은 librosa, numpy, pandas, matplotlib, seaborn, scikit-learn, soundfile, tqdm 입니다.
