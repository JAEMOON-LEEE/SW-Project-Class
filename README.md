# MusicLens 프로젝트 (Day 1: 환경 세팅 + 데이터 확보 + EDA)

## 1. 가상환경 세팅

```bash
cd MusicLens
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## 2. GTZAN 데이터셋 다운로드

Kaggle 계정 로그인 후 아래에서 다운로드 (약 1.2GB):
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

압축을 풀면 `Data/genres_original/` 폴더 안에 blues, classical, country, disco,
hiphop, jazz, metal, pop, reggae, rock 10개 하위 폴더(각 100곡, .wav)가 있습니다.

이 `genres_original` 폴더를 통째로 이 프로젝트의 `data/` 안에 넣어서
아래 구조가 되도록 맞춰주세요.

```
MusicLens/
├── data/
│   └── genres_original/
│       ├── blues/
│       ├── classical/
│       ├── ...
├── scripts/
│   ├── check_corrupted.py
│   └── eda_visualize.py
├── outputs/
├── requirements.txt
└── README.md
```

> 참고: jazz.00054.wav 파일이 손상되어 있는 것으로 널리 알려져 있습니다.
> `check_corrupted.py`로 먼저 확인하고 넘어가세요.

## 3. 실행 순서

```bash
# 1) 손상 파일 확인 (약 1~2분 소요)
python scripts/check_corrupted.py

# 2) 장르별 파형/스펙트로그램 시각화
python scripts/eda_visualize.py
```

결과 이미지는 `outputs/eda_overview.png`에 저장됩니다.

## 4. Day 1 체크리스트

- [ ] venv 생성 및 라이브러리 설치 완료
- [ ] GTZAN 다운로드 및 `data/genres_original/`에 배치
- [ ] `check_corrupted.py` 실행, 손상 파일 목록 확보 (팀 공유)
- [ ] `eda_visualize.py` 실행, 장르별 파형/스펙트로그램 결과 확인
- [ ] Git repo에 커밋 (data/ 폴더는 .gitignore로 제외됨)
