# MusicLens

유튜브 노래 링크를 넣으면, **그 곡이 어떤 장르인지** 알려주는 프로젝트입니다.

예를 들어 블루스 기타 연주 영상을 넣으면  
`blues 72% / rock 18% / jazz 10%` 처럼 **가능성이 높은 장르 3개**를 보여줍니다.

수업 과목: SW프로젝트응용  
팀: 이재문, 김승현

---

## 한 줄로 이해하기

사람은 노래를 듣고 “이건 재즈 같네”라고 느낍니다.  
컴퓨터는 귀가 없으니, 소리를 **숫자로 바꾼 뒤** 그 숫자 패턴을 보고 장르를 고릅니다.

MusicLens는 그 과정을 웹페이지로 만든 것입니다.

```
유튜브 링크
    → 소리만 내려받기
    → 소리에서 숫자(특징) 뽑기
    → 학습해 둔 모델이 장르 추측
    → 화면에 상위 3개 장르 표시
```

---

## 무엇을 배우게 했나

컴퓨터에게 장르를 가르치려면 **정답이 적힌 노래**가 필요합니다.

우리는 [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) 데이터셋을 썼습니다.

- 장르 10개: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- 각 장르 약 100곡, 곡당 약 30초
- 그중 깨진 파일 `jazz.00054.wav` 1개는 빼고 학습했습니다

이 노래들로 “이런 소리면 이 장르”를 익힌 뒤,  
**처음 보는 유튜브 곡**에도 같은 방식으로 장르를 붙여 봅니다.

---

## 웹앱이 하는 일 (사용자 기준)

1. 브라우저에서 카세트 모양 화면이 열립니다.
2. 유튜브 주소를 붙여넣고 분석을 누릅니다.
3. 영상 제목·썸네일은 미리보기용입니다. 장르 판단에는 쓰지 않습니다.
4. 서버가 실제 소리를 받아 **중간 30초**만 듣고 판단합니다.
5. 가장 비슷한 장르 3개와 확신 정도(%)를 보여줍니다.

> 왜 30초만 듣나요?  
> 학습에 쓴 노래가 원래 30초짜리이고, 영상 전체를 받으면 너무 오래 걸리기 때문입니다.  
> 그래서 **인트로·광고를 건너뛴 뒤의 30초**를 사용합니다. 곡 전체의 분위기와 다를 수는 있습니다.

---

## 컴퓨터는 소리를 어떻게 숫자로 바꾸나

어려운 용어를 일상 말로 바꾸면 이렇습니다.

| 우리가 뽑은 값 | 쉬운 설명 |
|---|---|
| MFCC | 목소리·악기가 “어떤 음색인지”를 숫자로 요약한 것 |
| 스펙트럴 특징 | 소리가 높은음 위주인지, 낮은음 위주인지 |
| 템포 | 박자가 빠른지 느린지 (BPM에 가까운 값) |
| 크로마 | 어떤 음정(도레미…)이 많이 나오는지 |

한 곡을 이 값들로 요약하면 **숫자 77개**가 됩니다.  
모델은 이 77개 숫자를 보고 10개 장르 중 하나를 고릅니다.

---

## 어떤 모델을 골랐나

같은 시험 문제(안 보여 준 노래 200곡)로 여러 방법을 비교했습니다.

| 방법 | 맞춘 비율 | 한 줄 설명 |
|---|---|---|
| **SVM (linear)** | **74%** | 숫자 77개를 보고 경계를 긋는 방법. **최종 채택** |
| SVM (rbf) | 71.5% | 경계를 더 구불구불하게 긋는 변형 |
| Random Forest | 71.5% | 작은 결정나무 여러 개의 투표 |
| CNN | 67.5% | 소리를 그림(멜스펙트로그램)처럼 보고 학습 |

10개 장르를 **아무거나 찍으면 약 10%**이므로, 74%는 “꽤 맞히지만 완벽하지는 않다” 수준입니다.  
록과 블루스처럼 비슷한 장르는 서로 헷갈리기 쉽습니다.

웹앱은 이 최종 모델(SVM linear)을 사용합니다.

---

## 폴더 안내

```
MusicLens_Day1_starter/
├── app.py                 웹앱 서버
├── templates/index.html   화면 (카세트 UI)
├── README.md              지금 보고 있는 설명서
├── README_WEBAPP.md       웹앱 설치·실행 상세 가이드
├── requirements.txt       필요한 파이썬 패키지 목록
├── scripts/               데이터 검사, 학습, 비교 코드
├── models/                학습이 끝난 모델 파일
├── outputs/               그래프, 성적표, 분석 결과
└── data/                  원본 노래(용량 커서 GitHub에는 안 올림)
```

원본 wav는 GitHub에 올리지 않습니다.  
쓰려면 Kaggle에서 받아 `data/genres_original/` 아래에 장르 폴더를 두면 됩니다.

---

## 우리가 한 일 (일정)

| 단계 | 무엇을 했나 |
|---|---|
| 1일차 | 환경 만들기, 데이터 받기, 깨진 파일 찾기, 장르별 파형 그려 보기 |
| 2일차 | 소리를 숫자로 바꾸는 함수 작성 (이재문: 음색·주파수 / 김승현: 박자·음정) |
| 3일차 | 숫자 테이블 통계·이상치 확인 |
| 4일차 | SVM, 랜덤포레스트로 첫 모델 학습 |
| 5일차 | 소리를 그림처럼 만든 멜스펙트로그램 준비 |
| 6일차 | CNN 학습 (그림으로 장르 맞히기) |
| 7일차 | 네 모델을 같은 조건으로 비교하고 최종 모델 선정 |
| 8일차 | 유튜브 링크를 넣는 웹앱 연결 |

---

## 웹앱만 실행해 보기

자세한 설치는 [README_WEBAPP.md](README_WEBAPP.md)에 있습니다. 요약만 적습니다.

필요한 것:

- Python (이 팀은 conda 환경 `nlp` 사용)
- ffmpeg (유튜브 소리를 wav로 바꿀 때 필요)
- yt-dlp (유튜브 다운로드)

프로젝트 폴더에서:

```powershell
conda activate nlp
pip install -r requirements.txt

$env:YTDLP_PATH = "C:\Users\사용자명\anaconda3\envs\ytdlp\Scripts\yt-dlp.exe"
$env:FFMPEG_LOCATION = "C:\ffmpeg\bin"
python app.py
```

브라우저에서 http://127.0.0.1:5000 을 엽니다.

참고:

- 이 팀의 학습 환경은 Python 3.8이라, 최신 yt-dlp는 별도 환경(`ytdlp`, Python 3.11)에 설치했습니다.
- 비공개·지역 제한 영상은 다운로드가 실패할 수 있습니다.

---

## 처음부터 다시 학습하려면

원본 데이터가 `data/genres_original/`에 있어야 합니다. 프로젝트 루트에서 순서대로:

```powershell
python scripts/check_corrupted.py
python scripts/eda_visualize.py
python scripts/extract_features.py
python scripts/analyze_features.py
python scripts/train_baseline.py
python scripts/generate_melspec.py
python scripts/train_cnn.py
python scripts/compare_models.py
python scripts/finalize_svm_linear.py
```

웹앱이 쓰는 파일이 최종 모델과 같은지 확인:

```powershell
python scripts/check_model_identity.py
```

---

## 한계 (솔직히)

- **유튜브 곡 전체가 아니라 30초**만 듣습니다.
- 학습 데이터는 오래된 서양 장르 중심이라, 최신 K-pop·OST·믹스장르는 약할 수 있습니다.
- 74%는 과제용 시작점이지, 상용 음악앱 수준은 아닙니다.
- CNN이 항상 더 좋은 것은 아니었습니다. 데이터가 작을 때는 단순한 모델이 더 나았습니다.

---

## 라이선스·데이터

코드는 수업 과제용입니다.  
GTZAN 데이터와 유튜브 음원은 각 저작권 조건을 따릅니다. 개인적인 실험 외에 음원을 재배포하지 마세요.
