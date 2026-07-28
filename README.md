# Predictive Maintenance Agent

[예지 보전 자율 의사결정 AI Agent]

현 진행상황
- Agent1(RUL 예측) → Agent2(판단) → Agent3(정비 계획, RAG) → Agent4(리포트 생성) **전체 연결 완료**
- Agent3/4는 조건부 라우팅으로 연결됨 — 판정이 "위험"/"주의"거나, "정상"이어도 임계값 경계선 근처면 Agent3가 호출되고, 그 외("정상" & 경계선 아님)에는 Agent3를 건너뛰고 바로 Agent4로 감
- Agent1 → CSV 로그 → Streamlit 실시간 대시보드 연동 완료

## 1. 사전 준비 - 별도로 받아야 하는 파일

깃허브에는 용량/보안 문제로 아래 파일들이 올라가 있지 않음
각 로컬에 있는 파일을 아래 위치에 넣어야 함

| 파일 | 넣을 위치 |
|---|---|
| `X_test.npy` | `preprocessed/X_test.npy` |
| `y_test.npy` | `preprocessed/y_test.npy` |
| `cutter_test.npy` | `preprocessed/cutter_test.npy` |
| `best_lstm_2d.pth` | `weights/best_lstm_2d.pth` |

전처리 npy 파일이 없어도 연결 테스트 자체는 랜덤 더미 데이터로 실행 가능함.
`cutter_test.npy`만 없는 경우엔 `data_source.py`가 자동으로 `"c6"`로 대체해서 진행함(이 데이터셋 자체가 전부 C6 test set이라 실제로도 맞는 값임) — 에러는 안 나니 급하게 안 구해도 됨.

## 2. 클론

macOS / Linux (터미널)
```
git clone https://github.com/greenxlouv/predictive-maintenance-agent.git
cd predictive-maintenance-agent
```

Windows (PowerShell 또는 명령 프롬프트)
```
git clone https://github.com/greenxlouv/predictive-maintenance-agent.git
cd predictive-maintenance-agent
```

명령어 자체는 macOS와 Windows가 동일

## 3. 파이썬 환경 준비

이미 conda(Anaconda/Miniconda)를 쓰고 있다면 새 가상환경 없이 base에 설치 가능함
아래는 새 가상환경을 만드는 경우를 가정하여 작성함

macOS / Linux
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell)
```
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Windows (명령 프롬프트, cmd)
```
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

설치 후 확인
```
pip list
```
`torch`, `numpy`, `langgraph`, `anthropic`, `google-generativeai`, `python-dotenv`, `langchain-chroma`, `langchain-huggingface`, `sentence-transformers`, `reportlab`, `streamlit`, `plotly`가 보이면 정상~~

⚠️ **`python`/`pip` 명령이 다른 환경을 가리키는 경우 주의** — 특히 맥에서 `python`이 쉘 alias(예: 시스템 파이썬 3.9.6)로 걸려있는 경우, 가상환경을 활성화해도 `python` 명령이 여전히 alias를 따라가서 엉뚱한 환경으로 실행될 수 있음. `which python`으로 경로에 `venv`가 포함되어 있는지 꼭 확인하고, 안 맞으면 `venv/bin/python`처럼 직접 경로를 지정해서 실행하는 걸 권장.

## 4. 받은 파일 배치

1번에서 전달받은 파일들을 아래처럼 넣어야함

```
predictive-maintenance-agent/
├── weights/
│   └── best_lstm_2d.pth       <- 여기에 넣기
├── preprocessed/
│   ├── X_test.npy             <- 여기에 넣기
│   ├── y_test.npy             <- 여기에 넣기
│   └── cutter_test.npy        <- 여기에 넣기 (없어도 c6로 자동 대체됨)
```

## 5. API 키 설정 (.env 파일)

### 5-1. .env 파일 만들기

프로젝트 루트 폴더(README가 있는 바로 이 위치, src 폴더 안이 아님)에 `.env` 파일을 만들어야 함.
`.env.example` 파일을 복사해서 작성 가능(본인만의 키 넣기 가능)

macOS / Linux
```
cp .env.example .env
```

Windows (PowerShell)
```
Copy-Item .env.example .env
```

Windows (명령 프롬프트)
```
copy .env.example .env
```

### 5-2. .env 파일 열어서 값 채우기

메모장, VSCode 등 아무 텍스트 편집기로 `.env` 파일을 열어서

macOS
```
open -e .env
```
또는
```
code .env
```

Windows
```
notepad .env
```
또는
```
code .env
```

### 5-3. 어떤 값을 넣어야 하는가

`.env` 파일 안에 아래 줄에

```
ANTHROPIC_API_KEY=your_api_key_here
GEMINI_API_KEY=your_api_key_here
LLM_PROVIDER=gemini 혹은 anthropic 혹은 ollama
```

- `LLM_PROVIDER`: Agent2(판단 설명), Agent3(정비 계획 요약), Agent4(최종 리포트) 전부 이 값 하나로 같이 스위칭됨. `anthropic`, `gemini`, `ollama` 중 하나.
- 선택한 provider에 맞는 키만 채우면 됨. 나머지는 빈 값으로 둬도 ㄱㅊ

Anthropic을 쓰는 경우
```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-실제키값
```
키 발급: https://console.anthropic.com 로그인 후 API Keys 메뉴에서 Create Key.
API 호출에는 Anthropic 계정에 크레딧(결제 등록 또는 충전)이 있어야 함. 크레딧이 없으면 `credit balance is too low` 에러 발생함.

Gemini를 쓰는 경우
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=실제키값
```
키 발급: https://aistudio.google.com 에서 Get API key.
학교 이메일(Google Workspace for Education) 계정은 관리자 정책으로 API 접근이 막혀 있는 경우가 있습니다. 개인 Gmail 계정으로 발급받는 것을 권장.

Ollama(로컬, 무료)를 쓰는 경우
```
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b
```
별도 API 키 필요 없음. 대신 로컬에 Ollama 설치와 모델 다운로드가 필요함(대략 15분정도 소요)

macOS
```
brew install ollama
ollama serve
ollama pull qwen2.5:7b
```

Windows
```
winget install Ollama.Ollama
ollama serve
ollama pull qwen2.5:7b
```
Windows는 Ollama 설치 후 앱이 자동으로 백그라운드 서버를 실행하는 경우도 있긴함. `ollama serve` 실행 시 이미 실행 중이라는 메시지가 뜨면 정상!

### 5-4. 주의사항

`.env` 파일은 절대 깃허브에 올리면 안 됨(털려요). `.gitignore`에 이미 등록되어 있어서 `git add .`을 해도 자동으로 제외되지만, 직접 `git add .env`처럼 강제로 추가하지 않도록 확인해주세여

## 6. 매뉴얼 RAG 인덱싱 (Agent3용, 최초 1회)

Agent3가 정비 계획 근거로 참고하는 매뉴얼 8개(`.md`)를 ChromaDB에 인덱싱하는 작업. **`manuals/` 폴더 채워진 뒤, Agent3를 처음 쓰기 전에 한 번만** 실행하면 됨.

```
manuals/
├── 1_위험대응.md
├── 2_주의대응.md
├── 3_정상대응.md
├── 4_모니터링절차.md
├── 5_경계선근처보정규칙.md
├── 6_교체부품정보.md
├── 7_정비절차.md
└── 8_소요시간.md
```

macOS / Linux
```
python src/index_manuals.py --manuals-dir manuals --persist-dir chroma_db
```

Windows
```
python src\index_manuals.py --manuals-dir manuals --persist-dir chroma_db
```

최초 실행 시 로컬 한국어 임베딩 모델(`jhgan/ko-sroberta-multitask`, 약 400MB)을 자동 다운로드하므로 시간이 좀 걸림. `chroma_db/` 폴더가 생기고 "N개 매뉴얼 문서 로드됨" 메시지가 뜨면 정상.

`manuals/`가 아직 준비 안 됐어도 Agent3 실행 자체는 에러 없이 됨 — 그냥 매뉴얼 배경 설명 없이("검색된 배경 설명 없음") 진행됨. 부품/절차/소요시간 같은 확정 수치(`manual_data.py` 기준)는 RAG랑 무관하게 항상 정확히 나옴.

## 7. 실행

### 7-1. Agent1-Agent2 연결 테스트 (가장 먼저 실행)

macOS / Linux
```
python src/test_agent1_agent2_connection.py
```

Windows
```
python src\test_agent1_agent2_connection.py
```

정상 동작 시 아래와 같은 형태로 출력됨.

```
[모델] weights/best_lstm_2d.pth 로드 완료
[데이터] 실제 X_test 사용 (index=0)

=== Agent 1 출력 ===
{'rul_pred': ..., 'rul_ci_lower': ..., 'rul_ci_upper': ...}

=== Agent 2 출력 ===
판정: ...
설명: ...
```

### 7-2. 전체 파이프라인(Agent1→2→3→4) 데모 실행

`main.py`가 이제 Agent3/4까지 전부 실행함. index를 커맨드라인 인자로 넘길 수 있음(기본값 0).

macOS / Linux
```
python main.py 0
```

Windows
```
python main.py 0
```

정상 판정(경계선 아님)이면 Agent3가 자동으로 스킵되고, 위험/주의/정상+경계선근처면 Agent3까지 실행되면서 `reports/` 폴더에 리포트(txt + pdf)가 저장됨.

어떤 index가 위험/주의로 나오는지 미리 알고 싶으면, 아래 스크립트로 전체를 빠르게 스캔해볼 수 있음(LLM 호출 없이 Agent1만 돌려서 가벼움):

```
python src/find_agent3_trigger_indices.py
```

`run_simulation()`(main.py 안, 현재 주석 처리됨)을 켜면 C6 test set(286개) 전체를 순회함 — 이 경우 위험/주의로 판정될 때마다 Agent3/4가 실제 LLM을 호출하니, 처음엔 `main.py <특정 index>`로 한 건씩 확인하는 걸 권장.

### 7-3. 실시간 대시보드 (Agent1 전용, Agent2~4와 무관)

Agent1의 예측 과정을 실시간 그래프로 보고 싶을 때 쓰는 별도 파이프라인. 두 터미널이 필요함.

**터미널 1 — Agent1 스트리밍 실행 (전체 test set 순회하며 CSV에 로그 기록):**
```
python src/run_agent1_stream.py --data-dir preprocessed --delay 0.2
```

**터미널 2 — 대시보드 실행:**
```
python -m streamlit run src/streamlit_dashboard.py
```

두 명령어 다 **프로젝트 루트에서** 실행해야 함(로그 CSV 경로가 상대경로라 실행 위치가 같아야 서로 같은 파일을 봄). 대시보드에는 RUL 예측+신뢰구간, 상태 게이지, 구간별 비율, 예측 보정도, 잔차 분포까지 총 5개 그래프가 실시간으로 채워짐.

## 8. 자주 나는 에러

`ModuleNotFoundError: No module named 'numpy'`
`python`과 `pip`이 서로 다른 환경을 가리키고 있을 때 발생함. `which python`(macOS/Linux) 또는 `where python`(Windows)으로 경로를 확인하고, `pip`이 설치한 환경의 파이썬으로 실행해주십셔. `python -m pip install -r requirements.txt`처럼 `-m pip`을 쓰면 지금 실행 중인 파이썬 기준으로 확실하게 설치됨.

`FileNotFoundError: weights/best_lstm_2d.pth`
4번 단계(받은 파일 배치)를 확인하십셔.

`FileNotFoundError: preprocessed/cutter_test.npy`
있으면 좋지만 없어도 자동으로 "c6"로 대체되어 진행됨(에러 아님). 정확한 파일이 필요하면 팀원한테 요청.

`ModuleNotFoundError: No module named 'langchain_chroma'` (또는 `langchain_huggingface`, `reportlab`)
`requirements.txt`에 있는 패키지가 아직 설치 안 된 것. `pip install -r requirements.txt` 다시 실행.

`RAG 검색 결과 없음` 경고
`manuals/` + `chroma_db/`가 아직 준비 안 된 상태에서 Agent3가 호출된 경우. 에러는 아니고, 6번 단계(매뉴얼 RAG 인덱싱)를 마치면 해결됨.

PDF에서 한글이 네모(□□□)로 깨져 나옴
`save_report_pdf()`가 한글 폰트를 못 찾아서 기본 폰트(Helvetica, 한글 미지원)로 대체된 경우. macOS는 시스템 기본 폰트(AppleGothic)를 자동으로 찾도록 되어 있음. 그래도 안 되면 `AGENT4_FONT_PATH` 환경변수로 직접 폰트 경로를 지정하거나, `assets/fonts/NanumGothic.ttf`에 폰트 파일을 넣어두면 팀원 전체 환경에서 동일하게 렌더링됨.

`anthropic.BadRequestError: credit balance is too low`
Anthropic 계정에 크레딧이 없는 경우, Plans & Billing에서 충전하거나, `.env`의 `LLM_PROVIDER`를 `gemini` 또는 `ollama`로 가능한걸로 바꾸십셔

`google.genai.errors.ClientError: 403 PERMISSION_DENIED`
Gemini API 키/프로젝트 접근이 막힌 경우, 학교 계정이면 개인 Gmail 계정으로 키를 재발급받아 시도하면 됩니당.

`numpy.dtype size changed, may indicate binary incompatibility` 같은 numpy 버전 충돌
conda `base`처럼 이 프로젝트랑 무관한 다른 패키지(tensorflow 등)가 같이 깔린 환경을 쓸 때 numpy 버전이 꼬이면서 발생. 이 프로젝트 전용 가상환경(3번 단계)을 새로 만들어서 격리하는 게 가장 확실한 해결책.
