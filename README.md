# Predictive Maintenance Agent

[예지 보전 자율 의사결정 AI Agent]

현 진행상황
- Agent1(RUL 예측) + Agent2(판단) 파이프라인

## 1. 사전 준비 - 별도로 받아야 하는 파일

깃허브에는 용량/보안 문제로 아래 파일들이 올라가 있지 않음
각 로컬에 있는 파일을 아래 위치에 넣어야 함

| 파일 | 넣을 위치 |
|---|---|
| `X_test.npy` | `preprocessed/X_test.npy` |
| `y_test.npy` | `preprocessed/y_test.npy` |

전처리 npy 파일이 없어도 연결 테스트 자체는 랜덤 더미 데이터로 실행 가능함.

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
`torch`, `numpy`, `langgraph`, `anthropic`, `google-genai`, `python-dotenv`가 보이면 정상~~

## 4. 받은 파일 배치

1번에서 전달받은 파일 3개를 아래처럼 넣어야함

```
predictive-maintenance-agent/
├── weights/
│   └── best_lstm_2d.pth       <- (그냥 넣었어요)
├── preprocessed/
│   ├── X_test.npy             <- 여기에 넣기
│   └── y_test.npy             <- 여기에 넣기
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

`.env` 파일 안에 아래 세 줄에

```
ANTHROPIC_API_KEY=your_api_key_here
GEMINI_API_KEY=your_api_key_here
LLM_PROVIDER=gemini 혹은 anthropic
```

- `LLM_PROVIDER`: 어떤 LLM으로 Agent2 설명을 생성할지 선택. `anthropic`, `gemini`, `ollama` 중 하나.
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

## 6. 실행

### 6-1. Agent1-Agent2 연결 테스트 (가장 먼저 실행)

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

### 6-2. 전체 파이프라인 데모 실행
‼️ 현재 agent 1->2만 연결되어있어서 데모 실행시 아래의 `python src/test_agent1_agent2_connection.py`로 1과 2의 연결 확인해보는게 확실하긴 합니당.

macOS / Linux
```
python main.py
python src/test_agent1_agent2_connection.py
```

Windows
```
python main.py
python src/test_agent1_agent2_connection.py
```

## 7. 자주 나는 에러

`ModuleNotFoundError: No module named 'numpy'`
`python`과 `pip`이 서로 다른 환경을 가리키고 있을 때 발생함. `which python`(macOS/Linux) 또는 `where python`(Windows)으로 경로를 확인하고, `pip`이 설치한 환경의 파이썬으로 실행해주십셔.

`FileNotFoundError: weights/best_lstm_2d.pth`
4번 단계(받은 파일 배치)를 확인하십셔.

`anthropic.BadRequestError: credit balance is too low`
Anthropic 계정에 크레딧이 없는 경우, Plans & Billing에서 충전하거나, `.env`의 `LLM_PROVIDER`를 `gemini` 또는 `ollama`로 가능한걸로 바꾸십셔

`google.genai.errors.ClientError: 403 PERMISSION_DENIED`
Gemini API 키/프로젝트 접근이 막힌 경우, 학교 계정이면 개인 Gmail 계정으로 키를 재발급받아 시도하면 됩니당.
