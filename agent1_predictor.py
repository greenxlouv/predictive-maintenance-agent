"""
Agent 1 - RUL 예측 Agent (Predictor)

기존 2-D 모델(hidden=64, layers=2, dropout=0.2, epochs=300,
Train Loss early stop, patience=30) 그대로 사용.
MC Dropout으로 100회 반복 추론 -> 평균(rul_pred) + 5th/95th percentile(CI) 산출.

[Agent 2로 전달되는 출력 - 기존과 동일, 변경 없음]
    {"rul_pred": float, "rul_ci_lower": float, "rul_ci_upper": float}
    (전부 0.0 클리핑, 단위: cut count)
    Agent 2의 score_action()이 rul_ci_lower를 보수적 판단 기준으로 사용함.

[신규 추가 - Agent 2와 무관, 대시보드 전용]
    run_stream(): X_test/y_test/cutter_test를 순회하며 한 윈도우씩 예측 →
    cutter_id, cut 순번, 실제 RUL, 예측 RUL, CI를 CSV 로그에 "한 행씩 append".
    이 로그 파일을 streamlit_dashboard.py 가 주기적으로 읽어서
    실시간처럼 그래프를 갱신함 (Agent 1 실행 = producer, 대시보드 = consumer,
    둘은 완전히 분리된 프로세스).
"""

import csv
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

# MPS도 실행 가능하게
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"device: {device}")

weight_path = os.environ.get(
    "AGENT1_WEIGHT_PATH",
    "/Users/bluecloud/workspace_vscode/vscode_research/agent1_pth/best_lstm_2d.pth",
)

# 위험 구간 임계값 (Agent 2의 score_action()과 동일한 값 - y_test.npy percentile 기준 산출)
# Agent 2와 대시보드가 이 상수 하나만 참조하도록 해서, 임계값을 바꿀 때 한 곳만 고치면 됨.
# (Agent 2 쪽 코드에서도 자체 상수 대신 `from agent1_predictor import RUL_THRESHOLDS`로 바꾸는 걸 권장)
RUL_THRESHOLDS = {"RED": 20.7, "YELLOW": 50.25}

# 대시보드가 읽는 로그 파일 경로 (환경변수로 덮어쓰기 가능)
LOG_PATH = os.environ.get("AGENT1_LOG_PATH", "agent1_stream_log.csv")
LOG_FIELDS = [
    "step",          # 처리 순번 (전체, 실시간 스트리밍 순서)
    "cutter_id",     # c1 / c4 / c6 ...
    "cutter_step",   # 해당 cutter 내에서의 순번 (x축으로 사용)
    "rul_true",      # 실제 RUL (있으면), 없으면 빈 값
    "rul_pred",
    "rul_ci_lower",
    "rul_ci_upper",
    "timestamp",     # ISO8601 UTC
]


# 1. 모델 정의 (기존 2-D 세팅 그대로) ──────────────────
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=61, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)  # MC Dropout용
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


# 2. 모델 로드
def load_model(weight_path=weight_path, input_size=61):
    model = LSTMPredictor(input_size=input_size).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model


# 3. MC Dropout 기반 예측 (기존 로직 그대로)
def predict_with_uncertainty(model, x, n_iter=100):
    # x: 단일 윈도우, shape (30, 61) 또는 (1, 30, 61).
    model.train()  # dropout 활성화 (BatchNorm 없어서 train()으로도 문제 없음)

    if x.ndim == 2:
        x = x[np.newaxis, ...]
    x_t = torch.tensor(x, dtype=torch.float32).to(device)

    preds = []
    with torch.no_grad():
        for _ in range(n_iter):
            preds.append(model(x_t).cpu().numpy())
    preds = np.stack(preds)  # (n_iter, 1)

    rul_pred = float(np.clip(preds.mean(), 0.0, None))
    ci_lower = float(np.clip(np.percentile(preds, 5), 0.0, None))
    ci_upper = float(np.clip(np.percentile(preds, 95), 0.0, None))

    return {
        "rul_pred": rul_pred,
        "rul_ci_lower": ci_lower,
        "rul_ci_upper": ci_upper,
    }


# 4. Agent 1 실행 지점 (LangGraph 노드에서 호출) - 기존과 동일, Agent 2로 그대로 전달
def run(x, model=None):
    """단일 윈도우 예측. Agent 2에 전달되는 값은 이 함수 반환값 그대로.
    LangGraph 노드에서 지금처럼 계속 이 함수만 호출하면 됨 (변경 없음)."""
    if model is None:
        model = load_model()
    return predict_with_uncertainty(model, x)


# ──────────────────────────────────────────────────────────────
# 5. [신규] 스트리밍 배치 실행 + 실시간 로그 기록 (대시보드 전용, Agent 2와 무관)
# ──────────────────────────────────────────────────────────────
def _init_log(log_path, reset_log):
    """로그 파일을 새로 만들거나(reset_log=True) 이어쓰기 위해 헤더만 확인."""
    file_exists = os.path.exists(log_path)
    if reset_log or not file_exists:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()


def _append_log(log_path, row):
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writerow(row)


def run_stream(
    X,
    cutter_ids,
    y=None,
    model=None,
    log_path=LOG_PATH,
    n_iter=100,
    delay=0.0,
    reset_log=True,
):
    """X (N, 30, 61) 전체를 한 윈도우씩 순회하며 예측하고,
    Agent 2 호출과 완전히 분리된 로그 파일(log_path)에 한 행씩 append.

    - cutter_ids: (N,) 배열, 각 윈도우의 cutter ID (cutter_test.npy 등)
    - y: (N,) 실제 RUL (y_test.npy 등). 없으면 None으로 두면 rul_true는 빈 값.
    - delay: 한 스텝마다 sleep할 시간(초). 실시간 모니터링을 흉내내고 싶을 때
      0보다 큰 값(예: 0.5)을 주면 됨. 실제 운영에서는 0으로 두고 센서 도착
      이벤트가 들어올 때마다 이 루프의 한 스텝만 호출하는 형태로 바꾸면 됨.
    - reset_log: True면 실행 시작 시 로그 파일을 새로 씀(처음부터 다시 스트리밍).
      False면 기존 로그 뒤에 이어서 append (중단 후 재개, 실 서비스 누적 등에 사용).

    반환값: 처리한 모든 행의 리스트 (dict). 대시보드는 이 반환값을 안 봐도 되고,
    log_path 파일만 폴링해서 그리면 됨.
    """
    if model is None:
        model = load_model()

    cutter_ids = np.asarray(cutter_ids)
    _init_log(log_path, reset_log)

    # cutter별 순번 카운터 (x축 = 해당 cutter 안에서 몇 번째 컷인지)
    cutter_step_counter = {}
    rows = []

    for step in range(len(X)):
        cutter_id = str(cutter_ids[step])
        cutter_step_counter[cutter_id] = cutter_step_counter.get(cutter_id, 0) + 1

        pred = predict_with_uncertainty(model, X[step], n_iter=n_iter)

        row = {
            "step": step,
            "cutter_id": cutter_id,
            "cutter_step": cutter_step_counter[cutter_id],
            "rul_true": float(y[step]) if y is not None else "",
            "rul_pred": pred["rul_pred"],
            "rul_ci_lower": pred["rul_ci_lower"],
            "rul_ci_upper": pred["rul_ci_upper"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _append_log(log_path, row)  # 한 스텝 끝날 때마다 바로 기록 -> 대시보드가 실시간으로 읽음
        rows.append(row)

        if delay > 0:
            time.sleep(delay)

    return rows


# 6. 실행 예시
if __name__ == "__main__":
    dummy_x = np.random.randn(30, 61).astype(np.float32)
    result = run(dummy_x)
    print(f"rul_pred={result['rul_pred']:.2f}, "
          f"ci_lower={result['rul_ci_lower']:.2f}, "
          f"ci_upper={result['rul_ci_upper']:.2f}")
