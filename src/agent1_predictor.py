"""
Agent 1 - RUL 예측 Agent (Predictor)

기존 2-D 모델(hidden=64, layers=2, dropout=0.2, epochs=300,
Train Loss early stop, patience=30) 그대로 사용.
MC Dropout으로 100회 반복 추론 -> 평균(rul_pred) + 5th/95th percentile(CI) 산출.

출력: {"rul_pred": float, "rul_ci_lower": float, "rul_ci_upper": float}
     (전부 0.0 클리핑, 단위: cut count)
     Agent 2의 score_action()이 rul_ci_lower를 보수적 판단 기준으로 사용함.
"""

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

weight_path="/Users/bluecloud/workspace_vscode/vscode_research/agent1_pth/best_lstm_2d.pth"


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
def load_model(weight_path=weight_path,input_size=61,):
    model = LSTMPredictor(input_size=input_size).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model

# 3. MC Dropout 기반 예측
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


# 4. Agent 1 실행 지점 (LangGraph 노드에서 호출)
def run(x, model=None):
    if model is None:
        model = load_model()
    return predict_with_uncertainty(model, x)


# 5. 실행 예시
if __name__ == "__main__":
    dummy_x = np.random.randn(30, 61).astype(np.float32)
    result = run(dummy_x)
    print(f"rul_pred={result['rul_pred']:.2f}, "
          f"ci_lower={result['rul_ci_lower']:.2f}, "
          f"ci_upper={result['rul_ci_upper']:.2f}")