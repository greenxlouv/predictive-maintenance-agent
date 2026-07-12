"""
Agent 1 -> Agent 2 연결 확인 스크립트.

실제 weight/데이터 있으면 그대로 쓰고, 없으면 더미로 배선(wiring)만 검증.

실행: python src/test_agent1_agent2_connection.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from agent1_predictor import load_model, predict_with_uncertainty
from agent2_decision import run as agent2_run

WEIGHT_PATH = Path("weights/best_lstm_2d.pth")
X_TEST_PATH = Path("preprocessed/X_test.npy")
Y_TEST_PATH = Path("preprocessed/y_test.npy")


def get_test_window(index=0):
    """실제 C6 test 윈도우 있으면 그거, 없으면 더미."""
    if X_TEST_PATH.exists():
        X_test = np.load(X_TEST_PATH)
        y_true = np.load(Y_TEST_PATH)[index] if Y_TEST_PATH.exists() else None
        print(f"[데이터] 실제 X_test 사용 (index={index})")
        return X_test[index], y_true
    print("[데이터] preprocessed/X_test.npy 없음 -> 랜덤 더미로 대체")
    return np.random.randn(30, 61).astype(np.float32), None


def main():
    # ── Agent 1 ──
    if WEIGHT_PATH.exists():
        model = load_model(str(WEIGHT_PATH))
        print(f"[모델] {WEIGHT_PATH} 로드 완료")
    else:
        raise FileNotFoundError(
            f"{WEIGHT_PATH} 없음 - best_lstm_2d.pth를 weights/ 아래에 넣어주세요"
        )

    x, y_true = get_test_window(index=0)
    agent1_output = predict_with_uncertainty(model, x)

    print("\n=== Agent 1 출력 ===")
    print(agent1_output)
    if y_true is not None:
        print(f"(참고, 실제 RUL: {y_true:.2f})")

    # ── Agent 1 -> Agent 2 스키마 확인 ──
    expected_keys = {"rul_pred", "rul_ci_lower", "rul_ci_upper"}
    assert set(agent1_output.keys()) == expected_keys, (
        f"Agent 1 출력 키 불일치: {agent1_output.keys()} vs {expected_keys}"
    )

    # ── Agent 2 ──
    result = agent2_run(agent1_output)

    print("\n=== Agent 2 출력 ===")
    print(f"판정: {result['action']} (경계선 근처: {result['is_near_boundary']})")
    print(f"설명: {result['explanation']}")


if __name__ == "__main__":
    main()
