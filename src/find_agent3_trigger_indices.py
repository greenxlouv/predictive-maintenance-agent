"""
find_agent3_trigger_indices.py

X_test 전체를 순회하며 각 인덱스의 action/is_near_boundary를 찍어서,
Agent3가 실제로 호출되는 인덱스(위험/주의/정상+경계선근처)를 찾아준다.
감으로 index 값을 올려가며 main.py의 run_demo()를 반복 실행하는 것보다
훨씬 빠름 — 모델 추론만 한 번 돌고 LLM은 호출 안 하니까 가벼움.

실행: python src/find_agent3_trigger_indices.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_source import load_c6_windows
from agent1_predictor import load_model, predict_with_uncertainty
from agent2_decision import score_action, is_near_boundary


def main():
    model = load_model()
    X_test, y_test, cutter_test = load_c6_windows()
    print(f"총 {len(X_test)}개 윈도우 (cutter_test.npy shape={cutter_test.shape})\n")

    trigger_indices = []
    for i in range(len(X_test)):
        pred = predict_with_uncertainty(model, X_test[i])
        state = {
            "rul_pred": pred["rul_pred"],
            "rul_ci_lower": pred["rul_ci_lower"],
            "rul_ci_upper": pred["rul_ci_upper"],
        }
        action = score_action(state)
        near = is_near_boundary(state)
        agent3_fires = action in ("위험", "주의") or (action == "정상" and near)

        mark = "  <- Agent3 호출됨" if agent3_fires else ""
        print(f"[{i:>3}] y_true={y_test[i]:>6.1f}  ci_lower={pred['rul_ci_lower']:>7.2f}  "
              f"action={action}{mark}")

        if agent3_fires:
            trigger_indices.append(i)

    print(f"\n=== Agent3가 호출되는 인덱스: {trigger_indices} ===")
    if trigger_indices:
        print(f"확인해보고 싶으면 main.py의 run_demo(index={trigger_indices[0]})로 돌려보면 됨.")
    else:
        print("전체 test set에서 위험/주의/경계선근처가 하나도 없음 -> "
              "임계값(RED=20.7, YELLOW=50.25)이나 모델 예측 자체를 다시 봐야 할 수도 있음.")


if __name__ == "__main__":
    main()