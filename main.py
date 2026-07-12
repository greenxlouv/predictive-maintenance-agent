'''
main.py
실행 진입점
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from graph import build_graph
from data_source import load_c6_windows, demo_single_window


def run_simulation():
    #C6 test set 전체를 순차로 흘려보내는 시뮬레이션
    app = build_graph()
    X_test, y_test = load_c6_windows()

    for i in range(len(X_test)):
        result = app.invoke({"x_input": X_test[i]})
        print(f"[cut {i}] pred={result['rul_pred']:.1f} "
              f"(true={y_test[i]:.1f}) action={result['action']}")


def run_demo(index=0):
    # 단일 윈도우 데모 - 결과 상세 출력.
    app = build_graph()
    x, y_true = demo_single_window(index)
    result = app.invoke({"x_input": x})

    print(f"실제 RUL: {y_true:.1f}")
    print(f"예측 RUL: {result['rul_pred']:.2f} "
          f"[{result['rul_ci_lower']:.2f}, {result['rul_ci_upper']:.2f}]")
    print(f"판정: {result['action']} (경계선 근처: {result['is_near_boundary']})")
    print(f"설명: {result['explanation']}")


if __name__ == "__main__":
    run_demo(index=0)
    # run_simulation()