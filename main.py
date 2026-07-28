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
    X_test, y_test, cutter_test = load_c6_windows()

    for i in range(len(X_test)):
        cutter_id = str(cutter_test[i])
        result = app.invoke({"x_input": X_test[i], "cutter_id": cutter_id})
        print(f"[cut {i}] cutter={cutter_id} pred={result['rul_pred']:.1f} "
              f"(true={y_test[i]:.1f}) action={result['action']}")


def run_demo(index=0):
    # 단일 윈도우 데모 - 결과 상세 출력.
    print(f"[디버그] 실행 파일: {__file__}")
    print(f"[디버그] 사용할 index: {index}")

    app = build_graph()
    x, y_true, cutter_id = demo_single_window(index)
    result = app.invoke({"x_input": x, "cutter_id": cutter_id})

    print(f"cutter_id: {cutter_id}")
    print(f"실제 RUL: {y_true:.1f}")
    print(f"예측 RUL: {result['rul_pred']:.2f} "
          f"[{result['rul_ci_lower']:.2f}, {result['rul_ci_upper']:.2f}]")
    print(f"판정: {result['action']} (경계선 근처: {result['is_near_boundary']})")
    print(f"설명: {result['explanation']}")

    plan = result.get("maintenance_plan")
    if plan is None:
        print("정비 계획: 없음 (Agent3 건너뜀 - 정상 & 경계선 아님)")
    else:
        print(f"정비 계획: effective_action={plan['effective_action']} "
              f"escalated={plan['escalated']}")
        print(f"  상황 요약: {plan['situation_summary']}")

    print(f"리포트 저장 경로: {result.get('report_paths')}")


if __name__ == "__main__":
    # index를 파일 수정 대신 커맨드라인 인자로 받음 (수정 후 저장 깜빡하는 실수 방지)
    # 사용법: python main.py 56
    demo_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_demo(index=demo_index)
    # run_simulation()