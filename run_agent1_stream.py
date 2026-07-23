"""
run_agent1_stream.py

Agent 1을 X_test/y_test/cutter_test 전체에 대해 실행하면서
한 윈도우 예측이 끝날 때마다 로그 CSV에 바로 기록한다.
(agent1_predictor.py의 run_stream() 사용)

이 스크립트를 실행하는 동안, 다른 터미널에서
    streamlit run streamlit_dashboard.py
를 띄워두면 그래프가 실시간으로 갱신되는 걸 볼 수 있음.

사용법:
    # 기본: preprocessed 폴더의 X_test.npy / y_test.npy / cutter_test.npy 사용
    python run_agent1_stream.py

    # 경로/딜레이 커스텀
    python run_agent1_stream.py --data-dir ./preprocessed --delay 0.3

    # 모델 가중치 경로는 환경변수로:
    AGENT1_WEIGHT_PATH=/path/to/best_lstm_2d.pth python run_agent1_stream.py
"""

import argparse
from pathlib import Path

import numpy as np

import agent1_predictor as agent1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=str, default=".",
        help="X_test.npy, y_test.npy, cutter_test.npy 가 있는 폴더",
    )
    parser.add_argument(
        "--delay", type=float, default=0.2,
        help="한 스텝마다 sleep할 시간(초). 실시간 스트리밍처럼 보이게 하는 용도. 0이면 빠르게 전체 처리",
    )
    parser.add_argument(
        "--n-iter", type=int, default=100,
        help="MC Dropout 반복 횟수 (기존 Agent1 설정과 동일하게 100 권장)",
    )
    parser.add_argument(
        "--no-reset", action="store_true",
        help="이 옵션을 주면 기존 로그 파일 뒤에 이어서 기록 (기본은 새로 시작)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="처음 N개 윈도우만 처리 (파이프라인만 빨리 확인하고 싶을 때). 기본은 전체 처리",
    )
    parser.add_argument(
        "--dummy-weights", action="store_true",
        help="실제 학습된 .pth 파일이 없을 때, 초기화만 된(학습 안 된) 랜덤 가중치로 대신 실행. "
             "예측값 자체는 의미 없지만 Agent1->로그->대시보드 파이프라인이 잘 도는지만 확인할 때 사용",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    cutter_test = np.load(data_dir / "cutter_test.npy", allow_pickle=True)

    if args.limit:
        X_test = X_test[: args.limit]
        y_test = y_test[: args.limit]
        cutter_test = cutter_test[: args.limit]

    print(f"X_test: {X_test.shape}, cutter 종류: {sorted(set(cutter_test.tolist()))}")
    print(f"로그 파일: {agent1.LOG_PATH}")

    if args.dummy_weights:
        print("[주의] --dummy-weights 옵션: 학습되지 않은 랜덤 가중치로 실행합니다. "
              "예측값은 의미 없고, 파이프라인 동작 확인용입니다.")
        model = agent1.LSTMPredictor(input_size=X_test.shape[-1]).to(agent1.device)
    else:
        model = agent1.load_model()

    agent1.run_stream(
        X_test,
        cutter_test,
        y=y_test,
        model=model,
        n_iter=args.n_iter,
        delay=args.delay,
        reset_log=not args.no_reset,
    )

    print("완료. streamlit_dashboard.py 를 실행해서 결과를 확인하세요.")


if __name__ == "__main__":
    main()
