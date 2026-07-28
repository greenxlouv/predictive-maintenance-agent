'''
data_source.py
시뮬용(C6 순차) + 데모용(단일 윈도우) 로더

변경사항: cutter_test.npy에서 cutter ID를 함께 로드하도록 확장.
agent3_planner.py / agent4_reporter.py가 state["cutter_id"]를 참조하므로
(없으면 "미상"으로 대체되긴 하지만) graph.py에서 초기 state를 만들 때
cutter_id를 같이 넘겨줘야 리포트에 실제 커터 ID가 찍힘.

주의: 기존 시그니처(반환값 2개)에서 3개로 바뀜.
    load_c6_windows()   : (X_test, y_test)          -> (X_test, y_test, cutter_test)
    demo_single_window(): (x, y_true)                -> (x, y_true, cutter_id)
이 함수를 쓰는 다른 코드(팀원 쪽 전처리/대시보드 연동 코드 등)가 있다면
호출부도 같이 맞춰야 함 -> 팀원 확인 필요.
'''
import numpy as np
from pathlib import Path

DATA_DIR = Path("preprocessed")  # 각자 데이터 경로 올려두기


def load_c6_windows():
    # 시뮬레이션용: C6 test set 전체 순차 로드.
    X_test = np.load(DATA_DIR / "X_test.npy")            # (286, 30, 61)
    y_test = np.load(DATA_DIR / "y_test.npy")              # 실제 RUL (검증용, Agent 입력엔 안 씀)

    cutter_path = DATA_DIR / "cutter_test.npy"
    if cutter_path.exists():
        cutter_test = np.load(cutter_path)                  # (286,) 각 윈도우의 cutter ID
    else:
        # cutter_test.npy가 아직 없으면(팀원 전처리 산출물 대기 중) 전부 "c6"로 대체.
        # 어차피 이 데이터 자체가 C6 test set이라 값 자체는 맞고, 정확한 커터별
        # 구분(추후 다른 cutter 섞일 경우 대비)만 못 함. 파일 생기면 자동으로 우선됨.
        print(f"[경고] {cutter_path} 없음 -> 전부 'c6'로 대체해서 진행합니다. "
              f"(정확한 cutter_id가 필요하면 팀원한테 cutter_test.npy를 받아 넣어주세요)")
        cutter_test = np.full(len(X_test), "c6")

    return X_test, y_test, cutter_test


def demo_single_window(index=0):
    # 데모용: 특정 인덱스 윈도우 하나만 + 해당 cutter_id.
    X_test, y_test, cutter_test = load_c6_windows()
    return X_test[index], y_test[index], str(cutter_test[index])