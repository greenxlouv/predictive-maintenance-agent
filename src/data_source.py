'''
data_source.py 
시뮬용(C6 순차) + 데모용(단일 윈도우) 로더
'''
import numpy as np
from pathlib import Path

DATA_DIR = Path("preprocessed") # 각자 데이터 경로 올려두기


def load_c6_windows():
    # 시뮬레이션용: C6 test set 전체 순차 로드.
    X_test = np.load(DATA_DIR / "X_test.npy")   # (286, 30, 61)
    y_test = np.load(DATA_DIR / "y_test.npy")   # 실제 RUL (검증용, Agent 입력엔 안 씀)
    return X_test, y_test


def demo_single_window(index=0):
    # 데모용: 특정 인덱스 윈도우 하나만.
    X_test, y_test = load_c6_windows()
    return X_test[index], y_test[index]