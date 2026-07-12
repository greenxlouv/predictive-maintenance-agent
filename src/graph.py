'''
graph.py
- State 정의 + LangGraph 노드/엣지
- Agent 1/Agent 2 노드 연결
- Agent3(Planner), Agent4(Reporter)는 아직 미구현.
- 추가 시 build_graph()에 add_node/add_edge만 이어붙이면 됨.
'''
import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from langgraph.graph import StateGraph, END

from agent1_predictor import load_model, predict_with_uncertainty
from agent2_decision import run as agent2_run


class PipelineState(TypedDict):
    x_input: np.ndarray
    rul_pred: float
    rul_ci_lower: float
    rul_ci_upper: float
    action: str
    is_near_boundary: bool
    explanation: str


_model = None  # 그래프 모듈 로드 시 1회만 로딩


def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def agent1_node(state: PipelineState) -> dict:
    return predict_with_uncertainty(_get_model(), state["x_input"])

def agent2_node(state: PipelineState) -> dict:
    agent1_output = {
        "rul_pred": state["rul_pred"],
        "rul_ci_lower": state["rul_ci_lower"],
        "rul_ci_upper": state["rul_ci_upper"],
    }
    return agent2_run(agent1_output)


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("agent1", agent1_node)
    graph.add_node("agent2", agent2_node)
    graph.set_entry_point("agent1")
    graph.add_edge("agent1", "agent2")
    graph.add_edge("agent2", END)
    return graph.compile()