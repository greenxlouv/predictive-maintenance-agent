'''
graph.py
- State 정의 + LangGraph 노드/엣지
- Agent 1 -> Agent 2 -> (조건부) Agent 3 -> Agent 4 전체 연결
- 조건부 라우팅: agent2_decision.py가 만든 action / is_near_boundary 값을
  agent3_planner.should_invoke_agent3()에 그대로 넘겨서 Agent 3 호출 여부를 판단한다.
  - action이 "위험"/"주의" -> Agent 3 호출
  - action이 "정상"인데 is_near_boundary=True -> Agent 3 호출 (매뉴얼 5부 경계선 보정 규칙)
  - 그 외("정상" & 경계선 아님) -> Agent 3 건너뛰고 바로 Agent 4
'''
import sys
from pathlib import Path
from typing import Optional, TypedDict

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from langgraph.graph import StateGraph, END

from agent1_predictor import load_model, predict_with_uncertainty
from agent2_decision import run as agent2_run
from agent3_planner import generate_maintenance_plan, should_invoke_agent3
from agent4_reporter import generate_report


class PipelineState(TypedDict, total=False):
    x_input: np.ndarray
    cutter_id: str

    # Agent 1 출력
    rul_pred: float
    rul_ci_lower: float
    rul_ci_upper: float

    # Agent 2 출력
    action: str
    is_near_boundary: bool
    explanation: str

    # Agent 3 출력 (should_invoke_agent3()가 False면 이 필드들은 채워지지 않음
    # -> agent3_planner.py / agent4_reporter.py 쪽에서 전부 state.get(...)으로 접근하므로 없어도 안전하게 None/기본값 처리됨)
    maintenance_plan: Optional[dict]
    maintenance_plan_raw_context: str

    # Agent 4 출력
    report_text: str
    report_paths: dict


_model = None  # 그래프 모듈 로드 시 1회만 로딩


def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


#  노드 정의 

def agent1_node(state: PipelineState) -> dict:
    print("[Agent1] RUL 예측 시작 (MC Dropout 100회)...")
    result = predict_with_uncertainty(_get_model(), state["x_input"])
    print(f"[Agent1] 완료 -> rul_pred={result['rul_pred']:.2f}")
    return result


def agent2_node(state: PipelineState) -> dict:
    print("[Agent2] 판정 + LLM 설명 생성 시작 (Ollama 첫 호출은 모델 로딩 때문에 느릴 수 있음)...")
    agent1_output = {
        "rul_pred": state["rul_pred"],
        "rul_ci_lower": state["rul_ci_lower"],
        "rul_ci_upper": state["rul_ci_upper"],
    }
    result = agent2_run(agent1_output)
    print(f"[Agent2] 완료 -> action={result['action']}")
    return result


def agent3_node(state: PipelineState) -> dict:
    print("[Agent3] 정비 계획 생성 시작 (RAG 검색 + LLM 호출)...")
    result = generate_maintenance_plan(state)
    print("[Agent3] 완료")
    return result


def agent4_node(state: PipelineState) -> dict:
    print("[Agent4] 리포트 생성 시작 (LLM 호출, formal 모드는 max_tokens=10000이라 특히 오래 걸릴 수 있음)...")
    result = generate_report(state, mode="formal")
    print("[Agent4] 완료")
    return result


# ── 조건부 라우팅 ──────────────────────────────────────

def route_after_agent2(state: PipelineState) -> str:
    """Agent 2 판정 직후 분기점.
    agent3_planner.should_invoke_agent3()를 그대로 재사용 — 판단 로직을
    이 파일에서 다시 구현하지 않고 단일 소스(agent3_planner.py)만 따른다.
    """
    return "agent3" if should_invoke_agent3(state) else "agent4_skip"


# ── 그래프 조립 ────────────────────────────────────────

def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("agent1", agent1_node)
    graph.add_node("agent2", agent2_node)
    graph.add_node("agent3", agent3_node)
    graph.add_node("agent4", agent4_node)

    graph.set_entry_point("agent1")
    graph.add_edge("agent1", "agent2")

    graph.add_conditional_edges(
        "agent2",
        route_after_agent2,
        {
            "agent3": "agent3",       # 위험/주의 또는 정상+경계선근처
            "agent4_skip": "agent4",  # 정상 & 경계선 아님 -> Agent 3 건너뜀
        },
    )

    graph.add_edge("agent3", "agent4")
    graph.add_edge("agent4", END)

    return graph.compile()


# 동작 확인용 
if __name__ == "__main__":
    from data_source import demo_single_window

    x, y_true, cutter_id = demo_single_window(index=0)

    app = build_graph()
    result = app.invoke({"x_input": x, "cutter_id": cutter_id})

    print(f"cutter_id={cutter_id} (참고용 실제 RUL: {y_true})")
    print(f"판정: {result['action']} (경계선 근처: {result['is_near_boundary']})")
    print(f"Agent 3 호출 여부: {result.get('maintenance_plan') is not None}")
    print(f"리포트 저장 경로: {result.get('report_paths')}")