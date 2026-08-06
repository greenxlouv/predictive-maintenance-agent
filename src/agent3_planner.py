"""
Agent 3 — 정비 제안 Agent (Planner)
====================================

역할
----
Agent 2(판단 Agent)의 판정을 받아, 정비 매뉴얼(1~8부) 기준으로 구체적인
정비 계획을 만든다. 부품/절차/소요시간 같은 확정 수치는 manual_data.py에
코드 상수로 고정돼 있고 (근거 없는 수치를 지어내지 않기 위함), LLM은
"이 케이스에 대한 상황 요약 + 특이사항"만 자연어로 생성한다.
ChromaDB RAG는 그 자연어 설명을 쓸 때 참고할 배경 근거(왜 이 임계값인지,
관련 표준이 뭔지)를 검색해오는 용도로 쓴다.

LLM 호출은 llm_client.py를 통해 agent2_decision.py와 동일한
LLM_PROVIDER(anthropic/gemini/ollama) 스위칭을 그대로 따른다.


입력으로 기대하는 state 필드
{
Agent 1:
    rul_pred, rul_ci_lower, rul_ci_upper : float

Agent 2:
    action           : str   "위험" | "주의" | "정상"
    is_near_boundary : bool
    explanation      : str

선택 필드:
    cutter_id : str   예) "c6". 없으면 "미상"으로 대체
}

출력으로 채워 넣는 state 필드
{
    maintenance_plan : dict | None
    maintenance_plan_raw_context : str
}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import manual_data
from llm_client import call_llm_json

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"  # index_manuals.py와 반드시 동일해야 함


@dataclass
class PlannerConfig:
    chroma_collection_name: str = "maintenance_manuals"
    chroma_persist_dir: str = "./chroma_db"
    top_k: int = 3


# ---------------------------------------------------------------------------
# 등급 판단 로직 (5부 경계선 보정 규칙 포함)
# ---------------------------------------------------------------------------

def should_invoke_agent3(state: dict[str, Any]) -> bool:
    """
    Agent 3를 호출해야 하는지 판단.
    - action이 "위험" 또는 "주의"면 항상 True.
    - action이 "정상"이어도 is_near_boundary=True면 True (5부 보정 규칙).
    """
    action = state.get("action")
    near = bool(state.get("is_near_boundary", False))

    if action in ("위험", "주의"):
        return True
    if action == "정상" and near:
        return True
    return False


def resolve_effective_action(state: dict[str, Any]) -> tuple[str, bool]:
    """실제로 적용할 등급(effective_action)과 escalated 여부 계산."""
    action = state.get("action", "정상")
    near = bool(state.get("is_near_boundary", False))

    if not near:
        return action, False

    escalated_action = manual_data.escalate_grade(action)
    if escalated_action is None:
        return action, False  # 이미 "위험"
    return escalated_action, True


# ---------------------------------------------------------------------------
# 결정론적 계획 골격 (manual_data.py 상수 그대로 사용)
# ---------------------------------------------------------------------------

def build_plan_skeleton(effective_action: str) -> dict[str, Any]:
    skeleton: dict[str, Any] = {
        "maintenance_timing": manual_data.ACTION_TIMING[effective_action],
        "monitoring_instruction": None,
        "recommended_part": None,
        "reference_parts": None,
        "procedure_steps": None,
        "estimated_time_minutes": None,
    }

    if effective_action == "주의":
        skeleton["monitoring_instruction"] = manual_data.MONITORING["집중"]
    elif effective_action == "정상":
        skeleton["monitoring_instruction"] = manual_data.MONITORING["정기"]

    if effective_action in ("위험", "주의"):
        skeleton["recommended_part"] = manual_data.RECOMMENDED_PART
        skeleton["reference_parts"] = manual_data.REFERENCE_PARTS

    if effective_action == "위험":
        skeleton["procedure_steps"] = manual_data.PROCEDURE_STEPS
        skeleton["estimated_time_minutes"] = {
            "total_range": manual_data.TOTAL_TIME_RANGE_MINUTES,
            "note": manual_data.TIME_ESTIMATE_NOTE,
        }

    return skeleton


# ---------------------------------------------------------------------------
# RAG: 배경 근거 검색 (로컬 한국어 임베딩, API 키 불필요)
# ---------------------------------------------------------------------------

def get_chroma_retriever(config: PlannerConfig):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        collection_name=config.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=config.chroma_persist_dir,
    )
    return vectorstore.as_retriever(search_kwargs={"k": config.top_k})


def build_retrieval_query(effective_action: str, escalated: bool) -> str:
    query = f"{effective_action} 등급 판정 근거와 표준 조치"
    if escalated:
        query += " 경계선 근처 보정 규칙 적용 사유"
    return query


def retrieve_narrative_context(query: str, config: PlannerConfig) -> str:
    retriever = get_chroma_retriever(config)
    docs = retriever.invoke(query)

    if not docs:
        logger.warning("RAG 검색 결과 없음 (query=%r)", query)
        return ""

    chunks = [f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in docs]
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# LLM 프롬프트 — situation_summary / notes 두 필드만 생성
# ---------------------------------------------------------------------------

SUMMARY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "situation_summary": {
            "type": "string",
            "description": "이번 케이스의 RUL/신뢰구간/판정 근거를 1~2문장으로 요약",
        },
        "notes": {
            "type": "string",
            "description": (
                "특이사항. 경계선 근처 보정이 적용됐다면 그 사실과 이유를 설명. "
                "N=10이 실측 근거 없는 초기값이라는 점, 소요시간이 추정치라는 점 등 "
                "관련 있으면 짧게 언급. 없으면 빈 문자열."
            ),
        },
    },
    "required": ["situation_summary", "notes"],
}

SYSTEM_PROMPT = """\
당신은 CNC 밀링 설비 예지보전(PHM) 시스템의 정비 계획 수립 에이전트입니다.

정비 계획의 확정된 항목(부품, 절차, 소요시간, 등급별 표준 조치)은 이미 \
매뉴얼에서 그대로 가져와 결정되어 있습니다. 당신의 역할은 그 확정된 \
내용을 재생성하는 것이 아니라, 이번 케이스의 구체적인 수치(RUL, \
신뢰구간)와 판정 근거를 바탕으로 "상황 요약"과 "특이사항"만 자연어로 \
작성하는 것입니다.

제공되는 매뉴얼 배경 설명을 참고해서 근거를 정확히 반영하되, 새로운 \
수치나 조치를 지어내지 마세요. 반드시 지정된 JSON 형식으로만 답하세요 \
(situation_summary, notes 두 개의 키만 있는 JSON).
"""

USER_PROMPT_TEMPLATE = """\
## 현재 설비 상태
- 커터: {cutter_id}
- 예측 RUL: {rul_pred} 사이클
- 신뢰구간: [{rul_ci_lower}, {rul_ci_upper}]
- Agent 2 판정: {action}
- 경계선 근처 판정 여부: {is_near_boundary}
- Agent 2 판단 근거: {explanation}

## 적용될 등급 (경계선 보정 반영 결과)
- 최종 적용 등급: {effective_action}
- 경계선 보정으로 인한 상향 적용 여부: {escalated}

## 이번에 확정된 정비 계획 골격 (그대로 참고만 할 것 — 재생성 금지)
{plan_skeleton_text}

## 매뉴얼 배경 설명 (참고용)
{manual_context}

## 요청
{{"situation_summary": "...", "notes": "..."}} 형식의 JSON으로만 답하세요.
"""


def _format_skeleton_for_prompt(skeleton: dict[str, Any]) -> str:
    lines = [f"- 정비 시점: {skeleton['maintenance_timing']}"]
    if skeleton["monitoring_instruction"]:
        lines.append(f"- 모니터링: {skeleton['monitoring_instruction']}")
    if skeleton["recommended_part"]:
        lines.append(f"- 추천 부품: {skeleton['recommended_part']['name']}")
    if skeleton["procedure_steps"]:
        lines.append(f"- 정비 절차: {len(skeleton['procedure_steps'])}단계")
    if skeleton["estimated_time_minutes"]:
        lo, hi = skeleton["estimated_time_minutes"]["total_range"]
        lines.append(f"- 예상 소요 시간: 약 {lo}~{hi}분")
    return "\n".join(lines)


def build_planner_user_prompt(
    state: dict[str, Any],
    effective_action: str,
    escalated: bool,
    skeleton: dict[str, Any],
    manual_context: str,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        cutter_id=state.get("cutter_id", "미상"),
        rul_pred=state.get("rul_pred", "N/A"),
        rul_ci_lower=state.get("rul_ci_lower", "N/A"),
        rul_ci_upper=state.get("rul_ci_upper", "N/A"),
        action=state.get("action", "UNKNOWN"),
        is_near_boundary=state.get("is_near_boundary", False),
        explanation=state.get("explanation", "(없음)"),
        effective_action=effective_action,
        escalated=escalated,
        plan_skeleton_text=_format_skeleton_for_prompt(skeleton),
        manual_context=manual_context if manual_context else "(검색된 배경 설명 없음)",
    )


# ---------------------------------------------------------------------------
# 메인 엔트리 포인트 — 노드 담당자가 이 함수 하나만 호출하면 됨
# ---------------------------------------------------------------------------

def generate_maintenance_plan(
    state: dict[str, Any],
    config: Optional[PlannerConfig] = None,
) -> dict[str, Any]:
    """
    Agent 3의 메인 함수.
    호출 조건: should_invoke_agent3(state)가 True일 때만.
    """
    config = config or PlannerConfig()

    if not should_invoke_agent3(state):
        logger.info(
            "should_invoke_agent3()가 False — 계획 없이 빈 결과 반환 (action=%r, near=%r)",
            state.get("action"),
            state.get("is_near_boundary"),
        )
        return {"maintenance_plan": None, "maintenance_plan_raw_context": ""}

    effective_action, escalated = resolve_effective_action(state)
    skeleton = build_plan_skeleton(effective_action)

    query = build_retrieval_query(effective_action, escalated)
    manual_context = retrieve_narrative_context(query, config)

    user_prompt = build_planner_user_prompt(state, effective_action, escalated, skeleton, manual_context)
    summary = call_llm_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        json_schema=SUMMARY_JSON_SCHEMA,
        tool_name="submit_summary",
    )

    plan = {
        "original_action": state.get("action"),
        "effective_action": effective_action,
        "escalated": escalated,
        "recheck_next_cycle": bool(state.get("is_near_boundary", False)),
        **skeleton,
        "situation_summary": summary["situation_summary"],
        "notes": summary["notes"],
    }

    return {
        "maintenance_plan": plan,
        "maintenance_plan_raw_context": manual_context,
    }


# ---------------------------------------------------------------------------
# 동작 확인용
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_cases = [
        {
            "cutter_id": "c6",
            "rul_pred": 18.4,
            "rul_ci_lower": 12.1,
            "rul_ci_upper": 24.7,
            "action": "위험",
            "is_near_boundary": False,
            "explanation": "ci_lower가 위험 임계값(20.7) 이하로 즉시 교체가 필요합니다.",
        },
        {
            "cutter_id": "c6",
            "rul_pred": 55.0,
            "rul_ci_lower": 52.0,
            "rul_ci_upper": 60.0,
            "action": "정상",
            "is_near_boundary": True,
            "explanation": "ci_lower가 주의 임계값(50.25)에 근접해 경계선 근처로 판정되었습니다.",
        },
    ]

    for i, dummy_state in enumerate(test_cases, start=1):
        print(f"\n=== 케이스 {i} ===")
        print("should_invoke_agent3:", should_invoke_agent3(dummy_state))
        eff, esc = resolve_effective_action(dummy_state)
        print("effective_action:", eff, "| escalated:", esc)
        print(_format_skeleton_for_prompt(build_plan_skeleton(eff)))

        # 실제 LLM/RAG 호출까지 확인하려면 .env 세팅 + chroma 인덱싱 후 주석 해제
        # result = generate_maintenance_plan(dummy_state)
        # print(result)


