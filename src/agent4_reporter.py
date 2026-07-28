"""
Agent 4 — 리포트 Agent (Reporter)
====================================

역할
----
Agent 1(예측) ~ Agent 3(정비 제안, 해당되는 경우)의 결과를 모두 모아
최종 리포트를 LLM으로 작성하고, TXT / PDF 파일로 저장한다. 파이프라인의
마지막 노드.

LLM 호출은 llm_client.py를 통해 agent2_decision.py와 동일한
LLM_PROVIDER(anthropic/gemini/ollama) 스위칭을 그대로 따른다.

이 파일도 agent3_planner.py와 마찬가지로 LangGraph 노드 연결과 무관한
순수 함수들로만 구성. 노드 담당자는 generate_report() 하나만 노드로
감싸서 쓰면 됨.

입력으로 기대하는 state 필드
----------------------------
Agent 1:
    rul_pred, rul_ci_lower, rul_ci_upper : float

Agent 2:
    action           : str   "위험" | "주의" | "정상"
    is_near_boundary : bool
    explanation      : str

Agent 3 (should_invoke_agent3()가 True였던 경우만; 아니면 None):
    maintenance_plan             : dict | None
        (agent3_planner.generate_maintenance_plan()의 반환 구조 그대로)
    maintenance_plan_raw_context : str
        (같은 함수가 반환하는 RAG 검색 원문 — 매뉴얼 배경 설명 인용에 사용)

선택 필드:
    cutter_id : str   예) "c6"

generate_report()의 mode 파라미터
----------------------------------
    mode="formal" (기본값) : 완결된 문장, 매뉴얼 배경 설명을 판정 근거
        섹션에서 1건 인용
    mode="quick"            : 화살표·가운뎃점 개조식, 매뉴얼 배경 설명
        인용 없음. 정보량은 formal과 동일

출력으로 채워 넣는 state 필드
----------------------------
    report_text   : str
    report_paths  : dict  {"txt": "...", "pdf": "..."}
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from llm_client import call_llm_text

logger = logging.getLogger(__name__)


@dataclass
class ReporterConfig:
    max_tokens: int = 1024
    output_dir: str = "./reports"
    filename_prefix: str = "phm_report"


ACTION_ICONS = {"위험": "🔴", "주의": "🟡", "정상": "🟢"}

# ---------------------------------------------------------------------------
# 서식 정규화 — LLM이 리포트를 자유롭게 쓰되, 구분선 문자만 고정 스타일로 통일
# ---------------------------------------------------------------------------

TITLE_DIVIDER = "=" * 44
SECTION_DIVIDER = "-" * 44

_DIVIDER_LINE_RE = re.compile(r"^[=\-─━*_]{4,}\s*$")
_SECTION_HEADING_RE = re.compile(r"^\d+\.\s")


def normalize_report_format(report_text: str) -> str:
    """LLM이 생성한 리포트에서 구분선처럼 보이는 줄만 고정 스타일로 통일.
    본문 문장 자체는 수정하지 않는다.
    """
    lines = report_text.split("\n")
    normalized: list[str] = []
    seen_title_divider = False
    prev_was_heading = False

    for line in lines:
        if _DIVIDER_LINE_RE.match(line.strip()):
            if not seen_title_divider:
                normalized.append(TITLE_DIVIDER)
                seen_title_divider = True
            else:
                normalized.append(SECTION_DIVIDER)
            prev_was_heading = False
            continue

        normalized.append(line)
        prev_was_heading = bool(_SECTION_HEADING_RE.match(line.strip()))

    return "\n".join(normalized)


# ---------------------------------------------------------------------------
# maintenance_plan(dict) → 프롬프트용 텍스트 변환
# ---------------------------------------------------------------------------

def _format_maintenance_plan_section(plan: Optional[dict[str, Any]]) -> str:
    if not plan:
        return "(정비 계획 없음 — 정상 운전 중이며 경계선 근처도 아님)"

    lines = []

    original = plan.get("original_action")
    effective = plan.get("effective_action")
    if plan.get("escalated"):
        lines.append(
            f"- 등급 보정: 1차 판정은 '{original}'이었으나 경계선 근처 판정으로 "
            f"'{effective}' 등급 조치까지 함께 적용됨"
        )
    else:
        lines.append(f"- 적용 등급: {effective}")

    lines.append(f"- 정비 시점: {plan.get('maintenance_timing', '')}")

    monitoring = plan.get("monitoring_instruction")
    if monitoring:
        lines.append(
            f"- 모니터링: {monitoring.get('적용 등급', '')} 등급 모니터링, "
            f"재점검 시점: {monitoring.get('재점검 시점', '')}"
        )
        for item in monitoring.get("점검 항목", []):
            lines.append(f"  · {item}")
        if monitoring.get("조기_교체_트리거"):
            lines.append(f"  · 조기 교체 트리거: {monitoring['조기_교체_트리거']}")

    recommended = plan.get("recommended_part")
    if recommended:
        lines.append(f"- 추천 부품(기본): {recommended['name']} ({recommended['grade_code']})")
        lines.append(f"  선정 이유: {recommended['reason']}")

    reference_parts = plan.get("reference_parts")
    if reference_parts:
        lines.append("- 참고 부품 옵션 (가공 조건에 따라 담당자가 선택):")
        for p in reference_parts:
            lines.append(f"  · {p['name']} ({p['grade_code']}) — {p['use_condition']}")

    steps = plan.get("procedure_steps")
    if steps:
        lines.append("- 정비 절차:")
        for s in steps:
            lo, hi = s["time_range_minutes"]
            lines.append(f"  {s['step']}. {s['title']} ({lo}~{hi}분): {s['content']}")

    time_est = plan.get("estimated_time_minutes")
    if time_est:
        lo, hi = time_est["total_range"]
        lines.append(f"- 예상 총 소요 시간: 약 {lo}~{hi}분 ({time_est['note']})")

    if plan.get("recheck_next_cycle"):
        lines.append("- 재판정 주기: 경계선 근처 판정이므로 다음 사이클에 즉시 재확인 필요")

    situation_summary = plan.get("situation_summary")
    if situation_summary:
        lines.append(f"- 상황 요약: {situation_summary}")

    notes = plan.get("notes")
    if notes:
        lines.append(f"- 특이사항: {notes}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 프롬프트 구성
# ---------------------------------------------------------------------------

FORMAL_SYSTEM_PROMPT = """\
당신은 CNC 밀링 설비 예지보전(PHM) 시스템의 최종 리포트 작성 에이전트입니다.
예측 Agent, 판단 Agent, (해당 시) 정비 제안 Agent의 결과를 받아서, 현장
엔지니어가 바로 읽고 행동할 수 있는 정식 보고서를 작성합니다.

- 2번(예측 결과), 3번(판정 결과 및 근거), 4번(정비 계획) 섹션은 아래
  2단 구성으로 작성하세요:
  ① 먼저 핵심 항목을 "항목 : 값" 형태의 짧은 줄로 나열합니다 (예: "예측
     RUL : 18.4 사이클", "판정 등급 : 위험 (🔴)", "적용 등급 : 위험").
     이 줄들에는 화살표(→)나 가운뎃점(·)을 쓰지 마세요.
  ② 그 아래에 한 줄 띄우고, 근거·해석·설명을 완결된 "-습니다"체 문장
     (2~4문장)으로 서술합니다. "판단 근거:"처럼 소제목을 붙여도 됩니다.
  4번 정비 계획은 하위 항목(등급 보정 여부/정비 시점/추천 부품/참고
  부품 옵션/정비 절차/소요시간)마다 이 2단 구성을 반복해서 적용하세요.
  1번 요약과 5번 특이사항은 완결된 문장으로만 작성하고 이 2단 구성을
  적용하지 마세요.
- 새로운 판단이나 수치를 지어내지 말고, 입력으로 주어진 값만 사용하세요.
- 정비 계획 섹션에 있는 부품명/절차/소요시간/모니터링 항목은 이미 매뉴얼에서
  확정된 값이니 그대로 정확히 인용하세요. 입력에 없는 절차나 점검 항목을
  새로 추가하지 마세요 (예: 입력에 없는 모니터링 지시를 임의로 만들지 말 것).
- 정비 계획이 없는 경우(정상 운전 중이며 경계선 근처도 아님)에는 그 사실을
  명시하고 간단히 마무리하세요.
- 등급이 보정(escalated)된 경우, 왜 원래 등급보다 더 신경써야 하는지
  명확히 설명하세요.
- "매뉴얼 배경 설명"이 주어지면, 반드시 그 내용 중 이번 판정과 가장
  밀접한 것을 최소 1건 인용하되, **오직 3번 판정 결과 및 근거 섹션
  안에서만** 인용하세요. 1번 요약, 2번 예측 결과, 4번 정비 계획에서는
  매뉴얼 배경 설명을 절대 언급하지 마세요 (그 섹션들은 각자의 정보만
  다룹니다). "이번 판정과 관련 없어 보인다"는 이유로 3번에서 인용을
  생략하면 안 됩니다. 배경 설명이 아예 비어 있는 경우("(검색된 배경
  설명 없음)")에만 언급하지 마세요.
- 매뉴얼 배경 설명을 인용할 때는 내용의 요지만 우리말 문장으로 자연스럽게
  풀어 쓰세요. "[1_위험대응.md]"나 "매뉴얼(1_위험대응)"처럼 원본 파일명·
  청크 출처 태그를 그대로 리포트에 노출하지 마세요 — 이건 내부 자료
  구조일 뿐 최종 문서에 나올 표현이 아닙니다. "매뉴얼 기준에 따르면"
  정도로만 자연스럽게 언급하세요. 새로운 수치나 조치를 지어내지도
  마세요.
- 보고서는 아래 순서로 구성하세요:
  1. 요약 (한 문단)
  2. 예측 결과 (RUL, 신뢰구간)
  3. 판정 결과 및 근거
  4. 정비 계획 (해당하는 경우: 등급 보정 여부 / 시점 / 모니터링 / 부품 / 절차 / 소요시간)
  5. 특이사항
- 5번 특이사항은 기본값이 "특이사항 없음"입니다. 1~4번에서 이미 다룬
  내용(신뢰구간 상·하한 관계, 가공 조건 정보 미확인으로 인한 부품 자동
  선택, 소요시간이 추정치라는 점 등)은 이미 해당 섹션에서 설명했으므로
  5번에서 그 내용을 다른 말로 바꿔 쓰지 마세요. 이런 것들은 특이사항이
  아니라 "이미 쓴 내용"입니다. 1~4번 어디에도 없는, 정말 새로운 정보나
  주의사항이 있을 때만(예: 이례적인 안전 경고, 데이터 자체의 결측·이상치
  등) 그 내용만 짧게 쓰세요. 조금이라도 애매하면 지어내지 말고
  "특이사항 없음"이라고 쓰세요.
- 5번 특이사항 다음, 리포트 맨 마지막 줄에 반드시 아래 형식 그대로
  마무리 문구를 한 줄 추가하세요 (다른 표현으로 바꾸지 말 것):
  "이상으로 커터 {cutter_id}에 대한 예지보전 정식 보고서를 마칩니다."
  {cutter_id} 자리에는 입력 데이터의 실제 커터 ID 값을 넣으세요.
- 한국어로 작성하고, 마크다운 문법(#, ** 등)은 쓰지 말고 일반 텍스트로
  작성하세요 (TXT/PDF로 그대로 저장됨).
- 반드시 1번 요약부터 5번 특이사항까지 다섯 섹션을 모두 끝까지 작성한
  뒤에만 응답을 마치세요. 정비 계획의 부품/절차/소요시간 항목이나
  문장을 도중에 끊지 말고 전부 포함하세요. 5번 특이사항까지 쓰기 전에
  응답을 종료하지 마세요.
"""

QUICK_SYSTEM_PROMPT = """\
당신은 CNC 밀링 설비 예지보전(PHM) 시스템의 "현장 담당자 본인 확인용"
리포트 작성 에이전트입니다. 담당자 본인이 이미 배경 지식을 아는 상태에서
빠르게 훑어보는 용도이므로, 아래 원칙을 지키세요.

- 모든 섹션을 완결된 문장이 아니라 화살표(→)나 가운뎃점(·)으로 핵심
  정보만 짧게 나열하세요. 정비 계획의 부품/절차/소요시간도 문장으로
  풀어 쓰지 말고 개조식으로 정리하세요 (단, 항목 자체는 빠짐없이 담을 것 —
  formal과 정보량은 같고 문체만 다름).
- 새로운 판단이나 수치를 지어내지 말고, 입력으로 주어진 값만 사용하세요.
- 정비 계획이 없는 경우(정상 운전 중이며 경계선 근처도 아님)에는 그 사실만
  짧게 표기하세요.
- 등급이 보정(escalated)된 경우, "등급 보정 적용됨(1차: X → 적용: Y)"처럼
  사실만 짧게 표기하세요.
- "매뉴얼 배경 설명"(왜 이 임계값·판정 기준인지 같은 이론적 근거)은 어느
  섹션에서도 인용하지 마세요. 담당자 본인이 이미 아는 내용이므로 생략합니다.
- 보고서는 아래 순서로 구성하세요 (formal과 섹션 구성은 동일, 서술 방식만 개조식):
  1. 요약
  2. 예측 결과 (RUL, 신뢰구간)
  3. 판정 결과 및 근거
  4. 정비 계획 (해당하는 경우: 등급 보정 여부 / 시점 / 모니터링 / 부품 / 절차 / 소요시간)
  5. 특이사항
- 5번 특이사항은 기본값이 "특이사항 없음"입니다. 1~4번에서 이미 다룬
  내용(가공 조건 미확인, 소요시간 추정치 등)을 다른 말로 바꿔 쓰지
  마세요. 1~4번 어디에도 없는 정말 새로운 내용이 있을 때만 짧게 쓰세요.
- 한국어로 작성하고, 마크다운 문법(#, ** 등)은 쓰지 마세요.
- 반드시 1번 요약부터 5번 특이사항까지 다섯 섹션을 모두 끝까지 작성한
  뒤에만 응답을 마치세요. 정비 계획의 부품/절차/소요시간 항목을 도중에
  끊지 말고 전부 포함하세요. 5번 특이사항까지 쓰기 전에 응답을 종료하지
  마세요.
"""

USER_PROMPT_TEMPLATE = """\
## 입력 데이터

### 설비 정보
- 커터: {cutter_id}
- 리포트 생성 시각: {timestamp}

### Agent 1 — 예측 결과
- 예측 RUL: {rul_pred} 사이클
- 신뢰구간: [{rul_ci_lower}, {rul_ci_upper}]

### Agent 2 — 판정 결과
- 판정: {action} ({action_icon})
- 경계선 근처 판정 여부: {is_near_boundary}
- 판단 근거: {explanation}

### Agent 3 — 정비 계획
{maintenance_plan_section}

### 매뉴얼 배경 설명 (RAG 검색 결과, 참고용)
{manual_reference_section}

## 요청
위 내용을 바탕으로 답변을 작성하세요.
"""


MAX_MANUAL_REFERENCE_CHARS = 1500  # 원문이 길어지면 프롬프트가 과도하게 커지는 것 방지


def _format_manual_reference_section(raw_context: Optional[str]) -> str:
    """agent3_planner.py가 RAG로 검색해온 maintenance_plan_raw_context를
    리포트 프롬프트에 넣기 좋은 형태로 변환.
    """
    if not raw_context or not raw_context.strip():
        return "(검색된 배경 설명 없음 — 이 섹션은 리포트에서 언급하지 말 것)"

    text = raw_context.strip()
    if len(text) > MAX_MANUAL_REFERENCE_CHARS:
        text = text[:MAX_MANUAL_REFERENCE_CHARS] + "\n...(이하 생략)"
    return text


def build_reporter_user_prompt(state: dict[str, Any]) -> str:
    action = state.get("action", "UNKNOWN")

    return USER_PROMPT_TEMPLATE.format(
        cutter_id=state.get("cutter_id", "미상"),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        rul_pred=state.get("rul_pred", "N/A"),
        rul_ci_lower=state.get("rul_ci_lower", "N/A"),
        rul_ci_upper=state.get("rul_ci_upper", "N/A"),
        action=action,
        action_icon=ACTION_ICONS.get(action, ""),
        is_near_boundary=state.get("is_near_boundary", False),
        explanation=state.get("explanation", "(없음)"),
        maintenance_plan_section=_format_maintenance_plan_section(state.get("maintenance_plan")),
        manual_reference_section=_format_manual_reference_section(state.get("maintenance_plan_raw_context")),
    )


# ---------------------------------------------------------------------------
# 파일 저장
# ---------------------------------------------------------------------------

def _build_filename(state: dict[str, Any], config: ReporterConfig, ext: str) -> Path:
    cutter_id = state.get("cutter_id", "unknown")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{config.filename_prefix}_{cutter_id}_{ts}.{ext}"


def save_report_txt(report_text: str, state: dict[str, Any], config: ReporterConfig) -> Path:
    path = _build_filename(state, config, "txt")
    path.write_text(report_text, encoding="utf-8")
    logger.info("TXT 리포트 저장 완료: %s", path)
    return path


def save_report_pdf(report_text: str, state: dict[str, Any], config: ReporterConfig) -> Path:
    """
    reportlab 기반 PDF 저장. 한글 폰트는 아래 순서로 찾는다:
      1. AGENT4_FONT_PATH 환경변수로 직접 지정한 경로
      2. 프로젝트에 번들된 폰트 (assets/fonts/NanumGothic.ttf) — 팀원 전체
         환경에서 동일하게 동작하길 원하면 이 방법이 가장 확실함
      3. OS별로 흔히 깔려있는 한글 폰트 경로 (맥: AppleGothic, 윈도우: 맑은 고딕,
         리눅스: NanumGothic)
    전부 없으면 Helvetica로 폴백(한글 깨짐, 경고 로그만 남김).
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    path = _build_filename(state, config, "pdf")

    font_name = "NanumGothic"
    candidate_paths = [
        os.environ.get("AGENT4_FONT_PATH"),
        str(Path(__file__).parent / "assets" / "fonts" / "NanumGothic.ttf"),
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",   # macOS
        "/Library/Fonts/AppleGothic.ttf",                        # macOS (구버전)
        "C:\\Windows\\Fonts\\malgun.ttf",                        # Windows (맑은 고딕)
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",       # Linux
    ]

    registered = False
    for font_path in candidate_paths:
        if not font_path or not Path(font_path).exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_path))
            registered = True
            logger.info("한글 폰트 등록: %s", font_path)
            break
        except Exception:
            continue

    if not registered:
        logger.warning(
            "한글 폰트를 어디서도 찾지 못해 기본 폰트로 대체합니다 (한글이 깨져 보입니다). "
            "AGENT4_FONT_PATH 환경변수로 직접 지정하거나, assets/fonts/NanumGothic.ttf로 폰트 파일을 받아 넣어주세요."
        )
        font_name = "Helvetica"

    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    margin = 20 * mm
    line_height = 6 * mm
    max_chars_per_line = 45

    c.setFont(font_name, 11)
    y = height - margin

    for raw_line in report_text.split("\n"):
        wrapped = [raw_line[i:i + max_chars_per_line] for i in range(0, len(raw_line), max_chars_per_line)] or [""]
        for line in wrapped:
            if y < margin:
                c.showPage()
                c.setFont(font_name, 11)
                y = height - margin
            c.drawString(margin, y, line)
            y -= line_height

    c.save()
    logger.info("PDF 리포트 저장 완료: %s", path)
    return path


# ---------------------------------------------------------------------------
# 메인 엔트리 포인트
# ---------------------------------------------------------------------------

def generate_report(
    state: dict[str, Any],
    config: Optional[ReporterConfig] = None,
    save_pdf: bool = True,
    mode: str = "formal",
) -> dict[str, Any]:
    if mode not in ("formal", "quick"):
        raise ValueError(f"알 수 없는 mode={mode!r} — 'formal' 또는 'quick'이어야 합니다.")

    config = config or ReporterConfig(max_tokens=10000 if mode == "formal" else 2500)

    system_prompt = QUICK_SYSTEM_PROMPT if mode == "quick" else FORMAL_SYSTEM_PROMPT
    user_prompt = build_reporter_user_prompt(state)
    raw_report_text = call_llm_text(system_prompt, user_prompt, max_tokens=config.max_tokens)
    report_text = normalize_report_format(raw_report_text)

    # quick 모드는 사용자가 filename_prefix를 직접 지정하지 않았을 때만 "_quick" 접미사 추가
    file_config = config
    if mode == "quick" and config.filename_prefix == ReporterConfig.filename_prefix:
        file_config = ReporterConfig(
            max_tokens=config.max_tokens,
            output_dir=config.output_dir,
            filename_prefix=f"{config.filename_prefix}_quick",
        )

    report_paths = {"txt": str(save_report_txt(report_text, state, file_config))}
    if save_pdf:
        try:
            report_paths["pdf"] = str(save_report_pdf(report_text, state, file_config))
        except Exception:
            logger.exception("PDF 저장 실패 — TXT는 정상 저장되었으니 이 부분만 확인 필요")

    return {
        "report_text": report_text,
        "report_paths": report_paths,
    }


# ---------------------------------------------------------------------------
# 동작 확인용
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import manual_data

    dummy_plan = {
        "original_action": "위험",
        "effective_action": "위험",
        "escalated": False,
        "recheck_next_cycle": False,
        "maintenance_timing": manual_data.ACTION_TIMING["위험"],
        "monitoring_instruction": None,
        "recommended_part": manual_data.RECOMMENDED_PART,
        "reference_parts": manual_data.REFERENCE_PARTS,
        "procedure_steps": manual_data.PROCEDURE_STEPS,
        "estimated_time_minutes": {
            "total_range": manual_data.TOTAL_TIME_RANGE_MINUTES,
            "note": manual_data.TIME_ESTIMATE_NOTE,
        },
        "situation_summary": "예측 RUL 신뢰구간 하한이 12.1 사이클로 위험 임계값(20.7) 이하입니다.",
        "notes": "",
    }

    # RAG 검색 결과 예시 — 실제로는 agent3_planner.generate_maintenance_plan()의
    # 반환값 중 maintenance_plan_raw_context를 그대로 state에 담아 넘기면 됨
    dummy_raw_context = (
        "[maintenance_manual_1부.md]\n"
        "위험 임계값 20.7 사이클은 C6 cutter의 y_test_alive 실측 데이터(n=198) 중 "
        "하위 10th percentile 값으로 산출되었다. 이는 통계적으로 전체 공구 중 약 "
        "90%가 이 시점 이후에도 정상 가동되었음을 의미하며, 하위 10%에 해당하는 "
        "조기 마모 사례를 놓치지 않기 위한 보수적 기준이다.\n\n"
        "[maintenance_manual_5부.md]\n"
        "경계선 근처 보정 규칙은 MC Dropout 기반 신뢰구간이 추론마다 흔들릴 수 "
        "있다는 한계를 감안해, 판정이 임계값에서 BOUNDARY_MARGIN(10) 이내일 때 "
        "한 단계 보수적인 등급을 함께 적용하도록 설계되었다."
    )

    dummy_state = {
        "cutter_id": "c6",
        "rul_pred": 18.4,
        "rul_ci_lower": 12.1,
        "rul_ci_upper": 24.7,
        "action": "위험",
        "is_near_boundary": False,
        "explanation": "ci_lower가 위험 임계값 이하로 즉시 교체가 필요합니다.",
        "maintenance_plan": dummy_plan,
        "maintenance_plan_raw_context": dummy_raw_context,
    }

    user_prompt = build_reporter_user_prompt(dummy_state)
    print("=== User Prompt (formal/quick 공통 — 시스템 프롬프트만 다름) ===")
    print(user_prompt)

    # 실제 LLM 호출까지 확인하려면 .env 세팅 후 아래 주석 해제
    # result_formal = generate_report(dummy_state, save_pdf=False, mode="formal")
    # print("\n=== formal 모드 ===")
    # print(result_formal["report_text"])
    #
    # result_quick = generate_report(dummy_state, save_pdf=False, mode="quick")
    # print("\n=== quick 모드 ===")
    # print(result_quick["report_text"])