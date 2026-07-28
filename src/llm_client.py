"""
llm_client.py
=============

Agent 2 / Agent 3 / Agent 4가 공통으로 쓰는 LLM 호출 창구.

.env의 LLM_PROVIDER 값(anthropic / gemini / ollama)에 따라 실제로
어느 서비스를 호출할지 갈라진다. 각 Agent 파일은 provider가 뭔지 몰라도
되고, 이 파일이 노출하는 call_llm_text() / call_llm_json() 두 함수만
쓰면 된다.

.env 예시 (.env.example 참고):
    LLM_PROVIDER=anthropic
    ANTHROPIC_API_KEY=sk-ant-...
    GEMINI_API_KEY=
    OLLAMA_MODEL=qwen2.5:7b

필요 패키지 (provider별로 실제 쓰는 것만 설치하면 됨):
    pip install python-dotenv --break-system-packages           # 공통
    pip install anthropic --break-system-packages                # LLM_PROVIDER=anthropic
    pip install google-generativeai --break-system-packages      # LLM_PROVIDER=gemini
    pip install requests --break-system-packages                 # LLM_PROVIDER=ollama (requests는 대부분 이미 있음)
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일을 읽어서 os.environ에 채워줌
# (find_dotenv 없이 load_dotenv()만 호출하면 현재 작업 디렉토리 기준으로
#  .env를 찾음 — 팀 README대로 "루트에서 실행"하면 문제 없음)
load_dotenv()

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()

# provider별 기본 모델 (필요하면 .env에 override 값을 추가해도 됨)
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def _strip_code_fence(text: str) -> str:
    """LLM이 JSON을 ```json ... ``` 로 감싸서 줄 때가 있어서 벗겨내는 용도."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


# ---------------------------------------------------------------------------
# provider별 실제 호출부 — 여기만 provider마다 다르게 구현됨
# ---------------------------------------------------------------------------

def _call_anthropic(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY가 .env에 없습니다. LLM_PROVIDER=anthropic이면 이 키가 필요합니다."
        )

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text


def _call_gemini(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY가 .env에 없습니다. LLM_PROVIDER=gemini면 이 키가 필요합니다."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_prompt)
    resp = model.generate_content(
        user_prompt,
        generation_config={"max_output_tokens": max_tokens},
    )
    return resp.text


def _call_ollama(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    import requests

    # 별도 API 키 없이 로컬에서 도는 ollama serve에 그대로 요청
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


_PROVIDER_FUNCS = {
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
    "ollama": _call_ollama,
}


def _dispatch(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    func = _PROVIDER_FUNCS.get(LLM_PROVIDER)
    if func is None:
        raise ValueError(
            f"알 수 없는 LLM_PROVIDER={LLM_PROVIDER!r} (.env 확인 필요). "
            f"anthropic / gemini / ollama 중 하나여야 합니다."
        )
    return func(system_prompt, user_prompt, max_tokens)


# ---------------------------------------------------------------------------
# 공개 함수 — Agent 2/3/4는 이 두 개만 호출하면 됨 (provider 몰라도 됨)
# ---------------------------------------------------------------------------

def call_llm_text(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    """자유 형식 텍스트가 필요할 때 (예: Agent 4의 리포트 본문)."""
    return _dispatch(system_prompt, user_prompt, max_tokens)


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    json_schema: Optional[dict[str, Any]] = None,
    tool_name: str = "submit",
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """JSON 응답이 필요할 때 (예: Agent 3의 situation_summary/notes).

    참고: anthropic의 tool_use 강제 JSON 기능은 gemini/ollama에는 똑같이
    없어서, provider 상관없이 "JSON만 출력하라"는 프롬프트 지시 + 파싱
    실패 시 에러로 통일했다. json_schema/tool_name은 프롬프트에 스키마를
    설명해 넣고 싶을 때 호출하는 쪽(agent3_planner.py)에서 참고용으로만
    쓰고, 이 함수 자체는 강제하지 않는다.
    """
    strict_prompt = (
        user_prompt
        + "\n\n반드시 지정된 JSON 형식으로만 답하세요. 코드블록(```)이나 "
        "다른 설명 텍스트 없이 JSON 객체 하나만 출력하세요."
    )
    raw = _dispatch(system_prompt, strict_prompt, max_tokens)
    cleaned = _strip_code_fence(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM 응답을 JSON으로 파싱하지 못함 (provider={LLM_PROVIDER}): {raw!r}"
        ) from e


# ---------------------------------------------------------------------------
# 동작 확인용
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"LLM_PROVIDER = {LLM_PROVIDER}")
    text = call_llm_text(
        system_prompt="당신은 짧게 답하는 어시스턴트입니다.",
        user_prompt="딱 한 문장으로: CNC 밀링이 뭔지 설명해줘.",
        max_tokens=200,
    )
    print(text)
