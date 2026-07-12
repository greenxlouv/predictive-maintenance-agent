import os
from dotenv import load_dotenv

load_dotenv()
# ── 0. Agent 1 모델 성능 참고값 (2-D 최종 채택 모델, Test=C6 기준) ──
# LLM 프롬프트에 투명하게 공개해서 판단 근거 설명이 과신하지 않도록 함
MODEL_TEST_RMSE = 31.10
MODEL_TEST_MAE = 28.38
DOMAIN_GAP_NOTE = "C6은 학습 데이터(C1, C4)와 마모 패턴이 다른 cutter로, 예측 오차가 클 수 있음"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini") 
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

# ── 1. 임계값 (y_test.npy 실측 percentile 기반, 감으로 잡은 값 아님) ──
# 계산 근거: y_test(C6, n=286)에서 RUL=0(이미 EOL 지난 컷, 88개=30.8%)을
#           제외한 "아직 살아있는" 컷들만 골라 percentile 계산
#           (RUL=0이 낀 채로 계산하면 p10/p25가 전부 0이 되어 왜곡됨)
#   y_test_alive = y_test[y_test > 0]
#   np.percentile(y_test_alive, 10) → 20.7
#   np.percentile(y_test_alive, 25) → 50.25
RED_THRESHOLD = 20.7     # ci_lower 이 값 이하 → 즉시 교체 (하위 10% 지점)
YELLOW_THRESHOLD = 50.25 # ci_lower 이 값 이하 → 집중 모니터링 (하위 25% 지점)
BOUNDARY_MARGIN = 10     # 경계값 근처(±10) 판정은 "경계선 근처"로 별도 표시
                         # (모델 Test RMSE ~31.10 스케일에 맞춰 넉넉히 설정)


def score_action(state: dict) -> str:
    """ci_lower(보수적 하한) 기준으로 위험 / 주의 / 정상 3단계 판정."""
    ci_lower = state["rul_ci_lower"]
    if ci_lower <= RED_THRESHOLD:
        return "위험"
    elif ci_lower <= YELLOW_THRESHOLD:
        return "주의"
    else:
        return "정상"


def is_near_boundary(state: dict) -> bool:
    """경계값 근처면 MC Dropout의 확률성 + 모델 오차 때문에 판정이 흔들릴 수 있음을 표시.
    대시보드/리포트에서 '참고용 판정'임을 알리는 용도."""
    ci_lower = state["rul_ci_lower"]
    return (
        abs(ci_lower - RED_THRESHOLD) <= BOUNDARY_MARGIN
        or abs(ci_lower - YELLOW_THRESHOLD) <= BOUNDARY_MARGIN
    )


# ── 2. LLM 클라이언트 (provider별로 지연 초기화)──────────────────────────────
def _get_client():
    if LLM_PROVIDER == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    elif LLM_PROVIDER == "gemini":
        from google import genai
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    elif LLM_PROVIDER == "ollama":
        import ollama
        return ollama  # 로컬 서버라 별도 client 객체 없이 모듈 자체 사용
    else:
        raise ValueError(f"알 수 없는 LLM_PROVIDER: {LLM_PROVIDER}")
client = _get_client()

def build_prompt(state: dict, action: str, near_boundary: bool) -> str:
    boundary_note = (
        "\n- 이 판정은 임계값 경계선 근처라 다음 추론 시 결과가 바뀔 수 있음"
        if near_boundary
        else ""
    )
    return f"""당신은 CNC 밀링 공구의 예지보전을 담당하는 판단 Agent입니다.

[예측 정보]
- 예측 RUL: {state['rul_pred']:.2f} 사이클
- 95% 신뢰구간: [{state['rul_ci_lower']:.2f}, {state['rul_ci_upper']:.2f}]
- 임계값 기반 1차 판정: {action}{boundary_note}

[모델 신뢰도 참고 — 반드시 설명에 반영]
- 이 모델은 검증용 test set(다른 cutter)에서 평균 {MODEL_TEST_RMSE:.1f} 사이클의 오차를 보였음
- {DOMAIN_GAP_NOTE}

위 정보를 종합해서, 정비 담당자에게 전달할 2~3문장짜리 한국어 판단 근거를 작성하세요.
신뢰구간이나 예측값을 과신하지 말고, 모델의 알려진 오차 범위를 함께 언급하세요.
경계선 근처(near_boundary=True)라면, 다음 추론에서 판정이 바뀔 수 있다는 점도 명시하세요.
경계선 근처가 아니라면(near_boundary=False) "경계", "근처" 같은 표현은 쓰지 마세요.
"""


def get_llm_explanation(state: dict, action: str, near_boundary: bool) -> str:
    prompt = build_prompt(state, action, near_boundary)

    if LLM_PROVIDER == "anthropic":
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    elif LLM_PROVIDER == "gemini":
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )
        return response.text

    elif LLM_PROVIDER == "ollama":
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]


# ── 3. Agent 1 → Agent 2 연결 지점 ────────────────────
def run(state: dict) -> dict:
    """Agent 1이 만든 state dict를 그대로 받아 Agent 2 결과를 반환.
    반환값도 dict라서 그대로 공유 State에 병합(update)하기 쉬움."""
    action = score_action(state)
    boundary = is_near_boundary(state)
    explanation = get_llm_explanation(state, action, boundary)

    return {
        "action": action,              # "위험" / "주의" / "정상"
        "is_near_boundary": boundary,  # True면 판정이 경계선 근처 (참고용 표시)
        "explanation": explanation,
    }


# ── 4. 실행 예시 (Agent 1이 준 예시 값 그대로 테스트) ──
if __name__ == "__main__":
    dummy_state = {
        "rul_pred": 42.3,
        "rul_ci_lower": 35.1,
        "rul_ci_upper": 49.5,
    }
    result = run(dummy_state)

    print(f"판정: {result['action']} (경계선 근처: {result['is_near_boundary']})")
    print(f"설명: {result['explanation']}")
