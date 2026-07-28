from __future__ import annotations

from typing import Any

# ── 1~3부: 등급 판정 임계값 (y_test_alive, C6, n=198 실측 percentile) ──
RED_THRESHOLD = 20.7      # ci_lower 이하 → 위험
YELLOW_THRESHOLD = 50.25  # ci_lower 이하 → 주의

# ── 2부: 주의 등급 교체 기한 (⚠️ 실측 근거 아닌 팀 초기 설정값) ──
YELLOW_REPLACEMENT_WINDOW_CYCLES = 10  # "N=10"
YELLOW_WINDOW_IS_ARBITRARY = True      # 나중에 실측 데이터로 조정될 수 있음

# ── 5부: 경계선 근처 보정 ──
BOUNDARY_MARGIN = 10
GRADE_ORDER = ["정상", "주의", "위험"]  # 낮은 순 → 높은 순


def escalate_grade(action: str) -> str | None:
    """한 단계 위 등급을 반환. 이미 최상위(위험)면 None."""
    idx = GRADE_ORDER.index(action)
    if idx >= len(GRADE_ORDER) - 1:
        return None
    return GRADE_ORDER[idx + 1]


# ── 1~3부: 등급별 표준 조치 텍스트 ──
ACTION_TIMING: dict[str, str] = {
    "위험": "즉시 교체 — 판정 확인 즉시 정비 절차 착수, 당일 내 교체 완료 목표",
    "주의": (
        f"향후 N={YELLOW_REPLACEMENT_WINDOW_CYCLES}사이클 이내 교체 완료 목표로 일정 확정 "
        "(⚠️ N값은 실측 근거가 아닌 팀 초기 설정값)"
    ),
    "정상": "별도 긴급 조치 없음 — 정기 모니터링 지속",
}

# ── 4부: 모니터링 절차 ──
MONITORING: dict[str, dict[str, Any]] = {
    "정기": {
        "적용 등급": "정상",
        "재점검 시점": "다음 Agent① 예측 사이클이 도래할 때",
        "점검 항목": [
            "일일: 공구 홀더·인서트 이상 마모/파손 육안 확인, 작업구역 정리, 냉각수 상태 확인, 웜업 이상 소음 확인",
            "주간: 베어링 상태 점검",
            "월간: 런아웃(runout) 측정",
            "분기: 진동 분석 (ballbar test 등)",
        ],
    },
    "집중": {
        "적용 등급": "주의",
        "재점검 시점": "매 사이클(컷)마다",
        "점검 항목": [
            "육안: flank wear, chipping 등 절삭날 마모/파손 징후 확인",
            "청각: 이상 소음(스크래핑음·채터) 청취",
            "표면 조도: 가공 부품의 표면 거칠기 저하 여부 확인",
        ],
        "조기_교체_트리거": (
            f"위 항목에서 이상 징후 조기 발견 시, N={YELLOW_REPLACEMENT_WINDOW_CYCLES}사이클 "
            "도래 전이라도 조기 교체 검토"
        ),
    },
}

# ── 6부: 교체 부품 정보 ──
TOOL_SPEC = (
    "6mm 볼노즈 텅스텐카바이드(초경합금) 엔드밀, 3-flute, 무급유(dry milling), "
    "스테인리스강(HRC52) 가공, 교체 기준 flank wear(VB) ≥ 0.3mm (ISO 8688-2, ISO 3685)"
)

RECOMMENDED_PART: dict[str, str] = {
    "name": "Kennametal UJBE0600A6AN",
    "grade_code": "KCSM15 (6날, 후막 PVD)",
    "reason": (
        "PHM2010 기본 가공 조건인 스테인리스강(HRC52)에 최적화 — 크레이터 마모/노칭 마모 억제에 특화. "
        "가공 조건 정보(정삭/황삭 여부)가 입력값에 없어 자동 선택 가능한 유일한 부품."
    ),
}

REFERENCE_PARTS: list[dict[str, str]] = [
    {
        "name": "Sandvik Coromant 2P070-0600-PB",
        "grade_code": "1610 (PVD-TiAlN)",
        "use_condition": "정삭(finishing) 가공 또는 스테인리스강보다 경도가 높은 소재 가공 시 담당자가 선택",
    },
    {
        "name": "Walter Prototyp MC482.6.35A2PC-WB10TG",
        "grade_code": "WB10TG (PVD-TiAlSiN)",
        "use_condition": "특별히 까다롭지 않은 범용 가공 상황에서 담당자가 선택",
    },
    {
        "name": "Sandvik Coromant 1B240-0600-XA",
        "grade_code": "1630 (2날, PVD-TiAlN)",
        "use_condition": "황삭(roughing) 가공처럼 절삭량이 많고 인성이 중요한 작업에서 담당자가 선택",
    },
]

PARTS_NOTE = "※ PHM2010 실제 사용 부품 아님 — 동일 스펙의 업계 예시 (실제 도입 시 현장 정확 부품번호로 교체 필요)"

# ── 7부: 정비 절차 (5단계) + 8부: 소요 시간 ──
PROCEDURE_STEPS: list[dict[str, Any]] = [
    {
        "step": 1,
        "title": "안전조치/전원차단",
        "content": "스핀들 정지 → 전원 차단(LOTO, OSHA 29 CFR 1910.147) → 드로우바 잔류 에너지 확인",
        "time_range_minutes": (2, 5),
    },
    {
        "step": 2,
        "title": "공구 탈거",
        "content": "공구 해제 버튼으로 홀더 분리. 절삭날 대신 V-플랜지 아래를 잡아 손끼임 방지",
        "time_range_minutes": (1, 2),
    },
    {
        "step": 3,
        "title": "테이퍼 청소 및 신품 삽입",
        "content": (
            "테이퍼 청소(이물질 잔류 시 런아웃 불량·수명 저하) → 돌출 길이 최소화 삽입 "
            "(돌출 절반 감소 시 강성 약 8배 증가) → 규정 토크 체결 "
            "(REGO-FIX 기준 ER32 콜릿너트 최대 136Nm, 리텐션 노브 25~100Nm)"
        ),
        "time_range_minutes": (5, 10),
    },
    {
        "step": 4,
        "title": "오프셋 설정",
        "content": "공구 길이 오프셋(H값) 재측정 필수 (생략 시 과도 절입으로 파손·충돌 위험)",
        "time_range_minutes": (2.5, 20),
    },
    {
        "step": 5,
        "title": "검증(Proving-out)",
        "content": "드라이런 → 싱글블록 → 통과 시 첫 부품 치수 검사(FAI). 생략 시 초기 불량 미발견 위험",
        "time_range_minutes": (5, 30),
    },
]

TOTAL_TIME_RANGE_MINUTES = (15.5, 67)
TIME_ESTIMATE_NOTE = "실측 데이터 아님 — OSHA/ISO 표준 및 공구홀더 제조사(REGO-FIX, BIG DAISHOWA 등) 자료 기반 추정치"
