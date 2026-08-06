import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # torch/numpy OpenMP 런타임 중복 충돌 방지

import sys
import time
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

# ---------------------------------------------------------------------------
# 실제 Agent 모듈 임포트 (src/ 폴더) — main.py와 동일한 방식
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "src"))

from data_source import load_c6_windows
from agent1_predictor import load_model, predict_with_uncertainty
from agent2_decision import (
    score_action,
    is_near_boundary,
    build_prompt as agent2_build_prompt,
    SYSTEM_PROMPT as AGENT2_SYSTEM_PROMPT,
    RED_THRESHOLD,
    YELLOW_THRESHOLD,
    BOUNDARY_MARGIN,
)
from llm_client import call_llm_text
from agent3_planner import generate_maintenance_plan, should_invoke_agent3
from agent4_reporter import generate_report

import matplotlib
from matplotlib import font_manager

# 한글 폰트 — Windows(Malgun Gothic)만 하드코딩되어 있으면 macOS/Linux에서
# 한글이 네모(tofu)로 깨짐. rcParams만 설정하면 legend() 등 일부 요소에
# 안정적으로 안 먹는 경우가 있어서, FontProperties 객체를 만들어 텍스트를
# 그리는 자리마다(legend/label/title/text) 명시적으로 넘기는 방식으로 강화함.
_KOREAN_FONT_CANDIDATES = [
    "Malgun Gothic",                                          # Windows (이름 매칭)
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",     # macOS
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",              # macOS (최신)
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",         # Linux
]

KOREAN_FONT_PROP = None
for _candidate in _KOREAN_FONT_CANDIDATES:
    try:
        if Path(_candidate).exists():
            font_manager.fontManager.addfont(_candidate)
            KOREAN_FONT_PROP = font_manager.FontProperties(fname=_candidate)
            plt.rcParams["font.family"] = KOREAN_FONT_PROP.get_name()
        else:
            # "Malgun Gothic"처럼 경로가 아니라 시스템에 설치된 폰트 이름인 경우
            # (Windows에서만 해당) — 존재 여부를 미리 확인할 수 없어 바로 시도
            KOREAN_FONT_PROP = font_manager.FontProperties(family=_candidate)
            plt.rcParams["font.family"] = _candidate
        break
    except Exception:
        continue

if KOREAN_FONT_PROP is None:
    print("[경고] 한글 폰트를 찾지 못해 그래프의 한글이 깨질 수 있습니다. "
          "AGENT4_FONT_PATH처럼 로컬 폰트 경로를 _KOREAN_FONT_CANDIDATES에 추가하세요.")

plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="PHM 예지보전 시스템", page_icon="🛠️", layout="wide")

st.markdown("""
<style>
.main .block-container {padding-top: 2rem; padding-bottom: 3rem;}
h1 {font-size: 2.1rem !important;}
h3 {margin-top: 0.3rem !important;}
[data-testid="stMetricValue"] {font-size: 1.8rem;}
.status-box {
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}
.status-danger  {background-color: #FDEDED; color: #C62828; border: 1px solid #F5B7B1;}
.status-warning {background-color: #FFF8E1; color: #B7791F; border: 1px solid #F6D97A;}
.status-normal  {background-color: #EAF7EE; color: #1E7B3C; border: 1px solid #A9DDB5;}
.section-caption {color: #6B7280; font-size: 0.9rem; margin-top: -0.4rem;}
</style>
""", unsafe_allow_html=True)

status_style = {
    "위험": ("status-danger", "🔴"),
    "주의": ("status-warning", "🟡"),
    "정상": ("status-normal", "🟢"),
}

st.title("🛠️ CNC 밀링 예지보전(PHM) 시스템")
st.caption("Agent 1(예측) → Agent 2(판정) → Agent 3(정비 계획) → Agent 4(리포트)")
st.divider()


# ---------------------------------------------------------------------------
# 캐싱 — 모델/데이터는 한 번만 로드
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_test_data():
    X_test, y_test, cutter_test = load_c6_windows()
    return X_test, y_test, cutter_test


def strip_markdown_headers(text: str) -> str:
    """LLM(Agent 2 등)이 붙인 '## 제목' 같은 마크다운 헤더 줄만 제거.
    본문 문장 내용은 그대로 둔다."""
    lines = text.split("\n")
    cleaned = [line for line in lines if not line.strip().startswith("#")]
    return "\n".join(cleaned).strip()


def render_pipeline_status(current_step: int):
    steps = ["Agent 1\n예측", "Agent 2\n판정", "Agent 3\n정비계획", "Agent 4\n리포트"]
    cols = st.columns(4)
    for i, (col, label) in enumerate(zip(cols, steps), start=1):
        with col:
            if i < current_step:
                st.success(f"✅ {label}")
            elif i == current_step:
                st.info(f"⏳ {label} 진행 중...")
            else:
                st.markdown(
                    f"<div style='opacity:0.4; padding:0.6rem; text-align:center;'>⬜ {label}</div>",
                    unsafe_allow_html=True,
                )
    st.progress(min(current_step / 4, 1.0))
    st.write("")


def draw_gauge(value, red_th, yellow_th, max_val=150):
    fig, ax = plt.subplots(figsize=(3, 1.9), subplot_kw={"aspect": "equal"})
    ax.add_patch(Wedge((0.5, 0), 0.4, 0, 180 * red_th / max_val, facecolor="#E74C3C"))
    ax.add_patch(Wedge((0.5, 0), 0.4, 180 * red_th / max_val, 180 * yellow_th / max_val, facecolor="#F1C40F"))
    ax.add_patch(Wedge((0.5, 0), 0.4, 180 * yellow_th / max_val, 180, facecolor="#2ECC71"))
    angle = np.deg2rad(180 * min(value, max_val) / max_val)
    ax.plot([0.5, 0.5 + 0.35 * np.cos(angle)], [0, 0.35 * np.sin(angle)], color="black", linewidth=2.5)
    ax.plot(0.5, 0, "ko", markersize=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 0.5)
    ax.axis("off")
    ax.text(0.5, -0.05, f"RUL = {value:.1f}", ha="center", fontsize=12, fontweight="bold",
            fontproperties=KOREAN_FONT_PROP)
    return fig


def build_real_log(selected_cutter, model, cutter_test, X_test, y_test, n_iter_preview=20, max_windows=None):
    """애니메이션 프레임용 — 이 커터의 윈도우를 순서대로 빠르게 추론(LLM 없음).
    max_windows를 주면 그 개수까지만 사용 (커터 수명 중 특정 시점까지만 보고 싶을 때)."""
    indices = np.where(cutter_test == selected_cutter)[0]
    if max_windows is not None:
        indices = indices[:max(1, max_windows)]
    rows = []
    for step, idx in enumerate(indices):
        result = predict_with_uncertainty(model, X_test[idx], n_iter=n_iter_preview)
        rows.append({
            "cutter_id": selected_cutter,
            "cutter_step": step,
            "data_index": idx,
            "rul_true": float(y_test[idx]),
            "rul_pred": result["rul_pred"],
            "rul_ci_lower": result["rul_ci_lower"],
            "rul_ci_upper": result["rul_ci_upper"],
        })
    return pd.DataFrame(rows)


def draw_agent1_frame(partial, cutter_df_full, n_total, ph_metrics, ph_graph1, selected_cutter):
    latest_pred = partial["rul_pred"].iloc[-1]
    latest_lower = partial["rul_ci_lower"].iloc[-1]
    latest_upper = partial["rul_ci_upper"].iloc[-1]

    with ph_metrics.container():
        m1, m2, m3 = st.columns(3)
        m1.metric("커터 ID", selected_cutter)
        m2.metric("최신 예측 RUL", f"{latest_pred:.1f} 사이클")
        m3.metric("95% 신뢰구간", f"[{latest_lower:.1f}, {latest_upper:.1f}]")

    fig1, ax1 = plt.subplots(figsize=(9, 3))
    ax1.fill_between(partial["cutter_step"], partial["rul_ci_lower"], partial["rul_ci_upper"],
                      color="#93C5FD", alpha=0.3, label="95% CI")
    ax1.plot(partial["cutter_step"], partial["rul_pred"], color="#2563EB", linewidth=1.5, label="예측 RUL")
    ax1.axhline(RED_THRESHOLD, color="#C62828", linestyle=":", linewidth=1, label=f"위험 임계값 ({RED_THRESHOLD})")
    ax1.axhline(YELLOW_THRESHOLD, color="#B7791F", linestyle=":", linewidth=1, label=f"주의 임계값 ({YELLOW_THRESHOLD})")
    ax1.set_xlim(0, n_total)
    ax1.set_ylim(0, max(cutter_df_full["rul_pred"].max(), 1) * 1.2)
    ax1.set_xlabel("cutter_step", fontproperties=KOREAN_FONT_PROP)
    ax1.set_ylabel("RUL", fontproperties=KOREAN_FONT_PROP)
    ax1.legend(loc="upper right", fontsize=8, frameon=False, prop=KOREAN_FONT_PROP)
    ax1.spines[["top", "right"]].set_visible(False)
    ph_graph1.pyplot(fig1)
    plt.close(fig1)

    return latest_pred, latest_lower, latest_upper


def render_agent2_frame(rul_pred, ci_lower, ci_upper, ph_badge, ph_gauge):
    """실시간 프레임용 — Agent 2의 실제 임계값 함수(LLM 제외)를 그대로 사용."""
    action = score_action({"rul_ci_lower": ci_lower})
    near_boundary = is_near_boundary({"rul_ci_lower": ci_lower})

    css_class, icon = status_style[action]

    with ph_badge.container():
        st.markdown(f'<div class="status-box {css_class}">{icon} {action} 등급</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="section-caption">근거: ci_lower({ci_lower:.1f}) 기준 · '
            f'위험 임계값 {RED_THRESHOLD} · 주의 임계값 {YELLOW_THRESHOLD}</p>',
            unsafe_allow_html=True,
        )
        if near_boundary:
            st.warning(f"⚠️ 경계선 근처 판정 (임계값과의 거리 {BOUNDARY_MARGIN} 이내)")

    fig2 = draw_gauge(rul_pred, RED_THRESHOLD, YELLOW_THRESHOLD, max_val=150)
    ph_gauge.pyplot(fig2)
    plt.close(fig2)

    return action, near_boundary


def draw_frame_graphs_validation(partial, cutter_df_full, n_total, ph_metrics, ph_graph1, ph_cal, ph_res, selected_cutter):
    latest_pred = partial["rul_pred"].iloc[-1]
    latest_lower = partial["rul_ci_lower"].iloc[-1]
    latest_upper = partial["rul_ci_upper"].iloc[-1]

    with ph_metrics.container():
        m1, m2, m3 = st.columns(3)
        m1.metric("커터 ID", selected_cutter)
        m2.metric("최신 예측 RUL", f"{latest_pred:.1f} 사이클")
        m3.metric("95% 신뢰구간", f"[{latest_lower:.1f}, {latest_upper:.1f}]")

    fig1, ax1 = plt.subplots(figsize=(9, 3))
    ax1.fill_between(partial["cutter_step"], partial["rul_ci_lower"], partial["rul_ci_upper"],
                      color="#93C5FD", alpha=0.3, label="95% CI")
    ax1.plot(partial["cutter_step"], partial["rul_pred"], color="#2563EB", linewidth=1.5, label="예측 RUL")
    ax1.plot(partial["cutter_step"], partial["rul_true"], color="#C62828", linestyle="--", linewidth=1, label="실제 RUL")
    ax1.axhline(RED_THRESHOLD, color="#C62828", linestyle=":", linewidth=1)
    ax1.axhline(YELLOW_THRESHOLD, color="#B7791F", linestyle=":", linewidth=1)
    ax1.set_xlim(0, n_total)
    ax1.set_ylim(0, max(cutter_df_full["rul_true"].max(), 1) * 1.1)
    ax1.set_xlabel("cutter_step", fontproperties=KOREAN_FONT_PROP)
    ax1.set_ylabel("RUL", fontproperties=KOREAN_FONT_PROP)
    ax1.legend(loc="upper right", fontsize=8, frameon=False, prop=KOREAN_FONT_PROP)
    ax1.spines[["top", "right"]].set_visible(False)
    ph_graph1.pyplot(fig1)
    plt.close(fig1)

    fig4, ax4 = plt.subplots(figsize=(4.3, 4))
    ax4.scatter(partial["rul_true"], partial["rul_pred"], s=10, alpha=0.5, color="#2563EB")
    lims = [0, max(cutter_df_full["rul_true"].max(), cutter_df_full["rul_pred"].max(), 1)]
    ax4.plot(lims, lims, "--", color="gray", linewidth=1, label="완벽한 예측 (y=x)")
    ax4.set_xlim(lims)
    ax4.set_ylim(lims)
    ax4.set_xlabel("실제 RUL", fontproperties=KOREAN_FONT_PROP)
    ax4.set_ylabel("예측 RUL", fontproperties=KOREAN_FONT_PROP)
    ax4.set_title("예측 보정도", fontsize=10, fontproperties=KOREAN_FONT_PROP)
    ax4.legend(fontsize=8, frameon=False, prop=KOREAN_FONT_PROP)
    ph_cal.pyplot(fig4)
    plt.close(fig4)

    residual = partial["rul_pred"] - partial["rul_true"]
    fig5, ax5 = plt.subplots(figsize=(4.3, 4))
    ax5.hist(residual, bins=15, color="#2563EB", alpha=0.8)
    ax5.axvline(residual.mean(), color="black", linestyle="--", linewidth=1)
    ax5.set_xlabel(f"예측 - 실제 (평균 오차: {residual.mean():.2f})", fontproperties=KOREAN_FONT_PROP)
    ax5.set_ylabel("빈도", fontproperties=KOREAN_FONT_PROP)
    ax5.set_title("잔차 분포", fontsize=10, fontproperties=KOREAN_FONT_PROP)
    ph_res.pyplot(fig5)
    plt.close(fig5)

    return latest_pred, latest_lower, latest_upper


def run_agent1_and_2(cutter_id, data_index, model, X_test):
    """정밀 추론(n_iter=100) + 실제 Agent2 LLM 설명까지만.

    Agent 2의 run()을 그대로 쓰지 않고 내부 로직(score_action/is_near_boundary/
    build_prompt)만 재사용하는 이유: agent2_decision.py의 get_llm_explanation()이
    max_tokens=300으로 고정돼있어 설명이 종종 중간에 잘리는 문제가 있음
    (Agent 4에서 겪었던 것과 동일한 유형). 팀 공용 파일은 건드리지 않고
    여기서만 max_tokens을 넉넉하게 줘서 우회함.
    """
    x = X_test[data_index]
    agent1_result = predict_with_uncertainty(model, x, n_iter=100)

    state = {"cutter_id": cutter_id, **agent1_result}

    action = score_action(state)
    near_boundary = is_near_boundary(state)
    user_prompt = agent2_build_prompt(state, action, near_boundary)
    explanation = call_llm_text(AGENT2_SYSTEM_PROMPT, user_prompt, max_tokens=600)

    state.update({
        "action": action,
        "is_near_boundary": near_boundary,
        "explanation": explanation,
    })
    return state


def run_agent3(state):
    """Agent 3(조건부) — Agent 2 결과가 담긴 state를 받아 정비 계획을 채워 반환."""
    if should_invoke_agent3(state):
        state.update(generate_maintenance_plan(state))  # 실제 RAG+LLM 호출
    else:
        state["maintenance_plan"] = None
        state["maintenance_plan_raw_context"] = ""
    return state


with st.sidebar:
    st.header("⚙️ 실행 설정")
    st.selectbox("커터 ID", ["c6"], index=0)
    cutter_id = "c6"
    sim_progress = st.slider(
        "시뮬레이션 진행 시점 (%)",
        min_value=10, max_value=100, value=100, step=10,
        help="이 커터 수명의 몇 %까지 진행된 시점을 확인할지 선택 (100% = 수명이 거의 다한 시점 → 위험이 나오기 쉬움, "
             "낮을수록 정상/주의가 나오기 쉬움)",
    )
    run_clicked = st.button("🚀 예측 실행", use_container_width=True, type="primary")

if run_clicked:
    st.session_state["show_results"] = True
    st.session_state["animated_cutter"] = None      # 탭1 애니메이션 새로 재생
    st.session_state["val_animated_cutter"] = None   # 탭2 애니메이션도 같이
    st.session_state["pipeline_state"] = None         # Agent2·3 최종 결과 새로 계산 필요
    st.session_state["report_cache"] = {}             # Agent4 리포트 캐시 초기화
    st.session_state["selected_cutter_for_run"] = cutter_id
    st.session_state["sim_progress_for_run"] = sim_progress

tab1, tab2 = st.tabs(["🔍 파이프라인 실행", "📊 모델 성능 검증"])

# ===========================================================================
# 탭 1 — 파이프라인 실행 (Agent 1~4, 실제 함수 연결)
# ===========================================================================
with tab1:
    if not st.session_state.get("show_results"):
        st.info("왼쪽 사이드바에서 시뮬레이션 진행 시점을 선택하고 '예측 실행'을 눌러주세요.")
    else:
        selected_cutter = st.session_state["selected_cutter_for_run"]
        sim_progress = st.session_state.get("sim_progress_for_run", 100)
        model = get_model()
        X_test, y_test, cutter_test = get_test_data()

        # Agent 1~4 전체를 하나의 자리에 담아서, 새로 실행될 때 이전 실행 내용이
        # 한 번에 깨끗이 지워지고 처음부터 다시 그려지게 함
        results_container = st.empty()
        results_container.empty()  # 이전 실행에서 남아있던 내용을 명시적으로 완전히 제거
        with results_container.container():
            status_ph = st.empty()
            with status_ph.container():
                render_pipeline_status(1)

            with st.container(border=True):
                st.subheader("1️⃣ 예측 결과 (Agent 1)")
                st.caption(f"커터 {selected_cutter} · 수명의 {sim_progress}% 지점까지 실제 test set 데이터를 순서대로 재생합니다")
                ph_metrics = st.empty()
                ph_graph1 = st.empty()

            with st.container(border=True):
                st.subheader("2️⃣ 판정 결과 (Agent 2)")
                col_text, col_gauge = st.columns([2, 1])
                ph_badge = col_text.empty()
                ph_gauge = col_gauge.empty()

            already_animated = st.session_state.get("animated_cutter") == selected_cutter

            if already_animated:
                cutter_df = st.session_state["cutter_df_cache"]
                n_total = len(cutter_df)
                rul_pred, ci_lower, ci_upper = draw_agent1_frame(
                    cutter_df, cutter_df, n_total, ph_metrics, ph_graph1, selected_cutter
                )
                render_agent2_frame(rul_pred, ci_lower, ci_upper, ph_badge, ph_gauge)
            else:
                total_windows_for_cutter = int(np.sum(cutter_test == selected_cutter))
                max_windows = max(1, int(total_windows_for_cutter * sim_progress / 100))

                with st.spinner(f"Agent 1 추론 중 (수명의 {sim_progress}% 지점까지, 윈도우별로 빠르게 미리보기)..."):
                    cutter_df = build_real_log(
                        selected_cutter, model, cutter_test, X_test, y_test,
                        n_iter_preview=20, max_windows=max_windows,
                    )
                st.session_state["cutter_df_cache"] = cutter_df
                n_total = len(cutter_df)

                step_size = max(1, n_total // 40)
                for n_visible in range(step_size, n_total + step_size, step_size):
                    partial = cutter_df.iloc[:min(n_visible, n_total)]
                    rul_pred, ci_lower, ci_upper = draw_agent1_frame(
                        partial, cutter_df, n_total, ph_metrics, ph_graph1, selected_cutter
                    )
                    render_agent2_frame(rul_pred, ci_lower, ci_upper, ph_badge, ph_gauge)
                    time.sleep(0.05)
                st.session_state["animated_cutter"] = selected_cutter

            st.caption(
                "⚠️ test 데이터(c6)는 도메인 시프트로 인해 예측이 실제보다 낮게 안 내려가고 "
                "정체되는 경향이 있음 (train 데이터에 RUL=0 샘플이 없어서 모델이 그 구간을 "
                "학습한 적이 없기 때문). 미리보기는 속도를 위해 n_iter=20으로 계산되어, "
                "아래 최종 판정(n_iter=100)과 신뢰구간이 약간 다를 수 있습니다."
            )

            with status_ph.container():
                render_pipeline_status(2)  # Agent 1 완료, Agent 2 진행 중 (아직 LLM 설명 전)

            # ---- Agent 2(LLM 설명) — 완료 시점에 맞춰 진행 표시기 갱신 ----
            if st.session_state.get("pipeline_state") is None:
                with st.spinner("Agent 2 판단 근거 생성 중 (LLM 호출)..."):
                    last_idx = int(cutter_df["data_index"].iloc[-1])
                    state = run_agent1_and_2(selected_cutter, last_idx, model, X_test)

                with status_ph.container():
                    render_pipeline_status(3)  # Agent 2 완료 (설명까지 다 나온 시점), Agent 3 진행 중

                # ---- Agent 3(RAG+LLM, 조건부) ----
                with st.spinner("Agent 3 정비 계획 생성 중 (RAG+LLM 호출)..."):
                    state = run_agent3(state)

                st.session_state["pipeline_state"] = state
            else:
                with status_ph.container():
                    render_pipeline_status(3)

            pipeline_state = st.session_state["pipeline_state"]
            action = pipeline_state["action"]
            is_near_boundary_final = pipeline_state["is_near_boundary"]

            # 최종 정밀 결과로 Agent 2 카드 갱신 (미리보기 마지막 프레임 대신 정확한 값)
            css_class, icon = status_style[action]
            with ph_badge.container():
                st.markdown(f'<div class="status-box {css_class}">{icon} {action} 등급 (최종)</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="section-caption">근거: ci_lower({pipeline_state["rul_ci_lower"]:.1f}) 기준 · '
                    f'위험 임계값 {RED_THRESHOLD} · 주의 임계값 {YELLOW_THRESHOLD}</p>',
                    unsafe_allow_html=True,
                )
                if is_near_boundary_final:
                    st.warning(f"⚠️ 경계선 근처 판정 (임계값과의 거리 {BOUNDARY_MARGIN} 이내)")
                st.markdown(f"**판단 근거(Agent 2 LLM):** {strip_markdown_headers(pipeline_state['explanation'])}")
            fig2_final = draw_gauge(pipeline_state["rul_pred"], RED_THRESHOLD, YELLOW_THRESHOLD, max_val=150)
            ph_gauge.pyplot(fig2_final)
            plt.close(fig2_final)

            # ---- 3. Agent 3 — 정비 계획 ----
            maintenance_plan = pipeline_state.get("maintenance_plan")

            with st.container(border=True):
                st.subheader("3️⃣ 정비 계획 (Agent 3)")

                if maintenance_plan is None:
                    st.success("정비 조치 불필요 — 정상 운전 중 (Agent 3 건너뜀)", icon="✅")
                else:
                    original_action = maintenance_plan["original_action"]
                    effective_action = maintenance_plan["effective_action"]
                    escalated = maintenance_plan["escalated"]
                    eff_css_class, eff_icon = status_style[effective_action]

                    if escalated:
                        st.markdown(
                            f'<div class="status-box {eff_css_class}">{eff_icon} 등급 보정 · {original_action} → {effective_action}</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption("🔁 다음 사이클에 즉시 재확인 필요")
                    else:
                        st.markdown(
                            f'<div class="status-box {eff_css_class}">{eff_icon} {effective_action} · 보정 없음</div>',
                            unsafe_allow_html=True,
                        )

                    recommended = maintenance_plan.get("recommended_part")
                    reference_parts = maintenance_plan.get("reference_parts")
                    if recommended:
                        c1, c2 = st.columns([1, 1.4])
                        with c1:
                            st.markdown("**🔧 기본 추천 부품**")
                            st.info(f"{recommended['name']}\n\n{recommended['grade_code']}", icon="🔩")
                            st.caption(recommended["reason"])
                        with c2:
                            if reference_parts:
                                st.markdown("**📋 참고 부품 옵션**")
                                st.dataframe(pd.DataFrame(reference_parts), use_container_width=True, hide_index=True)

                    procedure_steps = maintenance_plan.get("procedure_steps")
                    if procedure_steps:
                        st.markdown("**🛠️ 정비 절차**")
                        steps_df = pd.DataFrame([
                            {"단계": str(s["step"]), "작업": s["title"],
                             "소요시간(분)": f"{s['time_range_minutes'][0]}~{s['time_range_minutes'][1]}"}
                            for s in procedure_steps
                        ])
                        st.dataframe(steps_df, use_container_width=True, hide_index=True)

                    if maintenance_plan.get("situation_summary"):
                        st.caption(f"💬 {maintenance_plan['situation_summary']}")

            # ---- 4. Agent 4 — 최종 리포트 ----
            with status_ph.container():
                render_pipeline_status(4)

            with st.container(border=True):
                st.subheader("4️⃣ 최종 리포트 (Agent 4)")
                report_mode_choice = st.radio("리포트 형식", ["📄 formal (보고용)", "⚡ quick (확인용)"], horizontal=True)
                mode = "formal" if report_mode_choice.startswith("📄") else "quick"

                report_cache = st.session_state.setdefault("report_cache", {})
                if mode not in report_cache:
                    with st.spinner(f"{mode} 리포트 생성 중 (LLM 호출)..."):
                        report_cache[mode] = generate_report(pipeline_state, mode=mode, save_pdf=True)

                report_result = report_cache[mode]
                st.text_area("리포트 미리보기", report_result["report_text"], height=250)

                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        "⬇️ TXT 다운로드",
                        report_result["report_text"],
                        file_name=f"report_{selected_cutter}_{mode}.txt",
                        use_container_width=True,
                    )
                pdf_path = report_result.get("report_paths", {}).get("pdf")
                with dl_col2:
                    if pdf_path and Path(pdf_path).exists():
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "⬇️ PDF 다운로드",
                                f.read(),
                                file_name=f"report_{selected_cutter}_{mode}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                    else:
                        st.caption("PDF 저장 실패 — TXT만 이용 가능")

            # Agent 4까지 실제로 다 끝났으니 진행 표시기를 전부 완료 상태로 갱신
            with status_ph.container():
                render_pipeline_status(5)

# ===========================================================================
# 탭 2 — 모델 성능 검증 (test set 전체, 독립적인 대시보드)
# ===========================================================================
with tab2:
    st.subheader("test set 전체 기준 모델 성능 검증")
    st.caption("실제 rul_true를 알고 있는 test set(c6)으로 모델 예측 정확도를 확인하는 화면입니다.")

    model = get_model()
    X_test, y_test, cutter_test = get_test_data()
    val_selected_cutter = "c6"

    val_play_clicked = st.button("▶ 실시간 재생", key="val_play")

    val_should_animate = val_play_clicked or (st.session_state.get("val_animated_cutter") != val_selected_cutter)

    if val_should_animate or "val_cutter_df_cache" not in st.session_state or st.session_state.get("val_cutter_df_cutter") != val_selected_cutter:
        with st.spinner("test set 추론 중..."):
            val_cutter_df = build_real_log(val_selected_cutter, model, cutter_test, X_test, y_test, n_iter_preview=20)
        st.session_state["val_cutter_df_cache"] = val_cutter_df
        st.session_state["val_cutter_df_cutter"] = val_selected_cutter
    else:
        val_cutter_df = st.session_state["val_cutter_df_cache"]

    val_n_total = len(val_cutter_df)

    val_ph_metrics = st.empty()
    val_ph_graph1 = st.empty()
    val_col_cal, val_col_res = st.columns(2)
    val_ph_cal, val_ph_res = val_col_cal.empty(), val_col_res.empty()

    if val_should_animate:
        step_size = max(1, val_n_total // 40)
        for n_visible in range(step_size, val_n_total + step_size, step_size):
            partial = val_cutter_df.iloc[:min(n_visible, val_n_total)]
            draw_frame_graphs_validation(
                partial, val_cutter_df, val_n_total,
                val_ph_metrics, val_ph_graph1, val_ph_cal, val_ph_res,
                val_selected_cutter,
            )
            time.sleep(0.05)
        st.session_state["val_animated_cutter"] = val_selected_cutter
    else:
        draw_frame_graphs_validation(
            val_cutter_df, val_cutter_df, val_n_total,
            val_ph_metrics, val_ph_graph1, val_ph_cal, val_ph_res,
            val_selected_cutter,
        )

    st.caption(
        "참고: test set(c6)은 train(c1, c4)과 분포가 달라 예측 오차가 다소 크게 나타남 — "
        "전처리 단계에서부터 알려진 도메인 시프트 이슈. 여기 그래프도 속도를 위해 "
        "n_iter=20으로 계산된 미리보기입니다."
    )