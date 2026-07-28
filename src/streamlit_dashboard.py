import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# 설정
# =========================================================
st.set_page_config(page_title="Agent 1 - RUL 모니터링", layout="wide")

# run_agent1_stream.py / run_agent1_easy.ps1 과 동일한 환경변수를 읽어서
# 로그 경로를 맞춤 (환경변수 없으면 기존 기본값 그대로 사용)
LOG_PATH = os.environ.get("AGENT1_LOG_PATH", "agent1_stream_log.csv")
RUL_THRESHOLDS = {"RED": 20.7, "YELLOW": 50.25}
REFRESH_SEC = 0.5                        # 몇 초마다 화면을 새로고침할지

st.title("🔧 Agent 1 - RUL 예측 & 신뢰구간 실시간 모니터링")

# =========================================================
# 데이터 로드 (매 새로고침마다 CSV를 다시 읽음 -> 파일이 커지면 그래프도 커짐)
# =========================================================
if not Path(LOG_PATH).exists():
    st.warning(f"{LOG_PATH} 파일이 아직 없습니다. run_agent1_stream.py가 실행 중인지 확인해주세요.")
    st.stop()

log_df_full = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])

if log_df_full.empty:
    st.info("아직 기록된 데이터가 없습니다. 곧 첫 예측값이 들어올 거예요.")
    st.stop()

# ---------------------------------------------------------
# 점진적 애니메이션: producer가 이미 끝났어도 화면에는
# 한 번에 다 뿌리지 않고, 매 새로고침마다 몇 개씩만 더 보여줌
# ---------------------------------------------------------
POINTS_PER_REFRESH = 1   # 새로고침 한 번마다 몇 개씩 더 보여줄지

if "n_shown" not in st.session_state:
    st.session_state.n_shown = 1

# CSV에 실제로 쌓인 개수를 넘어서지 않게, 그리고 조금씩만 늘어나게
st.session_state.n_shown = min(
    len(log_df_full), st.session_state.n_shown + POINTS_PER_REFRESH
)

log_df = log_df_full.iloc[: st.session_state.n_shown].copy()

latest = log_df.iloc[-1]

# =========================================================
# 상단 요약 지표
# =========================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("처리된 윈도우 수", f"{len(log_df)} / {len(log_df_full)}")
col2.metric("최근 cutter", latest["cutter_id"])
col3.metric("최근 예측 RUL", f'{latest["rul_pred"]:.1f}')
col4.metric("최근 CI 폭", f'{latest["rul_ci_upper"] - latest["rul_ci_lower"]:.1f}')

st.markdown("---")

# =========================================================
# ① 실시간 RUL 예측 + 90% CI (cutter 선택 가능)
# =========================================================
st.subheader("① RUL 예측 & 신뢰구간")

cutter_options = sorted(log_df["cutter_id"].unique())
selected_cutters = st.multiselect("표시할 cutter 선택", options=cutter_options,
                                   default=[cutter_options[-1]])

fig1 = go.Figure()
for c_id in selected_cutters:
    sub = log_df[log_df["cutter_id"] == c_id].sort_values("cutter_step")
    fig1.add_trace(go.Scatter(
        x=sub["cutter_step"], y=sub["rul_ci_upper"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig1.add_trace(go.Scatter(
        x=sub["cutter_step"], y=sub["rul_ci_lower"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(70,70,150,0.35)", name=f"90% CI ({c_id})"
    ))
    fig1.add_trace(go.Scatter(
        x=sub["cutter_step"], y=sub["rul_pred"],
        mode="lines+markers", line=dict(color="#6a6aff", width=2),
        name=f"예측 RUL ({c_id})"
    ))
    fig1.add_trace(go.Scatter(
        x=sub["cutter_step"], y=sub["rul_true"],
        mode="lines", line=dict(color="#e34948", width=2, dash="dash"),
        name=f"실제 RUL ({c_id})"
    ))

fig1.update_layout(
    xaxis_title="cutter 내 순번 (cut)", yaxis_title="RUL",
    template="plotly_dark", height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig1, width='stretch')

# =========================================================
# ② 현재 상태 미터 (RED/YELLOW/GREEN 게이지)
# =========================================================
st.subheader("② 현재 상태 미터")

vmax = max(220, log_df["rul_pred"].max() * 1.1)
fig2 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=float(latest["rul_pred"]),
    number={"suffix": " RUL"},
    gauge={
        "axis": {"range": [0, vmax]},
        "bar": {"color": "black"},
        "steps": [
            {"range": [0, RUL_THRESHOLDS["RED"]], "color": "#e34948"},
            {"range": [RUL_THRESHOLDS["RED"], RUL_THRESHOLDS["YELLOW"]], "color": "#eda100"},
            {"range": [RUL_THRESHOLDS["YELLOW"], vmax], "color": "#008300"},
        ],
    },
))
fig2.update_layout(template="plotly_dark", height=300)
st.plotly_chart(fig2, width='stretch')

# =========================================================
# ③ 구간별 누적 시간 비율 (누적 막대)
# =========================================================
st.subheader("③ 구간별 누적 시간 비율")

bins = [-1, RUL_THRESHOLDS["RED"], RUL_THRESHOLDS["YELLOW"], np.inf]
zone = pd.cut(log_df["rul_pred"], bins=bins, labels=["RED", "YELLOW", "GREEN"])
zone_pct = zone.value_counts(normalize=True).reindex(["RED", "YELLOW", "GREEN"]).fillna(0) * 100

fig3 = go.Figure()
left = 0
for name, color in zip(["RED", "YELLOW", "GREEN"], ["#e34948", "#eda100", "#008300"]):
    fig3.add_trace(go.Bar(
        x=[zone_pct[name]], y=["전체 기간"], orientation="h",
        name=f"{name} {zone_pct[name]:.0f}%", marker_color=color,
    ))
    left += zone_pct[name]

fig3.update_layout(
    barmode="stack", xaxis=dict(range=[0, 100], title="비율 (%)"),
    template="plotly_dark", height=180,
    legend=dict(orientation="h", yanchor="bottom", y=1.1),
)
st.plotly_chart(fig3, width='stretch')

# =========================================================
# ④ 예측 보정도 (산점도, calibration check)
# =========================================================
st.subheader("④ 예측 보정도 (calibration)")

lim_max = max(log_df["rul_true"].max(), log_df["rul_pred"].max())
fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=log_df["rul_true"], y=log_df["rul_pred"],
    mode="markers", marker=dict(color="#2a78d6", size=6, opacity=0.6),
    name="예측 vs 실제",
))
fig4.add_trace(go.Scatter(
    x=[0, lim_max], y=[0, lim_max],
    mode="lines", line=dict(color="gray", dash="dash"),
    name="완벽한 예측 (y=x)",
))
fig4.update_layout(
    xaxis_title="실제 RUL", yaxis_title="예측 RUL",
    template="plotly_dark", height=400,
)
st.plotly_chart(fig4, width='stretch')

# =========================================================
# ⑤ 잔차 분포 (히스토그램)
# =========================================================
st.subheader("⑤ 잔차 분포")

residual = log_df["rul_pred"] - log_df["rul_true"]
fig5 = go.Figure(go.Histogram(x=residual, nbinsx=20, marker_color="#2a78d6"))
fig5.add_vline(x=0, line_dash="dash", line_color="gray")
fig5.update_layout(
    xaxis_title="예측 - 실제 (잔차)", yaxis_title="빈도",
    title=f"평균 오차: {residual.mean():.2f}",
    template="plotly_dark", height=350,
)
st.plotly_chart(fig5, width='stretch')

# =========================================================
# 원본 로그 데이터 보기
# =========================================================
with st.expander("원본 로그 데이터 보기"):
    st.dataframe(log_df.tail(50), width='stretch')

# =========================================================
# 자동 새로고침 (핵심: 이게 있어야 계속 늘어나는 걸 볼 수 있음)
# =========================================================
time.sleep(REFRESH_SEC)
st.rerun()
