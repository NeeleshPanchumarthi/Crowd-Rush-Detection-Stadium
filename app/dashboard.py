"""Streamlit dashboard: live video per gate, level, charts and the alert list.

Run:  streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crowdgate.config import load_all_gates  # noqa: E402
from crowdgate.response import EventLog  # noqa: E402
from crowdgate.runner import GateRunner  # noqa: E402

st.set_page_config(page_title="Crowd gate monitor", layout="wide")

COLORS = {"green": "#2e9e4f", "amber": "#f0a000", "red": "#d62f2f"}
CFGS = load_all_gates(ROOT / "configs" / "gates")
LOG = EventLog(ROOT / "outputs" / "dashboard")


@st.cache_resource
def session_store():
    return {"runners": {}, "shared": {}}


store = session_store()

# ---- sidebar: choose gates and start/stop -------------------------------------------------------
with st.sidebar:
    st.header("Gates")
    chosen = st.multiselect("Monitor", list(CFGS), default=["A"], format_func=lambda g: CFGS[g].name)
    speed = st.select_slider("Playback speed", [0.5, 1.0, 2.0, 4.0], value=1.0)
    c1, c2 = st.columns(2)
    start = c1.button("Start", type="primary")
    stop = c2.button("Stop")
    if st.button("Clear event log"):
        LOG.clear()
    heat = st.checkbox("Density heat-map overlay", value=True)
    st.caption("CPU only: YOLO11n + ByteTrack when sparse, DM-Count (SHB) when dense, Farnebäck flow always.")

if stop or start:
    for r in store["runners"].values():
        r.stop()
    store["runners"].clear()
    store["shared"].clear()
if start:
    for g in chosen:
        store["runners"][g] = GateRunner(CFGS[g], CFGS, store["shared"], LOG, speed=speed)

st.title("Stadium exit gates: crowd risk")


def banner(level: str, text: str):
    st.markdown(f"<div style='background:{COLORS[level]};color:white;padding:6px 12px;border-radius:6px;"
                f"font-weight:600'>{text}</div>", unsafe_allow_html=True)


@st.fragment(run_every=0.5)
def live():
    runners = store["runners"]
    if not runners:
        st.info("Pick one or more gates and press Start.")
        return
    cols = st.columns(len(runners))
    for col, (gid, r) in zip(cols, runners.items()):
        with col:
            stt = r.state
            if stt is None:
                st.write(f"{r.cfg.name}: starting…")
                continue
            banner(stt.level, f"{r.cfg.name}: {stt.level.upper()}" + ("  (clip ended)" if r.finished else ""))
            if r.frame is not None:
                frame = r.frame if heat else r.frame_plain
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("People/m²", f"{stt.density:.2f}")
            m2.metric("Speed m/s", "–" if stt.speed != stt.speed else f"{stt.speed:.2f}")
            m3.metric("Risk", f"{stt.score:.2f}")
            m4.metric("Mode", stt.mode)
            st.caption(f"{r.fps:.1f} FPS · frame→screen latency {stt.latency_ms:.0f} ms · "
                       f"video t={stt.t:.0f}s · build-up {stt.accumulation_ppm:+.0f}/min")
            if len(r.history) > 3:
                df = pd.DataFrame(list(r.history)).set_index("t")
                st.line_chart(df[["density"]], height=140)
                st.line_chart(df[["speed", "score"]], height=140)

    st.subheader("Alerts")
    rows = LOG.recent(30)
    if not rows:
        st.write("No level changes yet.")
    for e in rows:
        if e["level"] == "green":
            icon = "🟢"
        else:
            icon = "🔴" if e["level"] == "red" else "🟠"
        st.markdown(f"{icon} **{e['gate_name']}** · video t={e['video_time']:.0f}s · "
                    f"{e['previous']} → **{e['level']}** · {e['message']} "
                    f"<span style='color:gray'>(density {e['density']:.2f}/m², latency "
                    f"{(e['latency_ms'] or 0):.0f} ms)</span>", unsafe_allow_html=True)


live()
