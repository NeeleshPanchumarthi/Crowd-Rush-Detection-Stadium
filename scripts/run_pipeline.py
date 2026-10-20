"""Run one gate end to end without the dashboard (offline, repeatable).

Writes outputs/<gate>/metrics.csv (one row per processed frame), the event log, and optionally an
annotated video. DM-Count runs inline ("sync") so results do not depend on machine load.

Usage:
  python scripts/run_pipeline.py configs/gates/gate_A.yaml --video
  python scripts/run_pipeline.py configs/gates/gate_B.yaml --max-seconds 30 --every 2
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crowdgate.config import load_all_gates, load_gate_config  # noqa: E402
from crowdgate.pipeline import GatePipeline  # noqa: E402
from crowdgate.response import EventLog  # noqa: E402
from crowdgate.video import open_stream  # noqa: E402

COLS = ["frame_idx", "t", "level", "mode", "count", "density", "speed", "motion_variance", "inflow_ppm",
        "outflow_ppm", "accumulation_ppm", "score", "proc_ms", "latency_ms", "density_age_s", "alert"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--source", default=None, help="override the video source in the config")
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--video", action="store_true", help="also write an annotated mp4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_gate_config(args.config)
    all_cfgs = load_all_gates(Path(args.config).parent)
    src = args.source or cfg.resolve(cfg["source"])
    stream = open_stream(src, folder_fps=cfg.get("source_fps") or 2.0)
    out = Path(args.out or ROOT / "outputs" / f"gate_{cfg.gate_id}")
    out.mkdir(parents=True, exist_ok=True)
    log = EventLog(out)
    log.clear()
    pipe = GatePipeline(cfg, (stream.height, stream.width), sync_density=True, event_log=log, all_cfgs=all_cfgs)
    print(f"{cfg.name}: source={src} fps={stream.fps:.1f} frames={stream.n_frames} roi={pipe.roi} "
          f"zone={pipe.zone_area:.1f} m2, {len(pipe.grid_small)} flow samples")

    vw = None
    f = open(out / "metrics.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(COLS)
    n, t0 = 0, time.perf_counter()
    proc = []
    while True:
        ok, frame, idx = stream.read()
        if not ok:
            break
        if idx % args.every:
            continue
        t = idx / stream.fps
        if args.max_seconds and t > args.max_seconds:
            break
        st = pipe.process(frame, t, idx, capture_wall=time.time())
        proc.append(st.proc_ms)
        d = st.as_dict()
        w.writerow([round(d[c], 4) if isinstance(d[c], float) else d[c] for c in COLS])
        if st.alert:
            print(f"  t={t:6.1f}s  {st.alert.previous} -> {st.alert.level}: {st.alert.message}")
        if args.video:
            vis = pipe.draw(frame, st)
            if vw is None:
                vw = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     stream.fps / args.every, (vis.shape[1], vis.shape[0]))
            vw.write(vis)
        n += 1
        if n % 100 == 0:
            print(f"  t={t:6.1f}s level={st.level:5s} mode={st.mode:6s} n={st.count:5.1f} "
                  f"d={st.density:4.2f} v={st.speed:5.2f} var={st.motion_variance:.3f} "
                  f"acc={st.accumulation_ppm:6.1f} score={st.score:.2f}")
    f.close()
    if vw:
        vw.release()
    pipe.close()
    wall = time.perf_counter() - t0
    proc.sort()
    print(f"Processed {n} frames in {wall:.1f}s ({n / wall:.2f} FPS incl. inline DM-Count); "
          f"median per-frame {proc[len(proc) // 2]:.0f} ms, p95 {proc[int(len(proc) * 0.95)]:.0f} ms")
    print("Outputs in", out)


if __name__ == "__main__":
    main()
