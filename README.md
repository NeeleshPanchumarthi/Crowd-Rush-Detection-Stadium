# Crowd gate system: Part 1 (core prototype)

CPU-only, pretrained models only: YOLO11n + ByteTrack (sparse), DM-Count (dense), Farnebäck flow (always),
homography to people/m², risk score with hold time, alerts + event log, Streamlit dashboard.

## Setup (already done on this machine)
```
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install ultralytics streamlit pyyaml pandas lap scipy gdown
```
Weights: `models/yolo11n.pt`, `models/dmcount/model_{sh_B,sh_A,qnrf,nwpu}.pth` (DM-Count Google Drive folder).
Footage: `data/vtest.avi` (OpenCV sample), `data/mall_dataset/` (Mall), `data/synthetic/` (made by script).

## Run
| What | Command |
|---|---|
| Dashboard | `.venv\Scripts\streamlit run app/dashboard.py` then pick Gate A, press Start |
| Demo without UI | `.venv\Scripts\python scripts/run_pipeline.py configs/gates/gate_A.yaml --video` |
| Regenerate demo clip | `.venv\Scripts\python scripts/make_synthetic_gate.py` |
| Calibrate a camera | `.venv\Scripts\python scripts/calibrate.py configs/gates/gate_X.yaml` |
| Counting check (Mall) | `.venv\Scripts\python scripts/evaluate.py` |
| Unit tests | `.venv\Scripts\python tests/test_core.py` |

## Layout
`crowdgate/` config, video, calibration, geometry, perception/{detector,density,flow,mode}, metrics, risk,
response, pipeline (per-gate glue), runner (dashboard thread). `configs/gates/*.yaml` one file per gate.

## Results (i7-1255U, 16 GB, no GPU)
* **Demo (Gate A, synthetic clip, 150 s):** green until t=82 s, amber at 82 s (true density 1.4/m², stalled and
  building up), red at 128 s (true density 4.1/m²). Dispatch message "Send security to Gate A. Redirect crowd to
  Gate B." and both level changes are in `outputs/gate_A/events.csv` and on the dashboard.
* **Speed:** offline 8–13 FPS per gate incl. DM-Count inline; dashboard 5–8 FPS with DM-Count in a background thread.
  YOLO11n+ByteTrack ≈ 100 ms/frame, DM-Count ≈ 0.3 s on the gate crop, ≈ 1.9 s on a full 640×480 frame.
* **Alert latency** (frame captured → alert logged): 20–520 ms in offline runs, 60–250 ms in the dashboard.
* **Counting, Mall (100 frames, whole frame):** see `outputs/mall_counting_eval.txt`. DM-Count SHB MAE 10.4 (under-counts
  by a third), YOLO11n 13.0. Same frames with DM-Count **QNRF weights at 0.75× input scale: MAE ≈ 3.1**. Switch with
  `perception: {dm_weights: model_qnrf.pth, density_scale: 0.75}` in a gate file.

## Known limits
* The demo is synthetic (real pedestrian cut-outs on a simulated crowd), because no dataset labels dangerous compression.
  Real-clip checks (vtest, Mall) stay green, as they should.
* Gate B/C homographies are eyeballed, not surveyed. Calibrate with `scripts/calibrate.py` for real distances.
* ShanghaiTech B MAE was not run: the dataset needs a manual download (not in the project folder).
* Dense-mode speed and flux come from optical flow on a bird's-eye of heads; they are estimates, not tracks.
