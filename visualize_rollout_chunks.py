#!/usr/bin/env python3
"""Rerun visualizer for rollout trajectories vs. the policy's predicted action chunks.

Per inference cycle, overlays the predicted chunk (``policy_logs/episode_<n>.jsonl``)
on the trajectory in the rollout's replay buffer, for every config arm (the chunk
is ``n_arms`` equal blocks of pos + rotvec (+ gripper)). Both log schemas work:

- offline (``step_in_episode`` / ``n_waypoints``): the trajectory is the replayed
  demo, so chunk vs trajectory reads as prediction accuracy
- online (``cycle`` / ``obs_ts`` / ``sched_ts_offsets``): the trajectory is what the
  arm executed (plan vs execution); cycles are placed by matching ``obs_ts`` to the
  replay buffer timestamps, ``obs_lowdim`` (the model's eef input) drawn as blue dots

Online rollouts with width grippers get a gripper tab per arm: the RAW width of
chunks t-1 / t / t+1 (the log stores it after deploy.gripper_offset_m, inverted
here) and what was executed (sent / time-indexed / measured). Logs with
``gripper_schedule: ["target_time"]`` ran rows at their target_time; older logs
ran each chunk's LAST row on arrival.

Several paths overlay the same episode of each under ``world/<dataset>/``
(camera- and gripper-free). Layout::

    <path>/
        config.json             # own config (newer rollouts) or pass --config
        replay_buffer.zarr/
        policy_logs/episode_<n>.jsonl

Examples::

    python visualize_rollout_chunks.py <rollout> --episode 0 --serve
    python visualize_rollout_chunks.py <offline_rollout> --config <train_set>/config.json
    python visualize_rollout_chunks.py <rollout_a> <rollout_b> --episode 0   # A/B overlay
"""

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from viz_common import (
    DATASET_COLORS,
    add_arm_pose_args,
    add_camera_args,
    add_sink_args,
    add_tip_args,
    data_fps,
    decode_and_log_video,
    episode_range,
    load_config,
    load_depth_scale,
    load_store,
    load_tool_tip,
    log_world_axes,
    resolve_arms,
    resolve_cameras,
    rot_to_matrix,
    scene_column,
    select_sink,
    wait_if_serving,
)

CHUNK_COLOR = [255, 160, 0]        # full predicted chunk (incl. unused tail)
CHUNK_USED_COLOR = [255, 60, 200]  # portion actually executed before replanning
GT_FUTURE_COLOR = [0, 220, 220]    # what actually happened over that same horizon
PERTURB_GT_COLOR = [110, 220, 110]  # GT re-expressed in the perturbed origin frame
MODEL_OBS_COLOR = [80, 160, 255]   # eef pose(s) the model actually saw (obs_lowdim)
TIP_COLOR = [235, 235, 235]        # torch tip (--tip), grey against the orange chunks
SEAM_GT_COLOR = [60, 255, 120]     # operator-taught seam line (--seams), tip level

# Gripper panels: hand-drawn 2D plots, re-logged at every chunk's t_wall (a time-series
# view cannot move its window with the cursor). GRIP_SX px per s (x = 0 is chunk t's
# t_wall), GRIP_SY px per mm, y from 0 (bottom) to GRIP_YMAX mm.
GRIP_ROWS = (("tm1", -1, "chunk t-1"), ("t", 0, "chunk t"), ("tp1", 1, "chunk t+1"))
GRIP_ROW_COLORS = {-1: [70, 140, 255], 0: [255, 150, 30], 1: [70, 200, 70]}
GRIP_EXEC_COLOR = [240, 240, 240]  # what the gripper was sent
GRIP_TI_COLOR = [230, 50, 50]      # time-indexed value: chunk row whose timestamp is now
GRIP_MEAS_COLOR = [150, 150, 150]  # measured jaw width
GRIP_GRID_COLOR = [70, 70, 70]
GRIP_SX, GRIP_SY, GRIP_YMAX = 400.0, 4.0, 90.0


def load_policy_log(policy_logs_dir, episode):
    """(cycle entries, perturb, run_cfg); None for whatever is absent.

    ``perturb`` = the ``{"event": "perturb", "yaw_rad", "trans_m"}`` header of
    offline_rollout_with_server.py --perturb_* runs; ``run_cfg`` = the
    ``{"event": "run_config", ...}`` header of online eval rollouts (eval_pose_id).
    """
    path = Path(policy_logs_dir) / f"episode_{episode}.jsonl"
    if not path.exists():
        return None, None, None
    with open(path) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    perturb = next((e for e in entries if e.get("event") == "perturb"), None)
    run_cfg = next((e for e in entries if e.get("event") == "run_config"), None)
    return [e for e in entries if "action_chunk" in e], perturb, run_cfg


def synthesize_config(store, path):
    """Minimal config for rollout dirs without config.json (old offline rollouts):
    arms probed from the zarr's ``<name>_eef_pos`` / ``<name>_slam_eef_pos`` keys."""
    names = sorted({m.group(1) for k in store["data"].keys()
                    if (m := re.fullmatch(r"(.+?)_(?:slam_)?eef_pos", k))})
    if not names:
        raise ValueError(f"{path}: no config.json and no *_eef_pos keys in zarr to probe arms from")
    return {"name": Path(str(path).rstrip("/")).name,
            "arms": [{"obs_name": n} for n in names]}


# --------------------------------------------------------------------------- #
# Taught seam (--seams)
# --------------------------------------------------------------------------- #
def load_seam_line(seams_path, pose_id, tip_off):
    """(ws_tip, we_tip, weld_rotvec) of ``pose_id`` in a real_robot teach_seam.py
    seams.json, or None. Entries store canonical TCP endpoints
    (tip - R(weld_rotvec) @ tip_offset); adding R @ tip_offset recovers the tip line."""
    with open(seams_path) as f:
        entries = json.load(f)
    rec = next((e for e in entries if e.get("pose_id") == pose_id), None)
    if rec is None:
        return None
    rv = np.asarray(rec["weld_rotvec"], dtype=float)
    off = rot_to_matrix(rv) @ np.asarray(tip_off, dtype=float)
    return (np.asarray(rec["weld_start_xyz"], dtype=float) + off,
            np.asarray(rec["weld_end_xyz"], dtype=float) + off, rv)


def seam_overlay(args, run_cfg, arm, ent, prefix, tip_pos, tip_off, gt_rot):
    """Log the static taught seam; return per-step tip-to-seam errors and torch angles
    (same conventions as score_seam_session), or None when it cannot be drawn."""
    if arm.frame != "base":
        print(f"  [warn] --seams is base-frame; arm '{arm.name}' is in the "
              f"'{arm.frame}' frame — skipping seam overlay")
        return None
    pid = args.seam_pose if args.seam_pose is not None else (run_cfg or {}).get("eval_pose_id")
    if pid is None:
        print("  [warn] --seams: no eval_pose_id in the policy log and no "
              "--seam-pose given — skipping seam overlay")
        return None
    seam = load_seam_line(args.seams, int(pid), tip_off)
    if seam is None:
        print(f"  [warn] --seams: pose_id {pid} not in {args.seams} — skipping")
        return None
    ws, we, seam_rv = seam
    rs, ds_label = args.radius_scale, prefix.rstrip("/")
    rr.log(f"{ent}/seam_gt", rr.LineStrips3D([np.stack([ws, we])], colors=np.array([SEAM_GT_COLOR]),
                                             radii=rs * 0.0004), static=True)
    rr.log(f"{ent}/seam_gt_ends", rr.Points3D(np.stack([ws, we]), colors=np.array([SEAM_GT_COLOR]),
                                              radii=rs * 0.002), static=True)

    # lateral along n = z x u (horizontal), z = vertical residual
    u = (we - ws) / np.linalg.norm(we - ws)
    n = np.cross([0.0, 0.0, 1.0], u)
    n /= np.linalg.norm(n)
    foot = ws + ((tip_pos - ws) @ u)[:, None] * u  # closest point on the seam line
    d = tip_pos - foot
    # torch axis = R(rotvec) @ tip direction: work vs vertical, travel along the seam
    t_hat = np.asarray(tip_off, dtype=float) / np.linalg.norm(tip_off)
    axes = np.stack([rot_to_matrix(r) @ t_hat for r in gt_rot])
    taught_axis = rot_to_matrix(seam_rv) @ t_hat
    sm = SimpleNamespace(
        foot=foot, lat=d @ n, z=d[:, 2], dist=np.linalg.norm(d, axis=1), axes=axes,
        work=np.degrees(np.arccos(np.clip(-axes[:, 2], -1, 1))),
        travel=np.degrees(np.arcsin(np.clip(axes @ u, -1, 1))),
        taught_axis=taught_axis,
        taught_work=float(np.degrees(np.arccos(np.clip(-taught_axis[2], -1, 1)))),
        taught_travel=float(np.degrees(np.arcsin(np.clip(taught_axis @ u, -1, 1)))))
    for origin, names in (("seam_error", (("lateral_mm", "lateral"), ("z_mm", "z"),
                                          ("dist_mm", "3D dist"))),
                          ("seam_angles", (("work_deg", "work"), ("travel_deg", "travel"),
                                           ("work_taught_deg", "work (taught)"),
                                           ("travel_taught_deg", "travel (taught)")))):
        for suffix, label in names:
            rr.log(f"{origin}/{prefix}{suffix}",
                   rr.SeriesLines(names=[f"{ds_label} {label}" if ds_label else label]), static=True)
    low = tip_pos[:, 2] < tip_pos[:, 2].min() + 0.012
    print(f"  [seam] pose {pid}, low-z phase ({int(low.sum())} steps): "
          f"lateral {np.median(sm.lat[low])*1000:+.1f} mm  "
          f"z {np.median(sm.z[low])*1000:+.1f} mm  "
          f"3D dist median {np.median(sm.dist[low])*1000:.1f} "
          f"[{sm.dist[low].min()*1000:.1f}..{sm.dist[low].max()*1000:.1f}] mm  |  "
          f"work {np.median(sm.work[low]):.1f} deg (taught {sm.taught_work:.1f})  "
          f"travel {np.median(sm.travel[low]):+.1f} deg (taught {sm.taught_travel:+.1f})")
    return sm


def log_seam_step(sm, i, ent, prefix, tip_p, rs):
    lat_mm, z_mm, dist_mm = sm.lat[i] * 1000, sm.z[i] * 1000, sm.dist[i] * 1000
    rr.log(f"seam_error/{prefix}lateral_mm", rr.Scalars(lat_mm))
    rr.log(f"seam_error/{prefix}z_mm", rr.Scalars(z_mm))
    rr.log(f"seam_error/{prefix}dist_mm", rr.Scalars(dist_mm))
    rr.log(f"seam_angles/{prefix}work_deg", rr.Scalars(sm.work[i]))
    rr.log(f"seam_angles/{prefix}travel_deg", rr.Scalars(sm.travel[i]))
    rr.log(f"seam_angles/{prefix}work_taught_deg", rr.Scalars(sm.taught_work))
    rr.log(f"seam_angles/{prefix}travel_taught_deg", rr.Scalars(sm.taught_travel))
    # tip -> closest seam point, traffic-light colored
    col = [60, 220, 60] if dist_mm < 5.0 else [235, 200, 60] if dist_mm < 10.0 else [235, 70, 70]
    rr.log(f"{ent}/tip_err", rr.LineStrips3D([np.stack([sm.foot[i], tip_p])],
                                             colors=np.array([col]), radii=rs * 0.0003))
    # torch direction at the tip: actual (orange) vs taught (green)
    rr.log(f"{ent}/torch_axis",
           rr.Arrows3D(origins=[tip_p, tip_p], vectors=[-sm.axes[i] * 0.05, -sm.taught_axis * 0.05],
                       colors=np.array([[255, 140, 0], SEAM_GT_COLOR]), radii=rs * 0.0004))
    rr.log(f"seam_readout/{prefix}now",
           rr.TextDocument(f"lateral {lat_mm:+7.1f} mm\n"
                           f"z       {z_mm:+7.1f} mm\n"
                           f"3D dist {dist_mm:7.1f} mm\n"
                           f"work    {sm.work[i]:7.1f} deg (taught {sm.taught_work:.1f})\n"
                           f"travel  {sm.travel[i]:+7.1f} deg (taught {sm.taught_travel:+.1f})"))


# --------------------------------------------------------------------------- #
# Gripper panels
# --------------------------------------------------------------------------- #
def gripper_arms(config, data_keys):
    """{obs_name: max_width_m} for every width-gripper arm with a measured key."""
    out = {}
    for a in config.get("arms", []):
        g = a.get("gripper") or {}
        if g.get("type") in ("robotiq", "dynamixel") and f"{a['obs_name']}_gripper" in data_keys:
            out[a["obs_name"]] = float(g.get("max_width_m", 0.085))
    return out


def gripper_window(log):
    """(lo, hi) s around chunk t's t_wall that hold chunks t-1 .. t+1 (None without t_wall)."""
    if not log or "t_wall" not in log[0] or len(log) < 3:
        return None
    lo = [log[k - 1]["obs_ts"] - log[k]["t_wall"] for k in range(1, len(log))]
    hi = [log[k + 1]["obs_ts"] + log[k + 1]["action_ts_offsets"][-1] - log[k]["t_wall"]
          for k in range(len(log) - 1)]
    lo = max(-3.0, np.floor(np.percentile(lo, 5) * 10) / 10 - 0.1)
    hi = min(4.0, np.ceil(np.percentile(hi, 95) * 10) / 10 + 0.1)
    return float(lo), float(hi)


def gripper_bounds(window):
    """VisualBounds2D ranges of one gripper panel (room for axis labels)."""
    lo, hi = window
    return [lo * GRIP_SX - 70, hi * GRIP_SX + 10], [-28, GRIP_YMAX * GRIP_SY + 30]


def unoffset(y, hi, d, dead=None):
    """Inverse of real_robot common.gripper.offset_grip_columns for one width column (m).
    Exact except at >= open and <= d, which the offset flattens."""
    y = np.asarray(y, dtype=float)
    if dead is not None:
        return np.where(y >= hi - 1e-9, max(float(dead), hi), y + d)
    return np.where(y >= hi - 1e-9, hi, np.where(y >= hi - 2 * d, (y + hi) / 2, y + d))


def _clip_poly(t, v, a, b):
    """Polyline points of (t, v) inside [a, b] plus interpolated end points."""
    t, v = np.asarray(t, dtype=float), np.asarray(v, dtype=float)
    if len(t) == 0 or t[-1] < a or t[0] > b:
        return np.zeros((0, 2))
    m = (t > a) & (t < b)
    ends_t = [x for x in (a, b) if t[0] <= x <= t[-1]]
    pts = np.c_[np.r_[t[m], ends_t], np.r_[v[m], np.interp(ends_t, t, v)]]
    return pts[np.argsort(pts[:, 0], kind="stable")]


def log_gripper_panels(log, name, col, hi_m, meas, ts, t0, window, deploy):
    """Four plots per arm under gripper/<name>/{tm1,t,tp1,exec}, re-logged at every
    chunk's t_wall: the RAW width of chunks t-1 / t / t+1 over their row timestamps,
    and what was executed. With gripper_schedule "target_time" the white dots are the
    rows each chunk ran; otherwise the jaws got the chunk's LAST row on arrival (the
    arrow in each chunk row). x labels = s since episode start."""
    base = f"gripper/{name}"
    lo, hi = window
    X = lambda t, now: (np.asarray(t, dtype=float) - now) * GRIP_SX  # noqa: E731
    Y = lambda mm: (GRIP_YMAX - np.asarray(mm, dtype=float)) * GRIP_SY  # noqa: E731
    x_lo, x_hi = lo * GRIP_SX, hi * GRIP_SX
    subs = [r[0] for r in GRIP_ROWS] + ["exec"]

    d = float(deploy.get("gripper_offset_m") or 0.0)
    dead = deploy.get("gripper_open_deadband_m")
    logged = [np.asarray(e["action_chunk"], dtype=float)[:, col] * 1e3 for e in log]
    if deploy.get("gripper_binarize"):
        print(f"  [gripper] {name}: gripper_binarize on -- chunk rows are the binarized widths")
        raw = logged
    else:
        raw = [unoffset(v / 1e3, hi_m, d, dead) * 1e3 for v in logged]
    tw = np.array([e["t_wall"] for e in log]) - t0
    obs = np.array([e["obs_ts"] for e in log]) - t0
    row_t = [e["obs_ts"] - t0 + np.asarray(e["action_ts_offsets"], dtype=float) for e in log]
    n = len(log)
    t_next = np.r_[tw[1:], np.inf]  # each chunk runs until the next one arrives

    # what each chunk made the gripper do
    timed = "target_time" in (log[-1].get("gripper_schedule") or [])
    g_lat = float(log[-1].get("gripper_latency_s") or 0.0) if timed else 0.0
    ev_t, ev_v = [], []
    for k in range(n):
        if timed:  # rows at row_ts - latency
            tt = row_t[k] - g_lat
            m = (tt >= tw[k]) & (tt < t_next[k])
            ev_t.append(tt[m])
            ev_v.append(logged[k][m])
        else:  # last row on arrival
            ev_t.append(tw[k:k + 1])
            ev_v.append(logged[k][-1:])
    all_t, all_v = np.concatenate(ev_t), np.concatenate(ev_v)
    t_end = max(float(ts[-1] - t0), float(all_t[-1]) if len(all_t) else 0.0)
    v0 = meas[0] * 1e3  # before the first chunk the jaws hold the measured start width
    step_t = np.r_[0.0, np.repeat(all_t, 2), t_end]
    step_v = np.r_[v0, np.repeat(np.r_[v0, all_v], 2)[1:-1], all_v[-1] if len(all_v) else v0]
    starts = [t[0] if len(t) else np.nan for t in ev_t]

    def span(k):
        """[first command of chunk k, first command of the next chunk that commanded anything)."""
        nxt = [s for s in starts[k + 1:] if np.isfinite(s)]
        return starts[k], (nxt[0] if nxt else t_end)

    in_cycle = [(row_t[k] >= tw[k]) & (row_t[k] < t_next[k]) for k in range(n)]
    ti_t = np.concatenate([row_t[k][m] for k, m in enumerate(in_cycle)])
    ti_v = np.concatenate([logged[k][m] for k, m in enumerate(in_cycle)])
    meas_t, meas_mm = np.asarray(ts, dtype=float) - t0, meas * 1e3

    # static axes: y grid + labels, the "now" line
    ys = [0, 20, 40, 60, 80]
    for sub in subs:
        ent = f"{base}/{sub}"
        rr.log(f"{ent}/axes/grid_y",
               rr.LineStrips2D([[[x_lo, Y(y)], [x_hi, Y(y)]] for y in ys],
                               colors=GRIP_GRID_COLOR, radii=0.4), static=True)
        rr.log(f"{ent}/axes/labels_y",
               rr.Points2D([[x_lo - 8, Y(y)] for y in ys], labels=[f"{y} mm" for y in ys],
                           radii=0.01, colors=[200, 200, 200], show_labels=True), static=True)
        rr.log(f"{ent}/axes/now",
               rr.LineStrips2D([[[0, Y(0)], [0, Y(GRIP_YMAX)]]], colors=[255, 255, 255], radii=0.6),
               static=True)

    def vline(t, now):
        return [[X(t, now), Y(0)], [X(t, now), Y(GRIP_YMAX)]]

    for k in range(n):
        now = tw[k]
        rr.set_time("time", duration=now)
        a, b = now + lo, now + hi
        ticks = np.arange(np.ceil(a / 0.25) * 0.25, b, 0.25)
        for sub in subs:
            ent = f"{base}/{sub}"
            rr.log(f"{ent}/axes/grid_x",
                   rr.LineStrips2D([vline(t, now) for t in ticks], colors=GRIP_GRID_COLOR, radii=0.4))
            rr.log(f"{ent}/axes/labels_x",
                   rr.Points2D([[X(t, now), Y(0) + 14] for t in ticks], labels=[f"{t:.2f}" for t in ticks],
                               radii=0.01, colors=[200, 200, 200], show_labels=True))

        for sub, dk, label in GRIP_ROWS:
            kk = k + dk
            ent = f"{base}/{sub}/cyc"
            if not 0 <= kk < n:
                rr.log(ent, rr.Clear(recursive=True))
                continue
            c = GRIP_ROW_COLORS[dk]
            m = (row_t[kk] >= a) & (row_t[kk] <= b)
            pts = np.c_[X(row_t[kk][m], now), Y(raw[kk][m])]
            rr.log(f"{ent}/title",
                   rr.Points2D([[x_lo, -14]], labels=[f"{label} = cycle {kk}  (raw policy width)"],
                               radii=0.01, colors=c, show_labels=True))
            rr.log(f"{ent}/raw", rr.LineStrips2D([pts], colors=c, radii=1.2))
            rr.log(f"{ent}/rows", rr.Points2D(pts, colors=c, radii=2.2))
            rr.log(f"{ent}/obs_ts", rr.LineStrips2D([vline(obs[kk], now)], colors=c, radii=0.5))
            rr.log(f"{ent}/t_wall", rr.LineStrips2D([vline(tw[kk], now)], colors=c, radii=1.2))
            if timed:  # the rows this chunk actually ran
                m = (ev_t[kk] >= a) & (ev_t[kk] <= b)
                rr.log(f"{ent}/sent", rr.Points2D(np.c_[X(ev_t[kk][m], now), Y(ev_v[kk][m])],
                                                  colors=GRIP_EXEC_COLOR, radii=2.5))
            else:  # the last row, pulled back to t_wall
                sent = np.array([X(tw[kk], now), Y(ev_v[kk][0])])
                last = np.array([X(row_t[kk][-1], now), Y(raw[kk][-1])])
                rr.log(f"{ent}/handoff", rr.Arrows2D(origins=[last], vectors=[sent - last],
                                                     colors=GRIP_EXEC_COLOR, radii=0.8))
                rr.log(f"{ent}/sent", rr.Points2D([sent], labels=[f"sent {ev_v[kk][0]:.1f} mm"],
                                                  colors=GRIP_EXEC_COLOR, radii=5.0, show_labels=True))

        ent = f"{base}/exec/cyc"
        mode = f"rows at target_time - {g_lat * 1e3:.0f} ms" if timed else "last row on arrival"
        rr.log(f"{ent}/title",
               rr.Points2D([[x_lo, -14]],
                           labels=[f"executed ({mode}), t = cycle {k} (t_wall {now:.2f} s): white sent, "
                                   f"red time-indexed, grey measured"],
                           radii=0.01, colors=GRIP_EXEC_COLOR, show_labels=True))
        for key, (t, v), c, r in (("measured", (meas_t, meas_mm), GRIP_MEAS_COLOR, 1.0),
                                  ("time_indexed", (ti_t, ti_v), GRIP_TI_COLOR, 1.0),
                                  ("sent", (step_t, step_v), GRIP_EXEC_COLOR, 1.4)):
            p = _clip_poly(t, v, a, b)
            rr.log(f"{ent}/{key}", rr.LineStrips2D([np.c_[X(p[:, 0], now), Y(p[:, 1])]], colors=c, radii=r))
        segs, cols = [], []
        for _, dk, _ in GRIP_ROWS:
            kk = k + dk
            if 0 <= kk < n and np.isfinite(starts[kk]):
                s0, s1 = span(kk)
                s0, s1 = max(s0, a) + 1e-6, min(s1, b) - 1e-6
                if s1 > s0:
                    p = _clip_poly(step_t, step_v, s0, s1)
                    segs.append(np.c_[X(p[:, 0], now), Y(p[:, 1])])
                    cols.append(GRIP_ROW_COLORS[dk])
        rr.log(f"{ent}/from_chunk", rr.LineStrips2D(segs, colors=cols, radii=3.0))


def build_blueprint(cam_names=(), has_seam=False, gripper_names=(), gripper_window_s=None):
    right = [rrb.TimeSeriesView(origin="chunk_error", name="chunk endpoint error")]
    if has_seam:
        right = [rrb.TimeSeriesView(origin="seam_error", name="tip-seam error (mm)"),
                 rrb.TimeSeriesView(origin="seam_angles", name="torch angles (deg)"),
                 *right,
                 rrb.TextDocumentView(origin="seam_readout", name="live seam error")]
    right_col = rrb.Vertical(*right, row_shares=[3, 3, 2, 2] if has_seam else None)
    if gripper_names:
        # one tab per gripper arm: chunk t-1 / t / t+1 / executed, same x window
        x_range, y_range = gripper_bounds(gripper_window_s)
        tabs = [rrb.Vertical(*[rrb.Spatial2DView(origin=f"gripper/{g}/{sub}", name=f"{g} {label}",
                                                 visual_bounds=rrb.VisualBounds2D(x_range=x_range,
                                                                                  y_range=y_range))
                               for sub, label in [(r[0], r[2]) for r in GRIP_ROWS] + [("exec", "executed")]],
                             name=f"{g} gripper")
                for g in gripper_names]
        right_col = rrb.Tabs(*tabs, rrb.Vertical(*right, name="chunk error"))
    return rrb.Blueprint(
        rrb.Horizontal(scene_column(cam_names), right_col,
                       column_shares=[3, 3] if gripper_names else [3, 2]),
        collapse_panels=True,
    )


# --------------------------------------------------------------------------- #
# One arm of one dataset's episode
# --------------------------------------------------------------------------- #
def visualize(store, config, path, arm, episode, args, prefix="", gt_color=None, cams=(),
              arm_idx=0, n_arms=1, grip=None):
    """``prefix`` namespaces every entity (multi-dataset overlay) and ``gt_color``
    recolors the executed trajectory; ``cams`` are decoded here (single dataset);
    ``arm_idx`` of ``n_arms`` picks this arm's block of every chunk; ``grip`` =
    (max_width_m, window) logs this arm's gripper panels."""
    data = store["data"]
    s, e = episode_range(store, episode)
    ep_len = e - s
    fps = data_fps(args, config, 30.0)
    print(f"Episode {episode}: steps {s}..{e} (len={ep_len}) @ {fps}fps")
    ts = data["timestamp"][s:e] if "timestamp" in data else None
    t0 = float(ts[0]) if ts is not None else 0.0
    times = [float(ts[i] - t0) if ts is not None else i / fps for i in range(ep_len)]

    pos_key, rot_key = arm.keys["eef_pos"], arm.keys["eef_rot"]
    gt_pos = data[pos_key][s:e].astype(float)
    gt_rot = data[rot_key][s:e].astype(float) if rot_key else None

    policy_logs_dir = Path(args.policy_logs) if args.policy_logs else Path(path) / "policy_logs"
    log, perturb, run_cfg = load_policy_log(policy_logs_dir, episode)
    if log is None:
        print(f"  [warn] no policy log at {policy_logs_dir}/episode_{episode}.jsonl — "
              f"GT trajectory only, no predicted chunks")

    # Perturbation replay: chunks were logged back in the dataset frame plus a perturb
    # header; re-applying [R|t] restores the perturbed-frame GT and raw predictions.
    xf = None
    if perturb is not None:
        yaw = float(perturb["yaw_rad"])
        t = np.asarray(perturb["trans_m"], dtype=float)
        c, sn = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        xf = lambda P: P @ Rz.T + t  # noqa: E731
        print(f"  [perturb] yaw={np.degrees(yaw):+.1f} deg  "
              f"t=({t[0]*1000:+.0f}, {t[1]*1000:+.0f}, {t[2]*1000:+.0f}) mm — "
              f"rendering perturbed GT (green) + chunks in the perturbed frame")
    xfo = xf or (lambda P: P)

    gt_c = gt_color if gt_color is not None else arm.color
    cur_c = gt_color if gt_color is not None else [255, 255, 0]
    rs = args.radius_scale
    ent = f"world/{prefix}{arm.name}"
    # everything under `ent` stays in this arm's base frame; the viewer places it
    if arm.T_world is not None:
        rr.log(ent, rr.Transform3D(translation=arm.T_world[:3, 3], mat3x3=arm.T_world[:3, :3]),
               static=True)
    err_full_path = f"chunk_error/{prefix}{arm.name}/endpoint_m"
    err_used_path = f"chunk_error/{prefix}{arm.name}/executed_endpoint_m"
    label = f"{prefix.rstrip('/')} {arm.name}" if prefix else arm.name
    rr.log(err_full_path, rr.SeriesLines(names=[f"{label} full"]), static=True)
    rr.log(err_used_path, rr.SeriesLines(names=[f"{label} executed"]), static=True)

    depth_scale = load_depth_scale(path)
    for cam in cams:
        decode_and_log_video(f"world/cameras/{cam.name}", cam, fps, depth_scale)

    tip_pos = tip_off = None
    if args.tip or args.seams:
        if gt_rot is None:
            print(f"  [warn] --tip/--seams needs eef rotation; arm '{arm.name}' has "
                  f"none, skipping tip trajectory")
        else:
            tip_off = load_tool_tip(args.tip_config)
            tip_pos = np.stack([p + rot_to_matrix(r) @ tip_off for p, r in zip(gt_pos, gt_rot)])
    seam = None
    if args.seams and tip_pos is not None:
        seam = seam_overlay(args, run_cfg, arm, ent, prefix, tip_pos, tip_off, gt_rot)

    # executed trajectory (demo GT for offline rollouts), growing per step
    for i in range(ep_len):
        rr.set_time("time", duration=times[i])
        rr.log(f"{ent}/gt_traj",
               rr.LineStrips3D([gt_pos[:i + 1]], colors=np.array([gt_c]), radii=rs * 0.00008))
        rr.log(f"{ent}/gt_current",
               rr.Points3D(gt_pos[i].reshape(1, 3), colors=np.array([cur_c]), radii=rs * 0.0015))
        if tip_pos is not None:
            rr.log(f"{ent}/tip",
                   rr.Points3D(tip_pos[i].reshape(1, 3), colors=np.array([TIP_COLOR]), radii=rs * 0.0015))
            rr.log(f"{ent}/tip_traj",
                   rr.LineStrips3D([tip_pos[:i + 1]], colors=np.array([TIP_COLOR]), radii=rs * 0.00008))
        if seam is not None:
            log_seam_step(seam, i, ent, prefix, tip_pos[i], rs)
        if xf is not None:
            rr.log(f"{ent}/gt_traj_perturbed",
                   rr.LineStrips3D([xf(gt_pos[:i + 1])], colors=np.array([PERTURB_GT_COLOR]), radii=rs * 0.00008))
            rr.log(f"{ent}/gt_current_perturbed",
                   rr.Points3D(xf(gt_pos[i].reshape(1, 3)), colors=np.array([[255, 255, 0]]), radii=rs * 0.0015))

    # predicted action chunks, one per inference cycle
    if log and "step_in_episode" not in log[0] and ts is None:
        print("  [warn] online-format policy log needs replay_buffer 'timestamp' to place "
              "chunks — skipping chunk overlay")
        log = []
    if log:
        ts_abs = np.asarray(ts, dtype=float) if ts is not None else None

        def step_at(wall_t):
            """Replay-buffer step index for an absolute wall time (None past ep end)."""
            if wall_t > ts_abs[-1] + 1e-6:
                return None
            return int(min(np.searchsorted(ts_abs, wall_t), ep_len - 1))

        width = len(log[0]["action_chunk"][0]) // n_arms
        if width < 6:
            raise ValueError(f"action_chunk width {len(log[0]['action_chunk'][0])} is not "
                             f"{n_arms} arms x (pos3 + rotvec3 (+ extras))")
        for entry in log[:: args.chunk_stride]:
            chunk = np.asarray(entry["action_chunk"], dtype=float)[:, arm_idx * width:(arm_idx + 1) * width]
            h = chunk.shape[0]
            if "step_in_episode" in entry:  # offline: lockstep execution of the chunk prefix
                i = entry["step_in_episode"]
                lo, hi = 0, int(entry.get("n_waypoints", h))
                j_full = i + h - 1 if i + h - 1 < ep_len else None
                j_used = i + hi - 1 if i + hi - 1 < ep_len else None
            else:
                # online: place the cycle by its obs wall clock; the scheduler ran a
                # mid-chunk slice, recovered from sched_ts_offsets vs action_ts_offsets
                obs_t = float(entry["obs_ts"])
                i = step_at(obs_t)
                if i is None:
                    continue
                a_off = np.asarray(entry.get("action_ts_offsets", []), dtype=float)
                s_off = np.asarray(entry.get("sched_ts_offsets", []), dtype=float)
                if len(a_off) and len(s_off):
                    lo = int(np.argmin(np.abs(a_off - s_off[0])))
                    hi = min(lo + len(s_off), h)
                else:
                    lo, hi = 0, h
                j_full = step_at(obs_t + a_off[-1]) if len(a_off) else None
                j_used = step_at(obs_t + s_off[-1]) if len(s_off) else None
            if i >= ep_len:
                continue
            rr.set_time("time", duration=times[i])

            # model input (obs_lowdim history, else the 'eef' summary); its gap to the
            # yellow GT dot = obs staleness / camera latency
            obs_pos = (entry.get("obs_lowdim") or entry.get("eef") or {}).get(pos_key)
            if obs_pos is not None:
                rr.log(f"{ent}/model_obs_pos",
                       rr.Points3D(xfo(np.asarray(obs_pos, dtype=float).reshape(-1, 3)),
                                   colors=np.array([MODEL_OBS_COLOR]), radii=rs * 0.0008))

            # chunks span ~1 cm with 1-3 mm steps: thin tubes + waypoint dots
            chunk_pos = xfo(chunk[:, :3])
            rr.log(f"{ent}/chunk_pred",
                   rr.LineStrips3D([chunk_pos], colors=np.array([CHUNK_COLOR]), radii=rs * 0.00008))
            rr.log(f"{ent}/chunk_pred_pts",
                   rr.Points3D(chunk_pos, colors=np.array([CHUNK_COLOR]), radii=rs * 0.0003))
            rr.log(f"{ent}/chunk_pred_used",
                   rr.LineStrips3D([chunk_pos[lo:hi]], colors=np.array([CHUNK_USED_COLOR]), radii=rs * 0.00012))
            rr.log(f"{ent}/chunk_pred_used_pts",
                   rr.Points3D(chunk_pos[lo:hi], colors=np.array([CHUNK_USED_COLOR]), radii=rs * 0.0004))

            # what actually happened over the same horizon, when in-episode
            if j_full is not None:
                rr.log(f"{ent}/gt_future_horizon",
                       rr.LineStrips3D([xfo(gt_pos[i:j_full + 1])], colors=np.array([GT_FUTURE_COLOR]),
                                       radii=rs * 0.00008))
                rr.log(err_full_path, rr.Scalars(float(np.linalg.norm(chunk[-1, :3] - gt_pos[j_full]))))
            if j_used is not None and hi > lo:
                rr.log(err_used_path, rr.Scalars(float(np.linalg.norm(chunk[hi - 1, :3] - gt_pos[j_used]))))

    if grip is not None and log and "t_wall" in log[0] and arm.keys.get("gripper"):
        hi_m, window = grip
        width = len(log[0]["action_chunk"][0]) // n_arms
        if width > 6:
            log_gripper_panels(log, arm.name, arm_idx * width + 6, hi_m,
                               data[arm.keys["gripper"]][s:e].astype(float).reshape(-1),
                               ts, t0, window, config.get("deploy", {}))
            print(f"  [gripper] {arm.name}: {len(log)} cycles, panels under gripper/{arm.name}/")
    print("Done logging episode.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="+",
                   help="Rollout root(s) (replay_buffer.zarr + policy_logs/). Several paths "
                        "overlay the same episode of each, under world/<dataset>/")
    p.add_argument("--config", default=None,
                   help="config.json for every dataset (rollout dirs without one: point it at "
                        "the TRAINING set's config). Default: <path>/config.json")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--policy-logs", default=None,
                   help="Override policy_logs/ dir (default: <path>/policy_logs; single dataset)")
    p.add_argument("--chunk-stride", type=int, default=1,
                   help="Only plot every Nth predicted chunk (reduce clutter)")
    p.add_argument("--frame", choices=["auto", "base", "slam"], default="base",
                   help="Eef frame of the trajectory (action_chunk is always base-frame absolute)")
    p.add_argument("--data-fps", type=float, default=None, help="Default: config collection.frequency")
    p.add_argument("--gripper", dest="gripper", action="store_true", default=True,
                   help="Gripper tab per width-gripper arm (default on; single online rollout)")
    p.add_argument("--no-gripper", dest="gripper", action="store_false", help="Skip the gripper panels")
    p.add_argument("--radius-scale", type=float, default=1.0,
                   help="Multiply every line/dot radius (defaults suit cm-scale weld chunks; "
                        "~10 for room-scale UMI scenes)")
    add_camera_args(p)
    add_arm_pose_args(p.add_argument_group("arm placement"))
    tip = add_tip_args(p)
    tip.add_argument("--seams", default=None,
                     help="seams.json (real_robot teach_seam.py): the taught seam as a tip-level "
                          "line + tip-to-seam error plots; pose from the policy log's run_config "
                          "eval_pose_id. Implies tip computation")
    tip.add_argument("--seam-pose", type=int, default=None,
                     help="pose_id for --seams when the policy log has no eval_pose_id")
    add_sink_args(p, grpc_port=9876)
    return p


def main():
    args = build_arg_parser().parse_args()
    paths = args.path
    multi = len(paths) > 1
    if multi and args.policy_logs:
        raise SystemExit("--policy-logs only makes sense with a single dataset path")
    missing = [p for p in paths if not Path(p).is_dir()]
    if missing:  # shell brace expansion happily fabricates nonexistent paths
        raise SystemExit("[error] dataset path(s) not found: " + ", ".join(missing))

    app = (f"compare_{len(paths)}ds_ep{args.episode}" if multi
           else f"{Path(paths[0].rstrip('/')).name}_chunks_ep{args.episode}")
    rr.init(app, spawn=False)

    loaded = []  # everything up front: the blueprint needs the camera / gripper names
    for path in paths:
        store = load_store(path)
        ds = Path(path.rstrip("/")).name
        try:
            config = load_config(path, args.config)
        except FileNotFoundError:
            if args.config:
                raise
            config = synthesize_config(store, path)
            print(f"[{ds}] no config.json — probed arms {[a['obs_name'] for a in config['arms']]} "
                  f"from zarr keys (pass --config <training config> for exact geometry)")
        loaded.append((path, ds, store, config))

    # cameras and gripper panels: single dataset only
    path0, _, store0, config0 = loaded[0]
    keys0 = set(store0["data"].keys())
    cams = []
    if args.cameras and not multi:
        cams = [c for c in resolve_cameras(config0, keys0, path0, args.episode, args.camera_subset)
                if c.video is not None]
        print(f"Cameras: {[(c.name, c.video.name) for c in cams]}")
    grip_arms, grip_win = {}, None
    if args.gripper and not multi:
        logs_dir = Path(args.policy_logs) if args.policy_logs else Path(path0) / "policy_logs"
        grip_win = gripper_window(load_policy_log(logs_dir, args.episode)[0])
        if grip_win is not None:
            grip_arms = gripper_arms(config0, keys0)
        print(f"Gripper panels: {list(grip_arms)}"
              + (f" (window {grip_win[0]:+.1f}..{grip_win[1]:+.1f} s around chunk t's t_wall)"
                 if grip_arms else ""))

    select_sink(args, build_blueprint([c.name for c in cams], has_seam=bool(args.seams),
                                      gripper_names=list(grip_arms), gripper_window_s=grip_win))
    log_world_axes("world/origin")

    for k, (path, ds, store, config) in enumerate(loaded):
        arms = resolve_arms(path, config, set(store["data"].keys()), args, tag=f"[{ds}] ")
        for j, arm in enumerate(arms):
            visualize(store, config, path, arm, args.episode, args,
                      prefix=f"{ds}/" if multi else "",
                      gt_color=DATASET_COLORS[k % len(DATASET_COLORS)] if multi else None,
                      cams=cams if k == 0 and j == 0 else (),
                      arm_idx=j, n_arms=len(arms),
                      grip=(grip_arms[arm.name], grip_win) if k == 0 and arm.name in grip_arms else None)
    wait_if_serving(args)


if __name__ == "__main__":
    main()
