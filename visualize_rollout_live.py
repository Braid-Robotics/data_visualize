#!/usr/bin/env python3
"""Live Rerun view of a running rollout: the policy's action chunks as they arrive.

Tails the newest ``policy_logs/episode_<n>.jsonl`` under the given dir (the rollout
writes and flushes one line per inference cycle) and moves on when a newer
episode log appears. Per cycle, for every config arm:

- 3D: the eef the model saw (one trajectory point per cycle), the predicted chunk
  (orange), its scheduled rows (magenta), the previous chunk (grey) with the rows
  that ran before this one arrived (white)
- chunk switch jump: the first scheduled row vs. the previous chunk at the same
  timestamp
- gripper: the width each chunk sent (known when the next chunk arrives, so one
  cycle late) and the measured width
- timing (inference / round trip / camera latency / cycle) and, for compliant
  rollouts, the external force
- with an operator takeover (the rollout's --spacemouse): who drives each such arm,
  MODEL (green) or TAKEOVER (red) -- a labelled marker on the arm, its trail red where
  the operator drove, the dropped chunk grey, a banner in the status and a 0/1 plot

The replay buffer is only written at episode end, so the trajectory is the
per-cycle eef, not the 30 Hz state (visualize_rollout_chunks.py afterwards). The
policy log holds the measured width of the first gripper arm only; pass the
rollout's ``--dump-obs`` dir here too (a fresh one per run) for every arm's width
and the model's input images.

Layout::

    <path>/                          # one rollout session ...
        config.json
        policy_logs/episode_<n>.jsonl
    <path>/<session>/...             # ... or the deploy output dir: newest log wins

Examples::

    python visualize_rollout_live.py <rollout_dir> --radius-scale 10
    python visualize_rollout_live.py <deploy_output_dir> --dump-obs <dump_dir>
    python visualize_rollout_live.py <rollout_dir> --serve      # headless, view over gRPC
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from viz_common import (
    add_arm_pose_args,
    add_sink_args,
    load_config,
    log_world_axes,
    resolve_arms,
    scene_column,
    select_sink,
)

CHUNK_COLOR = [255, 160, 0]         # full predicted chunk
CHUNK_USED_COLOR = [255, 60, 200]   # rows scheduled for execution
CHUNK_PREV_COLOR = [120, 120, 120]  # previous chunk
CHUNK_RAN_COLOR = [240, 240, 240]   # previous chunk's rows that ran before this one arrived
CUR_COLOR = [255, 255, 0]           # eef at the newest observation
TAKEOVER_COLOR = [255, 40, 40]      # the operator drives the arm (rollout --spacemouse)
MODEL_COLOR = [60, 220, 60]         # the policy drives it
CHUNK_DROPPED_COLOR = [90, 90, 90]  # the policy's chunk while the operator has the arm (not executed)
TIMING_KEYS = (("inference_ms", "inference"), ("round_trip_ms", "round trip"),
               ("cam_lat_ms", "camera latency"), ("cycle_ms", "cycle"))
NEWEST_CHECK_S = 1.0                # how often to look for a newer episode log


def _mtime(p):
    try:
        return p.stat().st_mtime
    except OSError:
        return -1.0


def newest_log(root):
    """Newest policy_logs/episode_<n>.jsonl in ``root`` or one level below it."""
    logs = list(root.glob("policy_logs/episode_*.jsonl")) + list(root.glob("*/policy_logs/episode_*.jsonl"))
    return max(logs, key=_mtime) if logs else None


class LogTail:
    """Records of the complete lines appended to a jsonl since the last read.

    ``restarted`` is set when the file was truncated or replaced: a dropped episode's
    id is reused, so the next episode rewrites the same log."""

    def __init__(self, path):
        self.path = path
        self._f = open(path)
        self._ino = self.path.stat().st_ino
        self._buf = ""
        self.restarted = False

    def read(self):
        self.restarted = False
        try:
            st = self.path.stat()
        except OSError:
            return []
        if st.st_ino != self._ino or st.st_size < self._f.tell():
            self._f.close()
            self._f, self._ino, self._buf = open(self.path), st.st_ino, ""
            self.restarted = True
        self._buf += self._f.read()
        *lines, self._buf = self._buf.split("\n")
        out = []
        for line in lines:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  [warn] {self.path.name}: unreadable line skipped")
        return out

    def close(self):
        self._f.close()


class DumpTail:
    """Exchanges a rollout ``--dump-obs`` dir gained since the last read, starting at
    the newest one: (obs timestamp, {obs_name: width_m}, {cam: rgb})."""

    def __init__(self, path):
        self.dir = Path(path)
        done = sorted(self.dir.glob("action_*.npz"))
        self.next = int(done[-1].stem.split("_")[1]) if done else 0

    def read(self):
        import cv2
        out = []
        while (self.dir / f"action_{self.next:04d}.npz").exists():
            k = self.next
            try:
                with np.load(self.dir / f"obs_{k:04d}.npz") as obs:
                    ts = float(obs["timestamp"][-1])
                    widths = {key[:-len("_gripper")]: float(np.asarray(obs[key])[-1].reshape(-1)[0])
                              for key in obs.files if key.endswith("_gripper")}
            except (OSError, ValueError, KeyError, EOFError):
                break  # still being written
            frames = {}
            for png in sorted(self.dir.glob(f"obs_{k:04d}_*_t*.png")):
                cam, _, t = png.stem[len(f"obs_{k:04d}_"):].rpartition("_t")
                if cam not in frames or int(t) > frames[cam][0]:
                    frames[cam] = (int(t), png)
            imgs = {}
            for cam, (_, png) in frames.items():
                img = cv2.imread(str(png), cv2.IMREAD_COLOR)
                if img is not None:
                    imgs[cam] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out.append((ts, widths, imgs))
            self.next += 1
        return out


def chunk_slices(rec):
    """(row timestamps, lo, hi): every chunk row's absolute time and the scheduled
    rows [lo, hi), recovered from sched_ts_offsets vs action_ts_offsets."""
    n = len(rec["action_chunk"])
    a_off = np.asarray(rec.get("action_ts_offsets") or [], dtype=float)
    s_off = np.asarray(rec.get("sched_ts_offsets") or [], dtype=float)
    if len(a_off) != n:
        a_off = np.zeros(n)
    lo = int(np.argmin(np.abs(a_off - s_off[0]))) if len(s_off) else 0
    hi = min(lo + len(s_off), n) if len(s_off) else n
    return float(rec["obs_ts"]) + a_off, lo, hi


def ran_mask(rec, row_t, t_next, latency=0.0):
    """Rows of ``rec``'s chunk whose target time (minus ``latency``) fell between its
    arrival and the next chunk's."""
    tt = row_t - latency
    return (tt >= float(rec["t_wall"])) & (tt < t_next)


def lighter(c):
    return [(v + 255) // 2 for v in c]


def takeover_state(snap, t_lo, t_hi):
    """True if the operator had the arm at any time in (t_lo, t_hi] (the snapshot's flag or a
    logged interval, so a takeover shorter than a cycle still shows), False if the policy had
    it, None when the rollout runs no takeover on that arm."""
    if not snap or "takeover" not in snap:
        return None
    if snap["takeover"]:
        return True
    return any(t0 <= t_hi and (t1 is None or t1 > t_lo) for t0, t1 in snap.get("takeover_log") or [])


def mode_strips(pts, modes, color, rs):
    """The trail split into runs of one mode, each run starting at the previous run's last
    point: (strips, colors, radii) with the operator's runs red and thicker."""
    strips, colors, radii = [], [], []
    start = 0
    for k in range(1, len(pts) + 1):
        if k == len(pts) or modes[k] != modes[start]:
            strips.append(pts[max(start - 1, 0):k])
            colors.append(TAKEOVER_COLOR if modes[start] else color)
            radii.append(rs * (0.0003 if modes[start] else 0.00008))
            start = k
    return strips, np.array(colors), np.array(radii)


class LiveSession:
    """Arms, gripper columns and per-episode state of one rollout session dir."""

    def __init__(self, session, args):
        self.session = session
        self.args = args
        self.config = load_config(session)
        self.arms = None
        self.cfg_arms = self.config.get("arms", [])
        self.grip_names = [a["obs_name"] for a in self.cfg_arms
                           if (a.get("gripper") or {}).get("type") in ("robotiq", "dynamixel")]

    def resolve(self, rec):
        keys = set(rec.get("eef", {})) | {f"{n}_gripper" for n in self.grip_names}
        self.arms = resolve_arms(self.session, self.config, keys, self.args, tag=f"[{self.session.name}] ")
        for arm in self.arms:
            if arm.T_world is not None:
                rr.log(f"world/{arm.name}", rr.Transform3D(translation=arm.T_world[:3, 3],
                                                           mat3x3=arm.T_world[:3, :3]), static=True)
        for arm in self.arms:
            if arm.name in self.grip_names:
                rr.log(f"gripper/{arm.name}/sent", rr.SeriesLines(colors=[arm.color], names=[f"{arm.name} sent"]),
                       static=True)
                rr.log(f"gripper/{arm.name}/measured",
                       rr.SeriesLines(colors=[lighter(arm.color)], names=[f"{arm.name} measured"]), static=True)
            rr.log(f"chunk_jump/{arm.name}", rr.SeriesLines(colors=[arm.color], names=[arm.name]), static=True)
            rr.log(f"force/{arm.name}", rr.SeriesLines(colors=[arm.color], names=[f"{arm.name} |F|"]), static=True)
            rr.log(f"takeover/{arm.name}", rr.SeriesLines(colors=[TAKEOVER_COLOR], names=[f"{arm.name} operator"]),
                   static=True)
        for key, label in TIMING_KEYS:
            rr.log(f"timing/{key}", rr.SeriesLines(names=[label]), static=True)


class LiveEpisode:
    """Logs one episode's cycles as they are read."""

    def __init__(self, path, session):
        self.path = path
        self.s = session
        self.episode = int(path.stem.split("_")[-1])
        self.trail = {}
        self.modes = {}  # arm -> taken over at each trail point
        self.prev = None
        self.n = 0
        self.cleared = False

    def width(self, rec):
        return len(rec["action_chunk"][0]) // len(self.s.arms)

    def log_cycle(self, rec, has_dump):
        s, rs = self.s, self.s.args.radius_scale
        if s.arms is None:
            s.resolve(rec)
        t = float(rec["t_wall"])
        rr.set_time("wall", timestamp=t)
        if not self.cleared:
            for arm in s.arms:
                rr.log(f"world/{arm.name}/chunk_prev", rr.Clear(recursive=True))
            self.cleared = True
        if self.prev is not None:
            self.log_sent_gripper(self.prev, t)
            rr.set_time("wall", timestamp=t)

        chunk = np.asarray(rec["action_chunk"], dtype=float)
        w = self.width(rec)
        row_t, lo, hi = chunk_slices(rec)
        if self.prev is not None:
            p_chunk = np.asarray(self.prev["action_chunk"], dtype=float)
            p_t, _, _ = chunk_slices(self.prev)
            p_ran = ran_mask(self.prev, p_t, t)
        lines = [f"{s.session.name}  episode {self.episode}  cycle {rec.get('cycle', self.n)}",
                 f"inference {rec.get('inference_ms') or 0:.0f} ms  round trip {rec.get('round_trip_ms') or 0:.0f}  "
                 f"camera {rec.get('cam_lat_ms') or 0:.0f}  rows {rec.get('n_new', hi - lo)}/{len(chunk)}"
                 + ("  OVER BUDGET" if rec.get("over_budget") else "")]
        comp = rec.get("compliance") or {}
        t_lo = float(self.prev["t_wall"]) if self.prev is not None else t - 1.0
        headers = []
        for j, arm in enumerate(s.arms):
            ent = f"world/{arm.name}"
            blk = chunk[:, j * w:(j + 1) * w]
            c = comp.get(str(s.cfg_arms[j].get("name", j))) if j < len(s.cfg_arms) else None
            op = takeover_state(c, t_lo, t)
            eef = (rec.get("eef") or {}).get(arm.keys["eef_pos"])
            if eef is not None:
                self.trail.setdefault(arm.name, []).append(eef)
                self.modes.setdefault(arm.name, []).append(bool(op))
                pts = np.asarray(self.trail[arm.name], dtype=float)
                strips, cols, radii = mode_strips(pts, self.modes[arm.name], arm.color, rs)
                rr.log(f"{ent}/traj", rr.LineStrips3D(strips, colors=cols, radii=radii))
                if op is None:
                    rr.log(f"{ent}/current", rr.Points3D(pts[-1:], colors=np.array([CUR_COLOR]), radii=rs * 0.0015))
                else:
                    rr.log(f"{ent}/current",
                           rr.Points3D(pts[-1:], colors=np.array([TAKEOVER_COLOR if op else MODEL_COLOR]),
                                       radii=rs * (0.004 if op else 0.002),
                                       labels=[f"{arm.name} TAKEOVER" if op else f"{arm.name} MODEL"],
                                       show_labels=True))
            if op is not None:
                rr.log(f"takeover/{arm.name}", rr.Scalars(1.0 if op else 0.0))
                if op:
                    log = (c or {}).get("takeover_log") or []
                    since = t - log[-1][0] if c.get("takeover") and log else None
                    headers.append(f"## 🟥 {arm.name}: TAKEOVER (operator)"
                                   + (f"  {since:.1f} s" if since is not None else ""))
                else:
                    headers.append(f"## 🟩 {arm.name}: MODEL")
            pos = blk[:, :3]
            c_full, c_used = (CHUNK_DROPPED_COLOR, CHUNK_DROPPED_COLOR) if op else (CHUNK_COLOR, CHUNK_USED_COLOR)
            rr.log(f"{ent}/chunk_pred", rr.LineStrips3D([pos], colors=np.array([c_full]), radii=rs * 0.00008))
            rr.log(f"{ent}/chunk_pred_pts", rr.Points3D(pos, colors=np.array([c_full]), radii=rs * 0.0003))
            rr.log(f"{ent}/chunk_pred_used",
                   rr.LineStrips3D([pos[lo:hi]], colors=np.array([c_used]), radii=rs * 0.00012))
            rr.log(f"{ent}/chunk_pred_used_pts",
                   rr.Points3D(pos[lo:hi], colors=np.array([c_used]), radii=rs * 0.0004))
            line = f"{arm.name}  z {pos[lo, 2]:.3f} m"
            if self.prev is not None:
                p_pos = p_chunk[:, j * w:j * w + 3]
                rr.log(f"{ent}/chunk_prev",
                       rr.LineStrips3D([p_pos], colors=np.array([CHUNK_PREV_COLOR]), radii=rs * 0.00008))
                rr.log(f"{ent}/chunk_prev_ran",
                       rr.Points3D(p_pos[p_ran], colors=np.array([CHUNK_RAN_COLOR]), radii=rs * 0.0004))
                t0 = row_t[lo]
                if p_t[0] <= t0 <= p_t[-1]:
                    ref = np.array([np.interp(t0, p_t, p_pos[:, k]) for k in range(3)])
                    jump = float(np.linalg.norm(pos[lo] - ref)) * 1e3
                    rr.log(f"chunk_jump/{arm.name}", rr.Scalars(jump))
                    line += f"  jump {jump:4.1f} mm"
            if arm.name in s.grip_names and w > 6:
                line += f"  grip cmd {blk[lo, 6] * 1e3:4.1f} mm"
            if c:
                rr.log(f"force/{arm.name}", rr.Scalars(float(c.get("f_mag", 0.0))))
                flags = [f for f in ("in_contact", "blocked", "leaning") if c.get(f)]
                if c.get("halt"):
                    flags.append("HALT")
                line += f"  |F| {c.get('f_mag', 0.0):4.1f} N" + (f"  {' '.join(flags)}" if flags else "")
            lines.append(line)

        if not has_dump and s.grip_names and rec.get("gripper_qpos"):
            rr.set_time("wall", timestamp=float(rec["obs_ts"]))
            rr.log(f"gripper/{s.grip_names[0]}/measured", rr.Scalars(float(rec["gripper_qpos"][0]) * 1e3))
        rr.set_time("wall", timestamp=t)
        for key, _ in TIMING_KEYS[:3]:
            if rec.get(key) is not None:
                rr.log(f"timing/{key}", rr.Scalars(float(rec[key])))
        if self.prev is not None:
            rr.log("timing/cycle_ms", rr.Scalars((t - float(self.prev["t_wall"])) * 1e3))
        if headers:
            text = "\n\n".join(headers) + "\n\n```\n" + "\n".join(lines) + "\n```"
            rr.log("status", rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN))
        else:
            rr.log("status", rr.TextDocument("\n".join(lines)))
        self.prev = rec
        self.n += 1

    def log_sent_gripper(self, rec, t_next):
        """What ``rec``'s chunk sent the jaws before the next chunk (or the end) at
        ``t_next``: rows at target_time - latency, else the last row on arrival."""
        timed = "target_time" in (rec.get("gripper_schedule") or [])
        lat = float(rec.get("gripper_latency_s") or 0.0) if timed else 0.0
        chunk = np.asarray(rec["action_chunk"], dtype=float)
        w = self.width(rec)
        if w <= 6:
            return
        row_t, _, _ = chunk_slices(rec)
        m = ran_mask(rec, row_t, t_next, lat)
        for j, arm in enumerate(self.s.arms):
            if arm.name not in self.s.grip_names:
                continue
            v = chunk[:, j * w + 6] * 1e3
            tt, vv = (row_t[m] - lat, v[m]) if timed else (np.array([float(rec["t_wall"])]), v[-1:])
            if len(tt):
                rr.send_columns(f"gripper/{arm.name}/sent", indexes=[rr.TimeColumn("wall", timestamp=tt)],
                                columns=rr.Scalars.columns(scalars=vv))

    def end(self, rec):
        if self.prev is not None:
            self.log_sent_gripper(self.prev, float(rec.get("t_wall", self.prev["t_wall"])))
        rr.set_time("wall", timestamp=float(rec.get("t_wall", time.time())))
        rr.log("status", rr.TextDocument(f"{self.s.session.name}  episode {self.episode} ended "
                                         f"({rec.get('reason', '?')}, {self.n} cycles)"))
        print(f"  episode {self.episode} ended ({rec.get('reason', '?')}, {self.n} cycles)")


def log_dump(exchanges, session):
    for ts, widths, imgs in exchanges:
        rr.set_time("wall", timestamp=ts)
        for cam, img in imgs.items():
            rr.log(f"world/cameras/{cam}", rr.Image(img).compress(jpeg_quality=90))
        for name, wm in widths.items():
            if session is not None and name in session.grip_names:
                rr.log(f"gripper/{name}/measured", rr.Scalars(wm * 1e3))


def build_blueprint(cam_names=(), has_force=False, has_takeover=False, window_s=30.0):
    ranges = None
    if window_s > 0:
        ranges = rrb.VisibleTimeRange("wall", start=rrb.TimeRangeBoundary.cursor_relative(seconds=-window_s),
                                      end=rrb.TimeRangeBoundary.cursor_relative())
    right = [rrb.TextDocumentView(origin="status", name="now")]
    if has_takeover:
        right.append(rrb.TimeSeriesView(origin="takeover", name="takeover (1 = operator)", time_ranges=ranges,
                                        axis_y=rrb.ScalarAxis(range=(-0.1, 1.1), zoom_lock=True)))
    right += [rrb.TimeSeriesView(origin="gripper", name="gripper width (mm)", time_ranges=ranges),
              rrb.TimeSeriesView(origin="chunk_jump", name="chunk switch jump (mm)", time_ranges=ranges),
              rrb.TimeSeriesView(origin="timing", name="timing (ms)", time_ranges=ranges)]
    if has_force:
        right.append(rrb.TimeSeriesView(origin="force", name="external force (N)", time_ranges=ranges))
    shares = [2 if has_takeover else 1] + ([1] if has_takeover else []) + [2] * (len(right) - 1 - int(has_takeover))
    return rrb.Blueprint(
        rrb.Horizontal(scene_column(cam_names),
                       rrb.Vertical(*right, row_shares=shares),
                       column_shares=[3, 2]),
        rrb.TimePanel(state="collapsed", timeline="wall"),
        collapse_panels=True,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="Rollout session (config.json + policy_logs/) or the deploy output dir "
                                "holding sessions: follows the newest episode log below it")
    p.add_argument("--dump-obs", default=None,
                   help="The rollout's --dump-obs dir: model input images + every arm's measured width")
    p.add_argument("--window", type=float, default=30.0,
                   help="Seconds shown by the time-series views (0 = everything)")
    p.add_argument("--poll", type=float, default=0.05, help="Seconds between reads when nothing is new")
    p.add_argument("--radius-scale", type=float, default=1.0,
                   help="Multiply every line/dot radius (~10 for room-scale UMI scenes)")
    add_arm_pose_args(p.add_argument_group("arm placement"))
    add_sink_args(p, grpc_port=9877)
    p.set_defaults(frame="base")
    return p


def main():
    args = build_arg_parser().parse_args()
    root = Path(args.path).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"[error] not a directory: {root}")
    rr.init(f"{root.name}_live", spawn=False)

    dump = DumpTail(args.dump_obs) if args.dump_obs else None
    tail = ep = session = None
    started = False
    next_check = 0.0

    def feed(recs, episode):
        nonlocal started
        for rec in recs:
            if rec.get("event") == "episode_end":
                episode.end(rec)
            elif "action_chunk" in rec:
                if not started:
                    cams = [c["obs_name"] for c in session.config.get("cameras", [])] if dump else []
                    comp = rec.get("compliance") or {}
                    has_take = any("takeover" in (v or {}) for v in comp.values())
                    select_sink(args, build_blueprint(cams, has_force=bool(comp), has_takeover=has_take,
                                                      window_s=args.window))
                    log_world_axes("world/origin")
                    started = True
                episode.log_cycle(rec, has_dump=dump is not None)
        return len(recs)

    print(f"Waiting for policy_logs/episode_*.jsonl under {root} (Ctrl-C to stop)")
    try:
        while True:
            got = 0
            if time.monotonic() >= next_check:
                next_check = time.monotonic() + NEWEST_CHECK_S
                newest = newest_log(root)
                if newest is not None and (tail is None or newest != tail.path):
                    if tail is not None:  # drain the previous episode first
                        got += feed(tail.read(), ep)
                        tail.close()
                    tail = LogTail(newest)
                    if session is None or session.session != newest.parent.parent:
                        session = LiveSession(newest.parent.parent, args)
                    ep = LiveEpisode(newest, session)
                    print(f"Following {session.session.name}/{newest.name}")
            if tail is not None:
                recs = tail.read()
                if tail.restarted:
                    ep = LiveEpisode(tail.path, session)
                    print(f"Following {session.session.name}/{tail.path.name} (rewritten)")
                got += feed(recs, ep)
            if dump is not None and started:
                exchanges = dump.read()
                log_dump(exchanges, session)
                got += len(exchanges)
            if not got:
                time.sleep(args.poll)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
