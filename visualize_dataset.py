#!/usr/bin/env python3
"""Config-driven Rerun visualizer for Braid robot datasets (raw collection / rollout).

Reads the dataset's own ``config.json`` for arms, cameras, obs-key prefixes and
frequency, so one command works across dataset kinds::

    <path>/
        config.json            # arms[], cameras[], collection.frequency
        replay_buffer.zarr/    # data/<keys> + meta/episode_ends
        videos/<ep>/<cam>.mp4  # one video per camera obs_name

Examples::

    python visualize_dataset.py <path>                                # episode 0 + cameras
    python visualize_dataset.py <path> --episode 3 --no-cameras       # fast, no video decode
    python visualize_dataset.py <path> --episode all --pose-align last  # all episodes overlaid
    python visualize_dataset.py <path> --command --tip                # commanded eef + torch tip
    python visualize_dataset.py <raw> <raw>_gripfix --episode 3       # compare mode
    python visualize_dataset.py <raw> --key-suffix _online --episode 3
    python visualize_dataset.py <path> --serve                        # view over ssh -L
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from viz_common import (
    DATASET_COLORS,
    add_arm_pose_args,
    add_camera_args,
    add_sink_args,
    add_tip_args,
    apply_T,
    data_fps,
    decode_and_log_video,
    episode_bounds,
    episode_range,
    load_config,
    load_depth_scale,
    load_store,
    load_tool_tip,
    log_pose_axes,
    log_world_axes,
    matrix_to_aa,
    resolve_arms,
    resolve_cameras,
    rot_to_matrix,
    scene_column,
    select_sink,
    wait_if_serving,
)

CMD_COLOR = [190, 110, 255]  # commanded eef trajectory (--command)
TIP_COLOR = [255, 140, 0]

# (sliced arm key, plot group), in plot order
SERIES = [("eef_pos", "eef_pos"), ("eef_rot", "eef_rot"), ("eef_pos_vel", "eef_pos_vel"),
          ("tcp_force", "tcp_force"), ("joint_pos", "joint_pos"), ("joint_vel", "joint_vel"),
          ("gripper", "gripper"), ("slam_conf", "slam")]
GROUP_LABELS = {
    "eef_pos": ["x", "y", "z"],
    "eef_rot": ["rx", "ry", "rz"],
    "eef_pos_vel": ["vx", "vy", "vz", "wx", "wy", "wz"],
    "tcp_force": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
}
SINGLE_LINE = {"gripper": ["value"], "slam": ["conf"]}  # first column only
TIP_POS, TIP_ROT = ["tip_x", "tip_y", "tip_z"], ["tip_rx", "tip_ry", "tip_rz"]
CMD_POS, CMD_ROT = ["cmd_x", "cmd_y", "cmd_z"], ["cmd_rx", "cmd_ry", "cmd_rz"]
CMD_ERR = ["dx", "dy", "dz", "dpos", "drot_deg"]
COMPARE_DIFF = ["dx_mm", "dy_mm", "dz_mm", "dpos_mm", "drot_deg", "dgrip_mm"]
ALIGN_FRAME = {"first": 0, "first-pos": 0, "last": -1, "last-pos": -1}  # --pose-align modes


# --------------------------------------------------------------------------- #
# Scalars
# --------------------------------------------------------------------------- #
def series_dims(group, width):
    if group in SINGLE_LINE:
        return SINGLE_LINE[group]
    base = GROUP_LABELS.get(group)
    return base[:width] if base and width <= len(base) else [f"{group[:1]}{d}" for d in range(width)]


def has_series(sl, key):
    """Sliced signal present and plottable (quaternion eef_rot is not)."""
    return sl[key] is not None and (key != "eef_rot" or sl[key].shape[1] == 3)


def log_series_styles(views):
    """Static SeriesLines per line so plots get named, colored lines."""
    for origin, labels in views.items():
        for lab in labels:
            rr.log(f"{origin}/{lab}", rr.SeriesLines(names=[lab]), static=True)


def log_scalar_group(origin, labels, row):
    """One timestep of a multi-dim signal as child Scalars under ``origin``."""
    row = np.atleast_1d(np.asarray(row).ravel())
    for d, lab in enumerate(labels):
        if d < row.shape[0]:
            rr.log(f"{origin}/{lab}", rr.Scalars(float(row[d])))


def overlay_views(ns, views, pos_labels, rot_labels):
    """Extra lines in the arm's eef_pos / eef_rot plots (eef_rot only when plotted)."""
    out = {f"{ns}eef_pos": pos_labels}
    if f"{ns}eef_rot" in views:
        out[f"{ns}eef_rot"] = rot_labels
    return out


# --------------------------------------------------------------------------- #
# Display frame: world placement, origin_T_slamworld, --pose-align
# --------------------------------------------------------------------------- #
def to_display(arm, pos, rot, origin_T=None):
    """(pos (n,3), R (n,3,3) or None) after the optional per-frame origin_T_slamworld
    and the arm's world placement (3D only; scalar plots stay in the stored frame)."""
    pos = np.asarray(pos, dtype=float)
    R = None if rot is None else np.stack([rot_to_matrix(r) for r in rot])
    if origin_T is not None:
        pos = np.stack([apply_T(origin_T[i], pos[i].reshape(1, 3))[0] for i in range(len(pos))])
        if R is not None:
            R = origin_T[:, :3, :3] @ R
    if arm.T_world is not None:
        pos = apply_T(arm.T_world, pos)
        if R is not None:
            R = arm.T_world[:3, :3] @ R
    return pos, R


def align_arm_index(args, config):
    """Index of the --pose-align reference arm: config arms[].name or obs_name, default 0."""
    if not args.pose_align_arm:
        return 0
    ids = [(a.get("name"), a["obs_name"]) for a in config.get("arms", [])]
    ref = next((i for i, n in enumerate(ids) if args.pose_align_arm in n), None)
    if ref is None:
        raise SystemExit(f"--pose-align-arm {args.pose_align_arm!r}: not an arm "
                         f"(have {[o for _, o in ids]})")
    return ref


def align_from(mode, pos, R):
    """(R_ref, p_ref) of --pose-align ``mode`` on the reference arm's display trajectory
    ``pos``; display = R_ref^T (x - p_ref) for every arm, tip, command and camera.
    ``R`` is the 3x3 at that frame (None: translate only). ``*-pos`` translate only.
    """
    if mode == "none" or not len(pos):
        return None
    p = pos[ALIGN_FRAME[mode]].copy()
    if mode.endswith("-pos") or R is None:
        return np.eye(3), p
    return R, p


def aligned(pts, align):
    return pts if align is None else (pts - align[1]) @ align[0]


def aligned_rot(R, align):
    return R if align is None or R is None else align[0].T @ R


def pose_align(args, ref_arm, ref_pos, ref_R, label=""):
    """--pose-align transform for one episode from the reference arm's display trajectory."""
    if args.pose_align == "none":
        return None
    fi = ALIGN_FRAME[args.pose_align]
    if ref_R is None and not args.pose_align.endswith("-pos"):
        print(f"  [warn] --pose-align {args.pose_align}: '{ref_arm.name}' has no eef "
              f"rotation, translating only")
    print(f"  --pose-align {args.pose_align}: origin = {label}'{ref_arm.name}' eef at frame "
          f"{fi % len(ref_pos)} ({np.round(ref_pos[fi], 4).tolist()})")
    return align_from(args.pose_align, ref_pos, None if ref_R is None else ref_R[fi])


def arm_command(action, idx, n_arms):
    """(pos, rotvec) commanded for arm ``idx``: real_robot packs
    ``[eef_pos(3), eef_rot_axis_angle(3), gripper(g)]`` per arm contiguously.
    None when the row does not split into n_arms blocks of >= 6."""
    if action is None:
        return None
    per_arm = action.shape[1] // n_arms
    if per_arm < 6 or action.shape[1] % n_arms:
        return None
    off = idx * per_arm
    a = action.astype(float)
    return a[:, off:off + 3], a[:, off + 3:off + 6]


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
def obs_wh(config, cam_name):
    wh = next((c.get("obs_resolution_wh") for c in config["cameras"]
               if c["obs_name"] == cam_name), None)
    return tuple(wh) if wh else None


def log_videos(cams, config, path, fps, args):
    """Decode every camera's video into ``world/cameras/<name>``."""
    depth_scale = load_depth_scale(path)
    for cam in cams:
        if cam.video is not None:
            decode_and_log_video(f"world/cameras/{cam.name}", cam, args.video_fps or fps,
                                 depth_scale, (args.depth_near, args.depth_far),
                                 args.center_crop, obs_wh(config, cam.name) if args.center_crop else None,
                                 args.jpeg_quality)


def log_frusta(cams, arms, data, s, e, config, args, align, times):
    """Pinhole + per-frame cam_*_c2w pose per camera (c2w is in the base frame)."""
    if any(a.frame != "base" for a in arms):
        print("  [warn] --frusta needs base-frame eef, skipping camera frusta")
        return
    for cam in cams:
        if cam.c2w_key is None:
            continue
        c2w = np.array(data[cam.c2w_key][s:e], dtype=float)
        # Pinhole K must match the logged frames: obs-res zarr K when center-cropped
        if not args.center_crop:
            K, res = cam.K, cam.resolution_wh
        elif f"{cam.name}_K" in data:
            K, res = data[f"{cam.name}_K"][s], obs_wh(config, cam.name)
        else:
            print(f"  [warn] --center-crop frusta for {cam.name}: no {cam.name}_K, "
                  f"logging camera pose without Pinhole")
            K, res = None, None
        if align is not None:
            c2w[:, :3, 3] = aligned(c2w[:, :3, 3], align)
            c2w[:, :3, :3] = aligned_rot(c2w[:, :3, :3], align)
        entity = f"world/cameras/{cam.name}"
        if K is not None and res is not None:
            rr.log(entity, rr.Pinhole(image_from_camera=K, resolution=list(res)), static=True)
        for t, T in zip(times, c2w):
            rr.set_time("time", duration=float(t))
            rr.log(entity, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))


# --------------------------------------------------------------------------- #
# Blueprints
# --------------------------------------------------------------------------- #
def build_blueprint(cam_names, views):
    ts_views = [rrb.TimeSeriesView(origin=o, name=o) for o in views]
    return rrb.Blueprint(
        rrb.Horizontal(
            scene_column(cam_names),
            rrb.Tabs(rrb.Grid(*ts_views)) if ts_views else rrb.Vertical(),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def build_compare_blueprint(cam_names, plots, n_arms):
    """One tab per signal group, a (dimension x arm) grid of plots, a line per dataset."""
    tabs = [rrb.Grid(*[rrb.TimeSeriesView(origin=o, name=o) for o in origins],
                     grid_columns=n_arms, name=group)
            for group, origins in plots.items()]
    return rrb.Blueprint(
        rrb.Horizontal(scene_column(cam_names), rrb.Tabs(*tabs), column_shares=[3, 2]),
        collapse_panels=True,
    )


# --------------------------------------------------------------------------- #
# Single episode
# --------------------------------------------------------------------------- #
def visualize_episode(store, config, path, arms, episode, args):
    data = store["data"]
    s, e = episode_range(store, episode)
    ep_len = e - s
    fps = data_fps(args, config)
    print(f"\nEpisode {episode}: steps {s}..{e} (len={ep_len}) @ {fps}fps")
    ts = data["timestamp"][s:e] if "timestamp" in data else None
    times = [float(ts[i] - ts[0]) if ts is not None else i / fps for i in range(ep_len)]

    # raw datasets leave action unfilled (all zero)
    action = data["action"][s:e] if "action" in data else None
    if action is not None and not np.any(action):
        print("  action is all-zero -> skipping action plots")
        action = None
    if args.command and action is None:
        print("  [warn] --command needs a non-zero 'action' key, skipping command overlay")

    cams = []
    if args.cameras:
        cams = resolve_cameras(config, set(data.keys()), path, episode, args.camera_subset)
        print(f"Cameras: {[(c.name, c.video.name if c.video else None) for c in cams]}")

    # plot views, namespaced per arm only when >1 arm
    multi = len(arms) > 1
    views = {}
    if action is not None:
        w = action.shape[1]
        views["action"] = (["ax", "ay", "az", "arx", "ary", "arz", "grip"] if w == 7
                           else [f"A{d}" for d in range(w)])
    sliced = []
    for arm in arms:
        ns = f"{arm.name}/" if multi else ""
        sl = {k: data[key][s:e] if key else None for k, key in arm.keys.items()}
        sliced.append(sl)
        for key, grp in SERIES:
            if has_series(sl, key):
                views[f"{ns}{grp}"] = series_dims(grp, sl[key].shape[1])
            if key == "eef_rot" and args.command and action is not None:
                views[f"{ns}cmd_err"] = CMD_ERR
        views[f"{ns}motion"] = ["speed"]

    def origin_T_for(arm):  # --anchor-origin, slam-frame arms only
        if args.anchor_origin and arm.frame == "slam" and "origin_T_slamworld" in data:
            return data["origin_T_slamworld"][s:e]
        return None

    align = None
    if args.pose_align != "none":
        ref = align_arm_index(args, config)
        ref_pos, ref_R = to_display(arms[ref], sliced[ref]["eef_pos"], sliced[ref]["eef_rot"],
                                    origin_T_for(arms[ref]))
        align = pose_align(args, arms[ref], ref_pos, ref_R)

    select_sink(args, build_blueprint([c.name for c in cams], views))
    log_world_axes("world/origin")
    log_series_styles(views)
    tip_off = load_tool_tip(args.tip_config) if args.tip else None
    log_videos(cams, config, path, fps, args)

    for j, (arm, sl) in enumerate(zip(arms, sliced)):
        ns, ent = f"{arm.name}/" if multi else "", f"world/{arm.name}"
        eef_pos, eef_rot = sl["eef_pos"].astype(float), sl["eef_rot"]
        disp_pos, disp_R = to_display(arm, eef_pos, eef_rot, origin_T_for(arm))
        disp_pos, disp_R = aligned(disp_pos, align), aligned_rot(disp_R, align)
        speed = np.concatenate([[0.0], np.linalg.norm(np.diff(disp_pos, axis=0), axis=1)])

        arm_tip = tip_off
        if arm_tip is not None and eef_rot is None:
            print(f"  [warn] --tip needs eef rotation; arm '{arm.name}' has none, skipping")
            arm_tip = None
        tip_views = overlay_views(ns, views, TIP_POS, TIP_ROT)
        if arm_tip is not None:
            log_series_styles(tip_views)
        tip_hist = []

        # commanded eef (action): a base-frame command, so base-frame state only
        cmd = None
        if args.command and action is not None:
            if arm.frame != "base":
                print(f"  [warn] --command needs base-frame eef; arm '{arm.name}' is "
                      f"'{arm.frame}', skipping command overlay")
            else:
                cmd = arm_command(action, j, len(arms))
                if cmd is None:
                    print(f"  [warn] --command: action width {action.shape[1]} is not 6+g per "
                          f"arm x {len(arms)} arms, skipping command overlay")
        cmd_views = overlay_views(ns, views, CMD_POS, CMD_ROT)
        if cmd is not None:
            cmd_pos, cmd_rot = cmd
            cmd_disp, cmd_R = to_display(arm, cmd_pos, cmd_rot)
            cmd_disp, cmd_R = aligned(cmd_disp, align), aligned_rot(cmd_R, align)
            log_series_styles(cmd_views)

        for i in range(ep_len):
            rr.set_time("time", duration=times[i])
            p = disp_pos[i]
            rr.log(f"{ent}/trajectory",
                   rr.LineStrips3D([disp_pos[:i + 1]], colors=np.array([arm.color]), radii=0.002))
            rr.log(f"{ent}/current",
                   rr.Points3D(p.reshape(1, 3), colors=np.array([[255, 255, 0]]), radii=0.008))
            if disp_R is not None:
                R = disp_R[i]
                log_pose_axes(f"{ent}/pose", p, R)
                if arm_tip is not None:
                    tip_p = p + R @ arm_tip
                    tip_hist.append(tip_p)
                    rr.log(f"{ent}/tip",
                           rr.Points3D(tip_p.reshape(1, 3), colors=np.array([TIP_COLOR]), radii=0.005))
                    rr.log(f"{ent}/tip_traj",
                           rr.LineStrips3D([np.asarray(tip_hist)], colors=np.array([TIP_COLOR]),
                                           radii=0.002))
                    # tip pose in the display frame (rigid mount: eef orientation)
                    for (origin, labels), v in zip(tip_views.items(), (tip_p, matrix_to_aa(R))):
                        log_scalar_group(origin, labels, v)

            if cmd is not None:
                cp = cmd_disp[i]
                rr.log(f"{ent}/command/trajectory",
                       rr.LineStrips3D([cmd_disp[:i + 1]], colors=np.array([CMD_COLOR]), radii=0.0015))
                rr.log(f"{ent}/command/current",
                       rr.Points3D(cp.reshape(1, 3), colors=np.array([CMD_COLOR]), radii=0.006))
                log_pose_axes(f"{ent}/command/pose", cp, cmd_R[i], scale=0.04)
                rr.log(f"{ent}/command/gap",  # state -> command tracking error
                       rr.LineStrips3D([np.stack([p, cp])], colors=np.array([CMD_COLOR]), radii=0.001))
                for (origin, labels), v in zip(cmd_views.items(), (cmd_pos[i], cmd_rot[i])):
                    log_scalar_group(origin, labels, v)
                dp = cmd_pos[i] - eef_pos[i]
                drot = 0.0
                if eef_rot is not None:
                    dR = rot_to_matrix(eef_rot[i]).T @ rot_to_matrix(cmd_rot[i])
                    drot = float(np.degrees(np.linalg.norm(matrix_to_aa(dR))))
                log_scalar_group(f"{ns}cmd_err", CMD_ERR,
                                 [dp[0], dp[1], dp[2], float(np.linalg.norm(dp)), drot])

            if action is not None:
                log_scalar_group("action", views["action"], action[i])
            for key, grp in SERIES:
                if f"{ns}{grp}" in views:
                    log_scalar_group(f"{ns}{grp}", views[f"{ns}{grp}"], sl[key][i])
            log_scalar_group(f"{ns}motion", ["speed"], speed[i])

    if args.frusta and args.cameras:
        log_frusta(cams, arms, data, s, e, config, args, align, times)
    print("Done logging episode.")


# --------------------------------------------------------------------------- #
# All episodes overlaid
# --------------------------------------------------------------------------- #
def visualize_overview(store, config, arms, args):
    """Every episode's eef trajectory in its own color (camera-free), every arm under
    ``world/<arm>/ep_<k>``: the first arm in the episode color, later arms darker."""
    import matplotlib
    data = store["data"]
    starts, ends = episode_bounds(store)
    n_ep = len(ends) if args.max_episodes is None else min(args.max_episodes, len(ends))
    ts_all = data["timestamp"][:] if "timestamp" in data else None
    fps = data_fps(args, config)
    t0_global = float(ts_all[0]) if ts_all is not None else 0.0

    cmap = matplotlib.colormaps["tab20" if n_ep <= 20 else "hsv"].resampled(n_ep)
    colors = [(np.array(cmap(i)[:3]) * 255).astype(int).tolist() for i in range(n_ep)]
    arm_colors = [[c if j == 0 else c * 11 // 20 for c in col]
                  for col in colors for j in range(len(arms))]  # [k * n_arms + j]

    ref = align_arm_index(args, config)
    if args.pose_align in ("first", "last") and arms[ref].keys["eef_rot"] is None:
        print(f"  [warn] --pose-align {args.pose_align}: '{arms[ref].name}' has no eef "
              f"rotation, translating only")

    select_sink(args, rrb.Blueprint(rrb.Spatial3DView(origin="world", name="all episodes"),
                                    collapse_panels=True))
    log_world_axes("world/origin", length=0.5)

    cmd_on = args.command and "action" in data and all(a.frame == "base" for a in arms)
    if args.command and not cmd_on:
        print("  [warn] --command needs an 'action' key and a base-frame eef, skipping command overlay")
    cmd_color = [(c + 255) // 2 for c in CMD_COLOR]

    eps = []  # per episode: (t, [(pos, cmd_pos) per arm])
    for k in range(n_ep):
        s, e = int(starts[k]), int(ends[k])
        placed = []  # (pos, rot, cmd_pos) per arm: world-placed, not yet aligned
        for j, arm in enumerate(arms):
            pos = data[arm.keys["eef_pos"]][s:e].astype(float)
            rot = data[arm.keys["eef_rot"]][s:e].astype(float) if arm.keys["eef_rot"] else None
            cmd = arm_command(data["action"][s:e], j, len(arms)) if cmd_on else None
            cmd_pos = cmd[0] if cmd is not None else None
            if arm.T_world is not None:
                pos = apply_T(arm.T_world, pos)
                if cmd_pos is not None:
                    cmd_pos = apply_T(arm.T_world, cmd_pos)
            placed.append((pos, rot, cmd_pos))

        # one transform per episode from the reference arm keeps the rig geometry
        align = None
        ref_pos, ref_rot, _ = placed[ref]
        if args.pose_align != "none" and len(ref_pos):
            R = None
            if ref_rot is not None:
                R = rot_to_matrix(ref_rot[ALIGN_FRAME[args.pose_align]])
                if arms[ref].T_world is not None:
                    R = arms[ref].T_world[:3, :3] @ R
            align = align_from(args.pose_align, ref_pos, R)

        def display(pts):
            pts = aligned(pts, align)
            if args.flip_z:
                pts = pts.copy()
                pts[:, 2] *= -1.0
            return pts

        per_arm = [(display(pos), display(cmd_pos) if cmd_pos is not None else None)
                   for pos, _, cmd_pos in placed]
        t = ts_all[s:e].astype(float) if ts_all is not None else np.arange(e - s) / fps + s / fps
        if args.time_align == "first":
            t = t - t[0]
        elif args.time_align == "last":
            t = t - t[-1]
        else:
            t = t - t0_global
        eps.append((t, per_arm))
        print(f"  ep {k}: {e - s} frames")

    max_len = max((len(t) for t, _ in eps), default=0)
    for i in range(max_len):
        for k, (t, per_arm) in enumerate(eps):
            if i >= len(t):
                continue
            rr.set_time("time", duration=float(t[i]))
            for j, (arm, (pos, cmd_pos)) in enumerate(zip(arms, per_arm)):
                ns = f"world/{arm.name}/ep_{k}"
                col = np.array([arm_colors[k * len(arms) + j]])
                rr.log(f"{ns}/trajectory", rr.LineStrips3D([pos[:i + 1]], colors=col, radii=0.002))
                rr.log(f"{ns}/current", rr.Points3D(pos[i].reshape(1, 3), colors=col, radii=0.008))
                if cmd_pos is not None:
                    rr.log(f"{ns}/command",
                           rr.LineStrips3D([cmd_pos[:i + 1]], colors=np.array([cmd_color]), radii=0.001))
    print(f"Overview: {n_ep} episodes x {len(arms)} arms, max_len={max_len}.")


# --------------------------------------------------------------------------- #
# Compare mode: one episode of several datasets / key sets
# --------------------------------------------------------------------------- #
def compare_labels(paths, labels=None, suffixes=None):
    """Short per-dataset names: --labels, else the dir names minus their shared prefix
    (cut at a '_') plus the entry's --key-suffix (``umi_x``, ``umi_x_online`` ->
    ``orig``, ``online``; ``umi_x`` read with ``_online`` -> ``orig_online``)."""
    suffixes = suffixes or [""] * len(paths)
    names = [Path(str(p).rstrip("/")).name for p in paths]
    if labels:
        out = [s.strip() for s in labels.split(",")]
        if len(out) != len(paths):
            raise SystemExit(f"--labels: {len(out)} names for {len(paths)} datasets")
    else:
        uniq = list(dict.fromkeys(names))  # a path read with several --key-suffix counts once
        pre = os.path.commonprefix(uniq)
        if not all(len(n) == len(pre) or n[len(pre)] in "_-" for n in uniq):
            pre = pre[:pre.rfind("_") + 1]
        out = [(n[len(pre):].lstrip("_-") or "orig") + sfx for n, sfx in zip(names, suffixes)]
        if len(set(out)) != len(out):  # e.g. same dir name under different parents
            out = [f"{n}{sfx}_{k}" for k, (n, sfx) in enumerate(zip(names, suffixes))]
    out = [re.sub(r"[^\w.-]", "_", s) for s in out]  # entity-path safe
    if len(set(out)) != len(out):
        raise SystemExit(f"--labels: names must be unique, got {out}")
    return out


def compare_diff(sl, ref_sl):
    """(dims, (n, d) array): one arm's per-frame difference to the reference, eef
    position (mm, stored frame), rotation angle (deg) and gripper (mm) when both have it."""
    dp = (sl["eef_pos"].astype(float) - ref_sl["eef_pos"].astype(float)) * 1000.0
    dims, cols = COMPARE_DIFF[:4], [dp, np.linalg.norm(dp, axis=1, keepdims=True)]
    rot, ref_rot = sl["eef_rot"], ref_sl["eef_rot"]
    if rot is not None and ref_rot is not None and rot.shape[1] == ref_rot.shape[1] == 3:
        drot = [np.degrees(np.linalg.norm(matrix_to_aa(rot_to_matrix(r0).T @ rot_to_matrix(r))))
                for r, r0 in zip(rot, ref_rot)]
        dims, cols = dims + ["drot_deg"], cols + [np.asarray(drot)[:, None]]
    if sl["gripper"] is not None and ref_sl["gripper"] is not None:
        dg = (sl["gripper"].astype(float) - ref_sl["gripper"].astype(float)).reshape(len(dp), -1)
        dims, cols = dims + ["dgrip_mm"], cols + [dg[:, :1] * 1000.0]
    return dims, np.concatenate(cols, axis=1)


def visualize_compare(loaded, episode, args):
    """Overlay one episode of several variants of the same recording.

    ``loaded`` = [(label, path, config, store, arms)], the first being the reference:
    its cameras are shown and every other dataset gets a frame-by-frame ``diff`` plot
    against it. 3D goes under ``world/<label>/<arm>``; scalars under
    ``<arm>/<group>/<dim>/<label>`` (one plot per arm x dimension, a line per dataset).
    """
    ref_label, ref_path, ref_config, ref_store, ref_arms = loaded[0]
    fps = data_fps(args, ref_config)
    use_ts = all("timestamp" in store["data"] for _, _, _, store, _ in loaded)
    for flag in ("command", "frusta", "tip"):
        if getattr(args, flag):
            print(f"  [warn] --{flag} is not supported in compare mode, ignoring")
    arm_names = [a.name for a in ref_arms]

    def ns(name):
        return f"{name}/" if len(arm_names) > 1 else ""

    eps = []  # per dataset: (label, color, t, {arm name: (arm, sliced, origin_T)})
    t0 = None
    for k, (label, path, config, store, arms) in enumerate(loaded):
        data = store["data"]
        s, e = episode_range(store, episode, f"[{label}] ")
        if use_ts:
            ts = data["timestamp"][s:e].astype(float)
            t0 = float(ts[0]) if t0 is None else t0
            t = ts - t0  # each dataset's own timestamps on the reference's clock
        else:
            t = np.arange(e - s) / fps
        per_arm = {}
        for arm in arms:
            if arm.name not in arm_names:
                print(f"  [warn] [{label}] arm '{arm.name}' is not in the reference, skipping")
                continue
            sl = {key: data[zk][s:e] if zk else None for key, zk in arm.keys.items()}
            origin_T = (data["origin_T_slamworld"][s:e] if args.anchor_origin
                        and arm.frame == "slam" and "origin_T_slamworld" in data else None)
            per_arm[arm.name] = (arm, sl, origin_T)
        eps.append((label, DATASET_COLORS[k % len(DATASET_COLORS)], t, per_arm))
        print(f"[{label}] episode {episode}: steps {s}..{e} (len={e - s})")

    # per-frame difference to the reference (same length + same eef frame)
    diffs = {}  # (dataset index, arm name) -> (dims, (n, d) array)
    ref_t, ref_per_arm = eps[0][2], eps[0][3]
    for k, (label, _, t, per_arm) in enumerate(eps[1:], start=1):
        if len(t) != len(ref_t):
            print(f"  [warn] [{label}] {len(t)} frames vs reference {len(ref_t)}, no diff plot")
            continue
        if use_ts and not np.allclose(t, ref_t, atol=1e-3):
            print(f"  [warn] [{label}] timestamps differ from the reference, diff is frame-by-frame")
        for name, (arm, sl, _) in per_arm.items():
            if name not in ref_per_arm:
                continue
            ref_arm, ref_sl, _ = ref_per_arm[name]
            if arm.frame != ref_arm.frame:
                print(f"  [warn] [{label}] arm '{name}' is '{arm.frame}' frame vs reference "
                      f"'{ref_arm.frame}', no diff plot")
                continue
            diffs[(k, name)] = compare_diff(sl, ref_sl)

    groups = {}  # group -> {arm name: dims}
    for name in arm_names:
        for _, _, _, per_arm in eps:
            if name in per_arm:
                sl = per_arm[name][1]
                for key, grp in SERIES:
                    if has_series(sl, key):
                        groups.setdefault(grp, {}).setdefault(name, series_dims(grp, sl[key].shape[1]))
        groups.setdefault("motion", {})[name] = ["speed"]
    if diffs:
        groups["diff"] = {name: COMPARE_DIFF for name in arm_names}
    plots = {grp: [f"{ns(name)}{grp}/{dim}" for dim in max(by_arm.values(), key=len)
                   for name in arm_names]
             for grp, by_arm in groups.items()}

    cams = []
    if args.cameras:
        cams = [c for c in resolve_cameras(ref_config, set(ref_store["data"].keys()), ref_path,
                                           episode, args.camera_subset)
                if c.video is not None]
        print(f"Cameras ({ref_label}): {[(c.name, c.video.name) for c in cams]}")

    select_sink(args, build_compare_blueprint([c.name for c in cams], plots, len(arm_names)))
    log_world_axes("world/origin")
    log_videos(cams, ref_config, ref_path, fps, args)

    # one --pose-align transform from the reference, shared so the offsets survive
    align = None
    if args.pose_align != "none":
        ref_arm, ref_sl, ref_oT = ref_per_arm[arm_names[align_arm_index(args, ref_config)]]
        ref_pos, ref_R = to_display(ref_arm, ref_sl["eef_pos"], ref_sl["eef_rot"], ref_oT)
        align = pose_align(args, ref_arm, ref_pos, ref_R, f"[{ref_label}] ")

    for k, (label, color, t, per_arm) in enumerate(eps):
        for j, name in enumerate(arm_names):
            if name not in per_arm:
                continue
            arm, sl, origin_T = per_arm[name]
            col = np.array([color if j == 0 else [c * 11 // 20 for c in color]])
            disp_pos, disp_R = to_display(arm, sl["eef_pos"], sl["eef_rot"], origin_T)
            disp_pos, disp_R = aligned(disp_pos, align), aligned_rot(disp_R, align)
            speed = np.concatenate([[0.0], np.linalg.norm(np.diff(disp_pos, axis=0), axis=1)])

            series = []  # (entity prefix, dims, (n, d) values)
            for key, grp in SERIES:
                if sl[key] is not None and name in groups.get(grp, {}):
                    arr = np.asarray(sl[key], dtype=float).reshape(len(t), -1)
                    series.append((f"{ns(name)}{grp}", groups[grp][name][:arr.shape[1]], arr))
            series.append((f"{ns(name)}motion", ["speed"], speed[:, None]))
            if (k, name) in diffs:
                series.append((f"{ns(name)}diff", *diffs[(k, name)]))
            for prefix, dims, _ in series:
                for dim in dims:
                    rr.log(f"{prefix}/{dim}/{label}",
                           rr.SeriesLines(names=[label], colors=[color]), static=True)

            ent = f"world/{label}/{name}"
            for i in range(len(t)):
                rr.set_time("time", duration=float(t[i]))
                rr.log(f"{ent}/trajectory", rr.LineStrips3D([disp_pos[:i + 1]], colors=col, radii=0.002))
                rr.log(f"{ent}/current", rr.Points3D(disp_pos[i].reshape(1, 3), colors=col, radii=0.008))
                if disp_R is not None:
                    log_pose_axes(f"{ent}/pose", disp_pos[i], disp_R[i])
                for prefix, dims, arr in series:
                    for d, dim in enumerate(dims):
                        rr.log(f"{prefix}/{dim}/{label}", rr.Scalars(float(arr[i, d])))
    print(f"Compare: {len(eps)} datasets x {len(arm_names)} arms, reference '{ref_label}'.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="+",
                   help="Dataset root (config.json + replay_buffer.zarr + videos/). Several "
                        "roots = compare mode, the first is the reference")
    p.add_argument("--config", default=None, help="Override config.json location")
    p.add_argument("--episode", default="0", help="Episode index, or 'all' for overview mode")
    p.add_argument("--labels", default=None,
                   help="Compare mode: comma list of dataset names, one per path (default: "
                        "dir names minus their shared prefix)")
    p.add_argument("--key-suffix", action="append", default=None, metavar="SUFFIX",
                   help="Compare mode: also read every path with <key>SUFFIX in place of <key> "
                        "where the zarr has it (e.g. _online); repeatable, one path is enough")

    cam = add_camera_args(p)
    cam.add_argument("--frusta", action="store_true",
                     help="Place cameras in 3D via per-frame cam_*_c2w (base-frame datasets)")
    cam.add_argument("--center-crop", action="store_true",
                     help="Center-crop+resize frames to obs_resolution_wh (matches obs pipeline)")
    cam.add_argument("--jpeg-quality", type=int, default=90,
                     help="JPEG quality for RGB frames (0 = store raw, much larger)")
    cam.add_argument("--depth-near", type=float, default=0.07, help="Depth colormap near (m)")
    cam.add_argument("--depth-far", type=float, default=0.5, help="Depth colormap far (m)")

    add_tip_args(p)
    p.add_argument("--command", action="store_true",
                   help="Overlay the commanded eef (action[:, :6] per arm) on the state: 3D "
                        "trajectory, cmd_* lines in eef_pos/eef_rot and a cmd_err plot")

    frame = p.add_argument_group("coordinate frame")
    frame.add_argument("--frame", choices=["auto", "base", "slam"], default="auto",
                       help="Which eef frame to plot (auto: base if present else slam)")
    frame.add_argument("--anchor-origin", action="store_true",
                       help="Apply per-frame origin_T_slamworld to slam-frame eef")
    add_arm_pose_args(frame)
    frame.add_argument("--pose-align", choices=["none", "first", "last", "first-pos", "last-pos"],
                       default="none",
                       help="Move the display origin to the reference arm's first/last eef pose "
                            "(one transform per episode for every arm, tip, command and camera); "
                            "first/last also rotate it to identity, *-pos translate only. 3D only")
    frame.add_argument("--pose-align-arm", default=None, metavar="ARM",
                       help="Reference arm for --pose-align, by config arms[].name (left/right) "
                            "or obs_name (robot_1); default: first arm")

    ov = p.add_argument_group("overview mode (--episode all)")
    ov.add_argument("--max-episodes", type=int, default=None)
    ov.add_argument("--time-align", choices=["none", "first", "last"], default="none")
    ov.add_argument("--flip-z", action="store_true", help="Negate z for display")

    t = p.add_argument_group("timing")
    t.add_argument("--data-fps", type=float, default=None, help="Default: config collection.frequency")
    t.add_argument("--video-fps", type=float, default=None, help="PTS fallback only")

    add_sink_args(p, grpc_port=8060)
    return p


def load_dataset(path, args, suffix=""):
    """(config, store, arms) of one dataset root; ``suffix`` reads ``<key><suffix>`` in
    place of ``<key>`` wherever the zarr has it (--key-suffix), other keys are shared."""
    config = load_config(path, args.config)
    store = load_store(path)
    keys = store["data"].keys()
    alias = {k: k for k in keys}
    if suffix:
        alias.update({k[:-len(suffix)]: k for k in keys if k.endswith(suffix)})
    return config, store, resolve_arms(path, config, keys, args, alias)


def main():
    args = build_arg_parser().parse_args()

    if len(args.path) > 1 or args.key_suffix:
        if str(args.episode).lower() == "all":
            raise SystemExit("compare mode (several paths / --key-suffix) shows one episode: "
                             "pass --episode N")
        variants = []  # (path, key suffix): every path as stored, then once per --key-suffix
        for path in args.path:
            variants.append((path, ""))
            data_keys = set(load_store(path)["data"].keys())
            for sfx in args.key_suffix or []:
                if any(k.endswith(sfx) for k in data_keys):
                    variants.append((path, sfx))
                else:
                    print(f"  [warn] {path}: no '*{sfx}' keys, --key-suffix {sfx} skipped there")
        if len(variants) < 2:
            raise SystemExit("compare mode needs at least two datasets / key suffixes")
        labels = compare_labels([p for p, _ in variants], args.labels, [s for _, s in variants])
        loaded = []
        for label, (path, sfx) in zip(labels, variants):
            print(f"[{label}] {path}" + (f"  (keys *{sfx})" if sfx else ""))
            loaded.append((label, path, *load_dataset(path, args, sfx)))
        rr.init(f"compare_{len(loaded)}ds_ep{args.episode}", spawn=False)
        visualize_compare(loaded, int(args.episode), args)
    else:
        path = args.path[0]
        config, store, arms = load_dataset(path, args)
        rr.init(f"{config.get('name', 'dataset')}_{args.episode}", spawn=False)
        if str(args.episode).lower() == "all":
            args.cameras = False
            visualize_overview(store, config, arms, args)
        else:
            visualize_episode(store, config, path, arms, int(args.episode), args)
    wait_if_serving(args)


if __name__ == "__main__":
    main()
