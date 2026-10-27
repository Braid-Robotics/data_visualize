"""Shared helpers for the dataset / rollout-chunk Rerun visualizers.

Config + zarr key resolution, arm placement, camera video decode, the output
sink and the CLI flags both scripts take.
"""

import json
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import zarr

HERE = Path(__file__).resolve().parent
DEVICES_ENV = "REAL_ROBOT_DEVICES"
VIZ_SIDECAR = "viz.json"

ARM_COLORS = [[255, 80, 80], [80, 80, 255], [80, 200, 80], [230, 180, 60]]  # cycled by arm index
AXIS_COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # X, Y, Z
DATASET_COLORS = [  # one per dataset when several are overlaid, cycled
    [255, 90, 90], [90, 120, 255], [90, 220, 90],
    [235, 200, 60], [200, 90, 235], [90, 220, 220],
]


@dataclass
class ArmSpec:
    name: str                       # config obs_name, e.g. "robot_0"
    color: list                     # trajectory color
    frame: str                      # "base" or "slam" (which eef frame we plot)
    keys: dict = field(default_factory=dict)  # logical name -> real zarr key (or None)
    T_world: np.ndarray = None      # optional 4x4 world<-base placing this arm's base


@dataclass
class CamSpec:
    name: str                       # config obs_name, e.g. "cam_0"
    video: Path = None              # resolved video file (mp4/mkv) for the episode
    is_depth: bool = False
    K: np.ndarray = None            # 3x3 intrinsics at capture resolution (for frusta)
    resolution_wh: tuple = None     # (w, h) matching K
    c2w_key: str = None             # zarr key holding per-frame cam-to-world 4x4 (or None)


# --------------------------------------------------------------------------- #
# Math (no scipy dependency)
# --------------------------------------------------------------------------- #
def rot_to_matrix(v):
    """Axis-angle 3-vector or quaternion (xyzw) 4-vector -> 3x3 rotation matrix."""
    v = np.asarray(v, dtype=float)
    if v.shape[-1] == 4:
        n = np.linalg.norm(v)
        if n < 1e-9:
            return np.eye(3)
        x, y, z, w = v / n
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    theta = float(np.linalg.norm(v))
    if theta < 1e-9:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def matrix_to_aa(R):
    """3x3 rotation matrix -> axis-angle 3-vector."""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(cos))
    if th < 1e-9:
        return np.zeros(3)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return ax / (2.0 * np.sin(th)) * th


def apply_T(T, pts):
    """Apply a 4x4 homogeneous transform to an (N,3) array of points."""
    return np.asarray(pts, dtype=float) @ T[:3, :3].T + T[:3, 3]


# --------------------------------------------------------------------------- #
# Config, arm placement, key resolution
# --------------------------------------------------------------------------- #
def load_config(path, config_override=None):
    cfg_path = Path(config_override) if config_override else Path(path) / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.json at {cfg_path}")
    with open(cfg_path) as f:
        return json.load(f)


def find_devices_json(explicit=None):
    """Locate real_robot/config/devices.json: --devices, $REAL_ROBOT_DEVICES, sibling repo."""
    for c in (explicit, os.environ.get(DEVICES_ENV),
              HERE.parent / "real_robot" / "config" / "devices.json"):
        if c and Path(c).exists():
            return Path(c)
    if explicit:
        raise FileNotFoundError(f"--devices {explicit}: not found")
    return None


def load_arm_world_poses(path, config, cli_poses, devices_path=None):
    """{obs_name: [x, y, z, rx, ry, rz]} placing each arm's base in a shared world
    (m / axis-angle rad, world <- base; 3D display only). Sources, later wins:

      1. real_robot devices.json ``arms[<config arms[].device>].base_pose``
         (rig frame := ur5e_left base frame)
      2. ``<dataset>/viz.json`` -> ``{"arm_world_pose": {"robot_1": [...]}}``
      3. ``--arm-pose robot_1=x,y,z,rx,ry,rz`` (repeatable)

    Not read from config.json: its schema rejects unknown arm keys.
    """
    poses = {}
    dev = find_devices_json(devices_path)
    if dev is not None:
        with open(dev) as f:
            dev_arms = json.load(f).get("arms", {})
        for a in config.get("arms", []):
            bp = dev_arms.get(a.get("device") or "", {}).get("base_pose")
            if bp is not None:
                poses[a["obs_name"]] = bp
        if poses:
            print(f"Arm placement from {dev} (arms[].device -> base_pose)")
    side = Path(path) / VIZ_SIDECAR
    if side.exists():
        with open(side) as f:
            poses.update(json.load(f).get("arm_world_pose", {}))
        print(f"Arm placement override from {side}")
    for spec in cli_poses or []:
        name, _, vals = spec.partition("=")
        nums = [float(v) for v in vals.split(",")] if vals else []
        if not name or len(nums) != 6:
            raise SystemExit(f"--arm-pose expects NAME=x,y,z,rx,ry,rz, got {spec!r}")
        poses[name] = nums
    return poses


def _first_present(data_keys, candidates):
    return next((c for c in candidates if c in data_keys), None)


def resolve_arm(arm_cfg, data_keys, idx, frame_pref, world_pose=None):
    """ArmSpec with every logical key resolved against the zarr's actual keys.

    Raw collection stores ``robot_0_slam_eef_pos``, rollouts ``robot_0_eef_pos``;
    'auto' prefers the base frame (it lines up with cam_*_c2w frusta). All eef keys
    come from the chosen frame so pose/velocity are never mixed across frames.
    """
    p = arm_cfg["obs_name"]
    base_pos, slam_pos = f"{p}_eef_pos", f"{p}_slam_eef_pos"
    frame = "base" if frame_pref == "auto" else frame_pref
    if frame == "base" and base_pos not in data_keys:
        frame = "slam"
    if frame == "slam" and slam_pos not in data_keys and base_pos in data_keys:
        frame = "base"

    infix = "" if frame == "base" else "slam_"
    keys = {
        "eef_pos": _first_present(data_keys, [f"{p}_{infix}eef_pos", base_pos, slam_pos]),
        "eef_rot": _first_present(data_keys, [
            f"{p}_{infix}eef_rot_axis_angle", f"{p}_{infix}eef_quat"]),
        "eef_pos_vel": _first_present(data_keys, [f"{p}_{infix}eef_pos_vel"]),
        "slam_conf": _first_present(data_keys, [f"{p}_slam_conf"]),
        "joint_pos": _first_present(data_keys, [f"{p}_joint_pos"]),
        "joint_vel": _first_present(data_keys, [f"{p}_joint_vel"]),
        "tcp_force": _first_present(data_keys, [f"{p}_tcp_force"]),
        "gripper": _first_present(data_keys, [(arm_cfg.get("gripper") or {}).get("obs_key"),
                                              f"{p}_gripper", f"{p}_gripper_width"]),
    }
    if keys["eef_pos"] is None:
        raise ValueError(f"Arm '{p}': no eef_pos key found in zarr (looked for "
                         f"{base_pos!r}/{slam_pos!r}). Available: {sorted(data_keys)}")

    T_world = None
    if world_pose is not None:
        wp = np.asarray(world_pose, dtype=float)
        if wp.shape != (6,):
            raise ValueError(f"Arm '{p}': world_pose must be 6 numbers, got {wp.tolist()}")
        T_world = np.eye(4)
        T_world[:3, :3] = rot_to_matrix(wp[3:6])
        T_world[:3, 3] = wp[:3]
    return ArmSpec(name=p, color=ARM_COLORS[idx % len(ARM_COLORS)], frame=frame,
                   keys=keys, T_world=T_world)


def resolve_arms(path, config, data_keys, args, alias=None, tag=""):
    """ArmSpec per config arm, placed in the shared world unless --no-arm-pose.

    ``alias`` maps the key names to probe onto the real zarr keys (--key-suffix);
    default: the zarr keys themselves.
    """
    alias = alias or {k: k for k in data_keys}
    world_poses = ({} if args.no_arm_pose else
                   load_arm_world_poses(path, config, args.arm_pose, args.devices))
    arms = [resolve_arm(a, set(alias), i, args.frame, world_poses.get(a["obs_name"]))
            for i, a in enumerate(config.get("arms", []))]
    if not arms:
        raise ValueError(f"{path}: config.json has no 'arms'")
    for a in arms:
        a.keys = {k: alias[v] if v else None for k, v in a.keys.items()}
        wp = f", world_pose t={np.round(a.T_world[:3, 3], 3).tolist()}" if a.T_world is not None else ""
        print(f"{tag}Arm '{a.name}' [{a.frame} frame{wp}]: "
              f"{ {k: v for k, v in a.keys.items() if v} }")
    return arms


def data_fps(args, config, default=25.0):
    return args.data_fps or float(config.get("collection", {}).get("frequency", default))


# --------------------------------------------------------------------------- #
# Zarr
# --------------------------------------------------------------------------- #
def load_store(path):
    zpath = Path(path) / "replay_buffer.zarr"
    store = zarr.open(str(zpath), mode="r")
    if "data" not in store:
        raise ValueError(f"No 'data' group in {zpath}")
    return store


def episode_bounds(store):
    ends = store["meta"]["episode_ends"][:].astype(int)
    return np.concatenate([[0], ends[:-1]]), ends


def episode_range(store, episode, label=""):
    """(start, end) step indices of one episode."""
    starts, ends = episode_bounds(store)
    if episode < 0 or episode >= len(ends):
        raise ValueError(f"{label}Episode {episode} out of range (0..{len(ends) - 1})")
    return int(starts[episode]), int(ends[episode])


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
def intrinsics_matrix(cam_cfg):
    intr = cam_cfg.get("intrinsics")
    if not intr or not all(k in intr for k in ("fx", "fy", "cx", "cy")):
        return None  # only needed for frusta
    return np.array([[intr["fx"], 0, intr["cx"]],
                     [0, intr["fy"], intr["cy"]],
                     [0, 0, 1.0]], dtype=float)


def resolve_cameras(config, data_keys, path, episode, subset=None):
    """CamSpec per config camera, resolved to ``<path>/videos/<episode>/<obs_name>.mp4|.mkv``.

    Missing ``<obs_name>`` files fall back to the episode's videos by position
    (datasets whose stems are device names, e.g. ``flir0.mp4``). ``subset`` is the
    --camera-subset comma list.
    """
    subset = set(subset.split(",")) if subset else None
    ep_dir = Path(path) / "videos" / str(episode)
    on_disk = sorted(ep_dir.glob("*.mp4")) + sorted(ep_dir.glob("*.mkv")) if ep_dir.exists() else []
    used = set()
    specs = []
    for c in config.get("cameras", []):
        name = c["obs_name"]
        if subset and name not in subset:
            continue
        video = next((f for f in (ep_dir / f"{name}.mp4", ep_dir / f"{name}.mkv") if f.exists()), None)
        if video is None:
            video = next((f for f in on_disk if f not in used), None)
            if video is not None:
                print(f"  [warn] camera '{name}': no <name>.mp4/.mkv, "
                      f"falling back to {video.name} by position")
        if video is not None:
            used.add(video)
        cap_wh = c.get("capture_resolution_wh")
        specs.append(CamSpec(
            name=name,
            video=video,
            is_depth=video is not None and video.suffix == ".mkv",
            K=intrinsics_matrix(c),
            resolution_wh=tuple(cap_wh) if cap_wh else None,
            c2w_key=f"{name}_c2w" if f"{name}_c2w" in data_keys else None,
        ))
    return specs


def load_depth_scale(path):
    intr = Path(path) / "realsense_intrinsics.json"
    if intr.exists():
        with open(intr) as f:
            return float(json.load(f).get("depth_scale", 1e-4))
    return 1e-4


def center_crop_resize(img, out_wh):
    """Center-crop to a square then resize to out_wh (matches obs preprocessing)."""
    import cv2
    h, w = img.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    sq = img[y0:y0 + side, x0:x0 + side]
    if out_wh:
        sq = cv2.resize(sq, tuple(out_wh), interpolation=cv2.INTER_AREA)
    return sq


def decode_and_log_video(entity, cam, video_fps, depth_scale, depth_range=(0.07, 0.5),
                         center_crop=False, crop_wh=None, jpeg_quality=90):
    """Decode a camera video onto the 'time' timeline (frame PTS, else n / video_fps).

    RGB is JPEG-compressed (``jpeg_quality``; 0 = raw uint8), depth is lossless.
    """
    import av
    n = 0
    with av.open(str(cam.video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            t = frame.time if frame.time is not None else n / video_fps
            rr.set_time("time", duration=t)
            if cam.is_depth:
                meter = 1.0 / depth_scale
                rr.log(entity, rr.DepthImage(
                    frame.to_ndarray(format="gray16le"), meter=meter,
                    depth_range=(depth_range[0] * meter, depth_range[1] * meter),
                    colormap="Turbo"))
            else:
                img = frame.to_ndarray(format="rgb24")
                if center_crop:
                    img = center_crop_resize(img, crop_wh)
                image = rr.Image(img)
                rr.log(entity, image.compress(jpeg_quality=jpeg_quality) if jpeg_quality else image)
            n += 1
    print(f"  {entity}: {n} {'depth' if cam.is_depth else 'rgb'} frames")


# --------------------------------------------------------------------------- #
# 3D scene
# --------------------------------------------------------------------------- #
def log_world_axes(origin_entity, length=0.3):
    for j, c in enumerate(AXIS_COLORS):
        vec = np.zeros((1, 3))
        vec[0, j] = length
        rr.log(f"{origin_entity}/{'XYZ'[j]}",
               rr.Arrows3D(origins=np.zeros((1, 3)), vectors=vec,
                           colors=np.array([c])), static=True)


def log_pose_axes(entity, p, R, scale=0.06):
    """Small RGB triad for rotation R (3x3, display frame) at position p."""
    rr.log(entity, rr.Arrows3D(
        origins=np.tile(np.asarray(p).reshape(1, 3), (3, 1)),
        vectors=R.T * scale,          # row k = world axis k
        colors=np.array(AXIS_COLORS)))


def load_tool_tip(cfg_path):
    """Tool-tip offset in the TCP frame from a tool-mount calibration json.

    The json gives TCP and tip in CAD coordinates (``mesh_scale`` units) plus the
    CAD->TCP rotation; the tip is then eef_pos + R_world_tcp @ offset.
    """
    with open(cfg_path) as f:
        cfg = json.load(f)
    s = float(cfg.get("mesh_scale", 1.0))
    R = rot_to_matrix(np.asarray(cfg.get("cad_to_tcp_rot_aa", [0.0, 0.0, 0.0])))
    tcp = np.asarray(cfg["tcp_in_cad"], dtype=float)
    tip = np.asarray(cfg["tip_in_cad"], dtype=float)
    return R @ ((tip - tcp) * s)


def scene_column(cam_names):
    """Left blueprint column: the 3D scene above one 2D view per camera."""
    left = [rrb.Spatial3DView(origin="world", name="3D scene")]
    if cam_names:
        left.append(rrb.Horizontal(
            *[rrb.Spatial2DView(origin=f"world/cameras/{n}", name=n) for n in cam_names]))
    return rrb.Vertical(*left)


# --------------------------------------------------------------------------- #
# Output sink
# --------------------------------------------------------------------------- #
def _port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
        sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sk.bind(("0.0.0.0", port))
            return False
        except OSError:
            return True


def select_sink(args, blueprint):
    if args.save:
        rr.save(args.save, default_blueprint=blueprint)
        print(f"Saving recording to {args.save}")
    elif args.serve:
        # rr.serve_grpc() hangs silently on a bound port (usually a previous --serve)
        if _port_in_use(args.grpc_port):
            raise SystemExit(
                f"[error] gRPC port {args.grpc_port} is already in use (a previous --serve?).\n"
                f"        Free it:  fuser -k {args.grpc_port}/tcp\n"
                f"        Or use:   --grpc-port {args.grpc_port + 1}")
        # serve_grpc(default_blueprint=...) deadlocks on newer rerun: send it separately
        rr.serve_grpc(grpc_port=args.grpc_port)
        rr.send_blueprint(blueprint)
        print(f"Serving on gRPC :{args.grpc_port}. View it with `rerun --connect` "
              f"(a bare `rerun <url>` starts its own empty server):\n"
              f"  # same network:  rerun --connect rerun+http://<this-host-ip>:{args.grpc_port}/proxy\n"
              f"  # via ssh:       ssh -L {args.grpc_port}:localhost:{args.grpc_port} <user>@<server>\n"
              f"                   rerun --connect rerun+http://localhost:{args.grpc_port}/proxy")
    else:
        rr.spawn()
        rr.send_blueprint(blueprint)


def wait_if_serving(args):
    if not args.serve:
        return
    print("Logging done. Server running — Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


# --------------------------------------------------------------------------- #
# CLI flags shared by both scripts
# --------------------------------------------------------------------------- #
def add_camera_args(p):
    g = p.add_argument_group("cameras")
    g.add_argument("--cameras", dest="cameras", action="store_true", default=True,
                   help="Show camera video (default on)")
    g.add_argument("--no-cameras", dest="cameras", action="store_false",
                   help="Skip all video decode (fast)")
    g.add_argument("--camera-subset", default=None, help="Comma list of obs_names, e.g. cam_0,cam_1")
    return g


def add_arm_pose_args(g):
    g.add_argument("--arm-pose", action="append", metavar="NAME=x,y,z,rx,ry,rz",
                   help="Place arm NAME's base in the shared world (m, axis-angle rad); "
                        "repeatable. Overrides devices.json and <dataset>/viz.json")
    g.add_argument("--devices", default=None,
                   help="real_robot devices.json holding arms[].base_pose (default: "
                        f"${DEVICES_ENV}, else ../real_robot/config/devices.json)")
    g.add_argument("--no-arm-pose", action="store_true",
                   help="Draw the eef as stored: ignore devices.json, viz.json and --arm-pose")


def add_tip_args(p):
    g = p.add_argument_group("torch tip")
    g.add_argument("--tip", action="store_true",
                   help="Draw the torch-tip point + trajectory from the eef pose and the "
                        "mount calibration")
    g.add_argument("--tip-config", default=str(HERE / "torch_mount.json"),
                   help="Tool-mount json: mesh_scale, tcp_in_cad, cad_to_tcp_rot_aa, tip_in_cad")
    return g


def add_sink_args(p, grpc_port):
    g = p.add_argument_group("output sink")
    g.add_argument("--serve", action="store_true", help="gRPC serve for headless/SSH viewing")
    g.add_argument("--grpc-port", type=int, default=grpc_port)
    g.add_argument("--save", default=None, help="Write .rrd instead of spawning a viewer")
