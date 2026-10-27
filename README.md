# data_visualize

[Rerun](https://rerun.io) visualizers for Braid robot datasets.

| script | shows |
|---|---|
| `visualize_dataset.py` | raw-collection / rollout datasets: 3D eef trajectories, cameras, scalar plots; all-episode overview; compare mode |
| `visualize_rollout_chunks.py` | a rollout's trajectory vs. the policy's predicted action chunks (`policy_logs/episode_<n>.jsonl`), gripper panels |
| `viz_common.py` | shared helpers (config / zarr keys, arm placement, video, sink, CLI flags) |

Both read the dataset's own `config.json` (arms, cameras, obs-key prefixes, frequency):

```
<path>/
    config.json            # arms[], cameras[], collection.frequency
    replay_buffer.zarr/    # data/<keys> + meta/episode_ends
    videos/<ep>/<cam>.mp4  # one video per camera obs_name
    policy_logs/           # rollouts only: episode_<n>.jsonl, one line per inference cycle
```

## Setup

```bash
uv sync                                    # rerun-sdk 0.26, zarr 2.x, av, opencv, matplotlib
uv run python visualize_dataset.py <path>
```

## visualize_dataset.py

```bash
python visualize_dataset.py <path>                              # episode 0 with cameras
python visualize_dataset.py <path> --episode 3 --no-cameras     # fast: no video decode
python visualize_dataset.py <path> --episode all --pose-align last --pose-align-arm right
python visualize_dataset.py <path> --command --no-cameras       # commanded eef vs. state
python visualize_dataset.py <path> --frusta                     # cameras placed via cam_*_c2w
python visualize_dataset.py <path> --tip                        # torch tip (torch_mount.json)
```

- Per-arm zarr keys are resolved automatically (`robot_0_slam_eef_pos` vs `robot_0_eef_pos`);
  `--frame auto` plots the base frame when present, else SLAM. An all-zero `action` is skipped.
- `--pose-align {first,last,first-pos,last-pos}` moves the display origin to one arm's first/last
  eef pose (one transform per episode for every arm, tip, command and camera). `first`/`last`
  also rotate that pose to identity; `*-pos` translate only. 3D only.
- `--episode all` overlays every episode (camera-free); `--max-episodes`, `--time-align`, `--flip-z`.

### Compare mode

Several dataset roots overlay the same episode of each. The first is the reference: its
cameras are shown and every other dataset gets a `diff` tab against it (position mm,
rotation deg, gripper mm; needs equal episode length).

```bash
python visualize_dataset.py $R/box_102126 $R/box_102126_gripfix $R/box_102126_online --episode 3 --no-arm-pose
# two key sets in ONE dataset: robot_*_slam_eef_* vs robot_*_slam_eef_*_online
python visualize_dataset.py $R/box_102126 --key-suffix _online --labels offline,online --episode 3 --no-arm-pose
```

Labels default to the dir names minus their shared prefix. `--command`, `--frusta`, `--tip`
and `--episode all` are single-dataset only.

## visualize_rollout_chunks.py

```bash
python visualize_rollout_chunks.py <rollout> --episode 0
python visualize_rollout_chunks.py <offline_rollout> --config <train_set>/config.json   # no own config.json
python visualize_rollout_chunks.py <rollout_a> <rollout_b> --episode 0                 # A/B overlay
```

Handles offline (`step_in_episode`) and online (`cycle` / `obs_ts`) policy logs. Online
rollouts with width grippers get a gripper tab per arm (raw width of chunks t-1 / t / t+1
and what was executed). `--radius-scale 10` for room-scale UMI scenes, `--chunk-stride N`
to thin the overlay, `--tip` / `--seams seams.json` for the torch tip and the taught seam.

## Multi-arm placement

Each arm's eef is stored in its own base frame. For the 3D view the bases are placed in a
shared world (`[x, y, z, rx, ry, rz]`, m + axis-angle rad, world <- base), later wins:

1. `real_robot/config/devices.json` `arms[<device>].base_pose`, via the dataset config's
   `arms[].device` (found with `--devices`, `$REAL_ROBOT_DEVICES`, or `../real_robot`)
2. `<dataset>/viz.json`: `{"arm_world_pose": {"robot_1": [-0.61, 0, 0, 0, 0, 0]}}`
3. `--arm-pose robot_1=-0.61,0,0,0,0,0` (repeatable)

`--no-arm-pose` draws the eef as stored (arms already in a shared frame, e.g. UMI tag world).

## Remote viewing

```bash
python visualize_dataset.py <path> --serve        # on the host (port 8060; rollout script 9876)
ssh -L 8060:localhost:8060 <user>@<host>          # on the laptop
rerun --connect rerun+http://localhost:8060/proxy # viewer version must match the SDK (0.26.x)
```

`--save out.rrd` writes a recording instead (`rerun out.rrd`); RGB frames are JPEG-compressed
(`--jpeg-quality`, 0 = raw).
