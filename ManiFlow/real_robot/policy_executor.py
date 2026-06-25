#!/usr/bin/env python3
"""
real_robot/policy_executor.py
==============================
Real-robot DP3 policy execution.

Architecture mirrors 3D-Diffusion-Policy/policy_executor.py:
  - ROS2 camera node in a background thread (depth + rgb + cam_info)
  - obs_worker thread: samples observation ring at obs_hz
  - inference_worker thread: runs DDIM policy async
  - Main control loop: executes action chunks via RTDE servoJ

Robot state (arm joints + gripper) is read directly via RTDE — no ROS2
joint state subscriptions required.

Run:
  python3 real_robot/policy_executor.py \
    --config   real_robot/real_config.yaml \
    --checkpoint /path/to/epoch=XXXX-val_loss=X.ckpt \
    --n_action_steps 8 --infer_steps 10 --no_kickstart
"""

import argparse
import collections
import queue as _queue
import socket as _socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo

import rtde_control
import rtde_receive
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

sys.path.insert(0, str(Path(__file__).parent.parent / "3D-Diffusion-Policy"))
sys.path.insert(0, str(Path("/home/rosi/maniflow/ManiFlow_Policy/ManiFlow")))

_DP3_AVAILABLE = False
_MANIFLOW_AVAILABLE = False
try:
    from train import TrainDP3Workspace
    _DP3_AVAILABLE = True
except ImportError:
    pass
try:
    import dill as _dill
    from maniflow.workspace.train_maniflow_robotwin_workspace import TrainManiFlowRoboTwinWorkspace
    _MANIFLOW_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Point cloud helpers — GPU FPS preferred (pytorch3d), CPU fallback
# ---------------------------------------------------------------------------

try:
    import pytorch3d.ops as _p3d_ops
    _HAVE_P3D = True
except ImportError:
    _HAVE_P3D = False

_obs_cuda_stream = None


def fps_or_pad(pts: np.ndarray, n: int) -> np.ndarray:
    """Downsample (N,C) to (n,C) via FPS on GPU if available, else CPU."""
    global _obs_cuda_stream
    n_feat = pts.shape[1]
    if len(pts) == 0:
        return np.zeros((n, n_feat), dtype=np.float32)
    if len(pts) <= n:
        pad = np.zeros((n - len(pts), n_feat), dtype=np.float32)
        return np.vstack([pts, pad]).astype(np.float32)

    if _HAVE_P3D and torch.cuda.is_available():
        if _obs_cuda_stream is None:
            _obs_cuda_stream = torch.cuda.Stream()
        n_pre = min(len(pts), max(n * 2, 16384))
        if len(pts) > n_pre:
            idx_pre = np.random.choice(len(pts), n_pre, replace=False)
            pts_pre = pts[idx_pre]
        else:
            pts_pre = pts
        with torch.cuda.stream(_obs_cuda_stream):
            pts_t = torch.from_numpy(pts_pre).float().unsqueeze(0).cuda()
            _, idx = _p3d_ops.sample_farthest_points(pts_t[..., :3], K=n)
            result = pts_t[0, idx[0]].cpu().numpy()
        _obs_cuda_stream.synchronize()
        return result.astype(np.float32)

    # CPU fallback
    n_pre = min(len(pts), max(n, 8192))
    if len(pts) > n_pre:
        pts = pts[np.random.choice(len(pts), n_pre, replace=False)]
    xyz = pts[:, :3]
    sel = np.zeros(n, dtype=np.int64)
    d   = np.full(len(pts), np.inf)
    cur = 0
    for i in range(n):
        sel[i] = cur
        nd  = np.sum((xyz - xyz[cur]) ** 2, axis=1)
        d   = np.minimum(d, nd)
        cur = int(np.argmax(d))
    return pts[sel].astype(np.float32)


def depth_to_pointcloud(depth_msg: Image, cam_msg: CameraInfo, rgb_msg: Image,
                         ws: dict, n_points: int) -> np.ndarray:
    """Reconstruct XYZRGB point cloud — same pipeline as training."""
    h, w = depth_msg.height, depth_msg.width
    enc_d = depth_msg.encoding
    raw_d = bytes(depth_msg.data)
    if enc_d == "32FC1":
        depth = np.frombuffer(raw_d, dtype=np.float32).reshape(h, w)
    elif enc_d == "16UC1":
        depth = np.frombuffer(raw_d, dtype=np.uint16).reshape(h, w).astype(np.float32) / 1000.0
    else:
        raise ValueError(f"Unsupported depth encoding: {enc_d!r}")

    fx, fy = cam_msg.k[0], cam_msg.k[4]
    cx, cy = cam_msg.k[2], cam_msg.k[5]
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = (np.isfinite(pts[:, 2])
             & (pts[:, 0] >= ws["x_min"]) & (pts[:, 0] <= ws["x_max"])
             & (pts[:, 1] >= ws["y_min"]) & (pts[:, 1] <= ws["y_max"])
             & (pts[:, 2] >= ws["z_min"]) & (pts[:, 2] <= ws["z_max"]))
    pts = pts[valid]

    raw_rgb = np.frombuffer(bytes(rgb_msg.data), dtype=np.uint8)
    enc = rgb_msg.encoding
    if enc in ("bgra8", "rgba8"):
        img = raw_rgb.reshape(h, w, 4)
        r_ch = img[:, :, 2 if enc == "bgra8" else 0]
        g_ch = img[:, :, 1]
        b_ch = img[:, :, 0 if enc == "bgra8" else 2]
    else:
        img = raw_rgb.reshape(h, w, 3)
        r_ch, g_ch, b_ch = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    r = r_ch.reshape(-1)[valid].astype(np.float32) / 255.0
    g = g_ch.reshape(-1)[valid].astype(np.float32) / 255.0
    b = b_ch.reshape(-1)[valid].astype(np.float32) / 255.0
    xyzrgb = np.column_stack([pts, r, g, b])
    return fps_or_pad(xyzrgb, n_points)


# ---------------------------------------------------------------------------
# ROS2 camera node — spun in background thread, RTDE handles robot state
# ---------------------------------------------------------------------------

class CameraNode(Node):
    def __init__(self, topics: dict):
        super().__init__("dp3_camera_node")
        be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.depth_msg    = None
        self.cam_info_msg = None
        self.rgb_msg      = None
        self.create_subscription(Image,      topics["depth"],    self._cb_d,  be)
        self.create_subscription(CameraInfo, topics["cam_info"], self._cb_ci, be)
        self.create_subscription(Image,      topics["rgb"],      self._cb_rgb, be)

    def _cb_d(self,  m): self.depth_msg    = m
    def _cb_ci(self, m): self.cam_info_msg = m
    def _cb_rgb(self,m): self.rgb_msg      = m

    def ready(self) -> bool:
        return (self.depth_msg is not None
                and self.cam_info_msg is not None
                and self.rgb_msg is not None)


# ---------------------------------------------------------------------------
# Robotiq gripper — direct TCP socket to port 63352 on the UR controller.
# Protocol: "SET POS <0-255>\n" / "SET GTO 1\n" (Robotiq ASCII protocol).
# This works without URCap; URScript rq_set_pos() requires URCap to be active.
# ---------------------------------------------------------------------------

class _RobotiqSocket:
    PORT = 63352

    def __init__(self, hostname: str):
        self._lock = threading.Lock()
        self._sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        self._sock.settimeout(3.0)
        self._sock.connect((hostname, self.PORT))

    def _cmd(self, cmd: str) -> str:
        self._sock.sendall(cmd.encode())
        return self._sock.recv(1024).decode().strip()

    def move(self, position_01: float, speed: int = 255, force: int = 10):
        pos = int(np.clip(position_01, 0.0, 1.0) * 255)
        with self._lock:
            self._cmd(f"SET POS {pos}\n")
            self._cmd(f"SET SPE {speed}\n")
            self._cmd(f"SET FOR {force}\n")
            self._cmd(f"SET GTO 1\n")

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# BimanualRTDE — direct RTDE connection for both arms + grippers
# (identical to 3D-Diffusion-Policy/policy_executor.py)
# ---------------------------------------------------------------------------

class BimanualRTDE:
    LOOKAHEAD = 0.1
    GAIN      = 300
    ACC       = 1.0
    VEL       = 1.0

    def __init__(self, robot1_ip: str, robot2_ip: str, rtde_hz: int):
        self.dt = 1.0 / rtde_hz
        print(f"Connecting to robot1 @ {robot1_ip} ...")
        self.rc1 = RTDEControl(robot1_ip)
        self.rr1 = RTDEReceive(robot1_ip)
        print(f"Connecting to robot2 @ {robot2_ip} ...")
        self.rc2 = RTDEControl(robot2_ip)
        self.rr2 = RTDEReceive(robot2_ip)
        self._g1_pos = 1.0   # gripper1 starts closed (matches training init)
        self._g2_pos = 0.0   # gripper2 starts open
        print("RTDE connected.")

        # Gripper socket connections (Robotiq TCP protocol, port 63352)
        self._grip1 = None
        self._grip2 = None
        for which, ip, attr in [(1, robot1_ip, "_grip1"), (2, robot2_ip, "_grip2")]:
            try:
                setattr(self, attr, _RobotiqSocket(ip))
                print(f"Gripper{which} socket connected ({ip}:63352).")
            except Exception as e:
                print(f"[WARN] Gripper{which} socket failed ({ip}:63352): {e} — gripper disabled")

    def get_state(self) -> np.ndarray:
        """14-D [r1(6), g1(1), r2(6), g2(1)] — arms from RTDE, grippers local."""
        r1 = np.array(self.rr1.getActualQ(), dtype=np.float32)
        r2 = np.array(self.rr2.getActualQ(), dtype=np.float32)
        g1 = np.array([self._g1_pos], dtype=np.float32)
        g2 = np.array([self._g2_pos], dtype=np.float32)
        return np.concatenate([r1, g1, r2, g2])

    def get_joints(self):
        return (np.array(self.rr1.getActualQ()),
                np.array(self.rr2.getActualQ()))

    def servoJ_step(self, q1: np.ndarray, q2: np.ndarray):
        self.rc1.servoJ(q1.tolist(), self.VEL, self.ACC, self.dt, self.LOOKAHEAD, self.GAIN)
        self.rc2.servoJ(q2.tolist(), self.VEL, self.ACC, self.dt, self.LOOKAHEAD, self.GAIN)

    def set_gripper(self, which: int, position_01: float, min_change: float = 0.05):
        cur = self._g1_pos if which == 1 else self._g2_pos
        if abs(position_01 - cur) < min_change:
            return
        pos_byte = int(np.clip(position_01, 0.0, 1.0) * 255)
        print(f"[GRIPPER] gripper{which}: {cur:.2f}→{position_01:.2f}  (byte={pos_byte})")
        grip = self._grip1 if which == 1 else self._grip2
        if grip is not None:
            def _send():
                try:
                    grip.move(position_01)
                except Exception as e:
                    print(f"[WARN] gripper{which} socket command failed: {e}")
            threading.Thread(target=_send, daemon=True).start()
        else:
            print(f"[WARN] gripper{which} not connected — command skipped")
        if which == 1:
            self._g1_pos = float(np.clip(position_01, 0.0, 1.0))
        else:
            self._g2_pos = float(np.clip(position_01, 0.0, 1.0))

    def stop(self):
        self.rc1.servoStop()
        self.rc2.servoStop()
        self.rc1.stopScript()
        self.rc2.stopScript()

    def disconnect(self):
        self.stop()
        for g in (self._grip1, self._grip2):
            if g is not None:
                g.close()
        self.rc1.disconnect(); self.rc2.disconnect()
        self.rr1.disconnect(); self.rr2.disconnect()


def interpolate_waypoints(q_from: np.ndarray, q_to: np.ndarray, n: int):
    return [q_from + (q_to - q_from) * (i + 1) / n for i in range(n)]


# ---------------------------------------------------------------------------
# Policy loader
# ---------------------------------------------------------------------------

def _read_n_points(cfg) -> int:
    from omegaconf import OmegaConf
    # ManiFlow: shape_meta lives under robotwin_task
    for path in ("robotwin_task.shape_meta.obs.point_cloud.shape",
                 "task.shape_meta.obs.point_cloud.shape",
                 "shape_meta.obs.point_cloud.shape"):
        try:
            s = OmegaConf.select(cfg, path)
            if s is not None:
                return int(s[0])
        except Exception:
            pass
    # DP3 fallback
    for path in ("n_points",):
        try:
            v = OmegaConf.select(cfg, path)
            if v is not None:
                return int(v)
        except Exception:
            pass
    return 1024


def _read_action_dim(cfg) -> int:
    from omegaconf import OmegaConf
    for path in ("robotwin_task.shape_meta.action.shape",
                 "task.shape_meta.action.shape",
                 "shape_meta.action.shape"):
        try:
            s = OmegaConf.select(cfg, path)
            if s is not None:
                return int(s[0])
        except Exception:
            pass
    return 14


def _is_maniflow_checkpoint(payload: dict) -> bool:
    """Detect whether a checkpoint was saved by ManiFlow vs DP3."""
    cfg = payload.get("cfg", {})
    try:
        from omegaconf import OmegaConf
        return OmegaConf.select(cfg, "robotwin_task") is not None
    except Exception:
        return False


def load_policy(checkpoint_path: str, inference_steps: int):
    # ManiFlow checkpoints are saved with dill; DP3 with plain pickle.
    # Try dill first (works for both), fall back to torch.load default.
    try:
        import dill as _pkl
    except ImportError:
        import pickle as _pkl

    payload = torch.load(checkpoint_path, map_location="cpu", pickle_module=_pkl)
    cfg = payload["cfg"]
    cfg.policy.num_inference_steps = inference_steps

    if _is_maniflow_checkpoint(payload):
        if not _MANIFLOW_AVAILABLE:
            raise RuntimeError(
                "ManiFlow checkpoint detected but ManiFlow package not importable. "
                "Check /home/rosi/maniflow/ManiFlow_Policy/ManiFlow is on sys.path."
            )
        print("[load_policy] ManiFlow checkpoint detected.")
        ws = TrainManiFlowRoboTwinWorkspace(cfg)
        ws.load_payload(payload)
    else:
        if not _DP3_AVAILABLE:
            raise RuntimeError("DP3 checkpoint detected but DP3 package not importable.")
        print("[load_policy] DP3 checkpoint detected.")
        ws = TrainDP3Workspace(cfg)
        ws.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = ws.ema_model if cfg.training.use_ema else ws.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device).eval()
    return policy, device, _read_n_points(cfg), _read_action_dim(cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         default=str(Path(__file__).parent / "real_config.yaml"))
    parser.add_argument("--checkpoint",     default=None)
    parser.add_argument("--robot1_ip",      default=None)
    parser.add_argument("--robot2_ip",      default=None)
    parser.add_argument("--hz",             type=float, default=None, help="Obs/control Hz")
    parser.add_argument("--rtde_hz",        type=int,   default=None, help="RTDE servo Hz")
    parser.add_argument("--n_action_steps", type=int,   default=None)
    parser.add_argument("--infer_steps",    type=int,   default=None)
    parser.add_argument("--max_step",       type=float, default=0.05, help="Max joint delta per step (rad)")
    parser.add_argument("--speed_scale",    type=float, default=1.0,  help="Scale all joint deltas (0.0-1.0); 0.5 = half speed")
    parser.add_argument("--no_kickstart",   action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    robot1_ip     = args.robot1_ip  or cfg["robots"]["robot1"]["ip"]
    robot2_ip     = args.robot2_ip  or cfg["robots"]["robot2"]["ip"]
    hz            = args.hz         or cfg["policy"]["obs_hz"]
    rtde_hz       = args.rtde_hz    or cfg["policy"]["rtde_hz"]
    n_action_steps= args.n_action_steps or cfg["policy"]["n_action_steps"]
    infer_steps   = args.infer_steps    or cfg["policy"]["inference_steps"]
    ckpt          = args.checkpoint     or cfg["policy"]["checkpoint_path"]
    ws            = cfg["workspace"]
    max_step      = args.max_step
    speed_scale   = float(np.clip(args.speed_scale, 0.05, 2.0))
    interp_steps  = max(1, rtde_hz // hz)

    print(f"Loading policy from: {ckpt}")
    policy, device, n_points, action_dim = load_policy(ckpt, infer_steps)
    print(f"Policy loaded on {device}  |  n_points={n_points}  action_dim={action_dim}")
    print(f"hz={hz}  rtde_hz={rtde_hz}  n_action_steps={n_action_steps}"
          f"  infer_steps={infer_steps}  interp_steps={interp_steps}"
          f"  speed_scale={speed_scale:.2f}")

    # ── ROS2 camera ──
    rclpy.init()
    camera = CameraNode(cfg["topics"])
    spin_thread = threading.Thread(target=rclpy.spin, args=(camera,), daemon=True)
    spin_thread.start()
    print("Waiting for ZED depth + RGB + camera_info ...")
    while not camera.ready():
        time.sleep(0.05)
    print("Camera ready.")

    # ── RTDE robot connections ──
    robots = BimanualRTDE(robot1_ip, robot2_ip, rtde_hz)
    r1_start = np.array(robots.rr1.getActualQ(), dtype=np.float64)
    if action_dim == 8:
        print(f"r1 locked at: {np.round(r1_start, 4).tolist()}")

    # ── GPU warm-up ──
    dummy_pc  = torch.zeros(1, 2, n_points, 6).to(device)
    dummy_pos = torch.zeros(1, 2, 14).to(device)
    for _ in range(3):
        t_w = time.time()
        with torch.no_grad():
            policy.predict_action({"point_cloud": dummy_pc, "agent_pos": dummy_pos})
    infer_ms = (time.time() - t_w) * 1000
    print(f"Inference: {infer_ms:.0f} ms  ({1000/infer_ms:.1f} Hz)\n")

    # ── obs_worker thread: samples at exactly hz ──
    obs_lock = threading.Lock()
    obs_ring = collections.deque(maxlen=2)
    stop_obs = threading.Event()

    def obs_worker():
        interval = 1.0 / hz
        while not stop_obs.is_set():
            t0 = time.time()
            try:
                pc    = depth_to_pointcloud(camera.depth_msg, camera.cam_info_msg,
                                             camera.rgb_msg, ws, n_points)
                state = robots.get_state()
                with obs_lock:
                    obs_ring.append((pc, state))
            except Exception as e:
                print(f"[obs_worker] {e}", flush=True)
            elapsed = time.time() - t0
            wait = interval - elapsed
            if wait > 0:
                time.sleep(wait)

    obs_thread = threading.Thread(target=obs_worker, daemon=True)
    obs_thread.start()
    time.sleep(0.15)   # fill obs ring (2 frames @ 50 ms spacing)

    # ── Optional kickstart ──
    if not args.no_kickstart:
        print("Kickstart: nudging robot2 wrist +0.08 rad ...")
        q1_cur, q2_cur = robots.get_joints()
        q2_nudge = q2_cur.copy(); q2_nudge[5] += 0.08
        for q1_wp, q2_wp in zip(interpolate_waypoints(q1_cur, q1_cur, interp_steps),
                                 interpolate_waypoints(q2_cur, q2_nudge, interp_steps)):
            t_s = time.time()
            robots.servoJ_step(q1_wp, q2_wp)
            wait = (1.0 / rtde_hz) - (time.time() - t_s)
            if wait > 0:
                time.sleep(wait)
        time.sleep(0.06)
        print("Kickstart done.\n")
    else:
        print("Kickstart disabled.\n")

    # ── inference_worker thread ──
    action_queue = _queue.Queue(maxsize=2)
    stop_infer   = threading.Event()

    LATCH_COUNT    = 5
    g1_latch_count = 0
    g2_latch_count = 0
    g1_latched     = False
    g2_latched     = False
    g1_abs_carry   = float(robots._g1_pos)
    g2_abs_carry   = float(robots._g2_pos)

    def inference_worker():
        nonlocal g1_abs_carry, g2_abs_carry
        nonlocal g1_latch_count, g2_latch_count, g1_latched, g2_latched
        while not stop_infer.is_set():
            with obs_lock:
                if len(obs_ring) < 2:
                    time.sleep(0.01)
                    continue
                obs_a, obs_b = list(obs_ring)

            state_now = obs_b[1]
            pc_t    = torch.from_numpy(
                          np.stack([obs_a[0], obs_b[0]], axis=0)
                      ).float().unsqueeze(0).to(device)
            state_t = torch.from_numpy(
                          np.stack([obs_a[1], obs_b[1]], axis=0)
                      ).float().unsqueeze(0).to(device)

            t_inf = time.time()
            with torch.no_grad():
                result = policy.predict_action({"point_cloud": pc_t, "agent_pos": state_t})
            infer_ms = (time.time() - t_inf) * 1000
            actions  = result["action"].squeeze(0).cpu().numpy()   # (n_action_steps, 14)

            raw_d = actions[0].copy()
            if action_dim == 8:
                print(
                    f"  infer={infer_ms:4.0f}ms"
                    f"  r2_Δmax={np.abs(raw_d[0:6]).max():.4f}"
                    f"  g1={raw_d[6]:.3f}  g2={raw_d[7]:.3f}",
                    flush=True,
                )
            else:
                print(
                    f"  infer={infer_ms:4.0f}ms"
                    f"  r1_Δmax={np.abs(raw_d[0:6]).max():.4f}"
                    f"  r2_Δmax={np.abs(raw_d[7:13]).max():.4f}"
                    f"  g1_Δ={raw_d[6]:.3f}  g2_Δ={raw_d[13]:.3f}",
                    flush=True,
                )

            # Accumulate deltas → absolute targets; apply gripper handling
            g1_abs = g1_abs_carry
            g2_abs = g2_abs_carry

            if action_dim == 8:
                # 8D: [r2j0-5(0:6), g1(6), g2(7)] — r1 locked, grippers as absolute states
                q2_base = state_now[7:13].copy().astype(np.float64)
                for i in range(len(actions)):
                    for j in range(6):
                        d = np.clip(float(actions[i, j]), -max_step, max_step)
                        actions[i, j] = q2_base[j] + d
                    q2_base = actions[i, 0:6].copy()
                    actions[i, 6] = 1.0 if float(actions[i, 6]) > 0.5 else 0.0
                    actions[i, 7] = 1.0 if float(actions[i, 7]) > 0.5 else 0.0
                g1_abs_carry = float(actions[-1, 6])
                g2_abs_carry = float(actions[-1, 7])
            else:
                # 14D: store raw clipped deltas — accumulation to absolute targets
                # happens at execution time using the actual robot position then,
                # not the stale state_now captured 800ms before the chunk runs.
                for i in range(len(actions)):
                    for j in range(6):
                        actions[i, j]   = np.clip(float(actions[i, j]),   -max_step, max_step) * speed_scale
                    for j in range(6):
                        actions[i, 7+j] = np.clip(float(actions[i, 7+j]), -max_step, max_step) * speed_scale

                    g1_d   = np.clip(float(actions[i, 6]),  -max_step, max_step)
                    g1_abs = float(np.clip(g1_abs + g1_d, 0.0, 1.0))
                    if g1_abs >= 0.99:
                        g1_latch_count += 1
                    if g1_latch_count >= LATCH_COUNT:
                        g1_latched = True
                    if g1_latched:
                        g1_abs = 1.0
                    actions[i, 6] = g1_abs

                    g2_d   = np.clip(float(actions[i, 13]), -max_step, max_step)
                    g2_abs = float(np.clip(g2_abs + g2_d, 0.0, 1.0))
                    if g2_abs >= 0.5:
                        g2_latch_count += 1
                    if g2_latch_count >= LATCH_COUNT:
                        g2_latched = True
                    if g2_latched:
                        g2_abs = 1.0
                    actions[i, 13] = g2_abs

                g1_abs_carry = g1_abs
                g2_abs_carry = g2_abs

            try:
                action_queue.put(actions, timeout=0.1)
            except _queue.Full:
                pass   # execution behind — drop stale chunk

    infer_thread = threading.Thread(target=inference_worker, daemon=True)
    infer_thread.start()

    print("Waiting for first inference result ...")
    first_actions = action_queue.get()
    print("First action ready — starting execution.\n")

    # ── Start-position check against training distribution ──
    if action_dim != 8:
        _TRAIN_R1_MEAN_DEG = np.array([-54.88, -109.99,  144.20, -196.41, -115.81,   14.32])
        _TRAIN_R2_MEAN_DEG = np.array([-123.12,  -82.98, -140.53,   33.87, -235.15,  -13.22])
        _TRAIN_R1_STD_DEG  = np.array([0.04, 0.06, 0.04, 1.03, 0.70, 0.31])
        _TRAIN_R2_STD_DEG  = np.array([0.94, 1.31, 0.57, 0.57, 0.62, 0.12])
        _q1_now, _q2_now = robots.get_joints()
        _q1_deg = np.rad2deg(np.array(_q1_now))
        _q2_deg = np.rad2deg(np.array(_q2_now))
        _r1_err = _q1_deg - _TRAIN_R1_MEAN_DEG
        _r2_err = _q2_deg - _TRAIN_R2_MEAN_DEG
        _r1_sigma = np.abs(_r1_err) / _TRAIN_R1_STD_DEG
        _r2_sigma = np.abs(_r2_err) / _TRAIN_R2_STD_DEG
        print("─" * 70)
        print("START POSITION CHECK  (values in degrees, Δ = actual − training mean)")
        print(f"{'Joint':<8} {'R1 actual':>10} {'R1 Δ':>8} {'σ':>5}   {'R2 actual':>10} {'R2 Δ':>8} {'σ':>5}")
        print("─" * 70)
        for j in range(6):
            r1_flag = " !" if _r1_sigma[j] > 3 else ("  " if _r1_sigma[j] <= 1 else " ~")
            r2_flag = " !" if _r2_sigma[j] > 3 else ("  " if _r2_sigma[j] <= 1 else " ~")
            print(f"  j{j}    {_q1_deg[j]:>10.2f} {_r1_err[j]:>+8.2f} {_r1_sigma[j]:>4.1f}σ{r1_flag}"
                  f"  {_q2_deg[j]:>10.2f} {_r2_err[j]:>+8.2f} {_r2_sigma[j]:>4.1f}σ{r2_flag}")
        print("─" * 70)
        r1_ok = np.all(_r1_sigma <= 3)
        r2_ok = np.all(_r2_sigma <= 3)
        print(f"R1: {'OK' if r1_ok else 'OUT OF DISTRIBUTION — adjust before running'}   "
              f"R2: {'OK' if r2_ok else 'OUT OF DISTRIBUTION — adjust before running'}")
        print("─" * 70)
        print()

    # ── Main control loop ──
    try:
        print(f"=== DP3 Execution  ({n_action_steps} steps/chunk @ {hz} Hz, Ctrl+C to stop) ===\n")
        actions = first_actions
        while True:
            # 14D: re-base delta accumulation to actual robot position at chunk start.
            # inference_worker stores raw clipped deltas; we accumulate here so that
            # each chunk starts from where the robot actually is, not from the stale
            # state_now captured ~800 ms before this chunk runs.
            if action_dim != 8:
                _q1c, _q2c = robots.get_joints()
                q1_exec_base = np.array(_q1c, dtype=np.float64)
                q2_exec_base = np.array(_q2c, dtype=np.float64)

            for step_idx in range(n_action_steps):
                act = actions[step_idx]
                if action_dim == 8:
                    q1_target = r1_start
                    q2_target = act[0:6]
                    g1_target = float(act[6])
                    g2_target = float(act[7])
                else:
                    q1_target = q1_exec_base + act[0:6]
                    q2_target = q2_exec_base + act[7:13]
                    q1_exec_base = q1_target.copy()
                    q2_exec_base = q2_target.copy()
                    g1_target = float(act[6])
                    g2_target = float(act[13])

                q1_cur, q2_cur = robots.get_joints()
                for q1_wp, q2_wp in zip(
                        interpolate_waypoints(q1_cur, q1_target, interp_steps),
                        interpolate_waypoints(q2_cur, q2_target, interp_steps)):
                    t_s = time.time()
                    robots.servoJ_step(q1_wp, q2_wp)
                    wait = (1.0 / rtde_hz) - (time.time() - t_s)
                    if wait > 0:
                        time.sleep(wait)

                robots.set_gripper(1, g1_target)
                robots.set_gripper(2, g2_target)

            try:
                actions = action_queue.get(timeout=0.5)
            except _queue.Empty:
                print("[WARN] inference too slow — holding last action")

    except KeyboardInterrupt:
        print("\nStopping ...")
        if action_dim != 8:
            try:
                _q1f, _q2f = robots.get_joints()
                _q2f_deg = np.round(np.rad2deg(np.array(_q2f)), 2)
                _q2s_deg = np.array([-123.12, -82.98, -140.53, 33.87, -235.15, -13.22])
                _q2_off  = np.round(_q2f_deg - _q2s_deg, 2)
                print(f"\nR2 terminal joints (deg): {_q2f_deg.tolist()}")
                print(f"R2 offset from training start: {_q2_off.tolist()}")
                print("→ Place cable at the R2 terminal position if gripper missed.")
            except Exception:
                pass
    finally:
        stop_infer.set()
        stop_obs.set()
        infer_thread.join(timeout=1.0)
        obs_thread.join(timeout=1.0)
        robots.disconnect()
        camera.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
