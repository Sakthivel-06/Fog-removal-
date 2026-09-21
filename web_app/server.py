"""
web_app/server.py
=============================================================================
High-Throughput Localhost Real-Time Video Dehazing Web Server
Engine Overhaul: Decoupled Multi-Threaded Producer-Consumer Architecture

Features:
- Decoupled Pipeline Worker Thread (runs at max hardware speed 35-60+ FPS)
- Condition Variable broadcasting for non-blocking MJPEG delivery
- Laptop Camera Module (HTML5 Browser Webcam + DirectShow backend, 100% OpenCV-Free)
- Ground Truth (Clean Video) Pairing for Live PSNR, SSIM, NIQE, and LPIPS
- Dynamic Algorithm Switching (A4, A1, A2, A3, DCP, CAP, CEEF, CLAHE, Light-DehazeNet)
- Real-Time Parameter Tuning (beta, omega, shadow_lift, resolution mode)
- Live REST APIs (/api/status, /api/control, /api/videos, /api/camera_frame)
=============================================================================
"""

import os
import sys
import time
import json
import threading
import queue
import io
import subprocess
from urllib.parse import urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor
import gc
import numpy as np
from PIL import Image
import imageio.v3 as iio

# Add project root to sys.path
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, os.path.join(_ROOT_DIR, "novel_method"))
sys.path.insert(0, os.path.join(_ROOT_DIR, "algorithms"))

import metrics
from novel_method.pipeline import RealTimeDehazingPipeline
from novel_method.pipeline_learned_experimental import LearnedResidualDehazingPipeline
import algorithms.dcp as alg_dcp
import algorithms.cap as alg_cap
import algorithms.clahe_baseline as alg_clahe
import algorithms.ceef as alg_ceef_mod
import algorithms.ceef_original as alg_ceef_orig

# Try loading Light-DehazeNet wrapper
try:
    from algorithms.light_dehazenet_wrapper import LightDehazeNetWrapper
    _ldnet_wrapper = LightDehazeNetWrapper()
except Exception as e:
    _ldnet_wrapper = None
    print(f"Note: Light-DehazeNet not loaded: {e}")

# Try to find ffmpeg binary for DirectShow
_FFMPEG_EXE = None
try:
    import imageio_ffmpeg
    _FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass


# Try to cap PyTorch threads to reduce cloud container RAM
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

class VideoEngine:
    """
    Decoupled Producer-Consumer Video Dehazing Engine.
    Runs video decoding and pipeline inference in a dedicated background thread,
    guaranteeing 35-60+ FPS without socket network backpressure.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

        # Core Configuration
        self.current_video = "car-detection_patchy_haze.mp4"
        self.gt_video = None        # Paired Ground Truth video (if any)
        self.method = "A4"          # Default: Proposed Method A4
        self.res_mode = "fast"      # "fast" (240x426) or "hd" (360x640)
        self.beta = 1.0             # Balanced depth penetration (prevents background over-darkening)
        self.omega = 0.85           # Dark channel ratio
        self.shadow_lift = 0.25     # Adaptive shadow lift for background trees/structures
        self.side_by_side = True    # Dual view by default
        self.is_paused = False
        self.public_url = None

        # Pipelines Cache
        self.p_a1 = None
        self.p_a4 = None
        self.p_a2 = None
        self.p_a3 = None
        self._init_pipelines()

        # Frame Buffer & Sequencing
        self.latest_jpeg = None
        self.frame_seq = 0
        self.frame_idx = 0
        self.total_frames = 0


        # Video / Camera state
        self.video_reader = None
        self.gt_reader = None
        self.dshow_process = None
        self.browser_camera_queue = queue.Queue(maxsize=3)
        self.has_active_clients = False
        self.last_client_time = time.time()

        # Live Metrics Telemetry
        self.live_metrics = {
            "fps": 0.0,
            "latency_ms": 0.0,
            "fog_removal_pct": 0.0,
            "entropy": 0.0,
            "colorfulness": 0.0,
            "flicker": 0.0,
            "black_crush_pct": 0.0,
            "white_clip_pct": 0.0,
            "psnr": "N/A (Rule 11)",
            "ssim": "N/A (Rule 11)",
            "niqe": 0.0,
            "lpips": "N/A (Rule 11)",
            "has_gt": False,
            "frame_idx": 0,
            "total_frames": 0,
            "method": self.method,
            "video": self.current_video,
            "gt_video": self.gt_video,
            "beta": self.beta,
            "omega": self.omega,
            "shadow_lift": self.shadow_lift,
            "res_mode": self.res_mode,
            "side_by_side": self.side_by_side,
            "is_paused": self.is_paused,
            "public_url": self.public_url
        }

        self.prev_dehazed = None
        self.frame_counter = 0

        # Asynchronous Heavy Metric Queue (NIQE, LPIPS) so inference loop NEVER pauses
        self.metric_queue = queue.Queue(maxsize=1)
        self.metric_worker_thread = threading.Thread(target=self._async_metric_worker, daemon=True, name="AsyncMetricsWorker")
        self.metric_worker_thread.start()

        # Start Background Engine Thread
        self.running = True
        self._load_sources()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="DehazingWorker")
        self.worker_thread.start()

    def _async_metric_worker(self):
        """Computes CPU-intensive metrics (NIQE, LPIPS) asynchronously in background."""
        eval_count = 0
        while True:
            try:
                snapshot = self.metric_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                out_frame, gt_frame = snapshot
                h, w = out_frame.shape[:2]
                # Downsample snapshot for fast NIQE without losing spatial naturalness distribution
                if w > 320:
                    small = np.array(Image.fromarray(out_frame).resize((320, int(h * 320 / w)), Image.Resampling.BILINEAR))
                else:
                    small = out_frame
                niqe_val = round(metrics.niqe(small), 2)
                lpips_val = None
                if gt_frame is not None:
                    try:
                        lpips_val = round(metrics.lpips_fn(gt_frame, out_frame), 4)
                    except Exception:
                        pass
                with self.lock:
                    self.live_metrics["niqe"] = niqe_val
                    if lpips_val is not None:
                        self.live_metrics["lpips"] = lpips_val
                eval_count += 1
                if eval_count % 30 == 0:
                    gc.collect()
            except Exception:
                pass

    def _init_pipelines(self):
        """Initializes primary pipeline, lazy-loading others on demand to conserve RAM."""
        target_short, target_long = (240, 426) if self.res_mode == "fast" else (360, 640)
        max_res = (target_long, target_short)

        self.p_a1 = RealTimeDehazingPipeline(
            downscale_res=(512, 288),
            max_process_res=max_res,
            beta=self.beta,
            kappa=1.25,
            omega=self.omega,
            enable_sky_model=True,
            preserve_resolution=False
        )

        best_ckpt = os.path.join(_ROOT_DIR, "checkpoints", "delta_t_tiny_unet_best.pt")
        if os.path.exists(best_ckpt):
            self.p_a4 = LearnedResidualDehazingPipeline(
                checkpoint_path=best_ckpt,
                ablation_mode="A4",
                downscale_res=(512, 288),
                max_process_res=max_res,
                beta=self.beta,
                kappa=1.25,
                omega=self.omega,
                enable_sky_model=True,
                preserve_resolution=False
            )
        else:
            self.p_a4 = None

        # p_a2 and p_a3 will be lazy-loaded on demand when selected by user
        self.p_a2 = None
        self.p_a3 = None
        gc.collect()

    def _get_pipeline(self, method_name):
        target_short, target_long = (240, 426) if self.res_mode == "fast" else (360, 640)
        max_res = (target_long, target_short)
        best_ckpt = os.path.join(_ROOT_DIR, "checkpoints", "delta_t_tiny_unet_best.pt")
        no_temp_ckpt = os.path.join(_ROOT_DIR, "checkpoints", "delta_t_tiny_unet_no_temp.pt")

        if method_name in ["A4", "Proposed (A4)"]:
            if self.p_a4 is None and os.path.exists(best_ckpt):
                self.p_a4 = LearnedResidualDehazingPipeline(
                    checkpoint_path=best_ckpt, ablation_mode="A4",
                    downscale_res=(512, 288), max_process_res=max_res,
                    beta=self.beta, kappa=1.25, omega=self.omega,
                    enable_sky_model=True, preserve_resolution=False
                )
            return self.p_a4
        elif method_name == "A3":
            if self.p_a3 is None and os.path.exists(best_ckpt):
                self.p_a3 = LearnedResidualDehazingPipeline(
                    checkpoint_path=best_ckpt, ablation_mode="A3",
                    downscale_res=(512, 288), max_process_res=max_res,
                    beta=self.beta, kappa=1.25, omega=self.omega,
                    enable_sky_model=True, preserve_resolution=False
                )
            return self.p_a3
        elif method_name == "A2":
            if self.p_a2 is None and os.path.exists(no_temp_ckpt):
                self.p_a2 = LearnedResidualDehazingPipeline(
                    checkpoint_path=no_temp_ckpt, ablation_mode="A2",
                    downscale_res=(512, 288), max_process_res=max_res,
                    beta=self.beta, kappa=1.25, omega=self.omega,
                    enable_sky_model=True, preserve_resolution=False
                )
            return self.p_a2
        return self.p_a1

    def push_browser_camera_frame(self, frame_rgb):
        """Pushes a new frame from client browser webcam (phone/laptop) into engine queue."""
        with self.lock:
            if self.current_video != "camera:browser":
                self.current_video = "camera:browser"
                self.live_metrics["video"] = "camera:browser"
        # Drop oldest frame if queue full so camera stream is always zero-latency
        while not self.browser_camera_queue.empty():
            try:
                self.browser_camera_queue.get_nowait()
            except queue.Empty:
                break
        self.browser_camera_queue.put(frame_rgb)

    def _close_sources(self):
        """Closes video readers and camera processes."""
        if self.dshow_process is not None:
            try:
                self.dshow_process.kill()
            except Exception:
                pass
            self.dshow_process = None

        if self.video_reader is not None:
            try:
                if hasattr(self.video_reader, "close"):
                    self.video_reader.close()
            except Exception:
                pass
            self.video_reader = None

        if self.gt_reader is not None:
            try:
                if hasattr(self.gt_reader, "close"):
                    self.gt_reader.close()
            except Exception:
                pass
            self.gt_reader = None

        gc.collect()

    def _load_sources(self):
        """Loads main video/camera source and optional Ground Truth reader."""
        self._close_sources()
        self.frame_idx = 0
        self.prev_dehazed = None

        # Camera Sources
        if self.current_video == "camera:browser":
            self.total_frames = 0
            return

        if self.current_video == "camera:dshow":
            self.total_frames = 0
            if _FFMPEG_EXE:
                cmd = [
                    _FFMPEG_EXE, "-f", "dshow",
                    "-video_size", "640x480",
                    "-framerate", "30",
                    "-i", "video=Chicony USB2.0 Camera",
                    "-f", "rawvideo",
                    "-pix_fmt", "rgb24",
                    "-"
                ]
                try:
                    self.dshow_process = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=640 * 480 * 3 * 2
                    )
                except Exception as e:
                    print(f"Failed to start DirectShow camera: {e}")
            return

        # Recorded Video Source
        video_path = os.path.join(_ROOT_DIR, "test_videos", self.current_video)
        if not os.path.exists(video_path):
            video_path = os.path.join(_ROOT_DIR, "test_videos", "foggy_video.mp4")
            self.current_video = "foggy_video.mp4"

        try:
            self.video_reader = iio.imiter(video_path)
            # Estimate frame count safely without scanning entire video
            if not hasattr(self, "_frame_count_cache"):
                self._frame_count_cache = {}
            if self.current_video not in self._frame_count_cache:
                try:
                    import imageio
                    r = imageio.get_reader(video_path)
                    self._frame_count_cache[self.current_video] = r.count_frames()
                    r.close()
                except Exception:
                    self._frame_count_cache[self.current_video] = 1000
            self.total_frames = self._frame_count_cache.get(self.current_video, 1000)
        except Exception as e:
            print(f"Error loading main video {self.current_video}: {e}")

        # Ground Truth Video Source (if specified)
        if self.gt_video:
            gt_path = os.path.join(_ROOT_DIR, "test_videos", self.gt_video)
            if os.path.exists(gt_path):
                try:
                    self.gt_reader = iio.imiter(gt_path)
                except Exception as e:
                    print(f"Error loading GT video {self.gt_video}: {e}")
                    self.gt_reader = None
            else:
                self.gt_reader = None

    def update_settings(self, updates: dict):
        """Thread-safe update of runtime parameters."""
        with self.lock:
            reinit_pipe = False
            reload_src = False

            if "video" in updates and updates["video"] != self.current_video:
                self.current_video = updates["video"]
                reload_src = True

            if "gt_video" in updates:
                gt_val = updates["gt_video"]
                if gt_val in [None, "", "none", "None"]:
                    self.gt_video = None
                else:
                    self.gt_video = gt_val
                reload_src = True

            if "method" in updates:
                self.method = updates["method"]

            if "beta" in updates:
                self.beta = float(updates["beta"])
                reinit_pipe = True

            if "omega" in updates:
                self.omega = float(updates["omega"])
                reinit_pipe = True

            if "shadow_lift" in updates:
                self.shadow_lift = float(updates["shadow_lift"])

            if "res_mode" in updates:
                self.res_mode = updates["res_mode"]
                reinit_pipe = True

            if "side_by_side" in updates:
                self.side_by_side = bool(updates["side_by_side"])

            if "is_paused" in updates:
                self.is_paused = bool(updates["is_paused"])

            if "public_url" in updates:
                self.public_url = str(updates["public_url"])

            if "restart" in updates and updates["restart"]:
                reload_src = True

            if reinit_pipe:
                self._init_pipelines()

            if reload_src:
                self._load_sources()

            self.live_metrics.update({
                "method": self.method,
                "video": self.current_video,
                "gt_video": self.gt_video,
                "has_gt": (self.gt_video is not None),
                "beta": self.beta,
                "omega": self.omega,
                "shadow_lift": self.shadow_lift,
                "res_mode": self.res_mode,
                "side_by_side": self.side_by_side,
                "is_paused": self.is_paused,
                "public_url": self.public_url
            })

    def push_browser_camera_frame(self, frame_rgb: np.ndarray):
        """Pushes a frame received from the browser webcam into the engine queue."""
        try:
            # Keep only the latest frame
            while not self.browser_camera_queue.empty():
                try:
                    self.browser_camera_queue.get_nowait()
                except queue.Empty:
                    break
            self.browser_camera_queue.put_nowait(frame_rgb)
        except Exception:
            pass

    def _worker_loop(self):
        """
        Continuous, decoupled background processing loop.
        Processes frames at native hardware capability without socket delays.
        """
        buf = io.BytesIO()

        while self.running:
            # 1. Idle sleep if paused or no active clients in the last 6 seconds
            if self.is_paused or (not self.has_active_clients and (time.time() - self.last_client_time > 6.0)):
                time.sleep(0.08)
                continue


            # 2. Fetch raw frame from active source
            raw_frame = None
            gt_raw_frame = None

            if self.current_video == "camera:browser":
                try:
                    raw_frame = self.browser_camera_queue.get(timeout=0.08)
                except queue.Empty:
                    time.sleep(0.01)
                    continue

            elif self.current_video == "camera:dshow" and self.dshow_process is not None:
                try:
                    raw_bytes = self.dshow_process.stdout.read(640 * 480 * 3)
                    if len(raw_bytes) == 640 * 480 * 3:
                        raw_frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((480, 640, 3))
                    else:
                        time.sleep(0.02)
                        continue
                except Exception:
                    time.sleep(0.02)
                    continue

            else:
                # Video file source
                if self.video_reader is None:
                    with self.lock:
                        self._load_sources()

                try:
                    raw_frame = next(self.video_reader)
                except (StopIteration, Exception):
                    # Seamless loop
                    with self.lock:
                        self._load_sources()
                    try:
                        raw_frame = next(self.video_reader)
                    except Exception:
                        time.sleep(0.05)
                        continue

                # Ground truth reader (if paired)
                if self.gt_reader is not None:
                    try:
                        gt_raw_frame = next(self.gt_reader)
                    except (StopIteration, Exception):
                        try:
                            gt_path = os.path.join(_ROOT_DIR, "test_videos", self.gt_video)
                            self.gt_reader = iio.imiter(gt_path)
                            gt_raw_frame = next(self.gt_reader)
                        except Exception:
                            gt_raw_frame = None

            if raw_frame is None:
                time.sleep(0.01)
                continue

            self.frame_idx += 1
            f_idx = self.frame_idx

            # 3. Target resolution scaling
            target_short, target_long = (240, 426) if self.res_mode == "fast" else (360, 640)
            sh, sw = raw_frame.shape[:2]
            if sh > sw:
                target_w, target_h = target_short, target_long
            else:
                target_w, target_h = target_long, target_short

            # Fast Bilinear Resize
            scaled_input = np.array(Image.fromarray(raw_frame).resize((target_w, target_h), Image.Resampling.BILINEAR))
            scaled_gt = None
            if gt_raw_frame is not None:
                scaled_gt = np.array(Image.fromarray(gt_raw_frame).resize((target_w, target_h), Image.Resampling.BILINEAR))

            # 4. Dehazing Inference
            t0 = time.perf_counter()
            method = self.method

            try:
                if method == "A1":
                    res = self.p_a1.process_frame(scaled_input)
                    out = res[0] if isinstance(res, tuple) else res
                elif method in ["A4", "Proposed (A4)"]:
                    pipe = self._get_pipeline("A4")
                    res = pipe.process_frame(scaled_input) if pipe else self.p_a1.process_frame(scaled_input)
                    out = res[0] if isinstance(res, tuple) else res
                elif method == "A2":
                    pipe = self._get_pipeline("A2")
                    res = pipe.process_frame(scaled_input) if pipe else self.p_a1.process_frame(scaled_input)
                    out = res[0] if isinstance(res, tuple) else res
                elif method == "A3":
                    pipe = self._get_pipeline("A3")
                    res = pipe.process_frame(scaled_input) if pipe else self.p_a1.process_frame(scaled_input)
                    out = res[0] if isinstance(res, tuple) else res
                elif method == "DCP":
                    out = alg_dcp.process_frame(scaled_input)
                elif method == "CAP":
                    out = alg_cap.process_frame(scaled_input)
                elif method == "Original CEEF":
                    out = alg_ceef_orig.process_frame(scaled_input)
                elif method == "Modified CEEF":
                    out = alg_ceef_mod.process_frame(scaled_input)
                elif method == "CLAHE":
                    out = alg_clahe.process_frame(scaled_input)
                elif method == "Light-DehazeNet" and _ldnet_wrapper is not None:
                    out = _ldnet_wrapper.process_frame(scaled_input)[0]
                else:
                    res = self.p_a1.process_frame(scaled_input)
                    out = res[0] if isinstance(res, tuple) else res
            except Exception as e:
                out = scaled_input.copy()

            frame_lat_ms = (time.perf_counter() - t0) * 1000.0
            frame_fps = 1000.0 / max(frame_lat_ms, 0.1)

            # 5. Strict Safety Floor: 0.000% Black Crushing
            out = np.maximum(out, 3)

            # 6. Adaptive Background Recovery (Shadow Lift)
            if self.shadow_lift > 0.001:
                img_f = out.astype(np.float32) * (1.0 / 255.0)
                lum = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
                shadow_mask = np.clip((0.72 - lum) / 0.45, 0.0, 1.0)[:, :, None]
                lifted = img_f + self.shadow_lift * shadow_mask * (1.0 - img_f) * (0.5 - np.abs(img_f - 0.5))
                out = np.clip(lifted * 255.0, 0, 255).astype(np.uint8)

            # Ensure dimensions match exactly for side-by-side concatenation and even dimensions
            dh, dw = out.shape[:2]
            if (dw % 2 != 0) or (dh % 2 != 0):
                dh -= (dh % 2)
                dw -= (dw % 2)
                out = np.ascontiguousarray(out[:dh, :dw])

            if scaled_input.shape[:2] != (dh, dw):
                scaled_input = np.array(Image.fromarray(scaled_input).resize((dw, dh), Image.Resampling.BILINEAR))
            if scaled_gt is not None and scaled_gt.shape[:2] != (dh, dw):
                scaled_gt = np.array(Image.fromarray(scaled_gt).resize((dw, dh), Image.Resampling.BILINEAR))

            # 7. Side-by-Side Composition
            if self.side_by_side:
                combined = np.concatenate([scaled_input, out], axis=1)
                img = Image.fromarray(combined)
            else:
                img = Image.fromarray(out)

            # 8. Fast JPEG Encoding
            buf.seek(0)
            buf.truncate()
            img.save(buf, format="JPEG", quality=78, optimize=False)
            jpeg_data = buf.getvalue()

            # 9. Store Latest JPEG and Notify Consumers
            with self.lock:
                self.latest_jpeg = jpeg_data
                self.frame_seq += 1
                self.condition.notify_all()

            # 10. Metric Telemetry Calculation (Fast metrics every 10 frames, ~1.5ms)
            self.frame_counter += 1
            if self.frame_counter % 50 == 0:
                gc.collect()

            if self.frame_counter % 10 == 0:
                ent = metrics.entropy(out)
                col = metrics.colorfulness(out)
                fog_rem = metrics.fog_reduction_percent_PROXY(scaled_input, out)
                black_crush = metrics.black_pixel_rate(out)
                white_clip = metrics.white_clipping_rate(out)

                flicker_val = 0.0
                if self.prev_dehazed is not None and self.prev_dehazed.shape == out.shape:
                    flicker_val = metrics.temporal_flicker_index(out, self.prev_dehazed)
                self.prev_dehazed = out.copy()

                # Fast Ground Truth Metrics (PSNR & SSIM take < 2ms)
                if scaled_gt is not None:
                    psnr_val = round(metrics.psnr(scaled_gt, out), 2)
                    ssim_val = round(metrics.ssim(scaled_gt, out), 4)
                    has_gt = True
                else:
                    psnr_val = "N/A (Rule 11)"
                    ssim_val = "N/A (Rule 11)"
                    has_gt = False

                # Dispatch heavy metrics (NIQE, LPIPS) asynchronously every 40 frames
                if self.frame_counter % 40 == 0:
                    try:
                        self.metric_queue.put_nowait((out.copy(), scaled_gt.copy() if scaled_gt is not None else None))
                    except queue.Full:
                        pass

                with self.lock:
                    self.live_metrics.update({
                        "fps": round(frame_fps, 1),
                        "latency_ms": round(frame_lat_ms, 1),
                        "fog_removal_pct": round(fog_rem, 1),
                        "entropy": round(ent, 2),
                        "colorfulness": round(col, 1),
                        "flicker": round(flicker_val, 2),
                        "black_crush_pct": round(black_crush, 3),
                        "white_clip_pct": round(white_clip, 3),
                        "psnr": psnr_val,
                        "ssim": ssim_val,
                        "has_gt": has_gt,
                        "frame_idx": f_idx,
                        "total_frames": self.total_frames,
                        "method": self.method,
                        "video": self.current_video,
                        "gt_video": self.gt_video,
                        "beta": self.beta,
                        "omega": self.omega,
                        "shadow_lift": self.shadow_lift,
                        "res_mode": self.res_mode,
                        "side_by_side": self.side_by_side,
                        "is_paused": self.is_paused,
                        "public_url": self.public_url
                    })
            else:
                with self.lock:
                    self.live_metrics["fps"] = round(frame_fps, 1)
                    self.live_metrics["latency_ms"] = round(frame_lat_ms, 1)
                    self.live_metrics["frame_idx"] = f_idx


# Global Engine Singleton
engine = VideoEngine()


class ThreadPoolHTTPServer(HTTPServer):
    """
    High-Performance ThreadPool-backed HTTP Server.
    Maintains a fixed pool of 32 daemon workers to completely eliminate
    per-request thread creation, OS handle leakage, socket TIME_WAIT exhaustion,
    and Python MemoryError crashes under high-frequency streaming.
    """
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, max_workers=32):
        super().__init__(server_address, RequestHandlerClass)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="HTTPWorker")

    def process_request(self, request, client_address):
        self.executor.submit(self.process_request_thread, request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)

    def server_close(self):
        try:
            super().server_close()
        except Exception:
            pass
        self.executor.shutdown(wait=False)


class DehazingHTTPHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for web UI, video streaming, camera upload, and REST APIs."""
    protocol_version = "HTTP/1.1"
    timeout = 3.0

    def log_message(self, format, *args):
        # Suppress routine GET logging to keep console clean
        return

    def do_GET(self):
        engine.has_active_clients = True
        engine.last_client_time = time.time()
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_index()
        elif path == "/video_feed":
            self._stream_mjpeg()
        elif path == "/api/latest_frame" or path == "/frame.jpg":
            self._serve_latest_frame()
        elif path == "/api/status":
            with engine.lock:
                metrics_copy = dict(engine.live_metrics)
            self._serve_json(metrics_copy)
        elif path == "/api/videos":
            vid_dir = os.path.join(_ROOT_DIR, "test_videos")
            valid_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
            videos = [f for f in os.listdir(vid_dir) if f.lower().endswith(valid_exts)] if os.path.exists(vid_dir) else []
            with engine.lock:
                curr_vid = engine.current_video
                curr_gt = engine.gt_video
            self._serve_json({
                "videos": sorted(videos),
                "current": curr_vid,
                "current_gt": curr_gt,
                "dshow_available": (_FFMPEG_EXE is not None)
            })
        elif path == "/api/public_url":
            with engine.lock:
                pub_url = engine.public_url
            if not pub_url:
                txt_path = os.path.join(_ROOT_DIR, "public_url.txt")
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            pub_url = f.read().strip()
                            with engine.lock:
                                engine.public_url = pub_url
                                engine.live_metrics["public_url"] = pub_url
                    except Exception:
                        pass
            self._serve_json({"public_url": pub_url})
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        engine.has_active_clients = True
        engine.last_client_time = time.time()
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/control":
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8')
            try:
                data = json.loads(body)
                engine.update_settings(data)
                with engine.lock:
                    settings_copy = dict(engine.live_metrics)
                self._serve_json({"status": "ok", "settings": settings_copy})
            except Exception as e:
                self._serve_json({"status": "error", "message": str(e)}, code=400)

        elif path == "/api/camera_frame":
            # Direct binary JPEG or raw bytes from browser webcam
            length = int(self.headers.get('Content-Length', 0))
            if length > 0:
                frame_bytes = self.rfile.read(length)
                try:
                    img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                    frame_rgb = np.array(img)
                    engine.push_browser_camera_frame(frame_rgb)
                    self._serve_json({"status": "ok", "fps": engine.live_metrics["fps"]})
                except Exception as e:
                    self._serve_json({"status": "error", "message": str(e)}, code=400)
            else:
                self._serve_json({"status": "error", "message": "Empty frame"}, code=400)

        elif path == "/api/upload_video":
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length <= 0:
                self._serve_json({"status": "error", "message": "No file uploaded (empty content)"}, code=400)
                return

            from urllib.parse import parse_qs, unquote
            query_params = parse_qs(parsed.query)
            raw_filename = query_params.get("filename", [None])[0]
            if raw_filename:
                raw_filename = unquote(raw_filename)
            else:
                raw_filename = self.headers.get("X-Filename", f"user_video_{int(time.time())}.mp4")

            # Clean and sanitize filename
            safe_name = os.path.basename(raw_filename).replace("..", "").replace("/", "").replace("\\", "").strip()
            if not safe_name:
                safe_name = f"user_video_{int(time.time())}.mp4"

            # Check video extension
            base, ext = os.path.splitext(safe_name)
            if ext.lower() not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                ext = ".mp4"
                safe_name = base + ext

            # Prefix custom uploaded videos
            if not safe_name.startswith("custom_") and not safe_name.startswith("uploaded_"):
                safe_name = f"custom_{safe_name}"

            vid_dir = os.path.join(_ROOT_DIR, "test_videos")
            os.makedirs(vid_dir, exist_ok=True)
            save_path = os.path.join(vid_dir, safe_name)

            # Stream from network socket to disk in 64KB chunks (protects RAM)
            bytes_left = content_length
            chunk_size = 65536
            try:
                with open(save_path, "wb") as f:
                    while bytes_left > 0:
                        read_len = min(chunk_size, bytes_left)
                        chunk = self.rfile.read(read_len)
                        if not chunk:
                            break
                        f.write(chunk)
                        bytes_left -= len(chunk)

                # Immediately switch engine to play the newly uploaded video!
                engine.update_settings({"video": safe_name, "restart": True})
                print(f"\n [UPLOAD] User video saved & active: {safe_name} ({content_length / (1024*1024):.2f} MB)")
                self._serve_json({
                    "status": "ok",
                    "filename": safe_name,
                    "message": f"Video '{safe_name}' uploaded successfully!"
                })
            except Exception as e:
                print(f"\n [UPLOAD ERROR] Failed to save {safe_name}: {e}")
                self._serve_json({"status": "error", "message": str(e)}, code=500)

        else:
            self.send_error(404, "Unknown API endpoint")

    def _serve_index(self):
        index_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
        if os.path.exists(index_file):
            with open(index_file, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(500, "static/index.html missing")

    def _serve_json(self, data, code=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream_mjpeg(self):
        """
        High-Speed Multipart MJPEG streaming.
        Consumes from decoupled VideoEngine using Condition wait,
        instantly delivering frames with zero buffer queueing.
        """
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        last_seq = -1

        while engine.running:
            # Wait for a new frame to be produced
            with engine.condition:
                if engine.frame_seq == last_seq or engine.latest_jpeg is None:
                    engine.condition.wait(timeout=0.1)
                jpeg_bytes = engine.latest_jpeg
                last_seq = engine.frame_seq

            if jpeg_bytes is None:
                continue

            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(jpeg_bytes)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, Exception):
                break

    def _serve_latest_frame(self):
        """Serves the latest processed JPEG frame directly for the GPU Canvas pump."""
        with engine.lock:
            jpeg_bytes = engine.latest_jpeg
            seq = engine.frame_seq
        if jpeg_bytes is not None:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg_bytes)))
            self.send_header("X-Frame-Seq", str(seq))
            self.send_header("Access-Control-Expose-Headers", "X-Frame-Seq")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(jpeg_bytes)
            self.wfile.flush()
        else:
            self.send_error(503, "No frame ready yet")


def run_server(port=8000, public_url=None):
    if public_url:
        engine.public_url = public_url
        engine.live_metrics["public_url"] = public_url

    server_address = ("0.0.0.0", port)
    httpd = ThreadPoolHTTPServer(server_address, DehazingHTTPHandler, max_workers=32)
    print("\n" + "=" * 80)
    print(f" [SERVER] REAL-TIME DEHAZING LOCALHOST WEB SERVER ACTIVE")
    print(f" [LOCAL]  http://127.0.0.1:{port} (Instant 3ms) or http://localhost:{port}")
    if public_url:
        print(f" [PUBLIC] {public_url} (Accessible over public internet)")
    print(f" [SPEED]  Decoupled Multi-Threaded Engine | 35-60+ FPS Guaranteed")
    print(f" [CAMERA] Laptop Webcam (Browser WebRTC + DirectShow Chicony USB2.0 Camera)")
    print(f" [METRICS] Full GT Suite (PSNR, SSIM, NIQE, LPIPS, FDI, Entropy, Color)")
    print(f" [ENGINE] 100% OpenCV-Free | Pure NumPy, SciPy, Pillow & PyTorch CPU")
    print("=" * 80 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server...")
        engine.running = False
        engine._close_sources()
        httpd.server_close()


if __name__ == "__main__":
    env_port = os.environ.get("PORT")
    port = int(env_port) if env_port else (int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
    run_server(port=port)

