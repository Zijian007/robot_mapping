"""Mapping loop using normalized sensor data and an injectable LIO backend."""
from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from robot_mapping.ports import LioBackend, LioResult, SensorSource
from robot_mapping.backends.fastlio import BackendScan, prepare_scan


from robot_mapping.filters.body_filter import (
    BodyBox,
    MissingBodyTransform,
    filter_points_array,
    lidar_b_to_lidar_a_rt,
    transform_points_array,
)
from robot_mapping.config import MappingSettings
from robot_mapping.core.online_map import OnlineMap


@dataclass(frozen=True)
class FrontendScan:
    """Native scan plus the source time for TF and integration time for LIO."""
    scan: BackendScan
    timestamp_ns: int
    integration_stamp: float

def _fmt_box(box: BodyBox) -> str:
    return (
        f"[{box.min_x:.2f},{box.max_x:.2f}]x"
        f"[{box.min_y:.2f},{box.max_y:.2f}]x"
        f"[{box.min_z:.2f},{box.max_z:.2f}]"
    )


def apply_body_filter(scan: Any, rotation: np.ndarray, translation: np.ndarray, box: BodyBox) -> tuple[Any, int]:
    arr = scan.points_array
    kept, n_removed = filter_points_array(arr, rotation, translation, box)
    if n_removed == 0:
        return scan, 0
    return replace(scan, points_array=kept, dropped_points=int(scan.dropped_points) + n_removed), n_removed


def merge_back_into_front(
    front_scan: Any,
    back_scan: Any,
    *,
    rotation_front,
    translation_front,
    rotation_back,
    translation_back,
    body_box: BodyBox | None,
    max_scan_duration_s: float,
) -> tuple[Any, int, int]:
    """Put back-lidar points into the front lidar frame and concatenate.

    Returns (merged_scan, body_removed_on_back, n_back_kept).
    """
    back_arr = back_scan.points_array
    if len(back_arr) == 0:
        return front_scan, 0, 0
    body_removed = 0
    if body_box is not None:
        back_arr, body_removed = filter_points_array(
            back_arr, rotation_back, translation_back, body_box
        )
        if len(back_arr) == 0:
            return front_scan, body_removed, 0
    rotation_fb, translation_fb = lidar_b_to_lidar_a_rt(
        rotation_front, translation_front, rotation_back, translation_back
    )
    back_in_front = transform_points_array(back_arr, rotation_fb, translation_fb)
    if back_arr.shape[1] >= 8:
        back_in_front[:, 5:8] = back_arr[:, 5:8] @ rotation_fb.T + translation_fb
    dt_ms = (float(back_scan.frame_stamp) - float(front_scan.frame_stamp)) * 1000.0
    back_in_front[:, 4] += np.float32(dt_ms)
    # Keep actual capture times inside the primary scan's IMU interval. Clipping
    # times to the boundary gives out-of-window returns a fictitious pose.
    end_ms = float(front_scan.scan_duration_s) * 1000.0
    if max_scan_duration_s > 0.0:
        end_ms = min(end_ms, max_scan_duration_s * 1000.0)
    in_window = ((back_in_front[:, 4] >= 0.) & (back_in_front[:, 4] <= end_ms))
    time_removed = int(np.count_nonzero(~in_window))
    back_in_front = back_in_front[in_window]
    if len(back_in_front) == 0:
        return front_scan, body_removed, 0
    front_arr = front_scan.points_array
    if len(front_arr) == 0:
        merged = back_in_front
    else:
        merged = np.concatenate([front_arr, back_in_front], axis=0)
    return (
        replace(
            front_scan,
            points_array=merged,
            raw_points=int(front_scan.raw_points) + int(back_scan.raw_points),
            dropped_points=(int(front_scan.dropped_points) + int(back_scan.dropped_points)
                            + body_removed + time_removed),
        ),
        body_removed,
        len(back_in_front),
    )


class MappingFrontend:
    def __init__(
        self,
        source: SensorSource,
        settings: MappingSettings,
        *,
        online_map: OnlineMap | None = None,
        lidar_extrinsics: dict[str, np.ndarray],
        imu_from_lidar: np.ndarray | None = None,
        body_filter: Any = None,
        body_lookup: Callable[[str, str, int], np.ndarray] | None = None,
        clock_id: str = "sensor",
        imu_frame_id: str = "imu",
        lio: LioBackend | None = None,
        initialize_map: Callable[[np.ndarray, int], bool] | None = None,
    ) -> None:
        self._extrinsics = {k: np.array(v, dtype=np.float64, copy=True) for k, v in lidar_extrinsics.items()}
        self._imu_from_lidar = np.eye(4) if imu_from_lidar is None else np.array(imu_from_lidar, copy=True)
        self._source = source
        self._clock_id = clock_id
        self._imu_frame_id = imu_frame_id
        # Optional source capabilities are resolved once, outside the frame loop.
        self._flush_source = getattr(source, "flush", lambda: None)
        self._aux_lidar = getattr(source, "aux_lidar", lambda: None)
        self._settings = settings
        self._online_map = online_map
        self._initialize_map = initialize_map
        self._body_filter = body_filter
        self._body_lookup = body_lookup
        self._body_tf_missing_since: float | None = None
        self._lio = lio
        self._lio_lock = threading.Lock()
        self._reset_serial = 0
        self._flush_pending = False
        self.frame_count = 0
        self.skip_frame_count = 0
        self.body_removed_total = 0
        self.back_merged_frames = 0
        self.keyframe_count = 0
        self.match_failure_count = 0
        self.consecutive_match_failures = 0
        from .timing import FrameTimings
        self.timings = FrameTimings()
        self._frame_times = {}

    def _read_scan(self, timeout: float) -> FrontendScan | None:
        scan = self._source.next_lidar(timeout)
        if scan is None:
            return None
        if scan.clock_id != self._clock_id or scan.sensor_id != self._settings.primary_lidar:
            raise ValueError("primary scan has unexpected sensor or clock")
        native = prepare_scan(scan, self._settings.blind)
        # Preserve the arithmetic used by recorded IMU windows.
        return FrontendScan(native, scan.begin_ns, native.frame_stamp + native.scan_duration_s)

    def _read_aux(self) -> FrontendScan | None:
        scan = self._aux_lidar()
        if scan is None:
            return None
        if scan.clock_id != self._clock_id or scan.sensor_id != "back":
            raise ValueError("auxiliary scan has unexpected sensor or clock")
        native = prepare_scan(scan, self._settings.blind)
        # Preserve the arithmetic used by recorded IMU windows.
        return FrontendScan(native, scan.begin_ns, native.frame_stamp + native.scan_duration_s)

    def _read_imu(self, begin: float, end: float) -> np.ndarray:
        batch = self._source.imu_between(round(begin * 1e9), round(end * 1e9), clock_id=self._clock_id)
        if batch.clock_id != self._clock_id:
            raise ValueError("IMU and LiDAR clocks do not match")
        if batch.frame_id != self._imu_frame_id:
            raise ValueError("IMU frame does not match calibrated robot.imu_frame_id")
        return np.column_stack((batch.timestamps_ns * 1e-9, batch.angular_velocity, batch.linear_acceleration))

    def _imu_window_error(self, samples: np.ndarray, begin: float, end: float) -> str | None:
        """Validate coverage as well as count; small boundary extrapolation is allowed."""
        if not self._settings.require_imu:
            return None
        if len(samples) < self._settings.min_imu_samples:
            return f"样本不足 ({len(samples)} < {self._settings.min_imu_samples})"
        stamps = samples[:, 0]
        # Epoch timestamps lose a fraction of a microsecond in float arithmetic.
        tolerance = 1e-6
        if (not np.isfinite(samples).all() or np.any(np.diff(stamps) <= 0)
                or stamps[0] < begin - tolerance or stamps[-1] > end + tolerance):
            return "时间戳不在请求窗口内或样本无效"
        head, tail = stamps[0] - begin, end - stamps[-1]
        boundary = self._settings.max_imu_boundary_gap_s
        if max(head, tail) > boundary + tolerance:
            return f"首尾缺失过长 (head={head:.4f}s, tail={tail:.4f}s, limit={boundary:.4f}s)"
        gap = float(np.max(np.diff(stamps))) if len(stamps) > 1 else 0.
        if gap > self._settings.max_imu_gap_s + tolerance:
            return f"内部断点过长 ({gap:.4f}s > {self._settings.max_imu_gap_s:.4f}s)"
        return None

    def _filter_scan(self, scan: Any, lidar: str, timestamp_ns: int,
                     rotation: np.ndarray, translation: np.ndarray) -> tuple[Any, int]:
        settings = self._settings
        if not settings.body_filter_enabled:
            return scan, 0
        if len(scan.points_array) == 0:
            return scan, 0
        if settings.body_filter_mode == "box":
            box = settings.back_box if lidar == "back" else settings.front_box
            return apply_body_filter(scan, rotation, translation, box)
        points, removed = self._body_filter.filter_points(
            scan.points_array, rotation, translation, timestamp_ns, self._body_lookup)
        if removed:
            scan = replace(scan, points_array=points, dropped_points=int(scan.dropped_points) + removed)
        return scan, removed

    def _skip_missing_body_tf(self, exc: MissingBodyTransform) -> None:
        now = time.monotonic()
        self.skip_frame_count += 1
        if self._body_tf_missing_since is None:
            self._body_tf_missing_since = now
            print(f"本体过滤跳过点云：{exc}", file=sys.stderr)
        elif now - self._body_tf_missing_since >= self._settings.body_tf_timeout_s:
            raise RuntimeError(f"本体 TF 持续缺失，停止建图并保存已有地图：{exc}") from exc

    def setup_lio(self) -> None:
        from robot_mapping.backends.fastlio import create_backend
        if self._lio is None:
            self._lio = create_backend(self._settings, self._imu_from_lidar)

    def reset_lio(self) -> None:
        with self._lio_lock:
            if self._lio is not None:
                self._lio.reset()
                # Source/recorder state belongs to the acquisition thread.
                self._flush_pending = True
                self._reset_serial += 1
                self._body_tf_missing_since = None
                self.frame_count = 0
                self.keyframe_count = 0
                self.consecutive_match_failures = 0

    def run(self, stop_evt: threading.Event) -> None:
        settings = self._settings
        lio = self._lio
        source = self._source

        front = self._extrinsics.get("front", np.eye(4))
        back = self._extrinsics.get("back", np.eye(4))
        rotation_front, translation_front = front[:3, :3], front[:3, 3]
        rotation_back, translation_back = back[:3, :3], back[:3, 3]
        if settings.primary_lidar == "back":
            rotation, translation = rotation_back, translation_back
        else:
            rotation, translation = rotation_front, translation_front

        last_target_stamp = -math.inf
        self._last_report_time = -math.inf
        last_back_ns: int | None = None
        reset_serial = self._reset_serial
        use_back = settings.needs_back
        init_acc_sum = np.zeros(3)
        init_acc_count = 0
        map_alignment_ready = False
        self.last_status = "initializing"
        self.last_frame_monotonic = time.monotonic()

        print(f"等待传感器数据就绪 ({settings.lidar_ready:.1f}s)...")
        if stop_evt.wait(settings.lidar_ready):
            return
        self._flush_source()
        self.last_frame_monotonic = time.monotonic()
        imu_startup_started = self.last_frame_monotonic
        print(
            f"开始建图 lidar={settings.lidar} imu={settings.imu} "
            f"blind={settings.blind:.2f} cube={settings.cube_side_length:.0f} det={settings.det_range:.0f} "
            f"filter_map={settings.filter_size_map} filter_surf={settings.filter_size_surf} "
            f"body_filter={'on' if settings.body_filter_enabled else 'off'} "
            f"mode={settings.body_filter_mode} "
            + (f"urdf={settings.body_urdf_file} padding={settings.body_collision_threshold:.3f}m"
               if settings.body_filter_mode == "urdf" else
               f"front_box={_fmt_box(settings.front_box)} back_box={_fmt_box(settings.back_box)}")
        )

        while not stop_evt.is_set():
            if reset_serial != self._reset_serial:
                if self._flush_pending:
                    self._flush_source()
                    self._flush_pending = False
                last_target_stamp = -math.inf
                last_back_ns = None
                init_acc_sum = np.zeros(3)
                init_acc_count = 0
                map_alignment_ready = False
                imu_startup_started = time.monotonic()
                reset_serial = self._reset_serial
            generation = self._online_map.ingest_token if self._online_map is not None else None
            read_start = time.monotonic()
            tick = self._read_scan(0.05)
            if tick is None:
                if getattr(source, "exhausted", False):
                    return
                if settings.max_lidar_frame_gap > 0 and time.monotonic() - self.last_frame_monotonic > settings.max_lidar_frame_gap:
                    raise RuntimeError("LiDAR 断流，停止建图并保存已有地图")
                continue
            self.last_frame_monotonic = time.monotonic()
            handle_start = time.monotonic()
            self._frame_started = read_start
            self._frame_times = dict(read_ms=(handle_start-read_start)*1000., prepare_ms=0.,
                imu_ms=0., lio_ms=0., map_ms=0., grid_ms=0., voxel_ms=0., total_ms=0.)

            scan = tick.scan
            if settings.grid_ray_clearing:
                scan = replace(scan, points_array=np.column_stack((scan.points_array,
                    np.zeros((len(scan.points_array), 3), dtype=scan.points_array.dtype))))

            body_removed = 0
            back_n = 0
            if settings.body_filter_enabled:
                try:
                    scan, body_removed = self._filter_scan(
                        scan, settings.primary_lidar, tick.timestamp_ns, rotation, translation)
                except MissingBodyTransform as exc:
                    self._skip_missing_body_tf(exc)
                    continue
                self.body_removed_total += body_removed
                n_keep = len(scan.points_array)
                if n_keep == 0 and not use_back:
                    self.skip_frame_count += 1
                    continue

            if use_back:
                back_msg = self._read_aux()
                if back_msg is not None:
                    back_ns = int(back_msg.timestamp_ns)
                    back_scan = back_msg.scan
                    if settings.grid_ray_clearing:
                        back_scan = replace(back_scan, points_array=np.column_stack((back_scan.points_array,
                            np.zeros((len(back_scan.points_array), 3), dtype=back_scan.points_array.dtype))))
                    back_scan_time_offset_s = abs(float(back_scan.frame_stamp) - float(scan.frame_stamp))
                    if back_scan_time_offset_s <= settings.max_back_dt and back_ns != last_back_ns:
                        try:
                            back_scan, back_body_rm = self._filter_scan(
                                back_scan, "back", back_ns, rotation_back, translation_back)
                        except MissingBodyTransform as exc:
                            self._skip_missing_body_tf(exc)
                            continue
                        scan, _, back_n = merge_back_into_front(
                            scan,
                            back_scan,
                            rotation_front=rotation_front,
                            translation_front=translation_front,
                            rotation_back=rotation_back,
                            translation_back=translation_back,
                            body_box=None,  # Already filtered at the back lidar's own timestamp.
                            max_scan_duration_s=settings.max_scan_duration,
                        )
                        body_removed += back_body_rm
                        self.body_removed_total += back_body_rm
                        last_back_ns = back_ns
                        self.back_merged_frames += 1

            self._body_tf_missing_since = None
            n_keep = len(scan.points_array)
            if n_keep == 0:
                self.skip_frame_count += 1
                continue

            # TF uses scan BEGIN; LIO integrates at scan END.
            scan = replace(scan, frame_stamp=tick.integration_stamp)

            if reset_serial != self._reset_serial:
                continue

            target_stamp = scan.frame_stamp
            if (
                math.isfinite(last_target_stamp)
                and settings.max_lidar_frame_gap > 0.0
                and target_stamp - last_target_stamp > settings.max_lidar_frame_gap
            ):
                raise RuntimeError(
                    f"相邻处理帧时间跨度 {target_stamp - last_target_stamp:.3f}s 超过上限 "
                    f"{settings.max_lidar_frame_gap:.3f}s；可能为处理延迟跳帧或传感器间断，停止建图并保存当前地图")
            if math.isfinite(last_target_stamp) and target_stamp <= last_target_stamp:
                raise RuntimeError("LiDAR 时间戳倒退；停止建图并保留当前地图")

            if math.isfinite(last_target_stamp):
                window_start = last_target_stamp
            else:
                window_start = (
                    target_stamp - max(scan.scan_duration_s, settings.max_scan_duration, 0.1) - 0.02
                )

            self._frame_times["prepare_ms"] = (time.monotonic()-handle_start)*1000.
            imu_start = time.monotonic()
            imu_array = self._read_imu(window_start, target_stamp)
            self._frame_times["imu_ms"] = (time.monotonic()-imu_start)*1000.
            imu_error = self._imu_window_error(imu_array, window_start, target_stamp)
            if imu_error is not None:
                self.skip_frame_count += 1
                self.last_status = "waiting_for_imu"
                # The source has already waited for late data. Once tracking has
                # started, do not integrate across an unobserved interval.
                if (math.isfinite(last_target_stamp) or
                        time.monotonic() - imu_startup_started >= settings.imu_startup_timeout_s):
                    raise RuntimeError(f"IMU 时间覆盖不完整，停止建图并保存已有地图：{imu_error}")
                continue

            lio_start = time.monotonic()
            with self._lio_lock:
                if reset_serial != self._reset_serial:
                    continue
                odom: LioResult = lio.process(scan, imu_array)
            self._frame_times["lio_ms"] = (time.monotonic()-lio_start)*1000.
            self.last_status = odom["status"]
            if odom["status"] == "initializing":
                # Mirror native initialization's valid accelerometer samples.
                samples = np.asarray(imu_array, dtype=np.float64)
                if samples.size:
                    valid = samples[np.isfinite(samples[:, 1:7]).all(axis=1)]
                    init_acc_sum += valid[:, 4:7].sum(axis=0)
                    init_acc_count += len(valid)
            last_target_stamp = target_stamp
            self.frame_count += 1
            handle_ms = (time.monotonic() - handle_start) * 1000.0

            if odom["status"] != "initializing":
                valid_update = (odom["status"] == "scan_to_map"
                                and odom.get("lidar_update_valid") is True)
                if not valid_update:
                    self.match_failure_count += 1
                    self.consecutive_match_failures += 1
                    self._report_frame(odom, scan, handle_ms=handle_ms, imu_count=len(imu_array),
                        body_removed=body_removed, back_points=back_n, keyframe=False)
                    if self.consecutive_match_failures >= settings.max_match_failures:
                        raise RuntimeError(
                            f"激光匹配连续失败 {self.consecutive_match_failures} 帧，停止建图并保存已有地图")
                    continue
                self.consecutive_match_failures = 0

            keyframe = False
            if self._online_map is not None and generation is not None and odom["status"] == "scan_to_map":
                if not map_alignment_ready:
                    if self._initialize_map is None:
                        # The core defaults to the backend's coordinates. The
                        # application may inject a frame initialization policy.
                        map_alignment_ready = self._online_map.set_map_alignment(np.eye(4), generation=generation)
                    else:
                        if not init_acc_count:
                            raise RuntimeError("missing initialization IMU for map frame policy")
                        map_alignment_ready = self._initialize_map(init_acc_sum / init_acc_count, generation)
                    if not map_alignment_ready:
                        continue
                map_start = time.monotonic()
                keyframe = self._online_map.ingest_odom(odom, generation=generation)
                self._frame_times["map_ms"] = (time.monotonic()-map_start)*1000.
                self._frame_times.update(self._online_map.projection_timings)
                if keyframe:
                    self.keyframe_count += 1

            self._report_frame(odom, scan, handle_ms=handle_ms, imu_count=len(imu_array),
                body_removed=body_removed, back_points=back_n, keyframe=keyframe)

    def _report_frame(self, odom: LioResult, scan: BackendScan, *, handle_ms: float,
                      imu_count: int, body_removed: int, back_points: int, keyframe: bool) -> None:
        """Keep reporting and its throttle separate from the acquisition/estimation loop."""
        now = time.monotonic()
        self._frame_times["total_ms"] = (now-self._frame_started)*1000.
        self.timings.add(self._frame_times)
        if self._settings.print_interval > 0 and now - self._last_report_time < self._settings.print_interval:
            return

        prefix = "[warmup] " if self.frame_count <= self._settings.warmup_frames else ""
        online_bits = ""
        if self._online_map is not None:
            st = self._online_map.stats()
            online_bits = (
                f", phase={self._online_map.phase.value}, kf={st.keyframes}, "
                f"voxels={st.voxels}, occ={st.occupied_cells}, "
                f"ingest_ms={self._frame_times['map_ms']:.1f}"
            )
            online_bits += f", loops={st.loops}"
            if self._online_map.loop_error:
                online_bits += f", loop_error={self._online_map.loop_error}"
        print(
            f"{prefix}frame={self.frame_count - 1}, ts={odom['stamp']:.9f}, "
            f"handle_ms={handle_ms:.1f}, imu={imu_count}, "
            f"lio_ms={self._frame_times['lio_ms']:.1f}, grid_ms={self._frame_times['grid_ms']:.1f}, "
            f"voxel_ms={self._frame_times['voxel_ms']:.1f}, total_ms={self._frame_times['total_ms']:.1f}, "
            f"scan_dt={scan.scan_duration_s:.3f}, dropped={scan.dropped_points}, "
            f"body_rm={body_removed}, back_n={back_points}, matched={odom.get('matched_points', 0)}, "
            f"map={odom.get('map_points', 0)}, "
            f"res={float(odom.get('mean_abs_residual', 0.0)):.6f}, "
            f"status={odom['status']}, keyframe={int(keyframe)}"
            f"{online_bits}"
        )
        self._last_report_time = now

