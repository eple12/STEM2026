from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import neat
import numpy as np
import yaml


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class TrackData:
    name: str
    centerline_xy: np.ndarray
    width_right: np.ndarray
    width_left: np.ndarray
    raceline_xy: np.ndarray
    raceline_speed: np.ndarray
    raceline_kappa: np.ndarray
    map_yaml: dict


@dataclass(frozen=True)
class DifficultySettings:
    width_scale: float
    offtrack_scale: float
    step_scale: float
    lane_penalty_weight: float
    heading_penalty_weight: float
    speed_scale: float
    steer_penalty_weight: float
    checkpoint_pass_reward_scale: float
    start_speed: float
    downsample_stride: int
    brake_penalty_weight: float   # ★ 추가: 예측 브레이킹 페널티 가중치

    @staticmethod
    def from_name(name: str) -> "DifficultySettings":
        # brake_penalty_weight: 크게 설정할수록 코너 전 감속을 강하게 학습
        # 단, 너무 크면 직선에서도 과도하게 감속함 → 0.3~0.8 권장
        presets = {
            "normal":    DifficultySettings(1.00, 1.00, 1.00, 0.70, 0.40, 0.10, 0.028, 1.0,  2.0, 1, 0.50),
            "easy":      DifficultySettings(1.35, 1.25, 0.75, 0.45, 0.25, 0.12, 0.020, 1.35, 2.5, 2, 0.60),
            "very-easy": DifficultySettings(1.75, 1.60, 0.60, 0.25, 0.15, 0.14, 0.015, 1.65, 3.0, 3, 0.70),
            "overfit":   DifficultySettings(2.20, 2.00, 0.45, 0.12, 0.08, 0.18, 0.010, 2.0,  3.8, 4, 0.80),
            "ideal":     DifficultySettings(1.35, 1.25, 0.75, 0.50, 3.5,  0.12, 2.0,   1.35, 2.5, 2, 0.60),
        }
        return presets["ideal"]


class TrackLoader:
    @staticmethod
    def _read_numeric_table(path: Path, expected_cols: int) -> np.ndarray:
        rows: List[List[float]] = []
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                delimiter = ";" if ";" in line else ","
                parts = [p.strip() for p in line.split(delimiter)]
                if len(parts) < expected_cols:
                    continue
                rows.append([float(v) for v in parts[:expected_cols]])
        if not rows:
            raise ValueError(f"No usable numeric rows in {path}")
        return np.asarray(rows, dtype=np.float64)

    @staticmethod
    def load(track_dir: Path) -> TrackData:
        name = track_dir.name
        centerline_path = track_dir / f"{name}_centerline.csv"
        raceline_path = track_dir / f"{name}_raceline.csv"
        map_yaml_path = track_dir / f"{name}_map.yaml"
        if not centerline_path.exists():
            prefix = name.replace(" ", "")
            centerline_path = track_dir / f"{prefix}_centerline.csv"
            raceline_path = track_dir / f"{prefix}_raceline.csv"
            map_yaml_path = track_dir / f"{prefix}_map.yaml"

        centerline = TrackLoader._read_numeric_table(centerline_path, expected_cols=4)
        raceline = TrackLoader._read_numeric_table(raceline_path, expected_cols=7)
        with map_yaml_path.open("r", encoding="utf-8") as f:
            map_yaml = yaml.safe_load(f)

        return TrackData(
            name=name,
            centerline_xy=centerline[:, :2],
            width_right=centerline[:, 2],
            width_left=centerline[:, 3],
            raceline_xy=raceline[:, 1:3],
            raceline_speed=raceline[:, 5],
            raceline_kappa=raceline[:, 4],
            map_yaml=map_yaml,
        )


class CarEnv:
    def __init__(
        self,
        track: TrackData,
        settings: DifficultySettings,
        *,
        checkpoint_every: int,
        checkpoint_base_reward: float,
        checkpoint_miss_penalty: float,
        dt: float = 0.1,
    ):
        self.track = track
        self.settings = settings
        self.dt = dt
        stride = max(1, int(settings.downsample_stride))
        self.center = track.centerline_xy[::stride]
        self.raceline = track.raceline_xy[::stride]
        self.track_half_width = np.minimum(
            track.width_left[::stride], track.width_right[::stride]
        ) * settings.width_scale

        self.max_steer_rate = 1.8
        self.max_accel = 2.5
        self.min_speed = 0.5
        self.max_speed = max(10.0, float(np.max(track.raceline_speed)))

        # 곡률 기반 동적 목표 속도
        kappa = np.abs(track.raceline_kappa[::stride])
        a_lat = 5.0
        dynamic_speeds = np.sqrt(a_lat / (kappa + 1e-5))
        self.target_speeds = np.clip(dynamic_speeds, self.min_speed, self.max_speed)
        self.target_speeds = np.convolve(self.target_speeds, np.ones(5) / 5, mode="same")

        self.max_steps = max(
            160, int(max(400, len(self.center) // 3) * settings.step_scale)
        )
        interval = max(1, checkpoint_every // stride)
        n = len(self.center)
        self.checkpoint_indices = [0] + [
            i for i in range(interval, n, interval) if i != 0
        ]
        uniq: List[int] = []
        for i in self.checkpoint_indices:
            if i not in uniq:
                uniq.append(i)
        self.checkpoint_indices = (
            sorted(uniq) if len(uniq) >= 2 else list(range(min(8, n)))
        )
        self.checkpoint_every = checkpoint_every
        self.checkpoint_base_reward = checkpoint_base_reward
        self.checkpoint_miss_penalty = checkpoint_miss_penalty
        seg = np.sum(np.diff(self.center, axis=0) ** 2, axis=1)
        seg_med = float(np.percentile(seg, 50)) if len(seg) else 1.0
        gate_len = seg[0] if len(seg) else seg_med
        self.checkpoint_radius_sq = max(gate_len, seg_med) * 9.0
        self.checkpoint_radius_sq = max(36.0, self.checkpoint_radius_sq)
        self._build_checkpoint_geom()
        self.reset()

    def _build_checkpoint_geom(self) -> None:
        pts = []
        normals = []
        for j in self.checkpoint_indices:
            c = self.center[j]
            nxt = self.center[(j + 1) % len(self.center)]
            prv = self.center[(j - 1 + len(self.center)) % len(self.center)]
            tangent = nxt - prv
            tnorm = np.linalg.norm(tangent) + 1e-8
            tangent = tangent / tnorm
            normal = np.array([-tangent[1], tangent[0]])
            pts.append(np.array([c[0], c[1]], dtype=np.float64))
            normals.append(normal.astype(np.float64))
        self._checkpoint_points = pts
        self._checkpoint_normals = normals

    def reset(self) -> None:
        p0 = self.raceline[0]
        p1 = self.raceline[1]
        heading = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        self.x = float(p0[0])
        self.y = float(p0[1])
        self.heading = heading
        self.speed = self.settings.start_speed
        self.step_count = 0
        self.next_checkpoint_ord = 0
        self.cleared_checkpoint = [False] * len(self.checkpoint_indices)
        self.lap_started = False
        self.done = False
        self.path = [(self.x, self.y)]

    def _nearest_centerline_index(self, x: float, y: float) -> int:
        p = np.array([x, y], dtype=np.float64)
        d2 = np.sum((self.center - p) ** 2, axis=1)
        return int(np.argmin(d2))

    def _lane_error(self, idx: int, x: float, y: float) -> float:
        c = self.center[idx]
        next_idx = (idx + 1) % len(self.center)
        tangent = self.center[next_idx] - c
        tangent_norm = np.linalg.norm(tangent) + 1e-8
        tangent = tangent / tangent_norm
        normal = np.array([-tangent[1], tangent[0]])
        rel = np.array([x, y]) - c
        return float(np.dot(rel, normal))

    def _heading_error(self, idx: int, heading: float) -> float:
        p0 = self.raceline[idx % len(self.raceline)]
        p1 = self.raceline[(idx + 2) % len(self.raceline)]
        target_heading = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        return wrap_to_pi(target_heading - heading)

    def _crosses_checkpoint_segment(
        self, x0: float, y0: float, x1: float, y1: float, ord_idx: int
    ) -> bool:
        px = float(self._checkpoint_points[ord_idx][0])
        py = float(self._checkpoint_points[ord_idx][1])
        nx = float(self._checkpoint_normals[ord_idx][0])
        ny = float(self._checkpoint_normals[ord_idx][1])
        s0 = (x0 - px) * nx + (y0 - py) * ny
        s1 = (x1 - px) * nx + (y1 - py) * ny
        if s0 == 0.0 or s1 == 0.0:
            t = (-s0 / (s1 - s0 + 1e-18)) if s1 != s0 else 0.5
            t = float(np.clip(t, 0.0, 1.0))
            ix = x0 + t * (x1 - x0)
            iy = y0 + t * (y1 - y0)
            return math.hypot(ix - px, iy - py) ** 2 <= self.checkpoint_radius_sq
        if s0 * s1 >= 0.0:
            return False
        t = abs(s0) / (abs(s0 - s1) + 1e-12)
        ix = x0 + t * (x1 - x0)
        iy = y0 + t * (y1 - y0)
        return math.hypot(ix - px, iy - py) ** 2 <= self.checkpoint_radius_sq

    # ──────────────────────────────────────────────────────────────────────────
    # observe(): 입력 9개로 확장
    # [0] norm_lateral_err     현재 횡방향 오차
    # [1] norm_heading_err     현재 헤딩 오차
    # [2] speed_norm           현재 속도
    # [3] progress_norm        트랙 진행도
    # [4] speed_delta_now      현재 권장속도 대비 편차 (+면 가속 필요)
    # [5] min_future_speed_norm 전방 ~30m 중 최솟값 (코너 존재 여부)  ★ 핵심
    # [6] brake_urgency        현재속도 - 전방최소속도 (양수면 지금 브레이킹 필요)  ★ 핵심
    # [7] la_heading_err       중거리 전방 헤딩 오차
    # [8] curvature_ahead      전방 평균 곡률 절댓값 (코너 심각도)  ★ 핵심
    # ──────────────────────────────────────────────────────────────────────────
    def observe(self) -> np.ndarray:
        idx = self._nearest_centerline_index(self.x, self.y)
        lat_err = self._lane_error(idx, self.x, self.y)
        heading_err = self._heading_error(idx, self.heading)
        half_width = max(0.6, float(self.track_half_width[idx]))
        norm_err = lat_err / half_width
        speed_norm = self.speed / self.max_speed
        progress_norm = idx / max(1, len(self.center) - 1)

        n = len(self.center)
        lo1 = max(3, n // 40)
        lo2 = max(6, n // 20)
        lo3 = max(12, n // 10)

        idx1 = (idx + lo1) % n
        idx2 = (idx + lo2) % n
        idx3 = (idx + lo3) % n

        ts_now = self.target_speeds[idx]
        ts1 = self.target_speeds[idx1]
        ts2 = self.target_speeds[idx2]
        ts3 = self.target_speeds[idx3]

        # 전방 구간 최소 권장속도 (코너 존재 시 낮아짐)
        min_future_speed = min(ts1, ts2, ts3)

        speed_delta_now = (ts_now - self.speed) / self.max_speed
        min_future_speed_norm = min_future_speed / self.max_speed

        # ★ 브레이킹 긴급도: 현재속도 - 전방 최소 권장속도
        # 양수 = 지금 브레이크를 밟아야 함, 값이 클수록 급브레이킹 필요
        brake_urgency = max(0.0, self.speed - min_future_speed) / self.max_speed

        la_heading_err = self._heading_error(idx2, self.heading) / math.pi

        # ★ 전방 평균 곡률: 코너의 심각도를 직접 인지하게 함
        # kappa가 크면 = 앞에 급코너 있음
        kappa_vals = [
            abs(float(self.track.raceline_kappa[
                (idx + lo1 * k // lo1) % len(self.track.raceline_kappa)
            ])) for k in range(lo1, lo3, max(1, (lo3 - lo1) // 5))
        ]
        curvature_ahead = float(np.mean(kappa_vals)) * 100.0  # 스케일 조정 (값이 매우 작으므로)
        curvature_ahead = min(curvature_ahead, 1.0)  # 클리핑

        return np.array(
            [
                norm_err,                   # [0] 횡방향 오차
                heading_err / math.pi,      # [1] 헤딩 오차
                speed_norm,                 # [2] 현재 속도
                progress_norm,              # [3] 진행도
                speed_delta_now,            # [4] 현재 지점 속도 편차
                min_future_speed_norm,      # [5] 전방 최소 권장속도  ★
                brake_urgency,              # [6] 브레이킹 긴급도     ★
                la_heading_err,             # [7] 중거리 헤딩 오차
                curvature_ahead,            # [8] 전방 곡률 심각도    ★
            ],
            dtype=np.float64,
        )

    def step(self, steer_cmd: float, throttle_cmd: float) -> float:
        if self.done:
            return 0.0

        steer_norm = float(np.clip(steer_cmd, -1.0, 1.0))
        steer = steer_norm * self.max_steer_rate
        accel = float(np.clip(throttle_cmd, -1.0, 1.0)) * self.max_accel

        self.heading += steer * self.dt
        self.speed = float(
            np.clip(self.speed + accel * self.dt, self.min_speed, self.max_speed)
        )
        x_prev, y_prev = self.x, self.y
        self.x += self.speed * math.cos(self.heading) * self.dt
        self.y += self.speed * math.sin(self.heading) * self.dt
        self.path.append((self.x, self.y))
        self.step_count += 1

        idx = self._nearest_centerline_index(self.x, self.y)
        lat_err = abs(self._lane_error(idx, self.x, self.y))
        half_width = max(0.6, float(self.track_half_width[idx]))
        heading_err = abs(self._heading_error(idx, self.heading))

        # ── 다중 전방 탐색 ─────────────────────────────────────────────────────
        n_center = len(self.center)
        lo1 = max(3, n_center // 40)
        lo2 = max(6, n_center // 20)
        lo3 = max(12, n_center // 10)

        idx1 = (idx + lo1) % n_center
        idx2 = (idx + lo2) % n_center
        idx3 = (idx + lo3) % n_center

        target_speed_now = self.target_speeds[idx]
        ts1 = self.target_speeds[idx1]
        ts2 = self.target_speeds[idx2]
        ts3 = self.target_speeds[idx3]

        reward = 0.0
        if self.lap_started:
            o = self.next_checkpoint_ord
            if not self.cleared_checkpoint[o]:
                if self._crosses_checkpoint_segment(x_prev, y_prev, self.x, self.y, o):
                    if o == 0 and not any(self.cleared_checkpoint[1:]):
                        scale = self.settings.checkpoint_pass_reward_scale
                        reward += self.checkpoint_base_reward * scale * 0.6
                    else:
                        self.cleared_checkpoint[o] = True
                        if o == 0 and all(self.cleared_checkpoint[1:]):
                            self.cleared_checkpoint = [False] * len(self.checkpoint_indices)
                            self.next_checkpoint_ord = 0
                            scale = self.settings.checkpoint_pass_reward_scale
                            reward += self.checkpoint_base_reward * scale * 2.0
                        else:
                            self.next_checkpoint_ord = (o + 1) % len(self.checkpoint_indices)
                            scale = self.settings.checkpoint_pass_reward_scale
                            reward += self.checkpoint_base_reward * scale
            for other in range(len(self.checkpoint_indices)):
                if other == o or self.cleared_checkpoint[other]:
                    continue
                if self._crosses_checkpoint_segment(x_prev, y_prev, self.x, self.y, other):
                    reward -= self.checkpoint_miss_penalty
                    break
        else:
            if self._crosses_checkpoint_segment(x_prev, y_prev, self.x, self.y, 0):
                self.lap_started = True
                self.cleared_checkpoint[0] = True
                self.next_checkpoint_ord = 1 % len(self.checkpoint_indices)
                reward += (
                    self.checkpoint_base_reward
                    * self.settings.checkpoint_pass_reward_scale
                    * 0.35
                )

        lane_penalty = (lat_err / half_width) ** 2
        heading_penalty = (heading_err / math.pi) ** 2

        # ── 1. 속도 매칭 보상 ──────────────────────────────────────────────────
        speed_match_reward = (
            1.0 - abs(self.speed - target_speed_now) / self.max_speed
        ) * self.settings.speed_scale

        # ── 2. 예측 브레이킹 페널티 ★ 핵심 수정 ──────────────────────────────
        #
        # 아이디어: 전방 코너 권장속도보다 현재 속도가 높을수록 페널티.
        # 단계별(근/중/원) 탐지로 코너가 가까울수록 더 강하게 페널티.
        # ratio² 사용 → 약간의 과속은 용인, 심한 과속은 크게 처벌.
        #
        def _brake_penalty(current_spd: float, target_spd: float) -> float:
            excess = max(0.0, current_spd - target_spd)
            # max_speed 대비 비율로 정규화 (속도 스케일과 무관하게 일정한 신호)
            ratio = excess / self.max_speed
            return ratio ** 2

        bp1 = _brake_penalty(self.speed, ts1) * self.settings.brake_penalty_weight * 3.0
        bp2 = _brake_penalty(self.speed, ts2) * self.settings.brake_penalty_weight * 2.0
        bp3 = _brake_penalty(self.speed, ts3) * self.settings.brake_penalty_weight * 1.0
        brake_penalty = bp1 + bp2 + bp3

        # ── 3. 전방 헤딩 페널티 ──────────────────────────────────────────────
        heading_err_la = abs(self._heading_error(idx2, self.heading))
        lookahead_heading_penalty = (
            (heading_err_la / math.pi) * self.settings.heading_penalty_weight * 0.5
        )

        # ── 4. 조향 페널티 ──────────────────────────────────────────────────
        steer_penalty = self.settings.steer_penalty_weight * (steer_norm ** 2)

        reward += (
            speed_match_reward
            - self.settings.lane_penalty_weight * lane_penalty
            - self.settings.heading_penalty_weight * heading_penalty
            - brake_penalty
            - lookahead_heading_penalty
            - steer_penalty
        )

        # ── 트랙 이탈: 빠를수록 더 큰 페널티 ★ ─────────────────────────────
        if lat_err > half_width * (1.15 * self.settings.offtrack_scale):
            speed_factor = self.speed / self.max_speed
            reward -= 8.0 + 12.0 * speed_factor   # 최대 -20 (max_speed 시)
            self.done = True
        elif self.step_count >= self.max_steps:
            self.done = True

        return reward

    def rollout(self, net: neat.nn.FeedForwardNetwork) -> tuple[float, np.ndarray]:
        self.reset()
        fitness = 0.0
        while not self.done:
            obs = self.observe()
            out = net.activate(obs.tolist())
            steer, throttle = out[0], out[1]
            fitness += self.step(steer, throttle)
        return fitness, np.asarray(self.path)


class LiveVizReporter(neat.reporting.BaseReporter):
    def __init__(
        self,
        env: CarEnv,
        animate_best: bool = False,
        animation_step_pause: float = 0.01,
        animation_window_m: float = 18.0,
        show_net: bool = False,
    ):
        self.env = env
        self.animate_best = animate_best
        self.animation_step_pause = animation_step_pause
        self.animation_window_m = animation_window_m
        self.show_net = show_net
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.generation_ids: List[int] = []
        self.best_path: np.ndarray | None = None
        self.gen_best_path: np.ndarray | None = None
        self.best_so_far = -1e18
        self.current_generation = -1

        if self.show_net:
            self.fig, (self.ax_track, self.ax_fit, self.ax_net) = plt.subplots(
                1, 3, figsize=(18, 6)
            )
        else:
            self.fig, (self.ax_track, self.ax_fit) = plt.subplots(1, 2, figsize=(13, 6))
            self.ax_net = None

        self.fig.suptitle("NEAT F1TENTH Training Visualization")
        self._init_axes()
        plt.ion()
        plt.show(block=False)

    def _init_track_axes(self) -> None:
        c = self.env.track.centerline_xy
        self.ax_track.clear()
        self.ax_track.plot(
            c[:, 0], c[:, 1], color="gray", alpha=0.65, linewidth=1.0, label="centerline"
        )
        self.ax_track.plot(
            self.env.track.raceline_xy[:, 0],
            self.env.track.raceline_xy[:, 1],
            color="dodgerblue",
            alpha=0.6,
            linewidth=1.0,
            label="raceline",
        )
        self.ax_track.set_aspect("equal", adjustable="box")
        self.ax_track.set_title("Track + Best Genome Trajectory")
        self.ax_track.legend(loc="upper right", fontsize=8)

    def _init_fitness_axes(self) -> None:
        self.ax_fit.clear()
        self.ax_fit.set_title("Fitness Over Generations")
        self.ax_fit.set_xlabel("Generation")
        self.ax_fit.set_ylabel("Fitness")
        self.ax_fit.grid(True, alpha=0.3)

    def _init_net_axes(self) -> None:
        if not self.show_net or self.ax_net is None:
            return
        self.ax_net.clear()
        self.ax_net.set_title("Neural Network Activity")
        self.ax_net.set_ylim(-1.1, 1.1)
        # 입력 9개로 업데이트
        self.in_labels = [
            "LatErr", "HeadErr", "Speed", "Prog",
            "SpdΔNow", "FutSpd", "BrakeUrg", "LA-Head", "Curv"
        ]
        self.out_labels = ["Steer", "Throttle"]
        self.all_labels = self.in_labels + ["|"] + self.out_labels
        self.ax_net.set_xticks(range(len(self.all_labels)))
        self.ax_net.set_xticklabels(
            self.all_labels, rotation=45, ha="right", fontsize=8
        )
        self.ax_net.grid(True, axis="y", alpha=0.3)
        self.net_bars = self.ax_net.bar(
            range(len(self.all_labels)), [0] * len(self.all_labels), color="skyblue"
        )
        for i in range(len(self.in_labels) + 1, len(self.all_labels)):
            self.net_bars[i].set_color("salmon")

    def _init_axes(self) -> None:
        self._init_track_axes()
        self._init_fitness_axes()
        if self.show_net:
            self._init_net_axes()

    def _plot_fitness_curves(self) -> None:
        self.ax_fit.plot(
            self.generation_ids, self.best_fitness_history, color="green", label="best"
        )
        self.ax_fit.plot(
            self.generation_ids, self.avg_fitness_history, color="orange", label="average"
        )
        self.ax_fit.legend(loc="best", fontsize=8)

    def start_generation(self, generation: int) -> None:
        self.current_generation = generation

    def post_evaluate(self, config, population, species, best_genome) -> None:
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        if not fitnesses:
            return
        gen_best = float(max(fitnesses))
        gen_avg = float(sum(fitnesses) / len(fitnesses))
        self.generation_ids.append(self.current_generation)
        self.best_fitness_history.append(gen_best)
        self.avg_fitness_history.append(gen_avg)

        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        _, self.gen_best_path = self.env.rollout(net)
        if gen_best >= self.best_so_far:
            self.best_so_far = gen_best
            self.best_path = self.gen_best_path.copy()

        self._redraw()
        if (
            self.animate_best
            and self.gen_best_path is not None
            and len(self.gen_best_path) > 2
        ):
            self._animate_generation_best(self.gen_best_path, net)

    def _redraw(self) -> None:
        self._init_axes()
        if self.best_path is not None and len(self.best_path) > 1:
            self.ax_track.plot(
                self.best_path[:, 0],
                self.best_path[:, 1],
                color="crimson",
                linewidth=2.0,
                label="best trajectory",
            )
            self.ax_track.legend(loc="upper right", fontsize=8)

        self._plot_fitness_curves()
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.pump_events()

    def pump_events(self) -> None:
        plt.pause(0.001)
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def _animate_generation_best(
        self, path: np.ndarray, net: neat.nn.FeedForwardNetwork
    ) -> None:
        self.env.reset()
        self._init_track_axes()
        self._init_fitness_axes()
        self._init_net_axes()
        self._plot_fitness_curves()
        self.ax_track.set_title(f"Gen {self.current_generation} Best (Animated)")

        (line,) = self.ax_track.plot([], [], color="crimson", linewidth=2.2, label="gen best")
        (dot,) = self.ax_track.plot([], [], marker="o", color="red", markersize=6)
        self.ax_track.legend(loc="upper right", fontsize=8)

        step_limit = len(path)
        for i in range(step_limit):
            obs = self.env.observe()
            out = net.activate(obs.tolist())
            steer, throttle = out[0], out[1]

            partial = path[: i + 1]
            line.set_data(partial[:, 0], partial[:, 1])
            dot.set_data([path[i, 0]], [path[i, 1]])

            if self.show_net:
                activations = obs.tolist() + [0.0] + [steer, throttle]
                for bar, val in zip(self.net_bars, activations):
                    bar.set_height(val)

            if i % 2 == 0:
                x_c, y_c = path[i, 0], path[i, 1]
                self.ax_track.set_xlim(
                    x_c - self.animation_window_m, x_c + self.animation_window_m
                )
                self.ax_track.set_ylim(
                    y_c - self.animation_window_m, y_c + self.animation_window_m
                )

            self.fig.canvas.draw_idle()
            self.pump_events()
            plt.pause(self.animation_step_pause)

            self.env.step(steer, throttle)
            if self.env.done:
                break

        self.ax_track.set_title("Track + Best Genome Trajectory")
        if self.best_path is not None and len(self.best_path) > 1:
            self.ax_track.plot(
                self.best_path[:, 0],
                self.best_path[:, 1],
                color="crimson",
                linewidth=2.0,
                label="best trajectory",
            )
            self.ax_track.legend(loc="upper right", fontsize=8)
        self.ax_track.autoscale()
        self.ax_track.set_aspect("equal", adjustable="box")

        self._init_fitness_axes()
        self._plot_fitness_curves()
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.pump_events()


class SilentReporter(neat.reporting.BaseReporter):
    def __init__(self) -> None:
        self.best = -1e18

    def post_evaluate(self, config, population, species, best_genome) -> None:
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        if fitnesses:
            self.best = max(self.best, float(max(fitnesses)))


def eval_genomes(
    genomes: Sequence[tuple[int, neat.DefaultGenome]],
    config,
    env: CarEnv,
    live_reporter: LiveVizReporter | None = None,
) -> None:
    for idx, (_, genome) in enumerate(genomes):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness, _ = env.rollout(net)
        genome.fitness = fitness
        if live_reporter is not None and (idx + 1) % 6 == 0:
            live_reporter.pump_events()


def run_training(
    track_dir: Path,
    config_path: Path,
    generations: int,
    difficulty: str,
    *,
    checkpoint_every: int,
    checkpoint_pass_reward: float,
    checkpoint_miss_penalty: float,
    visualize: bool = True,
    animate_best: bool = False,
    animation_step_pause: float = 0.01,
    animation_window_m: float = 18.0,
    show_net: bool = False,
) -> neat.DefaultGenome:
    track = TrackLoader.load(track_dir)
    settings = DifficultySettings.from_name(difficulty)
    env = CarEnv(
        track,
        settings=settings,
        checkpoint_every=checkpoint_every,
        checkpoint_base_reward=checkpoint_pass_reward,
        checkpoint_miss_penalty=checkpoint_miss_penalty,
    )
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    live_reporter: LiveVizReporter | None = None
    if visualize:
        live_reporter = LiveVizReporter(
            env,
            animate_best=animate_best,
            animation_step_pause=animation_step_pause,
            animation_window_m=animation_window_m,
            show_net=show_net,
        )
        pop.add_reporter(live_reporter)
    else:
        pop.add_reporter(SilentReporter())

    winner = pop.run(
        lambda gs, cfg: eval_genomes(gs, cfg, env, live_reporter), generations
    )
    print("\nTraining complete.")
    print(f"Winner fitness: {winner.fitness:.3f}")
    return winner


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train NEAT on F1TENTH racetrack data with live visualization."
    )
    parser.add_argument(
        "--track-dir",
        type=Path,
        default=script_dir.parent / "f1tenth_racetracks-main" / "Austin",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=script_dir / "neat_config.ini",
    )
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy",
        choices=["normal", "easy", "very-easy", "overfit"],
    )
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--show-net", action="store_true")
    parser.add_argument("--animate-best", action="store_true")
    parser.add_argument("--animate-step-pause", type=float, default=0.01)
    parser.add_argument("--animate-window-m", type=float, default=18.0)
    parser.add_argument("--checkpoint-every", type=int, default=45)
    parser.add_argument("--checkpoint-pass-reward", type=float, default=72.0)
    parser.add_argument("--checkpoint-miss-penalty", type=float, default=48.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        args.track_dir.expanduser().resolve(),
        args.config.expanduser().resolve(),
        args.generations,
        args.difficulty,
        checkpoint_every=args.checkpoint_every,
        checkpoint_pass_reward=args.checkpoint_pass_reward,
        checkpoint_miss_penalty=args.checkpoint_miss_penalty,
        visualize=not args.no_gui,
        animate_best=args.animate_best,
        animation_step_pause=args.animate_step_pause,
        animation_window_m=args.animate_window_m,
        show_net=args.show_net,
    )
    if not args.no_gui:
        print("Close the visualization window to end.")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()