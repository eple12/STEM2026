from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

from tqdm import tqdm

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import neat
import numpy as np

from main import CRASH_FITNESS_PENALTY, CarEnv, DifficultySettings, TrackLoader


IN_LABELS = [
    "LatErr",
    "HeadErr",
    "Speed",
    "Prog",
    "SpdNow",
    "FutSpd",
    "BrakeUrg",
    "LA-Head",
    "Curv",
]
OUT_LABELS = ["Steer", "Throttle"]


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    required = {"genome", "config", "track_dir", "difficulty"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Missing keys in model file: {sorted(missing)}")
    return payload


def find_latest_model(models_root: Path) -> Path:
    candidates = list(models_root.rglob("generation_*_best.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No model files found under {models_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_env(payload: dict[str, Any], args: argparse.Namespace) -> CarEnv:
    track_dir = (
        args.track_dir.expanduser().resolve()
        if args.track_dir is not None
        else Path(payload["track_dir"]).expanduser().resolve()
    )
    difficulty = args.difficulty or payload.get("difficulty", "easy")
    follow_line = args.follow_line or payload.get("follow_line", "centerline")
    max_steps = args.max_steps if args.max_steps is not None else payload.get("max_steps", 0)

    track = TrackLoader.load(track_dir)
    settings = DifficultySettings.from_name(difficulty)
    return CarEnv(
        track,
        settings=settings,
        checkpoint_every=args.checkpoint_every,
        checkpoint_base_reward=args.checkpoint_pass_reward,
        checkpoint_miss_penalty=args.checkpoint_miss_penalty,
        max_steps=max_steps,
        follow_line=follow_line,
    )


def rollout_trace(
    env: CarEnv, net: neat.nn.FeedForwardNetwork
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    env.reset()
    fitness = 0.0
    observations = []
    actions = []
    while not env.done:
        obs = env.observe()
        out = net.activate(obs.tolist())
        steer, throttle = float(out[0]), float(out[1])
        observations.append(obs)
        actions.append([steer, throttle])
        fitness += env.step(steer, throttle)
    if env.crashed:
        fitness += CRASH_FITNESS_PENALTY
    return (
        np.asarray(env.path, dtype=np.float64),
        np.asarray(observations, dtype=np.float64),
        np.asarray(actions, dtype=np.float64),
        fitness,
    )


def init_track_axis(ax, env: CarEnv, title: str) -> None:
    center = env.track.centerline_xy
    raceline = env.track.raceline_xy
    ax.clear()
    ax.plot(center[:, 0], center[:, 1], color="gray", alpha=0.65, linewidth=1.0, label="centerline")
    ax.plot(
        raceline[:, 0],
        raceline[:, 1],
        color="dodgerblue",
        alpha=0.65,
        linewidth=1.0,
        label="raceline",
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8)


def init_net_axis(ax):
    labels = IN_LABELS + ["|"] + OUT_LABELS
    ax.clear()
    ax.set_title("Network Activity")
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    bars = ax.bar(range(len(labels)), [0.0] * len(labels), color="skyblue")
    bars[len(IN_LABELS)].set_color("lightgray")
    for i in range(len(IN_LABELS) + 1, len(labels)):
        bars[i].set_color("salmon")
    return bars


def animate_replay(
    env: CarEnv,
    path: np.ndarray,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    title: str,
    pause: float,
    window_m: float,
    zoom: bool,
    save_video: Path | None,
    video_fps: int,
    video_stride: int,
    show: bool,
) -> None:
    fig, (ax_track, ax_net) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title)
    init_track_axis(ax_track, env, "Track Replay")
    bars = init_net_axis(ax_net)

    (trail,) = ax_track.plot([], [], color="crimson", linewidth=2.2, label="trajectory")
    (car,) = ax_track.plot([], [], marker="o", color="red", markersize=6, label="car")
    ax_track.legend(loc="upper right", fontsize=8)

    if len(path) > 1:
        ax_track.plot(path[:, 0], path[:, 1], color="crimson", alpha=0.18, linewidth=1.0)

    writer = None
    save_context = None
    if save_video is not None:
        save_video = save_video.expanduser().resolve()
        save_video.parent.mkdir(parents=True, exist_ok=True)
        suffix = save_video.suffix.lower()
        if suffix == ".gif":
            writer = animation.PillowWriter(fps=video_fps)
        elif suffix in {".mp4", ".m4v", ".mov"}:
            if not animation.writers.is_available("ffmpeg"):
                raise RuntimeError(
                    "MP4/MOV export requires ffmpeg, but matplotlib cannot find it. "
                    "Use a .gif path or install ffmpeg."
                )
            writer = animation.FFMpegWriter(fps=video_fps, bitrate=1800)
        else:
            raise ValueError("Video path must end with .gif, .mp4, .m4v, or .mov")
        save_context = writer.saving(fig, str(save_video), dpi=120)
        save_context.__enter__()
        print(f"Saving replay video to: {save_video}")

    step_count = min(len(observations), max(0, len(path) - 1))
    stride = max(1, int(video_stride))
    progress_bar = None
    if writer is not None:
        progress_bar = tqdm(range(step_count), desc="Saving video", unit="frame")
        frame_iter = progress_bar
    else:
        frame_iter = range(step_count)
    try:
        for i in frame_iter:
            partial = path[: i + 2]
            trail.set_data(partial[:, 0], partial[:, 1])
            car.set_data([path[i + 1, 0]], [path[i + 1, 1]])

            values = observations[i].tolist() + [0.0] + actions[i].tolist()
            for bar, value in zip(bars, values):
                bar.set_height(float(value))

            ax_track.set_title(
                f"Track Replay | step {i + 1}/{step_count} | "
                f"speed {env.min_speed:.1f}-{env.max_speed:.1f} m/s"
            )
            if zoom:
                x_c, y_c = path[i + 1]
                ax_track.set_xlim(x_c - window_m, x_c + window_m)
                ax_track.set_ylim(y_c - window_m, y_c + window_m)

            fig.tight_layout()
            fig.canvas.draw_idle()
            if writer is not None and (i % stride == 0 or i == step_count - 1):
                writer.grab_frame()
            if show:
                plt.pause(pause)
            else:
                fig.canvas.draw()
    finally:
        if save_context is not None:
            save_context.__exit__(None, None, None)
        if progress_bar is not None:
            progress_bar.close()

    if not zoom:
        ax_track.autoscale()
        ax_track.set_aspect("equal", adjustable="box")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a saved NEAT racetrack model pickle with track and net animation."
    )
    parser.add_argument("model_path", nargs="?", type=Path, help="Path to generation_XXXX_best.pkl")
    parser.add_argument("--model", type=Path, default=None, help="Path to generation_XXXX_best.pkl")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
        help="Directory searched recursively when no model path is provided.",
    )
    parser.add_argument("--track-dir", type=Path, default=None, help="Override track folder from the pkl.")
    parser.add_argument("--difficulty", type=str, default=None, choices=["normal", "easy", "very-easy", "overfit", "ideal"])
    parser.add_argument("--follow-line", type=str, default=None, choices=["centerline", "raceline"])
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override replay step budget. By default, use max_steps saved in the pkl.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=45)
    parser.add_argument("--checkpoint-pass-reward", type=float, default=72.0)
    parser.add_argument("--checkpoint-miss-penalty", type=float, default=48.0)
    parser.add_argument("--pause", type=float, default=0.01, help="Animation pause per step in seconds.")
    parser.add_argument("--window-m", type=float, default=18.0, help="Zoom window half-size in meters.")
    parser.add_argument("--no-zoom", action="store_true", help="Show the whole track instead of following the car.")
    parser.add_argument("--summary-only", action="store_true", help="Load and roll out the model without opening a plot window.")
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="Save replay frames to a video file. Use .gif without extra tools; .mp4 requires ffmpeg.",
    )
    parser.add_argument("--video-fps", type=int, default=30, help="Frames per second for saved video.")
    parser.add_argument(
        "--video-stride",
        type=int,
        default=1,
        help="Save every Nth replay frame. Increase this to make smaller/faster videos.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window after/during video export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_arg = args.model or args.model_path
    if model_arg is None:
        model_path = find_latest_model(args.models_dir.expanduser().resolve())
        print(f"No model path provided; using latest model under {args.models_dir}:")
    else:
        model_path = model_arg.expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")

    payload = load_payload(model_path)
    env = build_env(payload, args)
    net = neat.nn.FeedForwardNetwork.create(payload["genome"], payload["config"])
    path, observations, actions, fitness = rollout_trace(env, net)

    generation = payload.get("generation", "?")
    saved_fitness = payload.get("fitness", None)
    print(f"Loaded model: {model_path}")
    print(f"Generation: {generation}")
    if saved_fitness is not None:
        print(f"Saved fitness: {float(saved_fitness):.3f}")
    print(f"Replay fitness: {fitness:.3f}")
    print(f"Steps: {len(observations)} / budget {env.max_steps}")
    print(f"Follow line: {env.follow_line}")
    print(f"Crashed: {env.crashed}")
    if args.summary_only:
        return

    title = f"Model Replay | gen {generation} | fitness {fitness:.3f}"
    animate_replay(
        env,
        path,
        observations,
        actions,
        title=title,
        pause=args.pause,
        window_m=args.window_m,
        zoom=not args.no_zoom,
        save_video=args.save_video,
        video_fps=args.video_fps,
        video_stride=args.video_stride,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
