"""Initialise, run, save, and optionally render one Mouse-network simulation.

MP4 export requires Body.render_mp4() from models.eigenmode_mjx.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import brainpy as bp
import brainpy.math as bm
import imageio.v2 as imageio

from models.mouse.network_body import Mouse


DEFAULT_DT = 0.025  # ms
DEFAULT_SEED = 0
DEFAULT_ANGLE_MU = -1.3
DEFAULT_ANGLE_AMP = 0.5


def make_monitors(net: Mouse) -> dict[str, Any]:
    """Variables retained by BrainPy during a run."""
    return {
        "pc_input": net.pc.input,
        "pc_spike": net.pc.spike,
        "cn_spike": net.cn.spike,
        "io_v_soma": net.io.neurons.V_soma,
        "body": net.body.state,
    }


def initialise_network(
    delta: Sequence[int | float],
    *,
    seed: int = DEFAULT_SEED,
    dt: float = DEFAULT_DT,
    angle_mu: float = DEFAULT_ANGLE_MU,
    angle_amp: float = DEFAULT_ANGLE_AMP,
    pf_to_pc: bool = True,
    jit: bool = True,
) -> tuple[Mouse, bp.DSRunner]:
    """Set seeds, construct the model, and make a runner without advancing time."""
    np.random.seed(seed)
    bm.random.seed(seed)

    net = Mouse(
        delta=list(delta),
        angle_mu=angle_mu,
        angle_amp=angle_amp,
        pf_to_pc=pf_to_pc,
    )
    runner = bp.DSRunner(
        net,
        monitors=make_monitors(net),
        dt=dt,
        jit=jit,
        progress_bar=False,
    )
    return net, runner


def run_network(runner: bp.DSRunner, duration: float) -> bp.DSRunner:
    """Advance an existing simulation for ``duration`` milliseconds."""
    if duration <= 0:
        raise ValueError("duration must be positive.")
    runner.run(duration)
    return runner


def _run_filename(run_id: int | str, suffix: str) -> str:
    """Create a stable filename from a numerical or descriptive run identifier."""
    stem = f"run_{run_id:04d}" if isinstance(run_id, int) else f"run_{run_id}"
    return f"{stem}{suffix}"


def save_run(
    runner: bp.DSRunner,
    delta: Sequence[int | float],
    *,
    run_id: int | str,
    output_dir: str | Path = "out",
) -> Path:
    """Save essential monitored arrays in a compressed NumPy archive."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _run_filename(run_id, ".npz")
    body_state = runner.mon["body"]

    np.savez_compressed(
        output_path,
        delta=np.asarray(delta),
        dt=runner.dt,
        pc_spike=np.asarray(runner.mon["pc_spike"]),
        pc_input=np.asarray(runner.mon["pc_input"]),
        cn_spike=np.asarray(runner.mon["cn_spike"]),
        io_v_soma=np.asarray(runner.mon["io_v_soma"]),
        q=np.asarray(body_state["qpos"]),
        qd=np.asarray(body_state["qvel"]),
    )
    return output_path

def render_mp4(self,mon,filename,*,fps=30,playback_speed=1.0,width=640,height=480,):
    """Render a monitored body trajectory to an H.264 MP4.

    Parameters
    ----------
    mon
        The monitored body-state dictionary: runner.mon["body"].
    filename
        Output MP4 path.
    fps
        Video frame rate.
    playback_speed
        1.0 is approximately real-time simulation playback.
        4.0 makes the video four times faster.
    width, height
        Output dimensions in pixels. H.264 requires even dimensions.
    """
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if playback_speed <= 0:
        raise ValueError("playback_speed must be positive.")

    # H.264/yuv420p requires even dimensions.
    width = width - (width % 2)
    height = height - (height % 2)

    qpos = np.asarray(mon["qpos"])
    qvel = np.asarray(mon["qvel"])

    if qpos.shape[0] == 0:
        raise ValueError("Cannot render an empty body trajectory.")

    # self.dt is in ms. Select simulation states so the MP4 has the
    # requested playback speed at the requested frame rate.
    frame_stride = max(
        1,
        round(1000.0 * playback_speed / (self.dt * fps)),
    )
    frame_indices = np.arange(0, qpos.shape[0], frame_stride)

    # Always include the final simulation state.
    if frame_indices[-1] != qpos.shape[0] - 1:
        frame_indices = np.append(frame_indices, qpos.shape[0] - 1)

    mj_data = mujoco.MjData(self.mj_model)
    renderer = mujoco.Renderer(
        self.mj_model,
        width=width,
        height=height,
    )

    n_written = 0
    try:
        with imageio.get_writer(
            str(filename),
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        ) as writer:
            for frame_idx in frame_indices:
                frame_qpos = qpos[frame_idx]

                # Do not let an unstable/NaN trajectory break encoding.
                if not np.all(np.isfinite(frame_qpos)):
                    continue

                mj_data.qpos[:] = frame_qpos
                mj_data.qvel[:] = qvel[frame_idx]
                mj_data.time = frame_idx * self.dt * 1e-3

                # Recompute body and geom positions from qpos/qvel.
                mujoco.mj_forward(self.mj_model, mj_data)

                # No named camera exists in mouse-free.xml, so this uses
                # MuJoCo's default free-camera view.
                renderer.update_scene(mj_data)
                writer.append_data(renderer.render())
                n_written += 1
    finally:
        renderer.close()

    if n_written == 0:
        raise RuntimeError("No valid frames were rendered; qpos may contain NaNs.")

    print(f"Saved MP4 ({n_written} frames): {filename}")



def render_video(
    net: Mouse,
    runner: bp.DSRunner,
    *,
    run_id: int | str,
    output_dir: str | Path = "out",
    fps: int = 30,
    playback_speed: float = 1.0,
    width: int = 640,
    height: int = 480,
) -> Path:
    """Render the monitored body trajectory to an H.264 MP4.

    ``playback_speed=1`` gives approximately real-time playback. For example,
    ``playback_speed=4`` turns four seconds of simulated motion into one
    second of video.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _run_filename(run_id, ".mp4")

   

    render_mp4(
        runner.mon["body"],
        filename=output_path,
        fps=fps,
        playback_speed=playback_speed,
        width=width,
        height=height,
    )
    return output_path


def init_and_run(
    delta: Sequence[int | float],
    *,
    duration: float = 10_000.0,
    seed: int = DEFAULT_SEED,
    dt: float = DEFAULT_DT,
    run_id: int | str | None = None,
    output_dir: str | Path = "out",
    save_data: bool = True,
    save_video: bool = False,
    video_fps: int = 30,
    video_playback_speed: float = 1.0,
    **network_kwargs: Any,
) -> tuple[Mouse, bp.DSRunner, Path | None, Path | None]:
    """Initialise and run one trial, then optionally save data and an MP4."""

    if (save_data or save_video) and run_id is None:
        raise ValueError("Provide run_id when saving data or video.")

    # Only genuine network/model parameters are passed here:
    net, runner = initialise_network(
        delta,
        seed=seed,
        dt=dt,
        **network_kwargs,
    )

    run_network(runner, duration)

    data_path = None
    video_path = None

    if save_data:
        data_path = save_run(
            runner,
            delta,
            run_id=run_id,
            output_dir=output_dir,
        )

    if save_video:
        video_path = render_video(
            net,
            runner,
            run_id=run_id,
            output_dir=output_dir,
            fps=video_fps,
            playback_speed=video_playback_speed,
        )

    return net, runner, data_path, video_path


if __name__ == "__main__":
    delta = [-1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1]

    _, _, data_file, video_file = init_and_run(
        delta,
        duration=5_000.0,
        seed=0,
        run_id=0,
        output_dir="out",
        save_video=True,
        video_fps=30,
    )
    print(f"Saved data:  {data_file}")
    print(f"Saved video: {video_file}")
