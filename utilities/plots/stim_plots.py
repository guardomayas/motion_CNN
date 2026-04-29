import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def show_batch_frames(batch, n_samples=4, frame_indices=None):
    """
    Show selected frames from several samples in the batch.

    batch["movie"]: (B, T, 1, H, W)
    """
    movies = batch["movie"]
    B, T, C, H, W = movies.shape

    n_samples = min(n_samples, B)

    if frame_indices is None:
        frame_indices = [0, min(14, T-1), min(T//2, T-1), T-1]
    else: 
        print("Using provided frame_indices:", frame_indices)
    n_rows = n_samples
    n_cols = len(frame_indices)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_samples):
        for j, t in enumerate(frame_indices):
            ax = axes[i, j]
            frame = movies[i, t, 0].detach().cpu().numpy()
            ax.imshow(frame, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Sample {i}, frame={t}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()
    

def show_one_sample_summary(dataset, sample_idx=0, frame_indices=None):
    """
    Visualize one sample:
    - first/mid/last movie frames
    - x,y position
    - vx,vy velocity
    """
    sample = dataset[sample_idx]
    movie = sample["movie"].detach().cpu().numpy()   # (T, 1, H, W) if TCHW
    pos = sample["fix_pos"].detach().cpu().numpy()       # (T, 2)
    vel = sample["scene_vel"].detach().cpu().numpy()       # (T, 2)

    T = movie.shape[0]
    if frame_indices is None:
        frame_indices = [0, min(T//2, T-1), T-1]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    # frames
    for j, t in enumerate(frame_indices):
        axes[0, j].imshow(movie[t, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(f"frame {t}")
        axes[0, j].axis("off")

    # position trace
    axes[1, 0].plot(pos[:, 0], label="x")
    axes[1, 0].plot(pos[:, 1], label="y")
    # axes[1, 0].invert_yaxis()  # invert y to match traj coordinates
    axes[1, 0].set_title("position")
    axes[1, 0].legend()
    # axes[1, 0].grid(True, alpha=0.3)

    # velocity trace
    axes[1, 1].plot(vel[:, 0], label="vx")
    axes[1, 1].plot(vel[:, 1], label="vy")
    axes[1, 1].set_title("velocity")
    axes[1,1].set_ylabel("pixels/sec")
    axes[1, 1].legend()
    # axes[1, 1].grid(True, alpha=0.3)

    # xy trajector
    axes[1, 2].plot(pos[:, 0], pos[:, 1], lw=1.5)
    axes[1, 2].scatter(pos[0, 0], pos[0, 1], label="start", s=40)
    axes[1, 2].scatter(pos[-1, 0], pos[-1, 1], label="end", s=40)
    axes[1, 2].set_title("trajectory")
    axes[1, 2].set_aspect("equal")
    axes[1, 2].legend()
    # axes[1, 2].grid(True, alpha=0.3)
    
    axes[1,2].set_xlim(-400, 400)
    axes[1,2].set_ylim(-400, 400)
    axes[1, 2].invert_yaxis()  # invert y to match traj coordinates

    plt.tight_layout()
    plt.show()
    
def sample_animation(dataset, idx=0, viz_fps=None, title=None, save_path=None):
    """
    Animate one sample from FixationMovieDataset.

    Parameters
    ----------
    dataset : FixationMovieDataset
    idx : int
        Dataset index
    fps : int or None
        If None, uses dataset.fps
    title : str or None
        Custom title
    save_path : str or None
        If given, save animation to mp4/gif

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    sample = dataset[idx]

    # pull arrays out of torch tensors
    movie = sample["movie"].detach().cpu().numpy()   # (T, 1, H, W) if TCHW
    pos = sample["fix_pos"].detach().cpu().numpy()       # (T, 2)
    vel = sample["scene_vel"].detach().cpu().numpy()       # (T, 2)


    # handle movie format
    if movie.ndim == 4:
        # (T, 1, H, W) -> (H, W, T)
        movie_hw_t = np.moveaxis(movie[:, 0, :, :], 0, 2)
    elif movie.ndim == 3:
        # assume already (H, W, T)
        movie_hw_t = movie
    else:
        raise ValueError(f"Unexpected movie shape: {movie.shape}")

    T = movie_hw_t.shape[2]
    t = np.arange(T) / dataset.fps

    source = sample.get("source", "unknown")
    run = sample.get("run", -1)
    segment_idx = sample.get("segment_idx", -1)

    fig = plt.figure(figsize=(9, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1.4], height_ratios=[1, 1])

    ax_img = fig.add_subplot(gs[:, 0])   # spans both rows
    ax_trace = fig.add_subplot(gs[0, 1]) # top-right
    ax_vel = fig.add_subplot(gs[1, 1])   # bottom-right

    im = ax_img.imshow(movie_hw_t[:, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
    ax_img.set_title(f"frame 1/{T}")
    ax_img.axis("off")

    ax_trace.plot(pos[:, 0], pos[:, 1], color="0.7", lw=1)
    trace_point, = ax_trace.plot(pos[0, 0], pos[0, 1], "ro")
    ax_trace.plot(pos[0, 0], pos[0, 1], "go", label="start")
    ax_trace.plot(pos[-1, 0], pos[-1, 1], "bx", label="end")
    ax_trace.set_title("Fixation trajectory")
    ax_trace.set_xlabel("x - x(0)")
    ax_trace.set_ylabel("y - y(0)")
    ax_trace.set_aspect("equal")
    ax_trace.legend()

    lim = max(np.max(np.abs(pos[:, 0])), np.max(np.abs(pos[:, 1])), 1.0) * 1.05
    ax_trace.set_xlim(-lim, lim)
    ax_trace.set_ylim(-lim, lim)
    ax_trace.invert_yaxis()

    ax_vel.plot(t, vel[:, 0], label="vx")
    ax_vel.plot(t, vel[:, 1], label="vy")
    vel_cursor = ax_vel.axvline(t[0], color="k", linestyle="--", alpha=0.7)
    ax_vel.set_title("Velocity traces")
    ax_vel.set_xlabel("time (s)")
    ax_vel.set_ylabel("velocity (px/s)")
    # ax_vel.grid(True, alpha=0.3)
    ax_vel.legend()

    plt.tight_layout()
    
    def update(frame):
        im.set_data(movie_hw_t[:, :, frame])
        ax_img.set_title(f"frame {frame + 1}/{T}")
        trace_point.set_data([pos[frame, 0]], [pos[frame, 1]])
        # pos_cursor.set_xdata([t[frame], t[frame]])
        vel_cursor.set_xdata([t[frame], t[frame]])
        return im, trace_point, vel_cursor

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / viz_fps, blit=False)

    if save_path is not None:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, fps=viz_fps, dpi=150)

    plt.close(fig)  # close the figure to avoid displaying static version in notebooks
    return anim