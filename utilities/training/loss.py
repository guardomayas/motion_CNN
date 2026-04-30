import torch

def direct_velocity_loss(
    pred_vel_z,
    target_vel,
    vel_mean,
    vel_std,
    gray_frames,
):
    """
    pred_vel_z : (B, T, 2)
        Predicted normalized velocity.

    target_vel : (B, T, 2)
        Target velocity in raw units, e.g. px/s.
    """
    target_vel_z = (target_vel - vel_mean) / vel_std

    loss = (
        (pred_vel_z[:, gray_frames:] - target_vel_z[:, gray_frames:]) ** 2
    ).mean()

    return loss

def compute_velocity_stats(dataset, indices=None, device="cpu"):
    """
    Fast velocity mean/std for CorrelationVelDataset.
    Does not reconstruct movies.
    
    If using a torch.utils.data.Subset for training, pass train_dataset.indices.
    """
    if indices is None:
        indices = range(len(dataset.index))

    sum_v = torch.zeros(2, device=device)
    sum_v2 = torch.zeros(2, device=device)
    n = 0

    for i in indices:
        meta = dataset.index[i]
        seed = meta["seed"]

        vel, pos = dataset.generate_velocity_trace_2d(seed=seed)

        # scene velocity is negative fixation/camera velocity
        scene_vel = -torch.from_numpy(vel).to(device)

        # no gray frames here, because generated vel is only real frames
        v = scene_vel.reshape(-1, 2)

        sum_v += v.sum(dim=0)
        sum_v2 += (v ** 2).sum(dim=0)
        n += v.shape[0]

    mean = sum_v / n
    var = sum_v2 / n - mean ** 2
    std = torch.sqrt(var + 1e-8)

    return mean, std