from torch.utils.data import Dataset
from skimage.transform import resize
import torch
import numpy as np
import h5py

class CorrelationVelDataset(Dataset):
    def __init__(
        self,
        file, 
        screen_size=(600, 800),
        crop_hw = (400, 400),
        downsample_hw = (96, 96),
        frames_per_segment = 75,
        fps = 75,   
        gray_sec = 0.2, 
        gray_value = 0.5,
        samples_per_image = 6,
        vel_half_life_s = 0.2, ## Velocity trace params
        vel_std = (500, 500),
        vel_corr = 0.0,
        random_seed = 0,
        include_frozen=True,
        include_running=True,
        running_runs=None,
        normalize_images=True,
        movie_format="TCHW",
        dtype=torch.float32,
        ):
        
        self.file = file
        self.screen_size = screen_size
        self.crop_hw = crop_hw
        self.downsample_hw = downsample_hw
        self.frames_per_segment = frames_per_segment
        self.fps = fps
        self.gray_sec = gray_sec
        self.gray_frames = int(round(gray_sec * fps))
        self.gray_value = gray_value
        self.include_frozen = include_frozen
        self.include_running = include_running
        self.running_runs = running_runs
        self.normalize_images = normalize_images
        self.movie_format = movie_format
        self.dtype = dtype
        self.vel_half_life_s = vel_half_life_s
        self.vel_std = vel_std
        self.vel_corr = vel_corr
        self.random_seed = random_seed
        self.samples_per_image = samples_per_image

        # Open once. This is convenient for now.
        # If dataloader workers later complain, we can switch to lazy file opening per worker.
        with h5py.File(self.file, "r") as h5:
            self.frozen_images = (
                np.array(h5["frozenImages"])
                .transpose((0, 2, 1))
                .astype(np.float32)
            )

            self.running_images = (
                np.array(h5["runningImages"])
                .transpose((0, 2, 1))
                .astype(np.float32)
            )

        self._h5 = None
        # Global normalization
        if self.normalize_images:
            global_min = min(self.frozen_images.min(), self.running_images.min())
            global_max = max(self.frozen_images.max(), self.running_images.max())
            scale = global_max - global_min + 1e-8

            self.frozen_images = (self.frozen_images - global_min) / scale
            self.running_images = (self.running_images - global_min) / scale

        # Build sample index
        self.index = []
        self._build_index()
    
    # ============================================================
    # Index building
    # ============================================================
    
    def _build_index(self):
        self.index = []
        sample_id = 0

        if self.include_frozen:
            n_images = self.frozen_images.shape[0]

            for image_idx in range(n_images):
                for rep in range(self.samples_per_image):
                    self.index.append({
                        "source": "frozen",
                        "image_idx": int(image_idx),
                        "rep": int(rep),
                        "sample_idx": sample_id,
                        "seed": self.random_seed + sample_id,
                    })
                    sample_id += 1

        if self.include_running:
            n_images = self.running_images.shape[0]

            for image_idx in range(n_images):
                for rep in range(self.samples_per_image):
                    self.index.append({
                        "source": "running",
                        "image_idx": int(image_idx),
                        "rep": int(rep),
                        "sample_idx": sample_id,
                        "seed": self.random_seed + sample_id,
                    })
                    sample_id += 1
                    
    # ============================================================
    # Velocity trace generation. 2D OU process with configurable correlation between x and y.
    # ============================================================
    
    def generate_velocity_trace_2d(
        self, seed):
        """
        Generate smooth 2D velocity traces vx(t), vy(t).

        Parameters
        ----------
        total_time_s : float
            Duration in seconds.
        sample_freq_hz : float
            Sampling frequency in Hz.
        half_life_s : float
            Temporal smoothness scale. Larger = smoother velocity.
        vel_std : float
            Stationary standard deviation of each velocity component,
            in pix/second.
        seed : int or None
            Random seed.

        Returns
        -------
        vel : ndarray, shape (T, 2)
            vel[:, 0] is vx, vel[:, 1] is vy.
        """

        T = self.frames_per_segment
        dt = 1.0 / self.fps

        tau = self.vel_half_life_s / np.log(2)
        alpha = np.exp(-dt / tau)
        
        # Allow scalar or tuple vel_std
        if np.isscalar(self.vel_std):
            sx = sy = float(self.vel_std)
        else:
            sx, sy = map(float, self.vel_std)

        rho = float(self.vel_corr)
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("vel_corr must be between -1 and 1.")

        cov = np.array([
            [sx**2, rho * sx * sy],
            [rho * sx * sy, sy**2],
        ], dtype=np.float64)

        rng = np.random.default_rng(seed)
        
        noise = rng.multivariate_normal(
            mean=np.zeros(2),
            cov=cov,
            size=T,
        ).astype(np.float32)

        vel = np.zeros((T, 2), dtype=np.float32)

        for t in range(1, T):
            vel[t] = alpha * vel[t - 1] + np.sqrt(1 - alpha**2) * noise[t]

        pos = np.cumsum(vel, axis=0) * dt
        pos -= pos[0]

        # if clip_position_px is not None:
        #     pos = np.clip(
        #         pos,
        #         -float(clip_position_px),
        #         float(clip_position_px),
        #     )

        #     # Recompute velocity after clipping, so pos and vel remain consistent
        #     vel = compute_velocity(pos)

        return vel.astype(np.float32), pos.astype(np.float32)
    
    
    # ============================================================
    # Movie construction helpers
    # ============================================================
    def return_fix_movie(self, image_stack, fixations):
        screen_h, screen_w = self.screen_size
        n_images, img_h, img_w = image_stack.shape
        n_frames = fixations.shape[0]

        # movie = np.zeros((screen_h, screen_w, n_frames), dtype=np.float32)
        movie = np.full(
                (screen_h, screen_w, n_frames),
                self.gray_value,
                dtype=np.float32
            )
        for i in range(n_frames):
            img_idx = int(fixations[i, 0]) - 1  # MATLAB -> Python indexing
            trX = float(fixations[i, 1])
            trY = float(fixations[i, 2])

            if img_idx < 0 or img_idx >= n_images:
                continue

            xdst, ydst, xsrc, ysrc = self.get_ranges(
                trX, trY, screen_w, screen_h, img_w, img_h
            )

            patch = image_stack[img_idx][np.ix_(ysrc, xsrc)]
            movie[np.ix_(ydst, xdst, [i])] = patch[:, :, None]

        return movie

    # @staticmethod
    # def center_trace(fixations):
    #     pos = fixations[:, 1:3].astype(np.float32)
    #     pos = pos.copy()
    #     pos -= pos[0]
    #     return pos

    # def compute_velocity(self, pos):
    #     dt = 1.0 / self.fps
    #     vel = np.zeros_like(pos, dtype=np.float32)
    #     vel[1:] = np.diff(pos, axis=0) / dt
    #     return vel

    def center_crop_movie(self, movie):
        H, W, T = movie.shape
        crop_h, crop_w = self.crop_hw
        y0 = (H - crop_h) // 2
        x0 = (W - crop_w) // 2
        return movie[y0:y0 + crop_h, x0:x0 + crop_w, :]

    def add_gray_padding(self, movie):
        H, W, _ = movie.shape
        gray_block = np.full(
            (H, W, self.gray_frames),
            self.gray_value,
            dtype=movie.dtype
        )
        return np.concatenate([gray_block, movie], axis=2)

    def pad_position_trace(self, pos):
        pre = np.repeat(pos[:1], self.gray_frames, axis=0)
        return np.concatenate([pre, pos], axis=0)

    def pad_velocity_trace(self, vel):
        pre = np.zeros((self.gray_frames, vel.shape[1]), dtype=vel.dtype)
        return np.concatenate([pre, vel], axis=0)

    def pad_fixations_pre(self, fixations):
        pre = np.repeat(fixations[:1], self.gray_frames, axis=0)
        return np.concatenate([pre, fixations], axis=0)
    
    @staticmethod
    def downsample_movie(movie, target_hw=(96, 96)):
        H2, W2 = target_hw
        T = movie.shape[2]

        small = np.empty((H2, W2, T), dtype=np.float32)
        for t in range(T):
            small[:, :, t] = resize(
                movie[:, :, t],
                (H2, W2),
                order=1,
                anti_aliasing=True,
                preserve_range=True,
            ).astype(np.float32)

        return small

    # # ============================================================
    # # Segment selection
    # # ============================================================
    # def _get_segment_fixations(self, meta):
    #     start, end = meta["start"], meta["end"]

    #     if meta["source"] == "frozen":
    #         seg = self.frozen_fixations[start:end]
    #         image_stack = self.frozen_images
    #     else:
    #         seg = self.running_fixations[meta["run"]][start:end]
    #         image_stack = self.running_images

    #     return seg.astype(np.float32), image_stack

    # ============================================================
    # Geometry helper
    # ============================================================
    @staticmethod
    def get_ranges(trX, trY, screen_w, screen_h, img_w, img_h):
        ymin = (screen_h / 2) - trY + 1
        ymax = img_h + (screen_h / 2) - trY
        xmin = (screen_w / 2) - trX + 1
        xmax = img_w + (screen_w / 2) - trX

        ymin = max(1, ymin)
        xmin = max(1, xmin)
        ymax = min(screen_h, ymax)
        xmax = min(screen_w, xmax)

        rx = np.arange(1, screen_w + 1)
        ry = np.arange(1, screen_h + 1)

        rxuse = (rx >= xmin) & (rx <= xmax)
        ryuse = (ry >= ymin) & (ry <= ymax)

        xsrc = trX - (screen_w / 2) + rx[rxuse]
        ysrc = trY - (screen_h / 2) + ry[ryuse]

        xsrc = np.round(xsrc).astype(int) - 1
        ysrc = np.round(ysrc).astype(int) - 1

        xdst = np.where(rxuse)[0]
        ydst = np.where(ryuse)[0]

        return xdst, ydst, xsrc, ysrc
    # ============================================================
    # Dataset interface
    # ============================================================
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        meta = self.index[idx]
        
        if meta["source"] == "frozen":
            image_stack = self.frozen_images
        else:
            image_stack = self.running_images
                
        image_idx = meta["image_idx"]
        seed = meta["seed"]
        # total_time_s = self.frames_per_segment / self.fps
        vel, pos = self.generate_velocity_trace_2d(seed=seed)
        
        vel = vel[:self.frames_per_segment]
        pos = pos[:self.frames_per_segment]
        
        img_h, img_w = image_stack.shape[1:]

        center_x = img_w / 2
        center_y = img_h / 2

        fixations = np.zeros((self.frames_per_segment, 3), dtype=np.float32)
        fixations[:, 0] = image_idx + 1  # MATLAB-style image index
        fixations[:, 1] = center_x + pos[:, 0]
        fixations[:, 2] = center_y + pos[:, 1]
        
        movie = self.return_fix_movie(image_stack, fixations)
        movie = self.center_crop_movie(movie)
        movie = self.downsample_movie(movie, target_hw=self.downsample_hw)
        movie = self.add_gray_padding(movie)
        pos =   self.pad_position_trace(pos)
        vel =   self.pad_velocity_trace(vel)
        fixations =   self.pad_fixations_pre(fixations)
        
        
        # convert movie format
        if self.movie_format == "TCHW":
            # (H, W, T) -> (T, H, W) -> (T, 1, H, W)
            movie = np.moveaxis(movie, 2, 0)
            movie = movie[:, None, :, :]
        elif self.movie_format == "HWT":
            pass
        else:
            raise ValueError(f"Unknown movie_format: {self.movie_format}")

        sample = {
            "movie": torch.from_numpy(movie).to(dtype=self.dtype),
            "fix_pos": torch.from_numpy(pos).to(dtype=self.dtype),
            "fix_vel": torch.from_numpy(vel).to(dtype=self.dtype),

            "scene_pos": torch.from_numpy(-pos).to(dtype=self.dtype),
            "scene_vel": torch.from_numpy(vel).to(dtype=self.dtype),
            "fixations": torch.from_numpy(fixations).to(dtype=self.dtype),
            "source": meta["source"],
            "image_idx": meta["image_idx"],       
        }
        return sample
        