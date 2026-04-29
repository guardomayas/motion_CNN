import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize


class FixationMovieDataset(Dataset):
    def __init__(
        self,
        file,
        screen_size=(600, 800),
        crop_hw=(400, 400),
        downsample_hw=(96, 96),
        frames_per_segment=75,
        fps=75,
        gray_sec=0.2,
        gray_value=0.5,
        include_frozen=True,
        include_running=True,
        running_runs=None,
        normalize_images=True,
        movie_format="TCHW",
        dtype=torch.float32,
    ):
        super().__init__()

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

        # Open once. This is convenient for now.
        # If dataloader workers later complain, we can switch to lazy file opening per worker.
        self._h5 = h5py.File(self.file, "r")

        # Load arrays into memory. Images are not huge enough to be crazy here, and the
        # reconstructed movies are the expensive part.
        self.frozen_images = np.array(self._h5["frozenImages"]).transpose((0, 2, 1)).astype(np.float32)
        self.frozen_fixations = np.array(self._h5["frozenfixations"]).astype(np.float32)
        self.running_images = np.array(self._h5["runningImages"]).transpose((0, 2, 1)).astype(np.float32)
        self.running_fixations = np.array(self._h5["runningfixations"]).astype(np.float32)

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
        if self.include_frozen:
            n_segments = self.frozen_fixations.shape[0] // self.frames_per_segment
            for seg_idx in range(n_segments):
                start = seg_idx * self.frames_per_segment
                end = start + self.frames_per_segment
                self.index.append({
                    "source": "frozen",
                    "run": -1,
                    "segment_idx": seg_idx,
                    "start": start,
                    "end": end,
                })

        if self.include_running:
            n_runs = self.running_fixations.shape[0]
            runs_to_use = range(n_runs) if self.running_runs is None else self.running_runs

            for run_idx in runs_to_use:
                run_fix = self.running_fixations[run_idx]
                n_segments = run_fix.shape[0] // self.frames_per_segment
                for seg_idx in range(n_segments):
                    start = seg_idx * self.frames_per_segment
                    end = start + self.frames_per_segment
                    self.index.append({
                        "source": "running",
                        "run": int(run_idx),
                        "segment_idx": seg_idx,
                        "start": start,
                        "end": end,
                    })

    # ============================================================
    # Geometry helpers
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
    # Reconstruction helpers
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

    @staticmethod
    def center_trace(fixations):
        pos = fixations[:, 1:3].astype(np.float32)
        pos = pos.copy()
        pos -= pos[0]
        return pos

    def compute_velocity(self, pos):
        dt = 1.0 / self.fps
        vel = np.zeros_like(pos, dtype=np.float32)
        vel[1:] = np.diff(pos, axis=0) / dt
        return vel

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

    # ============================================================
    # Segment selection
    # ============================================================
    def _get_segment_fixations(self, meta):
        start, end = meta["start"], meta["end"]

        if meta["source"] == "frozen":
            seg = self.frozen_fixations[start:end]
            image_stack = self.frozen_images
        else:
            seg = self.running_fixations[meta["run"]][start:end]
            image_stack = self.running_images

        return seg.astype(np.float32), image_stack

    # ============================================================
    # Dataset interface
    # ============================================================
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        meta = self.index[idx]
        seg, image_stack = self._get_segment_fixations(meta)

        # reconstruct movie
        movie = self.return_fix_movie(image_stack, seg)   # (H, W, T)

        # traces
        fix_pos = self.center_trace(seg)                      # (T, 2)
        fix_vel = self.compute_velocity(fix_pos)                  # (T, 2)

        # crop
        movie = self.center_crop_movie(movie)             # (Hc, Wc, T)

        #down sample
        movie = self.downsample_movie(movie, target_hw=self.downsample_hw)              # (Hd, Wd, T)
    
        # prepend gray
        movie = self.add_gray_padding(movie)              # (Hc, Wc, T_new)
        fix_pos = self.pad_position_trace(fix_pos)                # (T_new, 2)
        fix_vel = self.pad_velocity_trace(fix_vel)                # (T_new, 2)
        seg = self.pad_fixations_pre(seg)                 # (T_new, 3)

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
            "movie": torch.tensor(movie, dtype=self.dtype),
            "fix_pos": torch.tensor(fix_pos, dtype=self.dtype), #flip sign to convert from
            "fix_vel": torch.tensor(fix_vel, dtype=self.dtype), # fixation-centered to screen-centered coordinates
            #TODO: keep both fixation and movie coords.
                    # fix_pos = self.center_trace(seg)
                    # fix_vel = self.compute_velocity(fix_pos)

            "scene_pos" : torch.tensor(-fix_pos, dtype=self.dtype),
            "scene_vel" : torch.tensor(-fix_vel, dtype=self.dtype),
            "fixations": torch.tensor(seg, dtype=self.dtype),
            "source": meta["source"],
            "run": meta["run"],
            "segment_idx": meta["segment_idx"],
        }
        return sample

    def close(self):
        if hasattr(self, "_h5") and self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None

    def __del__(self):
        self.close()