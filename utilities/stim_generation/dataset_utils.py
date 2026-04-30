from torch.utils.data import Subset, DataLoader
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset

def split_by_source_image(
    dataset,
    train_frac=0.8,
    val_frac=0.2,
    seed=0,
):
    assert abs(train_frac + val_frac - 1.0) < 1e-8

    # Unique image identities: ("frozen", image_idx), ("running", image_idx)
    image_keys = sorted({
        (meta["source"], meta["image_idx"])
        for meta in dataset.index
    })

    rng = np.random.default_rng(seed)
    image_keys = np.array(image_keys, dtype=object)
    rng.shuffle(image_keys)

    n_total = len(image_keys)
    n_train = int(round(train_frac * n_total))

    train_keys = set(map(tuple, image_keys[:n_train]))
    val_keys = set(map(tuple, image_keys[n_train:]))

    train_indices = []
    val_indices = []

    for i, meta in enumerate(dataset.index):
        key = (meta["source"], meta["image_idx"])

        if key in train_keys:
            train_indices.append(i)
        elif key in val_keys:
            val_indices.append(i)
        else:
            raise RuntimeError("Sample did not match train or val split.")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset, train_keys, val_keys

def prerender_dataset_to_single_pt(
    dataset,
    indices,
    out_path,
    movie_dtype=torch.float16,
    overwrite=False,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        print(f"Cache already exists, skipping render: {out_path}")
        return torch.load(out_path, map_location="cpu")

    movies = []
    scene_vels = []
    scene_poss = []
    meta = []

    for idx in tqdm(indices, desc=f"Pre-rendering {out_path.name}"):
        sample = dataset[idx]

        movies.append(sample["movie"].to(dtype=movie_dtype).cpu())
        scene_vels.append(sample["scene_vel"].to(dtype=torch.float32).cpu())
        scene_poss.append(sample["scene_pos"].to(dtype=torch.float32).cpu())

        meta.append({
            "orig_idx": int(idx),
            "source": sample["source"],
            "image_idx": int(sample["image_idx"]),
        })

    cache = {
        "movie": torch.stack(movies, dim=0),
        "scene_vel": torch.stack(scene_vels, dim=0),
        "scene_pos": torch.stack(scene_poss, dim=0),
        "meta": meta,
    }

    torch.save(cache, out_path)
    print(f"Saved cache to: {out_path}")

    return cache

class TensorMovieDataset(Dataset):
    def __init__(self, path, movie_dtype=torch.float32):
        cache = torch.load(path, map_location="cpu")

        self.movies = cache["movie"]
        self.scene_vel = cache["scene_vel"]
        self.scene_pos = cache["scene_pos"]
        self.meta = cache["meta"]
        self.movie_dtype = movie_dtype

    def __len__(self):
        return self.movies.shape[0]

    def __getitem__(self, idx):
        return {
            "movie": self.movies[idx].to(dtype=self.movie_dtype),
            "scene_vel": self.scene_vel[idx],
            "scene_pos": self.scene_pos[idx],
            "source": self.meta[idx]["source"],
            "image_idx": self.meta[idx]["image_idx"],
            "orig_idx": self.meta[idx]["orig_idx"],
        }