import os
from tqdm import tqdm
import lmdb
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import safetensors.numpy as stnp
from pathlib import Path

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]


def load_data(input_data_path: str, output_lmdb_path: str, output_parquet_path: str):
    """
        Convert the given datasets to lmdb and parquet format.

        :param input_data_path: path to the source dataset
        :param output_lmdb_path: path to the destination lmdb file
        :param output_parquet_path: path to the destination  parquet file
        :return: None
        """
    input_data_path = Path(input_data_path)
    city_dirs = [city for city in input_data_path.iterdir() if city.is_dir()]
    city_to_files_paths = {city.name: [city.joinpath(files) for files in os.listdir(city) if files.endswith(".tif")] for city in city_dirs}
    if not os.path.exists(output_lmdb_path) or not os.path.exists(output_parquet_path):
        print("\nCreating LMDB...")
        keys = create_lmdb(city_to_files_paths, output_lmdb_path)
        print("\nCreating Metadata...")
        metadata = create_metadata(keys)
        metadata.to_parquet(output_parquet_path)


def create_metadata(keys):
    """
    Create metadata for the dataset.
    :param keys: list of keys for the patches
    :return: metadata dataframe
    """
    rows = []
    for f in tqdm(keys, desc="Building metadata rows", unit="patch"):
        city = f.split("_")[0]
        patch_idx = int(f.split("_")[1])
        rows.append({
            "patch_id": f,
            "city": city,
            "patch_idx": patch_idx
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["city", "patch_idx"]).reset_index(drop=True)

    N = len(df)
    n_train = int(0.7 * N)
    n_val = int(0.15 * N)
    splits = (
            ["train"] * n_train
            + ["validation"] * n_val
            + ["test"] * (N - n_train - n_val)
    )
    df["split"] = splits

    df["labels"] = df["city"].apply(lambda c: [c])

    return df[["patch_id", "split", "labels"]]


def reproject_to_reference(src_path: Path, dst_path: Path, ref_path: Path):
    """
    Reproject & resample src_path to have exactly the same
    CRS, transform, width & height as ref_path, saving to dst_path.
    """
    os.makedirs(dst_path.parent, exist_ok=True)
    with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
        transform, width, height = ref.transform, ref.width, ref.height
        kwargs = src.meta.copy()
        kwargs.update({
            "crs":       ref.crs,
            "transform": transform,
            "width":     width,
            "height":    height,
            "compress": "LZW",
            "predictor": 2,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "photometric": "RGB",
            "interleave": "PIXEL",
        })
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            dst.colorinterp = src.colorinterp
            for band in range(1, src.count + 1):
                reproject(
                    source      = rasterio.band(src, band),
                    destination = rasterio.band(dst, band),
                    src_transform = src.transform,
                    src_crs       = src.crs,
                    dst_transform = transform,
                    dst_crs       = ref.crs,
                    resampling    = Resampling.bilinear
                )


def create_lmdb(city_to_files_paths, output_lmdb_path, patch_size=64, stride=32, map_size=6.5e10):
    """
    Create a single LMDB with patches from all cities.
    :param city_to_files_paths: dict mapping city name -> dict with keys
                                'LCZ', 'PRISMA_30', 'S2' and Path values
    :param output_lmdb_path: directory for the LMDB
    :param patch_size: size of square patch (pixels)
    :param stride: step between patches (pixels)
    :param map_size: max size in bytes for LMDB
    :return: list of keys written
    """
    keys = []
    env = lmdb.open(output_lmdb_path, map_size=int(map_size))
    with env.begin(write=True) as txn:
        for city, paths in city_to_files_paths.items():
            print(f"\n▶ Processing city: {city}")
            prisma_src = Path(paths[1])
            prisma_res = prisma_src.parent / f"{prisma_src.stem}_toS2.tif"
            if not prisma_res.exists():
                reproject_to_reference(prisma_src, prisma_res, paths[2])

            with rasterio.open(paths[0]) as src_lcz, \
                 rasterio.open(prisma_res) as src_pr, \
                 rasterio.open(paths[2]) as src_s2:
                H, W = src_lcz.height, src_lcz.width
                coords = [
                    (r, c)
                    for r in range(0, H - patch_size + 1, stride)
                    for c in range(0, W - patch_size + 1, stride)
                ]
                for idx, (row_off, col_off) in enumerate(
                        tqdm(coords, desc=f"{city} patches", unit="patch")):
                    window = rasterio.windows.Window(col_off, row_off,
                                                     patch_size, patch_size)
                    s2_patch = src_s2.read(window=window)
                    pr_patch = src_pr.read(window=window)
                    lcz_patch = src_lcz.read(1, window=window)

                    x = np.concatenate([s2_patch, pr_patch], axis=0)
                    y = lcz_patch

                    key = f"{city}_{idx:06d}_{row_off}_{col_off}"
                    keys.append(key)
                    sample = {'data': x.astype(np.float32),
                              'label': y.astype(np.int64)}
                    txn.put(key.encode('ascii'), stnp.save(sample))
                    idx += 1
    env.close()
    return keys
