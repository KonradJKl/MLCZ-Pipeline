import os
from tqdm import tqdm
import lmdb
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds, Window
from rasterio.merge import merge
import gc
import safetensors.numpy as stnp
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings

# Band order of the S2 rasters (from the band descriptions of Milan's S2.tif, the other cities have none)
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def load_data(input_data_path: str, output_lmdb_path: str, output_parquet_path: str, stratify_by: str, split_strategy: str,
              map_size: float = 6.5e10):
    """
    Convert the given datasets to lmdb and parquet format, aligning the rasters of each city first.

    :param input_data_path: path to the source dataset
    :param output_lmdb_path: path to the destination lmdb file
    :param output_parquet_path: path to the destination parquet file
    :param stratify_by: 'city', 'labels', 'mixed', or 'none'
    :param split_strategy: 'city_based', 'patch_based', or 'hybrid'
    :param map_size: max size in bytes for the LMDB
    :return: None
    """
    input_data_path = Path(input_data_path)
    city_dirs = [city for city in input_data_path.iterdir() if city.is_dir()]
    city_to_files_paths = {
        city.name: [city.joinpath(files) for files in os.listdir(city) if files.endswith(".tif")]
        for city in city_dirs
    }

    if not os.path.exists(output_lmdb_path) or not os.path.exists(output_parquet_path):
        print("\nAligning and creating LMDB...")
        keys = create_lmdb_with_alignment(city_to_files_paths, output_lmdb_path, map_size=map_size)
        print(f"\nCreating metadata (split_strategy={split_strategy}, stratify_by={stratify_by})...")
        metadata = create_metadata(keys, stratify_by, split_strategy, lmdb_path=output_lmdb_path)
        metadata.to_parquet(output_parquet_path)


def check_spatial_alignment(lcz_path, prisma_path, s2_path, city_name):
    """
    Check spatial alignment of the three rasters and return alignment info.

    :param lcz_path: Path to LCZ raster
    :param prisma_path: Path to PRISMA raster
    :param s2_path: Path to Sentinel-2 raster
    :param city_name: Name of the city for logging
    :return: dict with alignment information
    """
    alignment_info = {
        'needs_alignment': False,
        'common_bounds': None,
        'target_transform': None,
        'target_crs': None,
        'target_width': None,
        'target_height': None,
        'issues': []
    }

    try:
        with rasterio.open(lcz_path) as lcz, \
                rasterio.open(prisma_path) as prisma, \
                rasterio.open(s2_path) as s2:

            # Check CRS alignment
            if not (lcz.crs == prisma.crs == s2.crs):
                alignment_info['issues'].append("CRS mismatch")
                alignment_info['needs_alignment'] = True
                print(f"  CRS mismatch: LCZ={lcz.crs}, PRISMA={prisma.crs}, S2={s2.crs}")
            else:
                print(f"  CRS aligned: {lcz.crs}")

            # Check resolution alignment
            lcz_res = (abs(lcz.transform.a), abs(lcz.transform.e))
            prisma_res = (abs(prisma.transform.a), abs(prisma.transform.e))
            s2_res = (abs(s2.transform.a), abs(s2.transform.e))

            # Allow small floating point differences
            res_tolerance = 1e-6
            if not (abs(lcz_res[0] - s2_res[0]) < res_tolerance and
                    abs(lcz_res[1] - s2_res[1]) < res_tolerance):
                alignment_info['issues'].append("LCZ-S2 resolution mismatch")
                alignment_info['needs_alignment'] = True
                print(f"  Resolution mismatch: LCZ={lcz_res}, S2={s2_res}")

            if not (abs(prisma_res[0] - s2_res[0]) < res_tolerance and
                    abs(prisma_res[1] - s2_res[1]) < res_tolerance):
                alignment_info['issues'].append("PRISMA-S2 resolution mismatch")
                alignment_info['needs_alignment'] = True
                print(f"  Resolution mismatch: PRISMA={prisma_res}, S2={s2_res}")

            if not alignment_info['issues'] or 'resolution' not in str(alignment_info['issues']):
                print(f"  Resolution aligned: {s2_res}")

            lcz_bounds = lcz.bounds
            prisma_bounds = prisma.bounds
            s2_bounds = s2.bounds

            # Intersection of all three rasters
            common_left = max(lcz_bounds.left, prisma_bounds.left, s2_bounds.left)
            common_bottom = max(lcz_bounds.bottom, prisma_bounds.bottom, s2_bounds.bottom)
            common_right = min(lcz_bounds.right, prisma_bounds.right, s2_bounds.right)
            common_top = min(lcz_bounds.top, prisma_bounds.top, s2_bounds.top)

            if common_left >= common_right or common_bottom >= common_top:
                raise ValueError(f"No spatial overlap between rasters for {city_name}")

            common_bounds = (common_left, common_bottom, common_right, common_top)
            print(f"  Common bounds: {common_bounds}")

            # Check if any raster extends beyond the common area
            bounds_match = (
                    abs(lcz_bounds.left - common_left) < res_tolerance and
                    abs(lcz_bounds.right - common_right) < res_tolerance and
                    abs(lcz_bounds.bottom - common_bottom) < res_tolerance and
                    abs(lcz_bounds.top - common_top) < res_tolerance and
                    abs(prisma_bounds.left - common_left) < res_tolerance and
                    abs(prisma_bounds.right - common_right) < res_tolerance and
                    abs(prisma_bounds.bottom - common_bottom) < res_tolerance and
                    abs(prisma_bounds.top - common_top) < res_tolerance and
                    abs(s2_bounds.left - common_left) < res_tolerance and
                    abs(s2_bounds.right - common_right) < res_tolerance and
                    abs(s2_bounds.bottom - common_bottom) < res_tolerance and
                    abs(s2_bounds.top - common_top) < res_tolerance
            )

            if not bounds_match:
                alignment_info['issues'].append("Bounds mismatch")
                alignment_info['needs_alignment'] = True
                print("  Bounds differ, cropping to the common area")

            # Use the S2 grid as reference
            target_crs = s2.crs
            target_res = s2_res

            # Target grid covering the common bounds at S2 resolution
            target_transform = rasterio.transform.from_bounds(
                common_left, common_bottom, common_right, common_top,
                width=int((common_right - common_left) / target_res[0]),
                height=int((common_top - common_bottom) / target_res[1])
            )

            target_width = int((common_right - common_left) / target_res[0])
            target_height = int((common_top - common_bottom) / target_res[1])

            alignment_info.update({
                'common_bounds': common_bounds,
                'target_transform': target_transform,
                'target_crs': target_crs,
                'target_width': target_width,
                'target_height': target_height
            })

    except Exception as e:
        print(f"  Could not check alignment for {city_name}: {e}")
        raise

    return alignment_info


def align_raster_to_target(src_path, target_transform, target_crs, target_width, target_height, output_path=None):
    """
    Align a raster to target spatial parameters.

    :param src_path: Path to source raster
    :param target_transform: Target transform
    :param target_crs: Target CRS
    :param target_width: Target width
    :param target_height: Target height
    :param output_path: Optional output path (if None, returns array)
    :return: Aligned raster data or None if saved to file
    """
    with rasterio.open(src_path) as src:
        aligned_data = np.full((src.count, target_height, target_width),
                               src.nodata if src.nodata is not None else 0,
                               dtype=src.dtypes[0])

        reproject(
            source=rasterio.band(src, list(range(1, src.count + 1))),
            destination=aligned_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear if src.count > 1 else Resampling.nearest  # Use nearest for labels
        )

        if output_path:
            profile = src.profile.copy()
            profile.update({
                'crs': target_crs,
                'transform': target_transform,
                'width': target_width,
                'height': target_height
            })

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(aligned_data)
            return None
        else:
            return aligned_data


def create_lmdb_with_alignment(city_to_files_paths, output_lmdb_path, patch_size=64, stride=32, map_size=6.5e10, batch_size=500):
    """
    Create LMDB with proper spatial alignment of all rasters.

    :param city_to_files_paths: dict mapping city name -> list of file paths
    :param output_lmdb_path: directory for the LMDB
    :param patch_size: size of square patch (pixels)
    :param stride: step between patches (pixels)
    :param map_size: max size in bytes for LMDB
    :param batch_size: number of patches written per LMDB transaction
    :return: list of keys written
    """

    keys = []
    env = lmdb.open(output_lmdb_path, map_size=int(map_size))

    for city, paths in city_to_files_paths.items():
        print(f"\nProcessing {city}...")

        # Identify the rasters by file name, falling back to the file order
        lcz_path = next((p for p in paths if 'lcz' in p.name.lower() or 'label' in p.name.lower()), paths[0])
        prisma_path = next((p for p in paths if 'prisma' in p.name.lower()), paths[1])
        s2_path = next((p for p in paths if 's2' in p.name.lower() or 'sentinel' in p.name.lower()), paths[2])

        print(f"  Files: LCZ={lcz_path.name}, PRISMA={prisma_path.name}, S2={s2_path.name}")

        alignment_info = check_spatial_alignment(lcz_path, prisma_path, s2_path, city)

        if alignment_info['needs_alignment']:
            print("  Aligning rasters...")
            lcz_aligned = align_raster_to_target(
                lcz_path,
                alignment_info['target_transform'],
                alignment_info['target_crs'],
                alignment_info['target_width'],
                alignment_info['target_height']
            )
            prisma_aligned = align_raster_to_target(
                prisma_path,
                alignment_info['target_transform'],
                alignment_info['target_crs'],
                alignment_info['target_width'],
                alignment_info['target_height']
            )
            s2_aligned = align_raster_to_target(
                s2_path,
                alignment_info['target_transform'],
                alignment_info['target_crs'],
                alignment_info['target_width'],
                alignment_info['target_height']
            )
        else:
            print("  Rasters already aligned, reading them directly...")
            with rasterio.open(lcz_path) as src:
                lcz_aligned = src.read()
            with rasterio.open(prisma_path) as src:
                prisma_aligned = src.read()
            with rasterio.open(s2_path) as src:
                s2_aligned = src.read()

        # Some S2 rasters hold digital numbers (reflectance x 10000, Berlin and Milan here) instead of reflectance.
        # Scale them so all cities and the PRISMA bands share the same 0-1 range
        if np.nanmedian(s2_aligned) > 10:
            print("  Scaling S2 digital numbers to reflectance")
            s2_aligned = s2_aligned / 10000

        H, W = lcz_aligned.shape[1], lcz_aligned.shape[2]
        print(f"  Extracting patches from {H}x{W} rasters...")

        # Top-left corner of every patch
        coords = [
            (r, c)
            for r in range(0, H - patch_size + 1, stride)
            for c in range(0, W - patch_size + 1, stride)
        ]

        # Write patches in batches to limit memory use
        valid_patches = 0
        batch_data = []

        for idx, (row_off, col_off) in enumerate(tqdm(coords, desc=f"{city} patches", unit="patch")):
            s2_patch = s2_aligned[:, row_off:row_off + patch_size, col_off:col_off + patch_size]
            prisma_patch = prisma_aligned[:, row_off:row_off + patch_size, col_off:col_off + patch_size]
            lcz_patch = lcz_aligned[0, row_off:row_off + patch_size, col_off:col_off + patch_size]

            # Skip patches with NaNs or without any labeled pixel
            if np.isnan(s2_patch).any() or np.isnan(prisma_patch).any() or np.isnan(lcz_patch).any():
                continue

            if np.all(lcz_patch == 0):
                continue

            # S2 bands first, then PRISMA bands
            x = np.concatenate([s2_patch, prisma_patch], axis=0)
            y = lcz_patch

            # Key format: {city}_{index}_{row_offset}_{col_offset}, parsed again in create_metadata
            key = f"{city}_{valid_patches:06d}_{row_off}_{col_off}"
            keys.append(key)

            sample = {
                'data': x.astype(np.float32),
                'label': y.astype(np.int64)
            }

            batch_data.append((key, sample))
            valid_patches += 1

            if len(batch_data) >= batch_size:
                with env.begin(write=True) as txn:
                    for k, v in batch_data:
                        txn.put(k.encode('ascii'), stnp.save(v))
                batch_data = []
                gc.collect()

        # Commit remaining patches
        if batch_data:
            with env.begin(write=True) as txn:
                for k, v in batch_data:
                    txn.put(k.encode('ascii'), stnp.save(v))

        print(f"  Created {valid_patches} patches for {city}")

        # Free the rasters before loading the next city
        del lcz_aligned, prisma_aligned, s2_aligned
        gc.collect()

    env.close()
    print(f"\nTotal patches: {len(keys)}")
    return keys


def analyze_patch_labels(lmdb_path, keys):
    """
    Analyze label distributions in patches for stratification.

    :param lmdb_path: Path to LMDB
    :param keys: List of patch keys to analyze
    :return: dict mapping patch key -> label statistics
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    patch_stats = {}

    with env.begin() as txn:
        for key in tqdm(keys, desc="Analyzing patches"):
            data = txn.get(key.encode())
            if data is None:
                continue

            tensors = stnp.load(data)
            labels = tensors['label']

            unique_labels, counts = np.unique(labels, return_counts=True)
            total_pixels = labels.size

            # Fraction of pixels per label (0 to 1, despite the name)
            label_percentages = {int(label): count / total_pixels for label, count in zip(unique_labels, counts)}

            # Most common label and its pixel fraction
            dominant_label = int(unique_labels[np.argmax(counts)])
            dominant_percentage = np.max(counts) / total_pixels

            label_diversity = len(unique_labels)

            # Labels covering less than 5% of the patch, ignoring background (0)
            rare_labels = [int(label) for label, pct in label_percentages.items() if pct < 0.05 and label != 0]

            patch_stats[key] = {
                'dominant_label': dominant_label,
                'dominant_percentage': dominant_percentage,
                'label_diversity': label_diversity,
                'label_percentages': label_percentages,
                'rare_labels': rare_labels,
                'has_rare_labels': len(rare_labels) > 0
            }

    env.close()
    return patch_stats


def create_metadata(keys, stratify_by, split_strategy, lmdb_path=None):
    """
    Create the patch metadata and assign the train/validation/test splits.

    :param keys: list of keys for the patches
    :param stratify_by: 'city', 'labels', 'mixed', or 'none'
    :param split_strategy: 'city_based', 'patch_based', or 'hybrid'
    :param lmdb_path: path to LMDB (needed for label-based stratification)
    :return: metadata dataframe
    """
    # Parse city, patch index and offsets from the keys
    rows = []
    for f in tqdm(keys, desc="Building basic metadata", unit="patch"):
        parts = f.split("_")
        city = parts[0]
        patch_idx = int(parts[1])

        # Fall back to 0 if a key has no offsets
        try:
            row_off = int(parts[2]) if len(parts) > 2 else 0
            col_off = int(parts[3]) if len(parts) > 3 else 0
        except (ValueError, IndexError):
            print(f"Warning: could not parse offsets from key {f}, using 0, 0")
            row_off = 0
            col_off = 0

        # The map visualizations use row_offset and col_offset to stitch patches back together
        rows.append({
            "patch_id": f,
            "city": city,
            "patch_idx": patch_idx,
            "row_offset": row_off,
            "col_offset": col_off
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["city", "patch_idx"]).reset_index(drop=True)

    # Label statistics are only needed for label-based stratification
    if stratify_by in ['labels', 'mixed'] and lmdb_path:
        patch_stats = analyze_patch_labels(lmdb_path, keys)

        df['dominant_label'] = df['patch_id'].map(lambda x: patch_stats.get(x, {}).get('dominant_label', 0))
        df['dominant_percentage'] = df['patch_id'].map(lambda x: patch_stats.get(x, {}).get('dominant_percentage', 1.0))
        df['label_diversity'] = df['patch_id'].map(lambda x: patch_stats.get(x, {}).get('label_diversity', 1))
        df['has_rare_labels'] = df['patch_id'].map(lambda x: patch_stats.get(x, {}).get('has_rare_labels', False))

        if stratify_by == 'labels':
            df['stratify_key'] = df['dominant_label'].astype(str)
        elif stratify_by == 'mixed':
            # Combine city and dominant label for more granular stratification
            df['stratify_key'] = df['city'] + '_' + df['dominant_label'].astype(str)
    else:
        df['stratify_key'] = df['city']

    if split_strategy == 'city_based':
        # Split by cities (good for domain adaptation experiments)
        cities = df['city'].unique()
        if len(cities) >= 3:
            # More than 4 cities: 70/15/15 by city
            # 3 or 4 cities: last city for test, the one before it for validation
            train_cities = cities[:int(0.7 * len(cities))] if len(cities) > 4 else cities[:-2]
            val_cities = cities[int(0.7 * len(cities)):int(0.85 * len(cities))] if len(cities) > 4 else cities[-2:-1]
            test_cities = cities[int(0.85 * len(cities)):] if len(cities) > 4 else cities[-1:]

            df['split'] = 'train'
            df.loc[df['city'].isin(val_cities), 'split'] = 'validation'
            df.loc[df['city'].isin(test_cities), 'split'] = 'test'
        else:
            # Fall back to patch-based if too few cities
            split_strategy = 'patch_based'
            print("Too few cities for a city-based split, falling back to patch_based")

    if split_strategy == 'patch_based':
        # Random 70/15/15 split, stratified by stratify_key where possible
        try:
            if stratify_by != 'none' and len(df['stratify_key'].unique()) > 1:
                train_df, temp_df = train_test_split(
                    df, test_size=0.30, stratify=df['stratify_key'], random_state=42
                )
                val_df, test_df = train_test_split(
                    temp_df, test_size=0.50, stratify=temp_df['stratify_key'], random_state=42
                )
            else:
                train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
                val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

            train_df['split'] = 'train'
            val_df['split'] = 'validation'
            test_df['split'] = 'test'
            df = pd.concat([train_df, val_df, test_df], axis=0)

        except ValueError as e:
            print(f"Stratification failed ({e}), using a random split")
            df['split'] = np.random.default_rng(42).choice(['train', 'validation', 'test'],
                                                           size=len(df), p=[0.7, 0.15, 0.15])

    elif split_strategy == 'hybrid':
        # Split each city separately so every city appears in all splits,
        # stratified by dominant label within the city if available
        splits = []
        for city in df['city'].unique():
            city_df = df[df['city'] == city].copy()

            if len(city_df) > 10:  # Only stratify if enough samples
                try:
                    if stratify_by != 'none' and 'stratify_key' in city_df.columns:
                        train_city, temp_city = train_test_split(
                            city_df, test_size=0.30,
                            stratify=city_df['dominant_label'] if 'dominant_label' in city_df.columns else None,
                            random_state=42
                        )
                        val_city, test_city = train_test_split(
                            temp_city, test_size=0.50,
                            stratify=temp_city['dominant_label'] if 'dominant_label' in temp_city.columns else None,
                            random_state=42
                        )
                    else:
                        train_city, temp_city = train_test_split(city_df, test_size=0.30, random_state=42)
                        val_city, test_city = train_test_split(temp_city, test_size=0.50, random_state=42)
                except:
                    # Fall back to a random split if stratification fails
                    train_city, temp_city = train_test_split(city_df, test_size=0.30, random_state=42)
                    val_city, test_city = train_test_split(temp_city, test_size=0.50, random_state=42)
            else:
                # Small cities are split in order, without shuffling
                n = len(city_df)
                train_city = city_df[:int(0.7 * n)]
                val_city = city_df[int(0.7 * n):int(0.85 * n)]
                test_city = city_df[int(0.85 * n):]

            train_city['split'] = 'train'
            val_city['split'] = 'validation'
            test_city['split'] = 'test'

            splits.extend([train_city, val_city, test_city])

        df = pd.concat(splits, axis=0)

    print("\nSplit statistics:")
    for split in ['train', 'validation', 'test']:
        split_df = df[df['split'] == split]
        print(f"{split.capitalize()}: {len(split_df)} patches")

        if 'dominant_label' in df.columns:
            print(f"  Dominant labels: {split_df['dominant_label'].value_counts().sort_index().to_dict()}")

        cities_in_split = split_df['city'].unique()
        print(f"  Cities: {list(cities_in_split)}")

    # Columns written to the parquet file
    final_columns = ['patch_id', 'city', 'split', 'row_offset', 'col_offset']
    if 'dominant_label' in df.columns:
        final_columns.extend(['dominant_label', 'label_diversity', 'has_rare_labels'])

    # Report missing columns, the selection below would fail with a KeyError
    for col in final_columns:
        if col not in df.columns:
            print(f"Warning: column {col} missing from the metadata")

    final_df = df[final_columns].reset_index(drop=True)

    return final_df