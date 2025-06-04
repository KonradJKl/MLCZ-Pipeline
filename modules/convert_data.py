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

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]


def load_data(input_data_path: str, output_lmdb_path: str, output_parquet_path: str, stratify_by: str,  split_strategy: str):
    """
    Convert the given datasets to lmdb and parquet format with proper spatial alignment.

    :param input_data_path: path to the source dataset
    :param output_lmdb_path: path to the destination lmdb file
    :param output_parquet_path: path to the destination parquet file
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
        keys = create_lmdb_with_alignment(city_to_files_paths, output_lmdb_path)
        print("\nCreating Metadata...")
        metadata = create_metadata(keys, stratify_by, split_strategy)
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
    print(f"\n🔍 Checking spatial alignment for {city_name}...")

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
                print(f"  ❌ CRS mismatch: LCZ={lcz.crs}, PRISMA={prisma.crs}, S2={s2.crs}")
            else:
                print(f"  ✅ CRS aligned: {lcz.crs}")

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
                print(f"  ❌ Resolution mismatch: LCZ={lcz_res}, S2={s2_res}")

            if not (abs(prisma_res[0] - s2_res[0]) < res_tolerance and
                    abs(prisma_res[1] - s2_res[1]) < res_tolerance):
                alignment_info['issues'].append("PRISMA-S2 resolution mismatch")
                alignment_info['needs_alignment'] = True
                print(f"  ❌ PRISMA resolution different: {prisma_res} vs S2={s2_res}")

            if not alignment_info['issues'] or 'resolution' not in str(alignment_info['issues']):
                print(f"  ✅ Resolution aligned: {s2_res}")

            # Check spatial bounds - find intersection
            lcz_bounds = lcz.bounds
            prisma_bounds = prisma.bounds
            s2_bounds = s2.bounds

            print(f"  📐 Bounds comparison:")
            print(f"    LCZ:    {lcz_bounds}")
            print(f"    PRISMA: {prisma_bounds}")
            print(f"    S2:     {s2_bounds}")

            # Calculate intersection of all three
            common_left = max(lcz_bounds.left, prisma_bounds.left, s2_bounds.left)
            common_bottom = max(lcz_bounds.bottom, prisma_bounds.bottom, s2_bounds.bottom)
            common_right = min(lcz_bounds.right, prisma_bounds.right, s2_bounds.right)
            common_top = min(lcz_bounds.top, prisma_bounds.top, s2_bounds.top)

            if common_left >= common_right or common_bottom >= common_top:
                raise ValueError(f"No spatial overlap between rasters for {city_name}")

            common_bounds = (common_left, common_bottom, common_right, common_top)
            print(f"  ✅ Common bounds: {common_bounds}")

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
                print(f"  ⚠️  Bounds need cropping to common area")

            # Use S2 as reference (typically has good resolution and coverage)
            target_crs = s2.crs
            target_res = s2_res

            # Calculate target transform and dimensions for common bounds
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

            print(f"  📏 Target dimensions: {target_width} x {target_height}")

    except Exception as e:
        print(f"  ❌ Error checking alignment: {e}")
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
        # Prepare output array
        aligned_data = np.full((src.count, target_height, target_width),
                               src.nodata if src.nodata is not None else 0,
                               dtype=src.dtypes[0])

        # Reproject each band
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
            # Save to file
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
    :return: list of keys written
    """

    keys = []
    env = lmdb.open(output_lmdb_path, map_size=int(map_size))

    for city, paths in city_to_files_paths.items():
        print(f"\n🏙️  Processing city: {city}")

        # Identify file paths (same as before)
        lcz_path = next((p for p in paths if 'lcz' in p.name.lower() or 'label' in p.name.lower()), paths[0])
        prisma_path = next((p for p in paths if 'prisma' in p.name.lower()), paths[1])
        s2_path = next((p for p in paths if 's2' in p.name.lower() or 'sentinel' in p.name.lower()), paths[2])

        print(f"  📁 Files identified:")
        print(f"    LCZ: {lcz_path.name}")
        print(f"    PRISMA: {prisma_path.name}")
        print(f"    S2: {s2_path.name}")

        # Check spatial alignment
        alignment_info = check_spatial_alignment(lcz_path, prisma_path, s2_path, city)

        # Load and align data
        if alignment_info['needs_alignment']:
            print(f"  🔧 Aligning rasters...")
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
            print(f"  ✅ Rasters already aligned, reading directly...")
            with rasterio.open(lcz_path) as src:
                lcz_aligned = src.read()
            with rasterio.open(prisma_path) as src:
                prisma_aligned = src.read()
            with rasterio.open(s2_path) as src:
                s2_aligned = src.read()

        H, W = lcz_aligned.shape[1], lcz_aligned.shape[2]
        print(f"  🔪 Creating patches from {H}x{W} aligned rasters...")

        # Create coordinates for patches
        coords = [
            (r, c)
            for r in range(0, H - patch_size + 1, stride)
            for c in range(0, W - patch_size + 1, stride)
        ]

        # Process patches in batches to manage memory
        valid_patches = 0
        batch_data = []

        for idx, (row_off, col_off) in enumerate(tqdm(coords, desc=f"{city} patches", unit="patch")):
            # Extract patches
            s2_patch = s2_aligned[:, row_off:row_off + patch_size, col_off:col_off + patch_size]
            prisma_patch = prisma_aligned[:, row_off:row_off + patch_size, col_off:col_off + patch_size]
            lcz_patch = lcz_aligned[0, row_off:row_off + patch_size, col_off:col_off + patch_size]

            # Quality checks
            if np.isnan(s2_patch).any() or np.isnan(prisma_patch).any() or np.isnan(lcz_patch).any():
                continue

            if np.all(lcz_patch == 0):
                continue

            # Combine spectral data
            x = np.concatenate([s2_patch, prisma_patch], axis=0)
            y = lcz_patch

            key = f"{city}_{valid_patches:06d}_{row_off}_{col_off}"
            keys.append(key)

            sample = {
                'data': x.astype(np.float32),
                'label': y.astype(np.int64)
            }

            batch_data.append((key, sample))
            valid_patches += 1

            # Commit batch when it reaches batch_size
            if len(batch_data) >= batch_size:
                with env.begin(write=True) as txn:
                    for k, v in batch_data:
                        txn.put(k.encode('ascii'), stnp.save(v))
                batch_data = []

                # Force garbage collection
                gc.collect()

                # Optional: print memory usage
                if idx % 500 == 0:
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    print(f"    Memory usage: {memory_mb:.1f} MB")

        # Commit remaining patches
        if batch_data:
            with env.begin(write=True) as txn:
                for k, v in batch_data:
                    txn.put(k.encode('ascii'), stnp.save(v))

        print(f"  ✅ Created {valid_patches} valid patches for {city}")

        # Clean up city data from memory
        del lcz_aligned, prisma_aligned, s2_aligned
        gc.collect()

    env.close()
    print(f"\n🎉 Total patches created: {len(keys)}")
    return keys


def analyze_patch_labels(lmdb_path, keys):
    """
    Analyze label distributions in patches for stratification.

    :param lmdb_path: Path to LMDB
    :param keys: List of patch keys to analyze
    :return: Dictionary with label statistics per patch
    """
    print("📊 Analyzing label distributions for stratification...")

    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    patch_stats = {}

    with env.begin() as txn:
        for key in tqdm(keys, desc="Analyzing patches"):
            data = txn.get(key.encode())
            if data is None:
                continue

            tensors = stnp.load(data)
            labels = tensors['label']

            # Calculate label statistics
            unique_labels, counts = np.unique(labels, return_counts=True)
            total_pixels = labels.size

            # Label distribution percentages
            label_percentages = {int(label): count/total_pixels for label, count in zip(unique_labels, counts)}

            # Dominant label (most common)
            dominant_label = int(unique_labels[np.argmax(counts)])
            dominant_percentage = np.max(counts) / total_pixels

            # Label diversity (number of different labels)
            label_diversity = len(unique_labels)

            # Rare label presence (labels with <5% coverage)
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


def create_metadata(keys, lmdb_path=None, stratify_by='mixed', split_strategy='hybrid'):
    """
    Create flexible metadata for the dataset with various stratification options.

    :param keys: list of keys for the patches
    :param lmdb_path: path to LMDB (needed for label-based stratification)
    :param stratify_by: 'city', 'labels', 'mixed', or 'none'
    :param split_strategy: 'city_based', 'patch_based', or 'hybrid'
    :return: metadata dataframe
    """
    print(f"🏗️  Creating metadata with stratify_by='{stratify_by}', split_strategy='{split_strategy}'")

    # Basic metadata
    rows = []
    for f in tqdm(keys, desc="Building basic metadata", unit="patch"):
        parts = f.split("_")
        city = parts[0]
        patch_idx = int(parts[1])
        row_off = int(parts[2]) if len(parts) > 2 else 0
        col_off = int(parts[3]) if len(parts) > 3 else 0

        rows.append({
            "patch_id": f,
            "city": city,
            "patch_idx": patch_idx,
            "row_offset": row_off,
            "col_offset": col_off
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["city", "patch_idx"]).reset_index(drop=True)

    # Add label-based features if needed
    if stratify_by in ['labels', 'mixed'] and lmdb_path:
        print("🔍 Analyzing patch labels for stratification...")
        patch_stats = analyze_patch_labels(lmdb_path, keys)

        # Add label statistics to dataframe
        df['dominant_label'] = df['patch_id'].map(lambda x: patch_stats.get(x, {}).get('dominant_label', 0))
        df['dominant_percentage'] = df['patch_id'].map(lambda x: patch_stats.get(x, {}).get('dominant_percentage', 1.0))
        df['label_diversity'] = df['patch_id'].map(lambda x: patch_stats.get(x, {}).get('label_diversity', 1))
        df['has_rare_labels'] = df['patch_id'].map(lambda x: patch_stats.get(x, {}).get('has_rare_labels', False))

        # Create stratification categories
        if stratify_by == 'labels':
            df['stratify_key'] = df['dominant_label'].astype(str)
        elif stratify_by == 'mixed':
            # Combine city and dominant label for more granular stratification
            df['stratify_key'] = df['city'] + '_' + df['dominant_label'].astype(str)
    else:
        df['stratify_key'] = df['city']

    # Create splits based on strategy
    if split_strategy == 'city_based':
        # Split by cities (good for domain adaptation experiments)
        cities = df['city'].unique()
        if len(cities) >= 3:
            # Use different cities for train/val/test
            train_cities = cities[:int(0.7*len(cities))] if len(cities) > 4 else cities[:-2]
            val_cities = cities[int(0.7*len(cities)):int(0.85*len(cities))] if len(cities) > 4 else cities[-2:-1]
            test_cities = cities[int(0.85*len(cities)):] if len(cities) > 4 else cities[-1:]

            df['split'] = 'train'
            df.loc[df['city'].isin(val_cities), 'split'] = 'validation'
            df.loc[df['city'].isin(test_cities), 'split'] = 'test'

            print(f"🏙️  City-based split:")
            print(f"   Train cities: {list(train_cities)}")
            print(f"   Val cities: {list(val_cities)}")
            print(f"   Test cities: {list(test_cities)}")
        else:
            # Fall back to patch-based if too few cities
            split_strategy = 'patch_based'
            print("⚠️  Too few cities for city-based split, falling back to patch-based")

    if split_strategy == 'patch_based':
        # Traditional patch-based split with stratification
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
            print(f"⚠️  Stratification failed ({e}), using random split")
            df['split'] = np.random.choice(['train', 'validation', 'test'],
                                         size=len(df), p=[0.7, 0.15, 0.15])

    elif split_strategy == 'hybrid':
        # Ensure each city appears in all splits, but maintain stratification within cities
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
                    # Fallback to random if stratification fails
                    train_city, temp_city = train_test_split(city_df, test_size=0.30, random_state=42)
                    val_city, test_city = train_test_split(temp_city, test_size=0.50, random_state=42)
            else:
                # For small cities, distribute randomly
                n = len(city_df)
                train_city = city_df[:int(0.7*n)]
                val_city = city_df[int(0.7*n):int(0.85*n)]
                test_city = city_df[int(0.85*n):]

            train_city['split'] = 'train'
            val_city['split'] = 'validation'
            test_city['split'] = 'test'

            splits.extend([train_city, val_city, test_city])

        df = pd.concat(splits, axis=0)

    # Print split statistics
    print(f"\n📈 Split Statistics:")
    for split in ['train', 'validation', 'test']:
        split_df = df[df['split'] == split]
        print(f"  {split.capitalize()}: {len(split_df)} patches")

        if 'dominant_label' in df.columns:
            print(f"    Label distribution: {dict(split_df['dominant_label'].value_counts())}")

        cities_in_split = split_df['city'].unique()
        print(f"    Cities: {list(cities_in_split)}")

    # Clean up final dataframe
    final_columns = ['patch_id', 'city', 'split']
    if 'dominant_label' in df.columns:
        final_columns.extend(['dominant_label', 'label_diversity', 'has_rare_labels'])

    return df[final_columns].reset_index(drop=True)