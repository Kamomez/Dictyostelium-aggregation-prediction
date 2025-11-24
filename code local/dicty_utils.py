import os
import random
import numpy as np
import torch
import zarr
from scipy.ndimage import gaussian_filter, label, center_of_mass

def set_seed(seed=7):
    #Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def read_zarr_data(path, z_projection='max'):
    try:
        store = zarr.open(path, mode='r')
        print(f"\n  Zarr shape: {store.shape}, dtype: {store.dtype}")

        # Sample frames to find valid data
        print(f"  Scanning for valid frames...")
        n_samples = min(50, store.shape[0])
        sample_indices = np.linspace(0, store.shape[0]-1, n_samples, dtype=int)
        samples = []
        for idx in sample_indices:
            sample = store[idx, 0, :, :, :].max()
            samples.append((idx, sample))
            if sample > 0:
                print(f"    Frame {idx}: max={sample}")

        # Find where data exists
        valid_samples = [(idx, val) for idx, val in samples if val > 0]
        if not valid_samples:
            print(f"  No valid data found")
            return None

        print(f"  Found {len(valid_samples)} valid samples")

        # Load range covering all valid data
        valid_start = max(0, min([idx for idx, _ in valid_samples]) - 5)
        valid_end = min(store.shape[0], max([idx for idx, _ in valid_samples]) + 6)

        print(f"  Loading frames {valid_start} to {valid_end}")
        data = store[valid_start:valid_end, 0, :, :, :]

        # Apply z-projection
        if z_projection == 'max':
            projected = data.max(axis=1)
        elif z_projection == 'mean':
            projected = data.mean(axis=1)
        else:
            projected = data[:, data.shape[1]//2, :, :]

        # Remove empty frames
        frame_means = projected.mean(axis=(1,2))
        valid_mask = frame_means > 0.001
        projected_clean = projected[valid_mask]

        print(f"  After cleaning: {projected_clean.shape}")
        return projected_clean.astype(np.float32)
    except Exception as e:
        print(f"  Error loading {path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_aggregation_centers(data, sigma=3, percentile=95, min_size=5):
    # Use final frames
    final_frames = data[-5:].mean(axis=0)
    smoothed = gaussian_filter(final_frames, sigma=sigma)
    threshold = np.percentile(smoothed, percentile)
    binary = smoothed > threshold
    labeled, num_features = label(binary)
    
    centers = []
    for i in range(1, num_features + 1):
        mask = labeled == i
        if mask.sum() > min_size:
            cy, cx = center_of_mass(mask)
            centers.append([cx, cy])
    
    centers = np.array(centers) if len(centers) > 0 else np.array([]).reshape(0, 2)
    prob_map = smoothed / (smoothed.max() + 1e-8)
    
    return centers, prob_map, smoothed, binary

def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min() + 1e-8)

if __name__ == '__main__':
    # Test utilities
    print("Utility Functions Test")
    print("=" * 80)
    
    # Test seed setting
    set_seed(7)
    
    # Test data loading (requires actual data)
    import sys
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        print(f"\nTesting data loading from: {test_path}")
        if os.path.exists(test_path):
            data = read_zarr_data(test_path)
            if data is not None:
                print(f"\nLoaded data shape: {data.shape}")
                print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
                
                # Test normalization
                normalized = normalize_data(data)
                print(f"\nNormalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")
                
                # Test center extraction
                centers, prob_map, smoothed, binary = extract_aggregation_centers(normalized)
                print(f"\nExtracted {len(centers)} aggregation centers")
                print(f"Probability map shape: {prob_map.shape}")
        else:
            print(f"Path not found: {test_path}")
    else:
        print("\nTo test data loading, run:")
        print("  python dicty_utils.py <path_to_zarr_file>")
