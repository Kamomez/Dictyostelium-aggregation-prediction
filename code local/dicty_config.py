import argparse

# Pixel size information (micrometers per pixel)
PIXEL_SIZE_MICRONS = {
    'mixin_test44': 0.325,  # 20x objective
    'mixin_test57': 0.65,   # 10x objective
    'mixin_test64': 0.65    # 10x objective
}

# Default dataset paths (relative to project root)
DEFAULT_DATA_PATH = './Data'
DEFAULT_OUTPUT_PATH = './Output'

def parse_args():
    parser = argparse.ArgumentParser(description='Train Dictyostelium aggregation prediction models')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH,
                        help='Path to data directory')
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT_PATH,
                        help='Path to output directory')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--K', type=int, default=8,
                        help='Number of input frames')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=7,
                        help='Random seed')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and only evaluate')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model')
    return parser.parse_args()

def get_dataset_paths(data_path):
    import os
    return {
        'mixin_test44': os.path.join(data_path, 'mixin_test44', '2024-01-17_ERH_23hr_ERH Red FarRed.zarr'),
        'mixin_test57': os.path.join(data_path, 'mixin_test57', '2024-02-29_mixin57_overnight_25um_ERH_Red_FarRed_25.zarr'),
        'mixin_test64': os.path.join(data_path, 'mixin_test64', 'ERH_2024-04-04_mixin64_wellC5_10x_overnight_ERH Red FarRed_1_t_subsampled.zarr')
    }

if __name__ == '__main__':
    # Test configuration
    print("Configuration Test")
    print("=" * 80)
    args = parse_args()
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"K frames: {args.K}")
    print(f"Learning rate: {args.lr}")
    print(f"Random seed: {args.seed}")
    print("\nDataset paths:")
    for name, path in get_dataset_paths(args.data_path).items():
        print(f"  {name}: {path}")
    print("\nPixel sizes (μm/pixel):")
    for name, size in PIXEL_SIZE_MICRONS.items():
        print(f"  {name}: {size}")
