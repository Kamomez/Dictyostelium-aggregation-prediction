import torch
from torch.utils.data import Dataset

class MultiDatasetAggregation(Dataset):
    def __init__(self, datasets_dict, prob_maps_dict, K=8):
        self.K = K
        self.samples = []

        for name in datasets_dict.keys():
            frames = datasets_dict[name]
            target = prob_maps_dict[name]

            for start_idx in range(max(1, len(frames) - K + 1)):
                self.samples.append({
                    'dataset': name,
                    'frames': torch.from_numpy(frames.astype('float32')),
                    'target': torch.from_numpy(target.astype('float32')),
                    'start_idx': start_idx
                })

        print(f"Created multi-dataset with {len(self.samples)} total samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        start = sample['start_idx']
        frames = sample['frames']
        target = sample['target']

        # Get K consecutive frames
        x = frames[start:start+self.K]
        x = x.unsqueeze(1)  # Add channel dimension: (K, 1, H, W)
        y = target.unsqueeze(0)  # Add channel dimension: (1, H, W)

        return x, y

if __name__ == '__main__':
    # Test dataset creation
    print("Dataset Test")
    print("=" * 80)
    
    import numpy as np
    from torch.utils.data import DataLoader, random_split
    
    # Create dummy data
    print("\nCreating dummy data...")
    dummy_datasets = {
        'test1': np.random.rand(50, 128, 128).astype('float32'),
        'test2': np.random.rand(60, 128, 128).astype('float32'),
    }
    
    dummy_prob_maps = {
        'test1': np.random.rand(128, 128).astype('float32'),
        'test2': np.random.rand(128, 128).astype('float32'),
    }
    
    # Create dataset
    K = 8
    print(f"\nCreating dataset with K={K} frames...")
    dataset = MultiDatasetAggregation(dummy_datasets, dummy_prob_maps, K=K)
    
    print(f"Total samples: {len(dataset)}")
    
    # Test data loading
    print("\nTesting data loading...")
    x, y = dataset[0]
    print(f"Input shape: {x.shape}")  # Should be (K, 1, H, W)
    print(f"Target shape: {y.shape}")  # Should be (1, H, W)
    
    # Test dataloader
    print("\nTesting DataLoader...")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch_x, batch_y in loader:
        print(f"Batch input shape: {batch_x.shape}")  # Should be (B, K, 1, H, W)
        print(f"Batch target shape: {batch_y.shape}")  # Should be (B, 1, H, W)
        break
    
    # Test train/val/test split
    print("\nTesting train/val/test split...")
    n_samples = len(dataset)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    n_test = n_samples - n_train - n_val
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Total: {len(train_ds) + len(val_ds) + len(test_ds)}")
    
    print("\nDataset test passed!")
