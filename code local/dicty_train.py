#Training functions for Dictyostelium aggregation prediction models

import os
import torch
from torch import nn

def train_model(model, train_loader, val_loader, epochs, lr, device, model_name="Model", save_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }
    best_model_state = None

    print(f"\nTraining {model_name}...")
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'LR':<12}")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Save best model
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch + 1
            best_model_state = model.state_dict().copy()
            
            # Save to disk if path provided
            if save_path:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch+1:<8} {train_loss:<15.6f} {val_loss:<15.6f} {current_lr:<12.2e}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"Best validation loss: {history['best_val_loss']:.6f} (epoch {history['best_epoch']})")
    
    return history


def load_model(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'val_loss': checkpoint.get('val_loss', 'unknown')
    }
    
    print(f"Loaded model from {model_path}")
    print(f"  Epoch: {info['epoch']}")
    print(f"  Val Loss: {info['val_loss']}")
    
    return model, info


if __name__ == '__main__':
    # Test training function
    print("Training Function Test")
    print("=" * 80)
    
    import sys
    sys.path.append(os.path.dirname(__file__))
    
    from dicty_models import CNN3DPredictor
    from dicty_dataset import MultiDatasetAggregation
    from torch.utils.data import DataLoader, random_split
    import numpy as np
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy data
    print("\nCreating dummy dataset...")
    dummy_datasets = {
        'test1': np.random.rand(50, 128, 128).astype('float32'),
    }
    dummy_prob_maps = {
        'test1': np.random.rand(128, 128).astype('float32'),
    }
    
    K = 8
    dataset = MultiDatasetAggregation(dummy_datasets, dummy_prob_maps, K=K)
    
    # Split dataset
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Create model
    print("\nCreating model...")
    model = CNN3DPredictor(K=K).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test training (few epochs)
    print("\nTesting training loop (5 epochs)...")
    history = train_model(
        model, train_loader, val_loader,
        epochs=5, lr=1e-3, device=device,
        model_name="Test Model",
        save_path="test_model.pth"
    )
    
    print("\nTraining test passed!")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"Best val loss: {history['best_val_loss']:.6f}")
    
    # Test loading
    if os.path.exists("test_model.pth"):
        print("\nTesting model loading...")
        loaded_model = CNN3DPredictor(K=K)
        loaded_model, info = load_model(loaded_model, "test_model.pth", device)
        print("Model loading test passed!")
        
        # Cleanup
        os.remove("test_model.pth")
        print("Cleaned up test files")
