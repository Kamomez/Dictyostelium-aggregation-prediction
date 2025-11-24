import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

def compute_center_error(pred_map, true_centers, top_k=10):
    pred_flat = pred_map.flatten()
    top_indices = np.argsort(pred_flat)[-top_k:]
    pred_centers = []
    for idx in top_indices:
        y, x = np.unravel_index(idx, pred_map.shape)
        pred_centers.append((y, x))

    min_errors = []
    for ty, tx in true_centers:
        distances = [np.sqrt((ty-py)**2 + (tx-px)**2) for py, px in pred_centers]
        if distances:
            min_errors.append(min(distances))

    return np.mean(min_errors) if min_errors else float('inf'), min_errors


def compute_auroc_and_ap(pred_map, true_map, threshold=0.1):
    y_true_flat = true_map.flatten()
    y_pred_flat = pred_map.flatten()
    y_true_binary = (y_true_flat > threshold).astype(int)

    if len(np.unique(y_true_binary)) < 2:
        return None, None

    try:
        auroc = roc_auc_score(y_true_binary, y_pred_flat)
        ap = average_precision_score(y_true_binary, y_pred_flat)
        return auroc, ap
    except:
        return None, None


def evaluate_model(model, dataloader, true_centers, dataset_name, device, pixel_size_microns=0.5):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            pred = model(xb)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(yb.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Metric 1: MSE
    mse = ((preds - targets) ** 2).mean()

    # Metric 2: Center Error (pixels)
    center_errors_px = []
    for i in range(len(preds)):
        error, _ = compute_center_error(preds[i, 0], true_centers, top_k=20)
        if error != float('inf'):
            center_errors_px.append(error)

    # Metric 3: Center Error (micrometers)
    center_errors_um = [e * pixel_size_microns for e in center_errors_px]

    # Metric 4: Spatial correlation
    pred_flat = preds.flatten()
    target_flat = targets.flatten()
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]

    # Metric 5: AUROC and Average Precision
    auroc_scores = []
    ap_scores = []
    for i in range(len(preds)):
        auroc, ap = compute_auroc_and_ap(preds[i, 0], targets[i, 0])
        if auroc is not None:
            auroc_scores.append(auroc)
            ap_scores.append(ap)

    return {
        'mse': mse,
        'center_error_px_mean': np.mean(center_errors_px) if center_errors_px else float('inf'),
        'center_error_px_std': np.std(center_errors_px) if center_errors_px else 0,
        'center_error_um_mean': np.mean(center_errors_um) if center_errors_um else float('inf'),
        'center_error_um_std': np.std(center_errors_um) if center_errors_um else 0,
        'correlation': correlation,
        'auroc_mean': np.mean(auroc_scores) if auroc_scores else None,
        'auroc_std': np.std(auroc_scores) if auroc_scores else None,
        'ap_mean': np.mean(ap_scores) if ap_scores else None,
        'ap_std': np.std(ap_scores) if ap_scores else None,
    }


def print_evaluation_results(results, model_name, dataset_name):
    print(f"\n{model_name} on {dataset_name}:")
    print(f"  MSE: {results['mse']:.6f}")
    print(f"  Center Error: {results['center_error_px_mean']:6.2f}±{results['center_error_px_std']:5.2f} px  "
          f"({results['center_error_um_mean']:6.2f}±{results['center_error_um_std']:5.2f} μm)")
    print(f"  Correlation: {results['correlation']:.4f}")
    if results['auroc_mean'] is not None:
        print(f"  AUROC: {results['auroc_mean']:.4f}±{results['auroc_std']:.4f}")
        print(f"  Average Precision: {results['ap_mean']:.4f}±{results['ap_std']:.4f}")


if __name__ == '__main__':
    # Test evaluation functions
    print("Evaluation Function Test")
    print("=" * 80)
    
    import os
    import sys
    sys.path.append(os.path.dirname(__file__))
    
    from dicty_models import CNN3DPredictor
    from dicty_dataset import MultiDatasetAggregation
    from torch.utils.data import DataLoader
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy data
    print("\nCreating dummy dataset...")
    K = 8
    H, W = 128, 128
    n_samples = 20
    
    dummy_datasets = {
        'test1': np.random.rand(50, H, W).astype('float32'),
    }
    dummy_prob_maps = {
        'test1': np.random.rand(H, W).astype('float32'),
    }
    
    dataset = MultiDatasetAggregation(dummy_datasets, dummy_prob_maps, K=K)
    loader = DataLoader(dataset, batch_size=4)
    
    # Create dummy centers
    true_centers = [(np.random.randint(0, H), np.random.randint(0, W)) for _ in range(5)]
    print(f"True centers: {len(true_centers)}")
    
    # Create and test model
    print("\nCreating model...")
    model = CNN3DPredictor(K=K).to(device)
    
    print("\nEvaluating model...")
    results = evaluate_model(
        model, loader, true_centers,
        dataset_name='test1',
        device=device,
        pixel_size_microns=0.5
    )
    
    print_evaluation_results(results, "3D CNN", "test1")
    
    print("\nEvaluation test passed!")
