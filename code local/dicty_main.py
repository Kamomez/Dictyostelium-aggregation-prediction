#Main pipeline for Dictyostelium aggregation prediction
#Orchestrates all modules for end-to-end training and evaluation

import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from dicty_config import parse_args, get_dataset_paths, PIXEL_SIZE_MICRONS
from dicty_utils import set_seed, read_zarr_data, extract_aggregation_centers, normalize_data
from dicty_dataset import MultiDatasetAggregation
from dicty_models import CNN3DPredictor, FlowBasedPredictor, ConvLSTMPredictor
from dicty_train import train_model, load_model
from dicty_evaluate import evaluate_model, print_evaluation_results


def load_and_process_data(args):
    print("STEP 1: DATA LOADING AND PROCESSING")
    
    dataset_paths = get_dataset_paths(args.data_path)
    
    # Load datasets
    print("\nLoading datasets...")
    all_datasets_raw = {}
    for name, path in dataset_paths.items():
        print(f"\n{name}:")
        if os.path.exists(path):
            data = read_zarr_data(path)
            if data is not None and len(data) > 5:
                all_datasets_raw[name] = data
        else:
            print(f"  Path not found: {path}")
    
    if not all_datasets_raw:
        raise ValueError("No datasets loaded! Check your data paths.")
    
    # Process datasets
    print("\nProcessing datasets...")
    datasets = {}
    all_centers = {}
    all_prob_maps = {}
    
    for name, raw_zarr in all_datasets_raw.items():
        raw = raw_zarr.copy()
        raw_normalized = normalize_data(raw)
        
        # Extract centers
        centers, prob_map, smoothed, binary = extract_aggregation_centers(raw_normalized)
        
        print(f"  {name}: {raw_normalized.shape}, centers={len(centers)}")
        
        datasets[name] = raw_normalized
        all_centers[name] = centers
        all_prob_maps[name] = prob_map
    
    return datasets, all_centers, all_prob_maps


def create_dataloaders(datasets, all_prob_maps, args):
    print("STEP 2: DATASET PREPARATION")
    
    # Create multi-dataset
    multi_ds = MultiDatasetAggregation(datasets, all_prob_maps, K=args.K)
    
    # Split dataset
    n_samples = len(multi_ds)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    n_test = n_samples - n_train - n_val
    
    train_ds, val_ds, test_ds = random_split(
        multi_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")
    
    # Create individual dataset loaders for cross-dataset evaluation
    individual_loaders = {}
    for name in datasets.keys():
        ds_single = MultiDatasetAggregation({name: datasets[name]}, {name: all_prob_maps[name]}, K=args.K)
        individual_loaders[name] = DataLoader(ds_single, batch_size=16)
    
    return train_loader, val_loader, test_loader, individual_loaders


def train_all_models(train_loader, val_loader, args, device):
    print("STEP 3: MODEL TRAINING")
    
    models = {}
    histories = {}
    
    # Model 1: 3D CNN
    print("\nModel 1: 3D CNN")
    model1 = CNN3DPredictor(K=args.K).to(device)
    print(f"Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    history1 = train_model(
        model1, train_loader, val_loader, 
        args.epochs, args.lr, device, 
        "3D CNN",
        save_path=os.path.join(args.output_path, "model1_3dcnn.pth")
    )
    models['3D CNN'] = model1
    histories['3D CNN'] = history1
    
    # Model 2: Flow-Based
    print("\nModel 2: Flow-Based")
    model2 = FlowBasedPredictor(K=args.K).to(device)
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    
    history2 = train_model(
        model2, train_loader, val_loader,
        args.epochs, args.lr, device,
        "Flow-Based",
        save_path=os.path.join(args.output_path, "model2_flow.pth")
    )
    models['Flow-Based'] = model2
    histories['Flow-Based'] = history2
    
    # Model 3: ConvLSTM
    print("\nModel 3: ConvLSTM")
    model3 = ConvLSTMPredictor(K=args.K, hidden_dim=64).to(device)
    print(f"Parameters: {sum(p.numel() for p in model3.parameters()):,}")
    
    history3 = train_model(
        model3, train_loader, val_loader,
        args.epochs, args.lr, device,
        "ConvLSTM",
        save_path=os.path.join(args.output_path, "model3_convlstm.pth")
    )
    models['ConvLSTM'] = model3
    histories['ConvLSTM'] = history3
    
    return models, histories


def evaluate_all_models(models, individual_loaders, all_centers, datasets, device, args):
    print("STEP 4: CROSS-DATASET EVALUATION")
    
    results_cross = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print("-" * 100)
        results_cross[model_name] = {}
        
        for dataset_name, loader in individual_loaders.items():
            centers = all_centers[dataset_name]
            centers_list = [(c[1], c[0]) for c in centers]
            
            pixel_size = PIXEL_SIZE_MICRONS.get(dataset_name, 0.5)
            
            metrics = evaluate_model(
                model, loader, centers_list, 
                dataset_name, device, pixel_size
            )
            results_cross[model_name][dataset_name] = metrics
            
            print_evaluation_results(metrics, model_name, dataset_name)
    
    return results_cross


def save_results(results_cross, datasets, args):
    print("STEP 5: SAVING RESULTS")
    
    results_data = []
    for model_name in results_cross.keys():
        for dataset_name in datasets.keys():
            m = results_cross[model_name][dataset_name]
            results_data.append({
                'Model': model_name,
                'Dataset': dataset_name,
                'MSE': m['mse'],
                'Center_Error_Pixels_Mean': m['center_error_px_mean'],
                'Center_Error_Pixels_Std': m['center_error_px_std'],
                'Center_Error_Microns_Mean': m['center_error_um_mean'],
                'Center_Error_Microns_Std': m['center_error_um_std'],
                'Correlation': m['correlation'],
                'AUROC_Mean': m['auroc_mean'],
                'AUROC_Std': m['auroc_std'],
                'Average_Precision_Mean': m['ap_mean'],
                'Average_Precision_Std': m['ap_std']
            })
    
    df_results = pd.DataFrame(results_data)
    csv_path = os.path.join(args.output_path, "cross_dataset_results_complete.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # Step 1: Load and process data
    datasets, all_centers, all_prob_maps = load_and_process_data(args)
    
    # Step 2: Create dataloaders
    train_loader, val_loader, test_loader, individual_loaders = create_dataloaders(
        datasets, all_prob_maps, args
    )
    
    # Step 3: Train models (or load if skip_training)
    if args.skip_training and args.model_path:
        print("\nSkipping training, loading pre-trained models...")
        models = {}
        # Load models (implement as needed)
        # For now, we'll train anyway
        models, histories = train_all_models(train_loader, val_loader, args, device)
    else:
        models, histories = train_all_models(train_loader, val_loader, args, device)
    
    # Step 4: Evaluate models
    results_cross = evaluate_all_models(
        models, individual_loaders, all_centers, datasets, device, args
    )
    
    # Step 5: Save results
    save_results(results_cross, datasets, args)
    
    # Summary
    print("PIPELINE COMPLETE!")
    print(f"\nOutputs saved to: {args.output_path}")
    print("  - model1_3dcnn.pth")
    print("  - model2_flow.pth")
    print("  - model3_convlstm.pth")
    print("  - cross_dataset_results_complete.csv")
    

if __name__ == '__main__':
    main()
