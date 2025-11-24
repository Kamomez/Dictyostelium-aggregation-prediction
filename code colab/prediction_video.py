"""
Prediction Video Generator for Slime Mold Aggregation
======================================================
This script generates temporal evolution videos showing how the model's predictions
evolve frame-by-frame, revealing the decision-making process.

Usage:
    python generate_prediction_video.py --data_path /path/to/data --model_path /path/to/model.pth --output_dir ./output

Requirements:
    - torch
    - numpy
    - matplotlib
    - opencv-python (cv2)
    - zarr
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import cv2
import zarr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as patches


# ============================================================================
# Model Definitions (ConvLSTM)
# ============================================================================

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, kernel_size=3, num_layers=2):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            cell_list.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size))
        self.cell_list = nn.ModuleList(cell_list)
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        device = x.device
        
        h = [torch.zeros(B, self.hidden_dim, H, W, device=device) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_dim, H, W, device=device) for _ in range(self.num_layers)]
        
        for t in range(T):
            x_t = x[:, t, :, :, :]
            for layer_idx in range(self.num_layers):
                h[layer_idx], c[layer_idx] = self.cell_list[layer_idx](
                    x_t if layer_idx == 0 else h[layer_idx - 1],
                    (h[layer_idx], c[layer_idx])
                )
        
        output = self.output_conv(h[-1])
        return output


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_zarr_data(zarr_path):
    """Load data from zarr file"""
    print(f"Loading data from {zarr_path}...")
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store)
    
    # Get the data array (usually at root['0'] or similar)
    data = None
    for key in root.array_keys():
        arr = root[key]
        if len(arr.shape) >= 3:  # Time series data
            data = arr[:]
            break
    
    if data is None:
        raise ValueError(f"Could not find valid data array in {zarr_path}")
    
    # Normalize to [0, 1]
    data = data.astype(np.float32)
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    
    print(f"Loaded data shape: {data.shape}")
    return data


def load_aggregation_centers(data_dir, dataset_name):
    """Load aggregation centers from metadata or compute from probability maps"""
    centers = []
    
    # Try to load from metadata.json
    for root_dir, dirs, files in os.walk(os.path.join(data_dir, dataset_name)):
        if 'metadata.json' in files:
            metadata_path = os.path.join(root_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                if 'center_x' in metadata and 'center_y' in metadata:
                    centers.append([metadata['center_x'], metadata['center_y']])
    
    if len(centers) > 0:
        print(f"Loaded {len(centers)} aggregation centers from metadata")
        return np.array(centers)
    
    # Fallback: use center of image
    print("No metadata found, using image center as fallback")
    return None


# ============================================================================
# Video Generation Function
# ============================================================================

def create_prediction_video(model, sequence, true_center,
                           output_path='prediction_evolution.gif',
                           fps=10, pixel_size=7.5, device='cuda'):
    """
    Generate video showing progressive prediction evolution
    
    Args:
        model: Trained prediction model
        sequence: Input sequence tensor (T, C, H, W)
        true_center: Ground truth aggregation center [x, y]
        output_path: Path to save the video
        fps: Frames per second
        pixel_size: Micrometers per pixel
        device: Computation device
    """
    sequence = sequence.to(device)
    T = sequence.shape[0]
    
    predictions = []
    flow_fields = []
    confidences = []
    
    print(f"Generating predictions for {T} frames...")
    model.eval()
    
    with torch.no_grad():
        for t in range(4, T+1):
            input_seq = sequence[:t].unsqueeze(0)
            pred_map = model(input_seq).squeeze().cpu().numpy()
            predictions.append(pred_map)
            
            # Calculate confidence (entropy)
            pred_norm = pred_map / (pred_map.sum() + 1e-8)
            entropy = -np.sum(pred_norm * np.log(pred_norm + 1e-8))
            confidences.append(entropy)
            
            # Compute optical flow
            if t >= 5:
                prev_frame = sequence[t-2, 0].cpu().numpy()
                curr_frame = sequence[t-1, 0].cpu().numpy()
                
                prev_norm = cv2.normalize(prev_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                curr_norm = cv2.normalize(curr_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                flow = cv2.calcOpticalFlowFarneback(
                    prev_norm, curr_norm, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                flow_fields.append(flow)
            else:
                flow_fields.append(None)
    
    print(f"Generated {len(predictions)} predictions. Creating animation...")
    
    # Create animation
    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.5])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ax5 = fig.add_subplot(gs[4])
    
    def update(frame_idx):
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
        
        t = frame_idx + 4
        curr_frame = sequence[t-1, 0].cpu().numpy()
        pred_map = predictions[frame_idx]
        
        pred_y, pred_x = np.unravel_index(np.argmax(pred_map), pred_map.shape)
        
        if true_center is not None:
            error_px = np.sqrt((pred_x - true_center[0])**2 + (pred_y - true_center[1])**2)
            error_um = error_px * pixel_size
        else:
            error_px = 0
            error_um = 0
        
        # 1. Input frame
        ax1.imshow(curr_frame, cmap='gray')
        ax1.set_title(f'Input Frame {t}/{T}\n({t} frames seen)', fontsize=10, fontweight='bold')
        ax1.axis('off')
        
        # 2. Prediction heatmap
        ax2.imshow(pred_map, cmap='hot')
        ax2.scatter(pred_x, pred_y, c='cyan', s=150, marker='x', linewidths=2, label='Predicted')
        if true_center is not None:
            ax2.scatter(true_center[0], true_center[1], c='lime', s=150, marker='+', 
                       linewidths=2, label='True')
        ax2.set_title(f'Prediction Heatmap\nError: {error_px:.1f}px ({error_um:.1f}μm)',
                     fontsize=10, fontweight='bold')
        ax2.legend(fontsize=8, loc='upper right')
        ax2.axis('off')
        
        # 3. Overlay
        ax3.imshow(curr_frame, cmap='gray', alpha=0.6)
        ax3.imshow(pred_map, cmap='hot', alpha=0.4)
        ax3.scatter(pred_x, pred_y, c='cyan', s=150, marker='x', linewidths=2)
        if true_center is not None:
            ax3.scatter(true_center[0], true_center[1], c='lime', s=150, marker='+', linewidths=2)
            circle = patches.Circle((pred_x, pred_y), error_px, fill=False,
                                   edgecolor='cyan', linewidth=2, linestyle='--', alpha=0.5)
            ax3.add_patch(circle)
        ax3.set_title('Overlay: Input + Prediction\n(Dashed circle = error)',
                     fontsize=10, fontweight='bold')
        ax3.axis('off')
        
        # 4. Flow vectors
        flow = flow_fields[frame_idx]
        if flow is not None:
            ax4.imshow(curr_frame, cmap='gray', alpha=0.5)
            step = 8
            h, w = flow.shape[:2]
            y_coords, x_coords = np.mgrid[step//2:h:step, step//2:w:step]
            u = flow[y_coords, x_coords, 0]
            v = flow[y_coords, x_coords, 1]
            magnitude = np.sqrt(u**2 + v**2)
            ax4.quiver(x_coords, y_coords, u, v, magnitude,
                      cmap='jet', scale=50, width=0.003, headwidth=4, headlength=5)
            ax4.scatter(pred_x, pred_y, c='cyan', s=100, marker='x', linewidths=2)
            ax4.set_title('Motion Flow Vectors\n(Movement cues)', fontsize=10, fontweight='bold')
        else:
            ax4.imshow(curr_frame, cmap='gray')
            ax4.set_title('Motion Flow\n(Computing...)', fontsize=10, fontweight='bold')
        ax4.axis('off')
        
        # 5. Confidence plot
        ax5.clear()
        ax5.plot(range(len(confidences[:frame_idx+1])), confidences[:frame_idx+1],
                'b-', linewidth=2)
        ax5.axhline(y=confidences[frame_idx], color='r', linestyle='--', linewidth=1)
        ax5.set_xlabel('Frames Seen', fontsize=9)
        ax5.set_ylabel('Entropy\n(Lower = More Confident)', fontsize=9)
        ax5.set_title('Confidence\nEvolution', fontsize=10, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, len(predictions))
        
        fig.suptitle(f'Progressive Prediction Evolution: Frame {t}/{T} ({t/T*100:.1f}% of data)',
                    fontsize=14, fontweight='bold')
        
        return [ax1, ax2, ax3, ax4, ax5]
    
    anim = FuncAnimation(fig, update, frames=len(predictions),
                        interval=1000//fps, blit=False, repeat=True)
    
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    print(f"✅ Video saved to {output_path}")
    
    plt.close()
    
    return predictions, confidences


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate prediction evolution video')
    parser.add_argument('--data_path', type=str, 
                       default='/content/drive/MyDrive/DictyProject/Data',
                       help='Path to data directory (containing zarr files)')
    parser.add_argument('--dataset_name', type=str, default='mixin_test44',
                       help='Dataset name (e.g., mixin_test44)')
    parser.add_argument('--model_path', type=str,
                       default='/content/drive/MyDrive/DictyProject/Output/model3_convlstm.pth',
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--output_dir', type=str, 
                       default='/content/drive/MyDrive/DictyProject/Output',
                       help='Output directory for video')
    parser.add_argument('--output_name', type=str, default='prediction_evolution.gif',
                       help='Output video filename')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for video')
    parser.add_argument('--pixel_size', type=float, default=7.5,
                       help='Micrometers per pixel')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data_dir = os.path.join(args.data_path, args.dataset_name)
    zarr_files = [f for f in os.listdir(data_dir) if f.endswith('.zarr') and 'subsampled' not in f]
    
    if len(zarr_files) == 0:
        raise ValueError(f"No zarr files found in {data_dir}")
    
    zarr_path = os.path.join(data_dir, zarr_files[0])
    data = load_zarr_data(zarr_path)
    
    # Convert to tensor
    sequence = torch.from_numpy(data).float()
    if len(sequence.shape) == 3:  # (T, H, W)
        sequence = sequence.unsqueeze(1)  # (T, C, H, W)
    
    # Load aggregation centers
    centers = load_aggregation_centers(args.data_path, args.dataset_name)
    true_center = centers[0] if centers is not None else None
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = ConvLSTM(input_dim=1, hidden_dim=64, kernel_size=3, num_layers=2)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Generate video
    output_path = os.path.join(args.output_dir, args.output_name)
    predictions, confidences = create_prediction_video(
        model, sequence, true_center,
        output_path=output_path,
        fps=args.fps,
        pixel_size=args.pixel_size,
        device=device
    )
    
    # Print statistics
    print("\n" + "=" * 80)
    print("VIDEO GENERATION COMPLETE")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Total frames: {len(predictions)}")
    print(f"Video duration: ~{len(predictions)/args.fps:.1f} seconds")
    print(f"Initial confidence (entropy): {confidences[0]:.4f}")
    print(f"Final confidence (entropy): {confidences[-1]:.4f}")
    print(f"Confidence improvement: {(confidences[0]-confidences[-1])/confidences[0]*100:.1f}%")
    print("=" * 80)


if __name__ == '__main__':
    main()
