"""
Utility scripts for ASL Recognition Pipeline
- Dataset preprocessing (landmark extraction)
- Model evaluation
- Export utilities
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_config, DataConfig, ModelConfig
from dataset import ASLDataset
from models import create_model
from featureExtractor import preprocess_dataset, HandLandmarkExtractor


def preprocess_all_data(data_dir: str, cache_dir: str, max_frames: int = 60):
    """
    Preprocess all videos by extracting and caching landmarks.
    
    This speeds up training significantly by avoiding repeated feature extraction.
    
    Args:
        data_dir: Directory containing video folders
        cache_dir: Directory to save cached features
        max_frames: Maximum frames to extract per video
    """
    print("=" * 60)
    print("Preprocessing Dataset - Extracting Hand Landmarks")
    print("=" * 60)
    
    preprocess_dataset(
        data_dir=data_dir,
        cache_dir=cache_dir,
        max_frames=max_frames,
        frame_skip=1,
        num_hands=2
    )
    
    # Print statistics
    cache_path = Path(cache_dir)
    num_cached = len(list(cache_path.glob("*.pkl")))
    print(f"\nCached {num_cached} video features to {cache_dir}")


def evaluate_model(
    model_path: str,
    data_dir: str,
    json_path: str,
    batch_size: int = 32,
    save_confusion: bool = True,
    output_dir: str = "evaluation"
):
    """
    Comprehensive model evaluation.
    
    Args:
        model_path: Path to trained model checkpoint
        data_dir: Directory containing test videos
        json_path: Path to test annotation JSON
        batch_size: Batch size for evaluation
        save_confusion: Whether to save confusion matrix plot
        output_dir: Directory to save evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    label_mapping = checkpoint.get('label_mapping', {})
    
    # Reconstruct model
    model_config = ModelConfig()
    for key, value in config.get('model', {}).items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
    model_config.num_classes = len(label_mapping)
    
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create dataset
    dataset = ASLDataset(
        data_dir=data_dir,
        json_path=json_path,
        label_mapping=label_mapping,
        augment=False
    )
    
    # Custom collate
    def collate_fn(batch):
        landmarks, labels, seq_lens = zip(*batch)
        return torch.stack(landmarks), torch.stack(labels), torch.tensor(seq_lens)
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for landmarks, labels, seq_lens in tqdm(loader):
            landmarks = landmarks.to(device)
            seq_lens = seq_lens.to(device)
            
            logits = model(landmarks, seq_lens)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    
    # Top-5 accuracy
    top5_correct = 0
    for i, probs in enumerate(all_probs):
        top5_indices = np.argsort(probs)[-5:]
        if all_labels[i] in top5_indices:
            top5_correct += 1
    top5_accuracy = top5_correct / len(all_labels)
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total samples: {len(all_labels)}")
    print(f"Top-1 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")
    
    # Per-class accuracy
    idx_to_label = {v: k for k, v in label_mapping.items()}
    
    # Classification report
    target_names = [idx_to_label.get(i, str(i)) for i in range(len(label_mapping))]
    report = classification_report(
        all_labels, all_preds, 
        target_names=target_names,
        output_dict=True
    )
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'top5_accuracy': float(top5_accuracy),
        'num_samples': len(all_labels),
        'classification_report': report
    }
    
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Confusion matrix (for smaller datasets)
    if save_confusion and len(label_mapping) <= 50:
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names[:len(cm)],
            yticklabels=target_names[:len(cm)]
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150)
        print(f"Confusion matrix saved to {cm_path}")
    
    # Find most confused pairs
    print("\nMost Confused Pairs:")
    cm = confusion_matrix(all_labels, all_preds)
    np.fill_diagonal(cm, 0)  # Remove correct predictions
    
    confused_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i, j] > 0:
                confused_pairs.append((i, j, cm[i, j]))
    
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for true_idx, pred_idx, count in confused_pairs[:10]:
        true_label = idx_to_label.get(true_idx, str(true_idx))
        pred_label = idx_to_label.get(pred_idx, str(pred_idx))
        print(f"  {true_label} -> {pred_label}: {count} times")
    
    return results


def export_to_onnx(
    model_path: str,
    output_path: str = "asl_model.onnx",
    seq_len: int = 60,
    input_dim: int = 126
):
    """
    Export trained model to ONNX format.
    
    Args:
        model_path: Path to trained model checkpoint
        output_path: Output ONNX file path
        seq_len: Expected sequence length
        input_dim: Input feature dimension
    """
    device = torch.device('cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    label_mapping = checkpoint.get('label_mapping', {})
    
    model_config = ModelConfig()
    for key, value in config.get('model', {}).items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
    model_config.num_classes = len(label_mapping)
    
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, seq_len, input_dim)
    dummy_seq_len = torch.tensor([seq_len])
    
    # Export
    torch.onnx.export(
        model,
        (dummy_input, dummy_seq_len),
        output_path,
        input_names=['landmarks', 'seq_len'],
        output_names=['logits'],
        dynamic_axes={
            'landmarks': {0: 'batch_size', 1: 'seq_len'},
            'seq_len': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        opset_version=14
    )
    
    print(f"Model exported to {output_path}")
    
    # Save label mapping alongside
    label_path = output_path.replace('.onnx', '_labels.json')
    with open(label_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"Labels saved to {label_path}")


def visualize_predictions(
    model_path: str,
    video_paths: List[str],
    output_dir: str = "visualizations"
):
    """
    Visualize model predictions on sample videos.
    
    Creates video with landmark overlay and prediction display.
    """
    import cv2
    from featureExtractor import HandLandmarkExtractor, normalize_landmarks
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    label_mapping = checkpoint.get('label_mapping', {})
    idx_to_label = {v: k for k, v in label_mapping.items()}
    
    model_config = ModelConfig()
    for key, value in config.get('model', {}).items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
    model_config.num_classes = len(label_mapping)
    
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    extractor = HandLandmarkExtractor(num_hands=2)
    
    for video_path in video_paths:
        video_name = Path(video_path).stem
        
        # Extract landmarks
        landmarks, num_frames = extractor.extract_from_video(video_path, max_frames=60)
        
        if num_frames == 0:
            print(f"No hands detected in {video_path}")
            continue
        
        # Predict
        normalized = normalize_landmarks(landmarks)
        x = torch.from_numpy(normalized).float().unsqueeze(0).to(device)
        seq_len = torch.tensor([num_frames], device=device)
        
        with torch.no_grad():
            logits = model(x, seq_len)
            probs = F.softmax(logits, dim=1)
            pred_idx = logits.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        pred_label = idx_to_label.get(pred_idx, "Unknown")
        
        print(f"{video_name}: {pred_label} ({confidence:.2%})")


def main():
    parser = argparse.ArgumentParser(description='ASL Recognition Utilities')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess dataset')
    preprocess_parser.add_argument('--data_dir', type=str, required=True)
    preprocess_parser.add_argument('--cache_dir', type=str, default='landmark_cache')
    preprocess_parser.add_argument('--max_frames', type=int, default=60)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model', type=str, required=True)
    eval_parser.add_argument('--data_dir', type=str, required=True)
    eval_parser.add_argument('--json', type=str, required=True)
    eval_parser.add_argument('--batch_size', type=int, default=32)
    eval_parser.add_argument('--output_dir', type=str, default='evaluation')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model to ONNX')
    export_parser.add_argument('--model', type=str, required=True)
    export_parser.add_argument('--output', type=str, default='asl_model.onnx')
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        preprocess_all_data(args.data_dir, args.cache_dir, args.max_frames)
    elif args.command == 'evaluate':
        evaluate_model(
            args.model, args.data_dir, args.json,
            args.batch_size, output_dir=args.output_dir
        )
    elif args.command == 'export':
        export_to_onnx(args.model, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()