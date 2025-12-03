"""
PyTorch Dataset for ASL Recognition
Loads video landmarks and labels for training
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import random

from featureExtractor import (
    HandLandmarkExtractor, 
    FeatureCache, 
    normalize_landmarks
)
from config import DataConfig, TrainingConfig


class ASLDataset(Dataset):
    """
    PyTorch Dataset for ASL video clips.
    
    Loads pre-extracted hand landmarks or extracts them on-the-fly.
    Supports the MS-ASL dataset format.
    """
    
    def __init__(
        self,
        data_dir: str,
        json_path: str,
        max_frames: int = 60,
        frame_skip: int = 1,
        num_hands: int = 2,
        use_cache: bool = True,
        cache_dir: str = "landmark_cache",
        normalize: bool = True,
        augment: bool = False,
        augment_prob: float = 0.5,
        label_mapping: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the ASL dataset.
        
        Args:
            data_dir: Directory containing video folders organized by label
            json_path: Path to MS-ASL JSON annotation file
            max_frames: Maximum sequence length
            frame_skip: Sample every nth frame
            num_hands: Number of hands to detect
            use_cache: Whether to use cached features
            cache_dir: Directory for cached features
            normalize: Whether to normalize landmarks
            augment: Whether to apply data augmentation
            augment_prob: Probability of applying augmentation
            label_mapping: Optional pre-defined label to index mapping
        """
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.frame_skip = frame_skip
        self.num_hands = num_hands
        self.normalize = normalize
        self.augment = augment
        self.augment_prob = augment_prob
        
        # Feature dimension: num_hands * 21 landmarks * 3 coords
        self.feature_dim = num_hands * 21 * 3
        
        # Load annotations
        self.samples = self._load_annotations(json_path)
        
        # Build or use provided label mapping
        if label_mapping is not None:
            self.label_to_idx = label_mapping
        else:
            self.label_to_idx = self._build_label_mapping()
        
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        
        # Setup caching
        self.use_cache = use_cache
        if use_cache:
            self.cache = FeatureCache(cache_dir)
        
        # Setup feature extractor (lazy initialization)
        self._extractor = None
        
        print(f"Loaded {len(self.samples)} samples with {self.num_classes} classes")
    
    @property
    def extractor(self):
        """Lazy initialization of feature extractor"""
        if self._extractor is None:
            self._extractor = HandLandmarkExtractor(num_hands=self.num_hands)
        return self._extractor
    
    def _load_annotations(self, json_path: str) -> List[Dict]:
        """Load annotations from MS-ASL JSON file"""
        samples = []
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                annotations = json.load(f)
            
            for i, ann in enumerate(annotations):
                label = ann['clean_text']
                video_name = f"{label}{i}.mp4"
                video_path = self.data_dir / label / video_name
                
                if video_path.exists():
                    samples.append({
                        'video_path': str(video_path),
                        'label': label,
                        'start_time': ann.get('start_time', 0),
                        'end_time': ann.get('end_time', 0)
                    })
        else:
            # Fallback: load from directory structure
            print(f"JSON not found at {json_path}, loading from directory structure")
            for label_dir in self.data_dir.iterdir():
                if label_dir.is_dir():
                    label = label_dir.name
                    for video_file in label_dir.glob("*.mp4"):
                        samples.append({
                            'video_path': str(video_file),
                            'label': label
                        })
        
        return samples
    
    def _build_label_mapping(self) -> Dict[str, int]:
        """Build label to index mapping"""
        labels = sorted(set(s['label'] for s in self.samples))
        return {label: idx for idx, label in enumerate(labels)}
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a single sample.
        
        Returns:
            landmarks: Tensor of shape (max_frames, feature_dim)
            label: Tensor with class index
            seq_len: Actual sequence length (before padding)
        """
        sample = self.samples[idx]
        video_path = sample['video_path']
        label = sample['label']
        
        # Try to load from cache first
        if self.use_cache and self.cache.exists(video_path):
            landmarks, seq_len = self.cache.load(video_path)
        else:
            # Extract features
            landmarks, seq_len = self.extractor.extract_from_video(
                video_path,
                max_frames=self.max_frames,
                frame_skip=self.frame_skip
            )
            
            # Cache for future use
            if self.use_cache:
                self.cache.save(video_path, landmarks, seq_len)
        
        # Normalize landmarks
        if self.normalize and seq_len > 0:
            landmarks = normalize_landmarks(landmarks)
        
        # Apply augmentation
        if self.augment and np.random.random() < self.augment_prob:
            landmarks = self._apply_augmentation(landmarks, seq_len)
        
        # Convert to tensors
        landmarks_tensor = torch.from_numpy(landmarks).float()
        label_idx = self.label_to_idx[label]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        
        return landmarks_tensor, label_tensor, seq_len
    
    def _apply_augmentation(self, landmarks: np.ndarray, seq_len: int) -> np.ndarray:
        """Apply temporal and spatial augmentation"""
        augmented = landmarks.copy()
        
        # Temporal augmentation: random speed variation
        if np.random.random() < 0.3 and seq_len > 10:
            speed_factor = np.random.uniform(0.8, 1.2)
            new_len = int(seq_len * speed_factor)
            new_len = min(max(new_len, 5), self.max_frames)
            
            old_indices = np.linspace(0, seq_len - 1, new_len)
            new_landmarks = np.zeros_like(augmented)
            for i, idx in enumerate(old_indices):
                lower = int(np.floor(idx))
                upper = min(int(np.ceil(idx)), seq_len - 1)
                weight = idx - lower
                new_landmarks[i] = (1 - weight) * augmented[lower] + weight * augmented[upper]
            augmented = new_landmarks
        
        # Spatial augmentation: mirror (flip left-right)
        if np.random.random() < 0.5:
            # Flip x coordinates
            augmented_reshaped = augmented.reshape(self.max_frames, self.num_hands, 21, 3)
            augmented_reshaped[:, :, :, 0] = 1.0 - augmented_reshaped[:, :, :, 0]
            
            # Swap hands
            if self.num_hands == 2:
                augmented_reshaped = augmented_reshaped[:, ::-1, :, :]
            
            augmented = augmented_reshaped.reshape(self.max_frames, -1)
        
        # Add random noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.02, augmented.shape)
            augmented += noise.astype(np.float32)
        
        return augmented
    
    def get_label_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced data.
        
        Returns:
            weights: Tensor of shape (num_classes,)
        """
        label_counts = defaultdict(int)
        for sample in self.samples:
            label_counts[sample['label']] += 1
        
        weights = []
        total = len(self.samples)
        for idx in range(self.num_classes):
            label = self.idx_to_label[idx]
            count = label_counts.get(label, 1)
            # Inverse frequency weighting
            weights.append(total / (self.num_classes * count))
        
        return torch.tensor(weights, dtype=torch.float32)


class ASLFingerspellingDataset(ASLDataset):
    """
    Specialized dataset for ASL fingerspelling (letters A-Z).
    
    For letter-level recognition, sequences are typically shorter
    and represent single letters rather than words.
    """
    
    LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    def __init__(self, *args, **kwargs):
        # Force label mapping to letters only
        letter_mapping = {letter: idx for idx, letter in enumerate(self.LETTERS)}
        kwargs['label_mapping'] = letter_mapping
        super().__init__(*args, **kwargs)
        
        # Filter samples to only include letter labels
        self.samples = [
            s for s in self.samples 
            if s['label'].upper() in self.LETTERS
        ]
        print(f"Filtered to {len(self.samples)} fingerspelling samples")


def create_data_loaders(
    data_config: DataConfig,
    training_config: TrainingConfig,
    train_json: Optional[str] = None,
    val_json: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Create training and validation data loaders.
    
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        label_mapping: Label to index mapping
    """
    # Create training dataset
    train_dataset = ASLDataset(
        data_dir=data_config.data_dir,
        json_path=train_json or data_config.json_path,
        max_frames=data_config.max_frames,
        frame_skip=data_config.frame_skip,
        num_hands=data_config.num_hands,
        use_cache=data_config.use_cache,
        cache_dir=data_config.cache_dir,
        normalize=True,
        augment=training_config.augment,
        augment_prob=training_config.augment_prob
    )
    
    # Create validation dataset with same label mapping
    val_dataset = ASLDataset(
        data_dir=data_config.data_dir,
        json_path=val_json or data_config.val_json_path,
        max_frames=data_config.max_frames,
        frame_skip=data_config.frame_skip,
        num_hands=data_config.num_hands,
        use_cache=data_config.use_cache,
        cache_dir=data_config.cache_dir,
        normalize=True,
        augment=False,  # No augmentation for validation
        label_mapping=train_dataset.label_to_idx
    )
    
    # Custom collate function to handle variable sequence lengths
    def collate_fn(batch):
        landmarks, labels, seq_lens = zip(*batch)
        landmarks = torch.stack(landmarks)
        labels = torch.stack(labels)
        seq_lens = torch.tensor(seq_lens, dtype=torch.long)
        return landmarks, labels, seq_lens
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, train_dataset.label_to_idx


def create_split_from_single_json(
    data_dir: str,
    json_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split a single JSON annotation file into train/val/test sets.
    
    Ensures balanced splits by stratifying by label.
    """
    random.seed(seed)
    
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    # Group by label
    label_samples = defaultdict(list)
    for i, ann in enumerate(annotations):
        label_samples[ann['clean_text']].append((i, ann))
    
    train_samples, val_samples, test_samples = [], [], []
    
    for label, samples in label_samples.items():
        random.shuffle(samples)
        n = len(samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_samples.extend(samples[:n_train])
        val_samples.extend(samples[n_train:n_train + n_val])
        test_samples.extend(samples[n_train + n_val:])
    
    # Shuffle final lists
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)
    
    return train_samples, val_samples, test_samples


if __name__ == "__main__":
    # Test the dataset
    from config import get_config
    
    config = get_config()
    data_config = config["data"]
    training_config = config["training"]
    
    print("Testing ASLDataset...")
    
    # Check if data exists
    data_path = Path(data_config.data_dir)
    if data_path.exists():
        dataset = ASLDataset(
            data_dir=data_config.data_dir,
            json_path=data_config.json_path,
            max_frames=data_config.max_frames,
            use_cache=data_config.use_cache
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of classes: {dataset.num_classes}")
        print(f"Feature dimension: {dataset.feature_dim}")
        
        # Test loading a sample
        if len(dataset) > 0:
            landmarks, label, seq_len = dataset[0]
            print(f"Sample landmarks shape: {landmarks.shape}")
            label_idx = int(label.item())
            print(f"Sample label: {label_idx} ({dataset.idx_to_label[label_idx]})")
            print(f"Sequence length: {seq_len}")
            
            # Test data loader
            loader = DataLoader(dataset, batch_size=4, shuffle=True)
            batch_landmarks, batch_labels, batch_lens = next(iter(loader))
            print(f"Batch landmarks shape: {batch_landmarks.shape}")
            print(f"Batch labels shape: {batch_labels.shape}")
    else:
        print(f"Data directory not found: {data_config.data_dir}")
        print("Please run download.py to fetch the MS-ASL dataset first.")
