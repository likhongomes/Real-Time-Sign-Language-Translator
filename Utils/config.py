#Config settings for training pipeline

from dataclasses import dataclass, field
from typing import List, Optional
import os 

@dataclass
class DataConfig:
    #Data related configurations
    data_dir: str = "MS-ASL-Train"
    json_path: str = "preprocess/MSASL_train.json"
    val_json_path: str = "preprocess/MSASL_val.json"
    test_json_path: str = "preprocess/MSASL_test.json"

    #Feature extraction
    max_frames: int = 60
    min_frames: int = 10
    frame_skip: int = 1

    #Hand landmark settings
    num_hands: int = 2 #maximum hands to detect
    landmarks_per_hand: int = 21 #MediaPipe outputs 21 landmarks per hand
    coords_per_landmark: int = 3 #x,y,z coordinates

    #Cache
    cache_dir: str = "landmark_cache"
    use_cache: bool = True 

@dataclass
class ModelConfig:
    #Model architecture configurations
    model_type: str = "transformer" 

    #Input dimensions
    input_dim: int = 126 #21 landmarks * 3 coordinates * 2 hands

    #LSTM settings
    hidden_dim: int = 256
    num_layers: int = 2
    bidirectional: bool = True 
    dropout: float = 0.3 

    #Transformer settings
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    transformer_dropout: float = 0.1

    #Classification head
    num_classes: int = 1000  #If we wanted to do letter by letter (fingerspelling) we could change this to 26

@dataclass
class TrainingConfig:
    #Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    #Learning rate scheduling
    scheduler: str = "cosine" 
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    #Early stopping
    patience: int = 15
    min_delta: float = 1e-4

    #Gradient clipping
    max_grad_norms: float = 1.0

    #Mixed precision training
    use_amp: bool = True 

    #Data augmentation
    augment: bool = True 
    augment_prob: float = 0.5

    #Checkpointing:
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5 #saving checkpoints every 5 epochs 

    #Logging
    log_dir: str = "logs"
    log_every: int = 100 #logging every 100 batches

    #Reproducibility
    seed: int = 42

    #Hardware
    num_workers: int = 4
    pin_memory: bool = True 

@dataclass
class InferenceConfig:
    #Real time inference configuration
    model_path: str = "checkpoints/best_model.pth"
    camera_id: int = 0
    confidence_threshold: float = 0.5

    #Smoothing predictions over time
    predictions_window: int = 10

    #Display settings
    show_landmarks: bool = True 
    show_confidence: bool = True 

def get_config():
    #Complete configuration
    return {
        "data": DataConfig(),
        "model": ModelConfig(),
        "training": TrainingConfig(),
        "inference": InferenceConfig()
    }

def setup_directories(config: dict):
    #Creating necessary directories
    dirs = [
        config["data"].cache_dir,
        config["training"].checkpoint_dir,
        config["training"].log_dir
    ]

    for d in dirs:
        os.makedirs(d, exist_ok = True)
