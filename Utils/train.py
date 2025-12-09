#Training Script for ASL Recognition
#Full training pipeline with validation, logging, and checkpointing

import os
import time
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Utils.config import get_config, setup_directories, DataConfig, ModelConfig, TrainingConfig
from Utils.dataset import ASLDataset, create_data_loaders
from Utils.models import create_model, ASLRecognitionModel
from Utils.export_gz import export_checkpoint_gz


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def get_scheduler(
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    num_training_steps: int
) -> Optional[LRScheduler]:
    """Create learning rate scheduler"""
    
    warmup_steps = config.warmup_epochs * (num_training_steps // config.num_epochs)
    
    if config.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=config.min_lr
        )
    elif config.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    elif config.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=config.min_lr
        )
    else:
        scheduler: Optional[LRScheduler] = None
    
    # Wrap with warmup if needed
    if warmup_steps > 0 and scheduler is not None:
        from torch.optim.lr_scheduler import LambdaLR
        
        def warmup_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        
        warmup_scheduler: LRScheduler = LambdaLR(optimizer, warmup_lambda)
        return warmup_scheduler  # Use warmup first
    
    return scheduler


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[LRScheduler],
    scaler: Optional[GradScaler],
    device: torch.device,
    config: TrainingConfig,
    epoch: int,
    writer: Optional[SummaryWriter] = None
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        avg_loss: Average training loss
        accuracy: Training accuracy
    """
    model.train()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
    
    for batch_idx, (landmarks, labels, seq_lens) in enumerate(pbar):
        landmarks = landmarks.to(device)
        labels = labels.to(device)
        seq_lens = seq_lens.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if config.use_amp and scaler is not None:
            with autocast():
                outputs = model(landmarks, seq_lens)
                loss = criterion(outputs, labels)
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.max_grad_norms > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norms)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(landmarks, seq_lens)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            if config.max_grad_norms > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norms)
            
            optimizer.step()
        
        # Update scheduler
        if scheduler is not None and config.scheduler != "plateau":
            scheduler.step()
        
        # Calculate accuracy
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        
        # Update meters
        batch_size = landmarks.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        # Log to tensorboard
        if writer is not None and batch_idx % config.log_every == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/batch_loss', loss.item(), global_step)
            writer.add_scalar('train/batch_acc', acc, global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
    
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: TrainingConfig
) -> Tuple[float, float, float]:
    """
    Validate the model.
    
    Returns:
        avg_loss: Average validation loss
        accuracy: Validation accuracy
        top5_accuracy: Top-5 accuracy
    """
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    top5_meter = AverageMeter()
    
    pbar = tqdm(val_loader, desc="Validation")
    
    for landmarks, labels, seq_lens in pbar:
        landmarks = landmarks.to(device)
        labels = labels.to(device)
        seq_lens = seq_lens.to(device)
        
        outputs = model(landmarks, seq_lens)
        loss = criterion(outputs, labels)
        
        # Top-1 accuracy
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        
        # Top-5 accuracy
        _, top5_preds = outputs.topk(5, dim=1)
        top5_correct = top5_preds.eq(labels.unsqueeze(1)).any(dim=1)
        top5_acc = top5_correct.float().mean().item()
        
        batch_size = landmarks.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        top5_meter.update(top5_acc, batch_size)
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}',
            'top5': f'{top5_meter.avg:.4f}'
        })
    
    return loss_meter.avg, acc_meter.avg, top5_meter.avg


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[LRScheduler],
    epoch: int,
    best_acc: float,
    label_mapping: Dict[str, int],
    config: dict,
    checkpoint_dir: str,
    is_best: bool = False
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc,
        'label_mapping': label_mapping,
        'config': config
    }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save periodic checkpoint
    if (epoch + 1) % 5 == 0:
        epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save(checkpoint, epoch_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"  New best model saved with accuracy: {best_acc:.4f}")

        gz_path = os.path.join(checkpoint_dir, "best_model.pth.gz")
        export_checkpoint_gz(checkpoint, gz_path)
        print(f"  Compressed weights exported to {gz_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[LRScheduler] = None
) -> Tuple[int, float, Dict[str, int]]:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return (
        checkpoint.get('epoch', 0),
        checkpoint.get('best_acc', 0.0),
        checkpoint.get('label_mapping', {})
    )


def train(
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    resume_from: Optional[str] = None
):
    """Main training function"""
    
    # Setup
    set_seed(training_config.seed)
    setup_directories({'data': data_config, 'training': training_config})
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nLoading dataset...")
    train_loader, val_loader, label_mapping = create_data_loaders(
        data_config, training_config
    )
    
    # Update model config with actual number of classes
    model_config.num_classes = len(label_mapping)
    print(f"Number of classes: {model_config.num_classes}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(model_config)
    model = model.to(device)
    
    # Loss function with class weights for imbalanced data
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    
    # Scheduler
    num_training_steps = len(train_loader) * training_config.num_epochs
    scheduler = get_scheduler(optimizer, training_config, num_training_steps)
    
    # Mixed precision scaler
    scaler = GradScaler() if training_config.use_amp and torch.cuda.is_available() else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    
    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from checkpoint: {resume_from}")
        start_epoch, best_acc, _ = load_checkpoint(
            resume_from, model, optimizer, scheduler
        )
        start_epoch += 1
        print(f"Resuming from epoch {start_epoch}, best accuracy: {best_acc:.4f}")
    
    # Tensorboard writer
    log_dir = os.path.join(
        training_config.log_dir,
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    writer = SummaryWriter(log_dir)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=training_config.patience,
        min_delta=training_config.min_delta,
        mode='max'
    )
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    config_dict = {
        'data': data_config.__dict__,
        'model': model_config.__dict__,
        'training': training_config.__dict__
    }
    
    for epoch in range(start_epoch, training_config.num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, training_config, epoch, writer
        )
        
        # Validate
        val_loss, val_acc, val_top5 = validate(
            model, val_loader, criterion, device, training_config
        )
        
        # Update plateau scheduler
        if training_config.scheduler == "plateau" and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Log to tensorboard
        writer.add_scalar('epoch/train_loss', train_loss, epoch)
        writer.add_scalar('epoch/train_acc', train_acc, epoch)
        writer.add_scalar('epoch/val_loss', val_loss, epoch)
        writer.add_scalar('epoch/val_acc', val_acc, epoch)
        writer.add_scalar('epoch/val_top5', val_top5, epoch)
        
        # Print summary
        print(f"\nEpoch {epoch + 1}/{training_config.num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Top-5: {val_top5:.4f}")
        
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_acc,
            label_mapping, config_dict, training_config.checkpoint_dir, is_best
        )
        
        # Early stopping
        if early_stopping(val_acc):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    writer.close()
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best validation accuracy: {best_acc:.4f}")
    print(f"Best model saved to: {training_config.checkpoint_dir}/best_model.pth")
    
    return best_acc


def main():
    parser = argparse.ArgumentParser(description='Train ASL Recognition Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='MS-ASL-Train',
                        help='Path to preprocessed video data')
    parser.add_argument('--train_json', type=str, default='../MS-ASL/MSASL_train.json',
                        help='Path to training annotation JSON')
    parser.add_argument('--val_json', type=str, default='../MS-ASL/MSASL_val.json',
                        help='Path to validation annotation JSON')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['lstm', 'transformer', 'tcn'],
                        help='Model architecture')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Get default config
    config = get_config()
    
    # Override with command line arguments
    config['data'].data_dir = args.data_dir
    config['data'].json_path = args.train_json
    config['data'].val_json_path = args.val_json
    
    config['model'].model_type = args.model_type
    
    config['training'].batch_size = args.batch_size
    config['training'].num_epochs = args.epochs
    config['training'].learning_rate = args.lr
    config['training'].seed = args.seed
    config['training'].augment = not args.no_augment
    
    # Train
    train(
        config['data'],
        config['model'],
        config['training'],
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()
