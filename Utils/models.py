#Neural Network Models for ASL Recognition
#Supports LSTM, Transformer, and Temporal Convolutional Networks

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from typing import Optional, Union

from config import ModelConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer"""
    
    pe: torch.Tensor
    d_model: int

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LSTMModel(nn.Module):
    """
    Bidirectional LSTM for sequence classification.
    
    Architecture:
    - Input projection
    - Bidirectional LSTM layers
    - Attention pooling
    - Classification head
    """
    
    def __init__(
        self,
        input_dim: int = 126,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 1000,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism for temporal pooling
        lstm_output_dim = hidden_dim * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        seq_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            seq_lens: Actual sequence lengths for masking
            
        Returns:
            logits: Classification logits of shape (batch, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        projected = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # Pack sequences if lengths provided
        lstm_input: Union[torch.Tensor, PackedSequence]
        if seq_lens is not None:
            lstm_input = pack_padded_sequence(
                projected, seq_lens.cpu(), batch_first=True, enforce_sorted=False
            )
        else:
            lstm_input = projected
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(lstm_input)
        
        # Unpack if needed
        if isinstance(lstm_out, PackedSequence):
            lstm_out, _ = pad_packed_sequence(
                lstm_out, batch_first=True, total_length=seq_len
            )
        
        # Attention-based pooling
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        
        # Mask attention for padded positions
        if seq_lens is not None:
            mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
            mask = mask >= seq_lens.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1), float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, lstm_output_dim)
        
        # Classification
        logits = self.classifier(context)
        
        return logits


class TransformerModel(nn.Module):
    """
    Transformer Encoder for sequence classification.
    
    Architecture:
    - Input projection
    - Positional encoding
    - Transformer encoder layers
    - Global average pooling or CLS token
    - Classification head
    """
    
    def __init__(
        self,
        input_dim: int = 126,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = 1000,
        dropout: float = 0.1,
        max_len: int = 100
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len + 1, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        seq_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            seq_lens: Actual sequence lengths for masking
            
        Returns:
            logits: Classification logits of shape (batch, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len + 1, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask for padded positions
        src_key_padding_mask = None
        if seq_lens is not None:
            # Account for CLS token
            mask = torch.arange(seq_len + 1, device=x.device).expand(batch_size, seq_len + 1)
            mask = mask > seq_lens.unsqueeze(1)  # CLS token (position 0) is always valid
            src_key_padding_mask = mask
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Use CLS token representation
        cls_output = x[:, 0, :]  # (batch, d_model)
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits


class TemporalConvBlock(nn.Module):
    """Single temporal convolutional block with residual connection"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, channels, seq_len)
        """
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out + residual)
        
        return out


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for sequence classification.
    
    Uses dilated causal convolutions to capture long-range dependencies.
    """
    
    def __init__(
        self,
        input_dim: int = 126,
        num_channels: list = [128, 256, 256, 512],
        kernel_size: int = 3,
        num_classes: int = 1000,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, num_channels[0], 1)
        
        # TCN blocks with increasing dilation
        layers = []
        for i in range(len(num_channels)):
            in_ch = num_channels[i] if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation = 2 ** i
            
            layers.append(TemporalConvBlock(
                in_ch, out_ch, kernel_size, dilation, dropout
            ))
        
        self.tcn = nn.Sequential(*layers)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, num_classes)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        seq_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            seq_lens: Actual sequence lengths (used for masking before pooling)
            
        Returns:
            logits: Classification logits of shape (batch, num_classes)
        """
        # Transpose for conv1d: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # TCN forward
        x = self.tcn(x)
        
        # Mask padded positions before pooling
        if seq_lens is not None:
            batch_size, channels, seq_len = x.shape
            mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
            mask = mask >= seq_lens.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1), 0)
            
            # Manual average over valid positions
            valid_lens = seq_lens.float().clamp(min=1).unsqueeze(1)
            x = x.sum(dim=2) / valid_lens
        else:
            x = self.global_pool(x).squeeze(-1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class ASLRecognitionModel(nn.Module):
    """
    Wrapper class to create the appropriate model based on config.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        if config.model_type == "lstm":
            self.model = LSTMModel(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_classes=config.num_classes,
                dropout=config.dropout,
                bidirectional=config.bidirectional
            )
        elif config.model_type == "transformer":
            self.model = TransformerModel(
                input_dim=config.input_dim,
                d_model=config.d_model,
                nhead=config.nhead,
                num_encoder_layers=config.num_encoder_layers,
                dim_feedforward=config.dim_feedforward,
                num_classes=config.num_classes,
                dropout=config.transformer_dropout
            )
        elif config.model_type == "tcn":
            self.model = TCNModel(
                input_dim=config.input_dim,
                num_classes=config.num_classes,
                dropout=config.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        self.model_type = config.model_type
    
    def forward(
        self, 
        x: torch.Tensor, 
        seq_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.model(x, seq_lens)
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: ModelConfig) -> ASLRecognitionModel:
    """Factory function to create model"""
    model = ASLRecognitionModel(config)
    print(f"Created {config.model_type.upper()} model with {model.count_parameters():,} parameters")
    return model


if __name__ == "__main__":
    # Test models
    from config import get_config
    
    config = get_config()
    model_config = config["model"]
    
    # Test input
    batch_size = 4
    seq_len = 60
    input_dim = model_config.input_dim
    
    x = torch.randn(batch_size, seq_len, input_dim)
    seq_lens = torch.tensor([60, 45, 30, 50])
    
    print("Testing models...")
    
    for model_type in ["lstm", "transformer", "tcn"]:
        print(f"\n--- {model_type.upper()} Model ---")
        model_config.model_type = model_type
        model = create_model(model_config)
        
        output = model(x, seq_lens)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Parameters: {model.count_parameters():,}")
