"""
Utility for exporting model weights as a compressed .gz file.
"""

import gzip
import torch
import io
from typing import Dict, Any


def export_checkpoint_gz(
    checkpoint: Dict[str, Any],
    output_path: str = "model_weights.pth.gz"
) -> str:
    """
    Compress a PyTorch checkpoint dictionary into a .gz file.

    Args:
        checkpoint: dict containing model_state_dict, optimizer_state_dict, config, etc.
        output_path: path where the .gz file should be saved

    Returns:
        output_path: the saved .gz file path
    """
    # Serialize checkpoint to in-memory buffer
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)

    # Compress
    with gzip.open(output_path, 'wb') as f:
        f.write(buffer.read())

    return output_path
