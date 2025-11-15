"""
Miscellaneous utility functions for the knight0 training pipeline.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to log file
        level: Logging level
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def ensure_dir(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def normalize_score(score_cp: float) -> float:
    """
    Normalize a centipawn score to [-1, 1] using tanh.
    
    Args:
        score_cp: Score in centipawns
        
    Returns:
        Normalized score in [-1, 1]
    """
    import math
    return math.tanh(score_cp / 100.0)

