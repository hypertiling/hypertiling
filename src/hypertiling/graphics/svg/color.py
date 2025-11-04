import numpy as np
from typing import Sequence, Optional, Tuple, Union
ColorLike = Union[str, Sequence[float], Sequence[Tuple[float, float, float]]]


def _is_scalar_sequence(x, n: int) -> bool:
    """Check if x is a sequence of n scalar values."""
    try:
        a = np.asarray(x)
        return a.ndim == 1 and a.size == n
    except Exception:
        return False


def _is_rgb_sequence(x, n: int) -> bool:
    """Check if x is a sequence of n RGB tuples."""
    try:
        a = np.asarray(x, dtype=float)
        return a.ndim == 2 and a.shape == (n, 3)
    except Exception:
        return False


def _is_color_string_sequence(obj, n: int) -> bool:
    """Check if obj is a sequence of n color strings (CSS names or hex codes)."""
    if not hasattr(obj, '__len__') or len(obj) != n:
        return False
    
    # Check if all elements are strings that look like colors
    for item in obj:
        if not isinstance(item, str):
            return False
        # Simple check: either starts with # (hex) or is a word (CSS name)
        if not (item.startswith('#') or item.isalpha()):
            return False
    
    return True


def _rgb_to_css(rgb: Sequence[float]) -> str:
    """
    Convert RGB tuple to CSS color string.
    Handles both 0-1 range and 0-255 range automatically.
    """
    arr = np.asarray(rgb, dtype=float)
    if arr.max() <= 1.0:
        arr = np.round(arr * 255.0)
    return f"rgb({int(arr[0])},{int(arr[1])},{int(arr[2])})"


def _resolve_facecolors(facecolors: ColorLike, n: int, cmap: str) -> Tuple[bool, Optional[Sequence[str]]]:
    """
    Returns (individual, colors_css). If individual=False, fill is group-level.
    
    Supports:
    - Single color string (CSS name or hex): "red", "#FF0000"
    - List of color strings: ["red", "#FF0000", "blue"]
    - List of RGB tuples: [(255, 0, 0), (0, 255, 0)] or [(1.0, 0.0, 0.0), ...]
    - List of scalars (mapped via colormap): [0.1, 0.5, 0.9]
    """
    # Single color string (CSS or hex)
    if isinstance(facecolors, str):
        return False, None
    
    # Check if it's a sequence
    if not hasattr(facecolors, '__len__'):
        return False, None
    
    # List of color strings (CSS names or hex)
    if _is_color_string_sequence(facecolors, n):
        return True, list(facecolors)
    
    # List of RGB tuples
    if _is_rgb_sequence(facecolors, n):
        return True, [_rgb_to_css(rgb) for rgb in facecolors]
    
    # List of scalars (map via colormap)
    if _is_scalar_sequence(facecolors, n):
        vals = np.asarray(facecolors, dtype=float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        vals = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
        ccmap = plt.get_cmap(cmap)
        rgba = ccmap(vals)[:, :3]
        return True, [_rgb_to_css(c) for c in rgba]
    
    return False, None