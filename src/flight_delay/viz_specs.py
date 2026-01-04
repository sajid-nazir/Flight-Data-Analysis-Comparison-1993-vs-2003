"""
Visualization specification utilities (Plotly JSON + PNG)
"""
import json
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional


def save_plotly_json(fig: go.Figure, out_json_path: str) -> None:
    """
    Save Plotly figure as JSON specification.
    
    Args:
        fig: Plotly figure object
        out_json_path: Path to output JSON file
    """
    output_path = Path(out_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use Plotly's built-in JSON encoder which handles numpy arrays
    fig.write_json(str(output_path))
    
    print(f"✓ Saved Plotly JSON to: {out_json_path}")


def save_plotly_png(fig: go.Figure, out_png_path: str, width: int = 1200, height: int = 800) -> None:
    """
    Export Plotly figure as PNG image.
    
    Args:
        fig: Plotly figure object
        out_png_path: Path to output PNG file
        width: Image width in pixels
        height: Image height in pixels
    """
    output_path = Path(out_png_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        fig.write_image(
            str(output_path),
            width=width,
            height=height,
            scale=2  # Higher resolution
        )
        print(f"✓ Saved PNG to: {out_png_path}")
    except Exception as e:
        print(f"⚠ Warning: Could not save PNG ({e}). Make sure kaleido is installed.")


def save_dual(
    fig: go.Figure,
    out_json_path: str,
    out_png_path: str,
    export_png: bool = True,
    width: int = 1200,
    height: int = 800
) -> None:
    """
    Save Plotly figure in both JSON and PNG formats.
    
    Args:
        fig: Plotly figure object
        out_json_path: Path to output JSON file
        out_png_path: Path to output PNG file
        export_png: Whether to export PNG (default: True)
        width: PNG width in pixels
        height: PNG height in pixels
    """
    save_plotly_json(fig, out_json_path)
    
    if export_png:
        save_plotly_png(fig, out_png_path, width=width, height=height)
