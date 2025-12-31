"""
Artifact I/O utilities
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary to JSON file with proper formatting.
    
    Args:
        data: Dictionary to save
        filepath: Path to output JSON file
        
    Raises:
        OSError: If file cannot be written
    """
    output_path = Path(filepath)
    
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved JSON to: {filepath}")


def get_file_metadata(filepath: str) -> Dict[str, Any]:
    """
    Get metadata for a file including size, hash, and modification time.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Dictionary with file metadata:
        - exists: bool
        - size_bytes: int (if exists)
        - sha256_hash: str (if exists)
        - modified_time: str ISO format (if exists)
    """
    file_path = Path(filepath)
    
    if not file_path.exists():
        return {
            "exists": False,
            "filename": file_path.name
        }
    
    # Get file size
    size_bytes = file_path.stat().st_size
    
    # Compute SHA256 hash
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read file in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    hash_hex = sha256_hash.hexdigest()
    
    # Get modification time
    modified_time = datetime.fromtimestamp(
        file_path.stat().st_mtime
    ).isoformat() + 'Z'
    
    return {
        "exists": True,
        "filename": file_path.name,
        "size_bytes": size_bytes,
        "sha256_hash": hash_hex,
        "modified_time": modified_time
    }
