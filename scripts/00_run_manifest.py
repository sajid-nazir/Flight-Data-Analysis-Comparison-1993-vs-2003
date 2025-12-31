#!/usr/bin/env python3
"""
Stage 00: Run manifest (metadata snapshot)

This script creates a metadata snapshot of the pipeline run, including:
- Execution timestamp
- Configuration parameters
- Raw input file metadata (size, hash, modification time)
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.flight_delay.config import load_config
from src.flight_delay.io_artifacts import save_json, get_file_metadata


def main():
    """Main execution function for Stage 00."""
    print("=" * 60)
    print("Stage 00: Run Manifest (Metadata Snapshot)")
    print("=" * 60)
    
    # Load configuration
    print("\n[1/4] Loading configuration...")
    try:
        config = load_config("config/params.yaml")
        print(f"✓ Loaded configuration from config/params.yaml")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        sys.exit(1)
    
    # Check for raw CSV files
    print("\n[2/4] Checking raw input files...")
    data_raw_dir = project_root / "data_raw"
    expected_files = ["1993.csv", "2003.csv"]
    
    raw_files_metadata = []
    for filename in expected_files:
        filepath = data_raw_dir / filename
        print(f"  Checking: {filename}...", end=" ")
        
        metadata = get_file_metadata(str(filepath))
        raw_files_metadata.append(metadata)
        
        if metadata["exists"]:
            size_mb = metadata["size_bytes"] / (1024 * 1024)
            print(f"✓ Found ({size_mb:.2f} MB)")
        else:
            print("✗ Not found")
    
    # Build manifest
    print("\n[3/4] Building manifest...")
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "0.1.0",
        "config": config,
        "raw_files": raw_files_metadata
    }
    
    # Save manifest
    print("\n[4/4] Saving manifest...")
    output_path = project_root / "outputs" / "tables" / "run_manifest.json"
    save_json(manifest, str(output_path))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Manifest Summary:")
    print("=" * 60)
    print(f"Timestamp: {manifest['timestamp']}")
    print(f"Pipeline Version: {manifest['pipeline_version']}")
    print(f"Configuration Parameters: {len(config)} items")
    print(f"Raw Files Checked: {len(raw_files_metadata)}")
    files_found = sum(1 for f in raw_files_metadata if f.get("exists", False))
    print(f"Raw Files Found: {files_found}/{len(raw_files_metadata)}")
    
    if files_found < len(raw_files_metadata):
        print("\n⚠ Warning: Some expected raw files are missing.")
        print("  The pipeline may fail in later stages if files are required.")
    
    print("\n✓ Stage 00 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
