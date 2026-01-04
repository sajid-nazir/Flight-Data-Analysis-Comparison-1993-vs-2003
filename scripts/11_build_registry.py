#!/usr/bin/env python3
"""
Stage 11: Build registries for future web app wiring

This script:
1. Scans all visualization files (Plotly JSON + PNG)
2. Creates viz_registry.json mapping charts to files and metadata
3. Creates drilldown_registry.json mapping drilldown types to tables
"""
import sys
from pathlib import Path
import json
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.flight_delay.config import load_config

def extract_chart_info(viz_path: Path) -> dict:
    """Extract chart information from file path."""
    # Pattern: viz_##_description.plotly.json
    match = re.match(r'viz_(\d+)_(.+)\.plotly\.json', viz_path.name)
    if match:
        chart_id = int(match.group(1))
        description = match.group(2).replace('_', ' ')
        return {
            'chart_id': chart_id,
            'description': description,
            'viz_file': str(viz_path.relative_to(project_root)),
            'png_file': str(viz_path.parent.parent.parent / 'figures' / viz_path.parent.name / f"fig_{chart_id:02d}_{match.group(2)}.png")
        }
    return None

def main():
    """Main execution function for Stage 11."""
    print("=" * 60)
    print("Stage 11: Build Registries")
    print("=" * 60)
    
    # Load configuration
    print("\n[1/3] Loading configuration...")
    try:
        config = load_config("config/params.yaml")
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        sys.exit(1)
    
    # Create output directory
    registry_dir = project_root / "outputs" / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    
    # Build visualization registry
    print("\n[2/3] Building visualization registry...")
    viz_dir = project_root / "outputs" / "viz"
    viz_registry = []
    
    # Scan all visualization directories
    for stage_dir in sorted(viz_dir.iterdir()):
        if not stage_dir.is_dir():
            continue
        
        stage_name = stage_dir.name
        for viz_file in sorted(stage_dir.glob("*.plotly.json")):
            chart_info = extract_chart_info(viz_file)
            if chart_info:
                chart_info['stage'] = stage_name
                chart_info['click_key'] = None  # Will be set based on chart type
                
                # Determine click key based on description
                desc_lower = chart_info['description'].lower()
                if 'carrier' in desc_lower:
                    chart_info['click_key'] = 'carrier'
                elif 'origin' in desc_lower or 'airport' in desc_lower:
                    chart_info['click_key'] = 'origin'
                elif 'route' in desc_lower:
                    chart_info['click_key'] = 'route'
                
                viz_registry.append(chart_info)
    
    # Sort by chart_id
    viz_registry.sort(key=lambda x: x['chart_id'])
    
    # Save visualization registry
    viz_registry_path = registry_dir / "viz_registry.json"
    with open(viz_registry_path, 'w') as f:
        json.dump(viz_registry, f, indent=2)
    print(f"  ✓ Saved: {viz_registry_path}")
    print(f"  Total charts: {len(viz_registry)}")
    
    # Build drilldown registry
    print("\n[3/3] Building drilldown registry...")
    drilldown_dir = project_root / "outputs" / "tables" / "drilldown"
    
    drilldown_registry = {
        'carrier': {
            'monthly': {
                '1993': str(drilldown_dir / "tbl_dd_01_carrier_monthly_1993.parquet"),
                '2003': str(drilldown_dir / "tbl_dd_02_carrier_monthly_2003.parquet")
            },
            'dep_hour': {
                '1993': str(drilldown_dir / "tbl_dd_03_carrier_dep_hour_1993.parquet"),
                '2003': str(drilldown_dir / "tbl_dd_04_carrier_dep_hour_2003.parquet")
            },
            'top_routes': {
                '1993': str(drilldown_dir / "tbl_dd_05_carrier_top_routes_1993.parquet"),
                '2003': str(drilldown_dir / "tbl_dd_06_carrier_top_routes_2003.parquet")
            }
        },
        'origin': {
            'monthly': {
                '1993': str(drilldown_dir / "tbl_dd_07_origin_monthly_1993.parquet"),
                '2003': str(drilldown_dir / "tbl_dd_08_origin_monthly_2003.parquet")
            },
            'dep_hour': {
                '1993': str(drilldown_dir / "tbl_dd_09_origin_dep_hour_1993.parquet"),
                '2003': str(drilldown_dir / "tbl_dd_10_origin_dep_hour_2003.parquet")
            },
            'top_dests': {
                '1993': str(drilldown_dir / "tbl_dd_11_origin_top_dests_1993.parquet"),
                '2003': str(drilldown_dir / "tbl_dd_12_origin_top_dests_2003.parquet")
            }
        },
        'route': {
            'monthly': {
                '1993': str(drilldown_dir / "tbl_dd_13_route_monthly_1993.parquet"),
                '2003': str(drilldown_dir / "tbl_dd_14_route_monthly_2003.parquet")
            },
            'dep_hour': {
                '1993': str(drilldown_dir / "tbl_dd_15_route_dep_hour_1993.parquet"),
                '2003': str(drilldown_dir / "tbl_dd_16_route_dep_hour_2003.parquet")
            }
        },
        'query_templates': {
            'sample_rows': 'SELECT * FROM read_parquet(\'parquet/clean/common/year={year}/**/*.parquet\') WHERE {key_col} = \'{key_value}\' ORDER BY RANDOM() LIMIT 500',
            'delay_histogram': 'SELECT FLOOR(ArrDelay / 10) * 10 as delay_bin, COUNT(*) as count FROM read_parquet(\'parquet/clean/common/year={year}/**/*.parquet\') WHERE {key_col} = \'{key_value}\' GROUP BY delay_bin ORDER BY delay_bin',
            'worst_routes': 'SELECT Origin || \'_\' || Dest as route, COUNT(*) as flights, AVG(ArrDelay) as avg_delay FROM read_parquet(\'parquet/clean/common/year={year}/**/*.parquet\') WHERE {key_col} = \'{key_value}\' GROUP BY route HAVING COUNT(*) >= 10 ORDER BY avg_delay DESC LIMIT 10'
        }
    }
    
    # Save drilldown registry
    drilldown_registry_path = registry_dir / "drilldown_registry.json"
    with open(drilldown_registry_path, 'w') as f:
        json.dump(drilldown_registry, f, indent=2)
    print(f"  ✓ Saved: {drilldown_registry_path}")
    
    # Summary
    print("\n[4/4] Registry Summary:")
    print("=" * 60)
    print(f"\nVisualization Registry:")
    print(f"  Total charts: {len(viz_registry)}")
    print(f"  Charts with click keys: {sum(1 for v in viz_registry if v['click_key'])}")
    print(f"\nDrilldown Registry:")
    print(f"  Drilldown types: carrier, origin, route")
    print(f"  Total drilldown tables: 16")
    print(f"  Query templates: 3")
    print("\n✓ Stage 11 completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
