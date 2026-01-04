"""
Registry building utilities for web app

NOTE: The actual registry building implementation is located in:
    - scripts/11_build_registry.py
    
Functions available:
    - extract_chart_info(): Extract chart information from visualization file paths
    
This script:
    1. Scans all visualization files (Plotly JSON + PNG)
    2. Creates viz_registry.json mapping charts to files and metadata
    3. Creates drilldown_registry.json mapping drilldown types to tables

The registries are used for future web app wiring and provide metadata about
available visualizations and drilldown tables.
"""

