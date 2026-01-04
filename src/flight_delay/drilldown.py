"""
Drilldown utilities for web app

NOTE: The actual drilldown implementation is located in:
    - scripts/12_build_drilldowns.py
    
Functions available:
    - create_carrier_monthly_drilldown(): Create carrier monthly drilldown table
    - create_carrier_dep_hour_drilldown(): Create carrier departure hour drilldown table
    - create_carrier_top_routes_drilldown(): Create carrier top routes drilldown table
    - create_origin_monthly_drilldown(): Create origin airport monthly drilldown table
    - create_origin_dep_hour_drilldown(): Create origin airport departure hour drilldown table
    - create_origin_top_dests_drilldown(): Create origin airport top destinations drilldown table
    - create_route_monthly_drilldown(): Create route monthly drilldown table
    - create_route_dep_hour_drilldown(): Create route departure hour drilldown table

This script generates drilldown tables for carrier, origin airport, and route dimensions,
with monthly, hourly, and top routes/destinations breakdowns for web UI panels.
"""

