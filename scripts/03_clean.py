#!/usr/bin/env python3
"""
Stage 03: Clean data and build clean Parquet

This script:
1. Removes records with invalid time values (outside 0-2400)
2. Removes cancelled and diverted flights
3. Removes records missing ArrDelay
4. Extracts extreme ArrDelay values (outside [-80, +150] minutes) to separate tables
5. Filters ArrDelay to [-80, +150] range
6. Applies winsorization if configured
7. Writes cleaned Parquet files (including extremes tables)
8. Generates cleaning ledger and visualizations
"""
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.flight_delay.config import load_config
from src.flight_delay.io_duckdb import get_duckdb_connection, write_partitioned_parquet
from src.flight_delay.io_artifacts import save_json
from src.flight_delay.clean import (
    create_clean_table, 
    get_cleaning_stats,
    create_common_columns_table,
    extract_extreme_arrdelay_values
)
from src.flight_delay.viz_specs import save_dual, save_plotly_json


def main():
    """Main execution function for Stage 03."""
    print("=" * 60)
    print("Stage 03: Clean Data and Build Clean Parquet")
    print("=" * 60)
    
    # Load configuration
    print("\n[1/7] Loading configuration...")
    try:
        config = load_config("config/params.yaml")
        export_png = config.get("export_png", True)
        winsorize = config.get("winsorize", False)
        winsor_q_low = config.get("winsor_q_low", 0.005)
        winsor_q_high = config.get("winsor_q_high", 0.995)
        on_time_threshold = config.get("on_time_threshold_min", 15)
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        sys.exit(1)
    
    # Connect to DuckDB
    print("\n[2/7] Connecting to DuckDB...")
    db_path = project_root / "db" / "flights.duckdb"
    conn = get_duckdb_connection(str(db_path))
    print(f"✓ Connected to database")
    
    # Check if raw tables exist
    try:
        conn.execute("SELECT COUNT(*) FROM raw_1993").fetchone()
        conn.execute("SELECT COUNT(*) FROM raw_2003").fetchone()
    except Exception as e:
        print(f"✗ Error: Raw tables not found. Please run Stage 01 first.")
        sys.exit(1)
    
    # Clean each year
    print("\n[3/7] Cleaning data for each year...")
    cleaning_ledger = []
    raw_vs_clean_kpis = []
    
    for year in [1993, 2003]:
        print(f"\n  Processing {year}...")
        source_table = f"raw_{year}"
        clean_table = f"clean_{year}"
        extremes_table = f"extremes_{year}"
        temp_table = f"temp_clean_{year}"
        
        # Get stats from ORIGINAL raw table (before any cleaning)
        # This is critical - we need stats from the source, not from intermediate tables
        print(f"    Getting cleaning stats from raw table...")
        raw_stats = get_cleaning_stats(conn, source_table, year)
        
        # Step 1: Create temp table with all filters EXCEPT ArrDelay range filter
        print(f"    Creating temp table (before ArrDelay filter)...")
        create_clean_table(
            conn,
            source_table,
            temp_table,
            year,
            winsorize=winsorize,
            winsor_q_low=winsor_q_low,
            winsor_q_high=winsor_q_high,
            drop_100pct_missing=True,
            drop_low_missingness_rows=True,
            low_missingness_threshold=0.02,  # 2%
            apply_arrdelay_filter=False  # Don't filter ArrDelay yet
        )
        
        # Step 2: Extract extreme ArrDelay values to separate table
        print(f"    Extracting extreme ArrDelay values...")
        extreme_count = extract_extreme_arrdelay_values(
            conn,
            temp_table,
            extremes_table,
            arrdelay_min=-80,
            arrdelay_max=150
        )
        
        # Step 3: Get count before ArrDelay filter (for ledger)
        # This is the ACTUAL count after all filters except ArrDelay range filter
        before_arrdelay_filter = conn.execute(f"SELECT COUNT(*) FROM {temp_table}").fetchone()[0]
        
        # Step 4: Create final clean table with ArrDelay filter applied
        print(f"    Creating final clean table (with ArrDelay filter)...")
        create_clean_table(
            conn,
            temp_table,
            clean_table,
            year,
            winsorize=False,  # Already applied if needed
            drop_100pct_missing=False,  # Already dropped
            drop_low_missingness_rows=False,  # Already dropped
            apply_arrdelay_filter=True,  # Now apply ArrDelay filter
            arrdelay_min=-80,
            arrdelay_max=150
        )
        
        # Clean up temp table
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        
        # Record cleaning ledger using stats from RAW table
        ledger_entry = {
            'year': year,
            'step': 'initial',
            'row_count': raw_stats['initial'],
            'rows_removed': 0,
            'pct_removed': 0.0
        }
        cleaning_ledger.append(ledger_entry)
        
        prev_count = raw_stats['initial']
        for step_name in ['after_invalid_times', 'after_cancelled', 'after_diverted']:
            current_count = raw_stats.get(step_name, prev_count)
            rows_removed = prev_count - current_count
            pct_removed = (rows_removed / prev_count * 100) if prev_count > 0 else 0
            
            step_display = {
                'after_invalid_times': 'Remove Invalid Times',
                'after_cancelled': 'Remove Cancelled',
                'after_diverted': 'Remove Diverted'
            }.get(step_name, step_name)
            
            ledger_entry = {
                'year': year,
                'step': step_display,
                'row_count': current_count,
                'rows_removed': rows_removed,
                'pct_removed': round(pct_removed, 2)
            }
            cleaning_ledger.append(ledger_entry)
            prev_count = current_count
        
        # Add "Remove Missing ArrDelay + Low Missingness" step
        # Use the ACTUAL count from temp table (before_arrdelay_filter) which includes low missingness filter
        # This is the count after all filters except ArrDelay range filter
        rows_removed_missing = prev_count - before_arrdelay_filter
        pct_removed_missing = (rows_removed_missing / prev_count * 100) if prev_count > 0 else 0
        ledger_entry = {
            'year': year,
            'step': 'Remove Missing ArrDelay + Low Missingness',
            'row_count': before_arrdelay_filter,
            'rows_removed': rows_removed_missing,
            'pct_removed': round(pct_removed_missing, 2)
        }
        cleaning_ledger.append(ledger_entry)
        
        # Add ArrDelay filter step
        after_arrdelay_filter = conn.execute(f"SELECT COUNT(*) FROM {clean_table}").fetchone()[0]
        rows_removed_arrdelay = before_arrdelay_filter - after_arrdelay_filter
        pct_removed_arrdelay = (rows_removed_arrdelay / before_arrdelay_filter * 100) if before_arrdelay_filter > 0 else 0
        ledger_entry = {
            'year': year,
            'step': 'Remove Extreme ArrDelay',
            'row_count': after_arrdelay_filter,
            'rows_removed': rows_removed_arrdelay,
            'pct_removed': round(pct_removed_arrdelay, 2)
        }
        cleaning_ledger.append(ledger_entry)
        
        # Add final step
        final_count = conn.execute(f"SELECT COUNT(*) FROM {clean_table}").fetchone()[0]
        ledger_entry = {
            'year': year,
            'step': 'Final (after winsorization)' if winsorize else 'Final',
            'row_count': final_count,
            'rows_removed': 0,
            'pct_removed': 0.0
        }
        cleaning_ledger.append(ledger_entry)
        
        # Calculate KPIs for raw vs clean
        # Raw KPIs
        raw_kpi_stats = conn.execute(f"""
            SELECT 
                COUNT(*) as total_flights,
                AVG(ArrDelay) as avg_arrdelay,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ArrDelay) as median_arrdelay,
                SUM(CASE WHEN ArrDelay <= {on_time_threshold} THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime15_pct
            FROM {source_table}
            WHERE ArrDelay IS NOT NULL
        """).fetchone()
        
        # Clean KPIs
        clean_kpi_stats = conn.execute(f"""
            SELECT 
                COUNT(*) as total_flights,
                AVG(ArrDelay) as avg_arrdelay,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ArrDelay) as median_arrdelay,
                SUM(CASE WHEN ArrDelay <= {on_time_threshold} THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ontime15_pct
            FROM {clean_table}
        """).fetchone()
        
        raw_vs_clean_kpis.append({
            'year': year,
            'dataset': 'raw',
            'total_flights': raw_kpi_stats[0] if raw_kpi_stats[0] else 0,
            'avg_arrdelay': round(raw_kpi_stats[1], 2) if raw_kpi_stats[1] else None,
            'median_arrdelay': round(raw_kpi_stats[2], 2) if raw_kpi_stats[2] else None,
            'ontime15_pct': round(raw_kpi_stats[3], 2) if raw_kpi_stats[3] else None
        })
        
        raw_vs_clean_kpis.append({
            'year': year,
            'dataset': 'clean',
            'total_flights': clean_kpi_stats[0] if clean_kpi_stats[0] else 0,
            'avg_arrdelay': round(clean_kpi_stats[1], 2) if clean_kpi_stats[1] else None,
            'median_arrdelay': round(clean_kpi_stats[2], 2) if clean_kpi_stats[2] else None,
            'ontime15_pct': round(clean_kpi_stats[3], 2) if clean_kpi_stats[3] else None
        })
        
        # Get final counts
        final_count = conn.execute(f"SELECT COUNT(*) FROM {clean_table}").fetchone()[0]
        extreme_count_final = conn.execute(f"SELECT COUNT(*) FROM {extremes_table}").fetchone()[0]
        print(f"    ✓ Cleaned {year}: {raw_stats['initial']:,} → {final_count:,} rows (removed {extreme_count_final:,} extreme ArrDelay rows)")
    
    # Save cleaning ledger
    print("\n[4/7] Saving cleaning ledger...")
    cleaning_dir = project_root / "outputs" / "tables" / "cleaning"
    cleaning_dir.mkdir(parents=True, exist_ok=True)
    
    ledger_df = pd.DataFrame(cleaning_ledger)
    ledger_path = cleaning_dir / "tbl_08_cleaning_ledger.csv"
    ledger_df.to_csv(ledger_path, index=False)
    print(f"✓ Saved: {ledger_path}")
    
    # Save raw vs clean KPIs
    kpis_df = pd.DataFrame(raw_vs_clean_kpis)
    kpis_path = cleaning_dir / "tbl_09_raw_vs_clean_kpis.csv"
    kpis_df.to_csv(kpis_path, index=False)
    print(f"✓ Saved: {kpis_path}")
    
    # Create common-columns versions (for fair comparison)
    print("\n[5/8] Creating common-columns versions...")
    
    # First create 1993 common (no reference needed)
    clean_table_1993 = "clean_1993"
    common_table_1993 = "clean_common_1993"
    create_common_columns_table(conn, clean_table_1993, common_table_1993, 1993)
    print(f"  ✓ Created common-columns version for 1993")
    
    # Then create 2003 common using 1993 as reference to ensure same columns
    clean_table_2003 = "clean_2003"
    common_table_2003 = "clean_common_2003"
    create_common_columns_table(conn, clean_table_2003, common_table_2003, 2003, 
                               reference_table=common_table_1993)
    print(f"  ✓ Created common-columns version for 2003 (matched to 1993 columns)")
    
    # Write cleaned Parquet files
    print("\n[6/8] Writing cleaned Parquet files...")
    clean_parquet_dir = project_root / "parquet" / "clean"
    
    # Clear existing clean parquet directory
    if clean_parquet_dir.exists():
        shutil.rmtree(clean_parquet_dir)
    clean_parquet_dir.mkdir(parents=True, exist_ok=True)
    
    for year in [1993, 2003]:
        # Write common-columns version (primary for comparison)
        common_table = f"clean_common_{year}"
        output_dir = str(clean_parquet_dir / "common" / f"year={year}")
        write_partitioned_parquet(
            conn,
            common_table,
            output_dir,
            partition_cols=['year', 'month'],
            overwrite=True
        )
        print(f"  ✓ Wrote common-columns Parquet for {year}")
        
        # Write full version for 2003 (with delay breakdown for diagnostic analysis)
        if year == 2003:
            clean_table = f"clean_{year}"
            output_dir = str(clean_parquet_dir / "full" / f"year={year}")
            write_partitioned_parquet(
                conn,
                clean_table,
                output_dir,
                partition_cols=['year', 'month'],
                overwrite=True
            )
            print(f"  ✓ Wrote full Parquet for {year} (with delay breakdown)")
        
        # Write extremes table
        extremes_table = f"extremes_{year}"
        extremes_count = conn.execute(f"SELECT COUNT(*) FROM {extremes_table}").fetchone()[0]
        if extremes_count > 0:
            output_dir = str(clean_parquet_dir / "extremes" / f"year={year}")
            write_partitioned_parquet(
                conn,
                extremes_table,
                output_dir,
                partition_cols=['year', 'month'],
                overwrite=True
            )
            print(f"  ✓ Wrote extremes Parquet for {year} ({extremes_count:,} rows)")
    
    # Generate visualizations
    print("\n[7/8] Generating visualizations...")
    viz_dir = project_root / "outputs" / "viz" / "cleaning"
    fig_dir = project_root / "outputs" / "figures" / "cleaning"
    viz_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualization 1 & 2: Cleaning waterfall for each year
    for year in [1993, 2003]:
        print(f"  Creating cleaning waterfall for {year}...", end=" ")
        year_ledger = ledger_df[ledger_df['year'] == year].copy().reset_index(drop=True)
        
        # Calculate proper waterfall measures and values
        # First step: absolute (starting point)
        # Intermediate steps: relative (change from previous)
        # Last step: total (final count)
        measures = []
        y_values = []
        
        for i in range(len(year_ledger)):
            if i == 0:
                # First step is absolute
                measures.append('absolute')
                y_values.append(year_ledger.iloc[i]['row_count'])
            elif i == len(year_ledger) - 1:
                # Last step is total
                measures.append('total')
                y_values.append(year_ledger.iloc[i]['row_count'])
            else:
                # Intermediate steps show the change (delta)
                prev_count = year_ledger.iloc[i-1]['row_count']
                current_count = year_ledger.iloc[i]['row_count']
                change = current_count - prev_count
                measures.append('relative')
                y_values.append(change)  # Negative for removals, positive if count increases
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=measures,
            x=year_ledger['step'].tolist(),
            textposition="outside",
            text=[f"{val:,.0f}" for val in year_ledger['row_count'].tolist()],
            y=y_values,
            connector={"line": {"color": "rgb(63, 63, 63)", "width": 2}},
        ))
        
        fig.update_layout(
            title=dict(
                text=f"Data Cleaning Waterfall - {year}<br><sub>Row Counts After Each Filtering Step</sub>",
                x=0.5,
                font=dict(size=22, family='Arial Black', color='#1a1a1a'),
                y=0.98
            ),
            xaxis=dict(
                title=dict(text="Cleaning Step", font=dict(size=14, family='Arial Black')),
                tickfont=dict(size=11, family='Arial'),
                tickangle=-45
            ),
            yaxis=dict(
                title=dict(text="Row Count", font=dict(size=14, family='Arial Black')),
                tickfont=dict(size=12, family='Arial')
            ),
            template="plotly_white",
            height=600,
            font=dict(family="Arial", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=40, t=100, b=120)
        )
        
        viz_num = 6 if year == 1993 else 7
        save_dual(
            fig,
            str(viz_dir / f"viz_{viz_num:02d}_cleaning_waterfall_{year}.plotly.json"),
            str(fig_dir / f"fig_{viz_num:02d}_cleaning_waterfall_{year}.png"),
            export_png=export_png
        )
        print("✓")
    
    # Visualization 3 & 4: ArrDelay distribution before/after cleaning
    for year in [1993, 2003]:
        print(f"  Creating ArrDelay distribution comparison for {year}...", end=" ")
        
        # Get raw distribution - USE ALL DATA, not a sample
        print(f"(loading all raw data...", end=" ")
        raw_data = conn.execute(f"""
            SELECT ArrDelay
            FROM raw_{year}
            WHERE ArrDelay IS NOT NULL
        """).fetchall()
        raw_delays = [r[0] for r in raw_data]
        print(f"{len(raw_delays):,} rows)", end=" ")
        
        # Get clean distribution - USE ALL DATA, not a sample
        print(f"(loading all clean data...", end=" ")
        clean_data = conn.execute(f"""
            SELECT ArrDelay
            FROM clean_{year}
        """).fetchall()
        clean_delays = [r[0] for r in clean_data]
        print(f"{len(clean_delays):,} rows)", end=" ")
        
        # Calculate proper axis ranges from actual data
        all_delays = raw_delays + clean_delays
        x_min = min(all_delays) if all_delays else -100
        x_max = max(all_delays) if all_delays else 200
        # Add padding to ensure negative values are clearly visible
        x_min = x_min - 20  # More padding for negatives
        x_max = x_max + 20
        
        fig = go.Figure()
        
        # Raw distribution
        fig.add_trace(go.Histogram(
            x=raw_delays,
            name='Raw',
            opacity=0.7,
            marker=dict(color='#E74C3C', line=dict(color='#C0392B', width=1)),
            histnorm='probability density',
            nbinsx=200  # More bins to show detail including negatives
        ))
        
        # Clean distribution
        fig.add_trace(go.Histogram(
            x=clean_delays,
            name='Clean',
            opacity=0.7,
            marker=dict(color='#2E86AB', line=dict(color='#1a5f7a', width=1)),
            histnorm='probability density',
            nbinsx=200  # More bins to show detail including negatives
        ))
        
        fig.update_layout(
            title=dict(
                text=f"Arrival Delay Distribution - {year}<br><sub>Raw vs Clean Data</sub>",
                x=0.5,
                font=dict(size=22, family='Arial Black', color='#1a1a1a'),
                y=0.98
            ),
            xaxis=dict(
                title=dict(text="Arrival Delay (minutes)", font=dict(size=14, family='Arial Black')),
                tickfont=dict(size=12, family='Arial'),
                range=[x_min, x_max],  # Force explicit range
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=3,
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1
            ),
            yaxis=dict(
                title=dict(text="Probability Density", font=dict(size=14, family='Arial Black')),
                tickfont=dict(size=12, family='Arial')
            ),
            template="plotly_white",
            height=600,
            barmode='overlay',
            font=dict(family="Arial", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=13, family='Arial Black')
            ),
            margin=dict(l=70, r=40, t=100, b=60)
        )
        
        viz_num = 8 if year == 1993 else 9
        save_dual(
            fig,
            str(viz_dir / f"viz_{viz_num:02d}_arrdelay_before_after_{year}.plotly.json"),
            str(fig_dir / f"fig_{viz_num:02d}_arrdelay_before_after_{year}.png"),
            export_png=export_png
        )
        print("✓")
    
    # Visualization 5: OnTime15 rate comparison
    print("  Creating OnTime15 rate comparison...", end=" ")
    kpis_pivot = kpis_df.pivot(index='year', columns='dataset', values='ontime15_pct')
    
    fig = go.Figure()
    
    years = kpis_pivot.index.astype(str)
    fig.add_trace(go.Bar(
        name='Raw',
        x=years,
        y=kpis_pivot['raw'].values,
        marker=dict(color='#E74C3C', line=dict(color='white', width=2)),
        text=[f"{val:.1f}%" for val in kpis_pivot['raw'].values],
        textposition="outside",
        textfont=dict(size=14, family='Arial Black', color='#1a1a1a')
    ))
    
    fig.add_trace(go.Bar(
        name='Clean',
        x=years,
        y=kpis_pivot['clean'].values,
        marker=dict(color='#2E86AB', line=dict(color='white', width=2)),
        text=[f"{val:.1f}%" for val in kpis_pivot['clean'].values],
        textposition="outside",
        textfont=dict(size=14, family='Arial Black', color='#1a1a1a')
    ))
    
    fig.update_layout(
        title=dict(
            text="On-Time Performance (≤15 min delay)<br><sub>Raw vs Clean Data Comparison</sub>",
            x=0.5,
            font=dict(size=22, family='Arial Black', color='#1a1a1a'),
            y=0.98
        ),
        xaxis=dict(
            title=dict(text="Year", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=13, family='Arial')
        ),
        yaxis=dict(
            title=dict(text="On-Time Rate (%)", font=dict(size=14, family='Arial Black')),
            tickfont=dict(size=12, family='Arial'),
            range=[0, 100]
        ),
        template="plotly_white",
        height=600,
        barmode='group',
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13, family='Arial Black')
        ),
        margin=dict(l=60, r=40, t=100, b=60)
    )
    
    save_dual(
        fig,
        str(viz_dir / "viz_10_ontime15_raw_vs_clean.plotly.json"),
        str(fig_dir / "fig_10_ontime15_raw_vs_clean.png"),
        export_png=export_png
    )
    print("✓")
    
    # Summary
    print("\n[8/8] Cleaning Summary:")
    print("=" * 60)
    for year in [1993, 2003]:
        year_ledger = ledger_df[ledger_df['year'] == year]
        initial = year_ledger[year_ledger['step'] == 'initial']['row_count'].iloc[0]
        # Get the last step (final) - look for "Final" step
        final_rows = year_ledger[year_ledger['step'].str.contains('Final', case=False, na=False)]
        if len(final_rows) > 0:
            final = final_rows.iloc[-1]['row_count']
        else:
            # Fallback to last row
            final = year_ledger.iloc[-1]['row_count']
        removed = initial - final
        pct_removed = (removed / initial * 100) if initial > 0 else 0
        
        print(f"\n{year}:")
        print(f"  Initial: {initial:,} rows")
        print(f"  Final: {final:,} rows")
        print(f"  Removed: {removed:,} rows ({pct_removed:.2f}%)")
    
    print("\n✓ Stage 03 completed successfully!")
    print("=" * 60)
    
    # Close connection
    conn.close()


if __name__ == "__main__":
    main()
