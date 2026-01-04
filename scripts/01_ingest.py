#!/usr/bin/env python3
"""
Stage 01: Ingest CSV → DuckDB scan → raw Parquet

This script:
1. Reads raw CSV files (1993.csv, 2003.csv) using DuckDB
2. Writes partitioned Parquet files (by year and month)
3. Generates audit tables (row counts, schemas)
4. Creates visualizations (row counts, schema presence matrix)
"""
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.flight_delay.config import load_config
from src.flight_delay.io_duckdb import (
    get_duckdb_connection,
    read_csv_to_duckdb,
    write_partitioned_parquet,
    get_table_schema,
    get_row_count,
    get_row_count_by_partition
)
from src.flight_delay.io_artifacts import save_json
from src.flight_delay.viz_specs import save_dual, save_plotly_json


def normalize_column_types(conn, table_name: str, year: int) -> None:
    """
    Normalize column types to ensure consistency between 1993 and 2003.
    Specifically, cast columns that are 100% missing in 1993 (VARCHAR) 
    to BIGINT to match 2003 types.
    
    Args:
        conn: DuckDB connection
        table_name: Table name to normalize
        year: Year being processed
    """
    # Columns that should be BIGINT (numeric) but may be inferred as VARCHAR
    # when 100% missing in 1993
    numeric_columns = [
        'AirTime',
        'TaxiIn',
        'TaxiOut',
        'CarrierDelay',
        'WeatherDelay',
        'NASDelay',
        'SecurityDelay',
        'LateAircraftDelay'
    ]
    
    # Get current schema
    schema_info = conn.execute(f"DESCRIBE {table_name}").fetchall()
    current_types = {row[0]: row[1] for row in schema_info}
    
    # Build SELECT clause with type casting for columns that need it
    all_columns = [row[0] for row in schema_info]
    select_parts = []
    
    for col in all_columns:
        if col in numeric_columns and col in current_types:
            current_type = current_types[col]
            # If it's VARCHAR (likely because it's 100% missing), cast to BIGINT
            if current_type.upper() == 'VARCHAR':
                select_parts.append(f"CAST({col} AS BIGINT) AS {col}")
            else:
                # Keep original column
                select_parts.append(col)
        else:
            # Keep original column
            select_parts.append(col)
    
    # Only recreate table if we need to cast any columns
    columns_to_cast = [
        col for col in numeric_columns 
        if col in current_types and current_types[col].upper() == 'VARCHAR'
    ]
    
    if columns_to_cast:
        select_clause = ", ".join(select_parts)
        
        # Create temporary table with corrected types
        temp_table = f"{table_name}_temp"
        conn.execute(f"""
            CREATE OR REPLACE TABLE {temp_table} AS
            SELECT {select_clause}
            FROM {table_name}
        """)
        
        # Replace original table
        conn.execute(f"DROP TABLE {table_name}")
        conn.execute(f"ALTER TABLE {temp_table} RENAME TO {table_name}")


def extract_month_from_data(conn, table_name: str) -> None:
    """
    Extract month from date column and add as separate column.
    Tries common date column names.
    """
    # Get column names (preserve original case for SQL queries)
    schema_info = conn.execute(f"DESCRIBE {table_name}").fetchall()
    column_names_lower = [col[0].lower() for col in schema_info]
    column_names_original = [col[0] for col in schema_info]
    
    # Check if month column already exists (case-insensitive)
    if 'month' in column_names_lower:
        # Find the actual column name (preserving case)
        month_col_idx = column_names_lower.index('month')
        month_col_name = column_names_original[month_col_idx]
        
        # If it's already lowercase 'month', we're done
        if month_col_name.lower() == 'month':
            # Just ensure it's integer type
            try:
                conn.execute(f"""
                    ALTER TABLE {table_name} ALTER COLUMN month TYPE INTEGER USING CAST(month AS INTEGER)
                """)
            except:
                # If conversion fails, create a new column
                conn.execute(f"""
                    ALTER TABLE {table_name} ADD COLUMN month_new INTEGER;
                    UPDATE {table_name} SET month_new = CAST({month_col_name} AS INTEGER) WHERE {month_col_name} IS NOT NULL;
                    ALTER TABLE {table_name} DROP COLUMN {month_col_name};
                    ALTER TABLE {table_name} RENAME COLUMN month_new TO month;
                """)
            return
    
    # Try to find date column
    date_col = None
    for possible_col in ['flightdate', 'date', 'yearmonthday', 'depdate', 'arrdate']:
        if possible_col in column_names_lower:
            # Find original case
            idx = column_names_lower.index(possible_col)
            date_col = column_names_original[idx]
            break
    
    if date_col is None:
        # Try to find any column with 'date' in the name
        for i, col_name_lower in enumerate(column_names_lower):
            if 'date' in col_name_lower:
                date_col = column_names_original[i]
                break
    
    if date_col:
        # Extract month from date column
        # Try different date formats
        try:
            conn.execute(f"""
                ALTER TABLE {table_name} ADD COLUMN month INTEGER;
                UPDATE {table_name} 
                SET month = EXTRACT(MONTH FROM CAST({date_col} AS DATE))
                WHERE {date_col} IS NOT NULL
            """)
        except:
            # If date parsing fails, try extracting from string
            try:
                conn.execute(f"""
                    ALTER TABLE {table_name} ADD COLUMN month INTEGER;
                    UPDATE {table_name} 
                    SET month = CAST(SUBSTRING(CAST({date_col} AS VARCHAR), 5, 2) AS INTEGER)
                    WHERE {date_col} IS NOT NULL AND LENGTH(CAST({date_col} AS VARCHAR)) >= 6
                """)
            except:
                print(f"⚠ Warning: Could not extract month from date column {date_col}")
                # Try Month column as fallback
                if 'month' in column_names_lower:
                    month_col_idx = column_names_lower.index('month')
                    month_col_name = column_names_original[month_col_idx]
                    conn.execute(f"""
                        ALTER TABLE {table_name} ADD COLUMN month INTEGER;
                        UPDATE {table_name} 
                        SET month = CAST({month_col_name} AS INTEGER)
                        WHERE {month_col_name} IS NOT NULL
                    """)
    else:
        # Try Month column directly (case-insensitive)
        if 'month' in column_names_lower:
            month_col_idx = column_names_lower.index('month')
            month_col_name = column_names_original[month_col_idx]
            conn.execute(f"""
                ALTER TABLE {table_name} ADD COLUMN month INTEGER;
                UPDATE {table_name} 
                SET month = CAST({month_col_name} AS INTEGER)
                WHERE {month_col_name} IS NOT NULL
            """)
        else:
            print(f"⚠ Warning: No date or month column found in {table_name}")


def main():
    """Main execution function for Stage 01."""
    print("=" * 60)
    print("Stage 01: Ingest CSV → DuckDB scan → raw Parquet")
    print("=" * 60)
    
    # Load configuration
    print("\n[1/6] Loading configuration...")
    try:
        config = load_config("config/params.yaml")
        export_png = config.get("export_png", True)
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        sys.exit(1)
    
    # Check for raw CSV files
    print("\n[2/6] Checking raw input files...")
    data_raw_dir = project_root / "data_raw"
    csv_files = {
        1993: data_raw_dir / "1993.csv",
        2003: data_raw_dir / "2003.csv"
    }
    
    for year, csv_path in csv_files.items():
        if csv_path.exists():
            size_mb = csv_path.stat().st_size / (1024 * 1024)
            print(f"✓ Found {year}.csv ({size_mb:.2f} MB)")
        else:
            print(f"✗ Missing {year}.csv")
            sys.exit(1)
    
    # Create DuckDB connection
    print("\n[3/6] Setting up DuckDB database...")
    db_path = project_root / "db" / "flights.duckdb"
    conn = get_duckdb_connection(str(db_path))
    print(f"✓ Connected to database: {db_path}")
    
    # Ingest each year
    print("\n[4/6] Ingesting CSV files and writing Parquet...")
    row_counts_data = []
    schemas = {}
    
    for year in [1993, 2003]:
        csv_path = csv_files[year]
        table_name = f"raw_{year}"
        
        print(f"\n  Processing {year}...")
        
        # Read CSV into DuckDB
        print(f"    Reading CSV...", end=" ")
        read_csv_to_duckdb(conn, str(csv_path), table_name, year=year)
        total_rows = get_row_count(conn, table_name)
        print(f"✓ ({total_rows:,} rows)")
        
        # Normalize column types (fix type mismatches)
        print(f"    Normalizing column types...", end=" ")
        normalize_column_types(conn, table_name, year)
        print("✓")
        
        # Extract month column
        print(f"    Extracting month...", end=" ")
        extract_month_from_data(conn, table_name)
        print("✓")
        
        # Write partitioned Parquet
        print(f"    Writing partitioned Parquet...", end=" ")
        parquet_dir = project_root / "parquet" / "raw"
        write_partitioned_parquet(
            conn,
            table_name,
            str(parquet_dir),
            partition_cols=["year", "month"]
        )
        print("✓")
        
        # Get row counts by month
        month_counts = get_row_count_by_partition(conn, table_name, ["year", "month"])
        for month_data in month_counts:
            row_counts_data.append({
                "year": month_data["year"],
                "month": month_data["month"],
                "rows_raw": month_data["row_count"]
            })
        
        # Get schema
        schemas[year] = get_table_schema(conn, table_name)
        print(f"    Schema: {schemas[year]['column_count']} columns")
    
    # Save audit tables
    print("\n[5/6] Generating audit tables...")
    audit_dir = project_root / "outputs" / "tables" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    # Row counts CSV
    row_counts_df = pd.DataFrame(row_counts_data)
    row_counts_df = row_counts_df.sort_values(["year", "month"])
    row_counts_path = audit_dir / "tbl_01_ingest_row_counts.csv"
    row_counts_df.to_csv(row_counts_path, index=False)
    print(f"✓ Saved: {row_counts_path}")
    
    # Schema JSON files
    for year in [1993, 2003]:
        schema_path = audit_dir / f"tbl_0{2 if year == 1993 else 3}_ingest_schema_{year}.json"
        save_json(schemas[year], str(schema_path))
    
    # Generate visualizations
    print("\n[6/6] Generating visualizations...")
    viz_dir = project_root / "outputs" / "viz" / "audit"
    fig_dir = project_root / "outputs" / "figures" / "audit"
    viz_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualization 1: Row counts by year with monthly breakdown
    print("  Creating row counts visualization...", end=" ")
    
    # Prepare data for monthly comparison
    row_counts_df['month_name'] = pd.to_datetime(row_counts_df['month'], format='%m').dt.strftime('%b')
    row_counts_1993 = row_counts_df[row_counts_df['year'] == 1993].sort_values('month')
    row_counts_2003 = row_counts_df[row_counts_df['year'] == 2003].sort_values('month')
    
    # Create subplot with monthly trends and year totals
    fig1 = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Monthly Flight Counts Comparison', 'Total Flights by Year'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Monthly comparison line chart
    fig1.add_trace(
        go.Scatter(
            x=row_counts_1993['month_name'],
            y=row_counts_1993['rows_raw'],
            mode='lines+markers',
            name='1993',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8, color='#2E86AB', symbol='circle'),
            fillcolor='rgba(46, 134, 171, 0.1)',
            hovertemplate='<b>1993</b><br>Month: %{x}<br>Flights: %{y:,}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig1.add_trace(
        go.Scatter(
            x=row_counts_2003['month_name'],
            y=row_counts_2003['rows_raw'],
            mode='lines+markers',
            name='2003',
            line=dict(color='#A23B72', width=3),
            marker=dict(size=8, color='#A23B72', symbol='diamond'),
            fillcolor='rgba(162, 59, 114, 0.1)',
            hovertemplate='<b>2003</b><br>Month: %{x}<br>Flights: %{y:,}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Year totals bar chart
    year_totals = row_counts_df.groupby("year")["rows_raw"].sum().reset_index()
    colors = ['#2E86AB', '#A23B72']
    
    # Calculate max value for y-axis padding
    max_val = year_totals["rows_raw"].max()
    y_padding = max_val * 0.15  # 15% padding for text visibility
    
    fig1.add_trace(
        go.Bar(
            x=year_totals["year"].astype(str),
            y=year_totals["rows_raw"],
            name='Year Total',
            marker_color=colors,
            text=[f"{val:,}" for val in year_totals["rows_raw"]],
            textposition="outside",
            textfont=dict(size=14, color='black', family='Arial Black'),
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Total Flights: %{y:,}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update y-axis range for bottom subplot to accommodate text
    fig1.update_yaxes(range=[0, max_val + y_padding], row=2, col=1)
    
    # Update layout
    fig1.update_xaxes(title_text="Month", row=1, col=1, tickangle=-45)
    fig1.update_xaxes(title_text="Year", row=2, col=1)
    fig1.update_yaxes(title_text="Number of Flights", row=1, col=1)
    fig1.update_yaxes(title_text="Total Flights", row=2, col=1)
    
    fig1.update_layout(
        title=dict(
            text="Flight Data Volume: 1993 vs 2003",
            x=0.5,
            font=dict(size=20, family='Arial Black', color='#1a1a1a')
        ),
        template="plotly_white",
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            traceorder="normal"
        ),
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Save combined visualization
    save_dual(
        fig1,
        str(viz_dir / "viz_01_row_counts_by_year.plotly.json"),
        str(fig_dir / "fig_01_row_counts_by_year.png"),
        export_png=export_png
    )
    
    # Save individual components
    viz1_dir = viz_dir / "viz_01_row_counts_by_year"
    viz1_dir.mkdir(exist_ok=True)
    
    # Individual: Monthly comparison line chart
    fig1a = go.Figure()
    fig1a.add_trace(fig1.data[0])  # 1993 line
    fig1a.add_trace(fig1.data[1])  # 2003 line
    fig1a.update_xaxes(title_text="Month", tickangle=-45)
    fig1a.update_yaxes(title_text="Number of Flights")
    fig1a.update_layout(
        title=dict(text="Monthly Flight Counts Comparison", x=0.5, font=dict(size=18, family='Arial Black')),
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        height=500
    )
    save_plotly_json(fig1a, str(viz1_dir / "monthly_comparison.plotly.json"))
    
    # Individual: Year totals bar chart
    fig1b = go.Figure()
    fig1b.add_trace(fig1.data[2])  # Year totals bar
    fig1b.update_xaxes(title_text="Year")
    fig1b.update_yaxes(title_text="Total Flights", range=[0, max_val + y_padding])
    fig1b.update_layout(
        title=dict(text="Total Flights by Year", x=0.5, font=dict(size=18, family='Arial Black')),
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        height=400
    )
    save_plotly_json(fig1b, str(viz1_dir / "year_totals.plotly.json"))
    
    print("✓")
    
    # Visualization 2: Schema comparison - detailed and aesthetic
    print("  Creating schema comparison visualization...", end=" ")
    
    # Analyze schema differences
    cols_1993 = {col["name"]: col["type"] for col in schemas[1993]["columns"]}
    cols_2003 = {col["name"]: col["type"] for col in schemas[2003]["columns"]}
    
    all_cols = set(cols_1993.keys()) | set(cols_2003.keys())
    
    # Categorize columns
    only_1993 = [col for col in all_cols if col in cols_1993 and col not in cols_2003]
    only_2003 = [col for col in all_cols if col in cols_2003 and col not in cols_1993]
    common_cols = [col for col in all_cols if col in cols_1993 and col in cols_2003]
    
    # Type differences for common columns
    type_differences = []
    for col in common_cols:
        if cols_1993[col] != cols_2003[col]:
            type_differences.append({
                'column': col,
                'type_1993': cols_1993[col],
                'type_2003': cols_2003[col]
            })
    
    # Create a more informative and aesthetic visualization
    # Layout: 2 columns - left: bar chart (taller), right: pie chart + table stacked
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Column Count', 
            'Schema Compatibility',
            None,
            'Type Differences'
        ),
        specs=[
            [{"type": "bar", "rowspan": 2}, {"type": "pie"}],
            [None, {"type": "table"}]
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.10,
        column_widths=[0.42, 0.58],
        row_heights=[0.40, 0.60]
    )
    
    # 1. Column count comparison (left)
    fig2.add_trace(
        go.Bar(
            x=['1993', '2003'],
            y=[len(cols_1993), len(cols_2003)],
            marker=dict(
                color=['#2E86AB', '#A23B72'],
                line=dict(color='white', width=2)
            ),
            text=[f"{len(cols_1993)}", f"{len(cols_2003)}"],
            textposition="inside",
            textfont=dict(size=18, color='white', family='Arial Black'),
            hovertemplate='<b>%{x}</b><br>Columns: %{y}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Schema compatibility pie chart (middle) - modern aesthetic
    compatible = len(common_cols) - len(type_differences)
    incompatible = len(type_differences)
    
    # Only show categories with values > 0
    labels = []
    values = []
    colors_pie = []
    
    if compatible > 0:
        labels.append('Fully Compatible')
        values.append(compatible)
        colors_pie.append('#2E86AB')  # Modern blue
    
    if incompatible > 0:
        labels.append('Type Mismatch')
        values.append(incompatible)
        colors_pie.append('#A23B72')  # Modern purple
    
    fig2.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(
                colors=colors_pie,
                line=dict(color='white', width=3)
            ),
            hole=0.5,
            textinfo='label+value',
            textfont=dict(size=11, family='Arial Black', color='white'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            showlegend=False,
            rotation=90
        ),
        row=1, col=2
    )
    
    # 3. Type differences detail (right) - use table
    if type_differences:
        # Prepare table data
        table_data = []
        for diff in type_differences:
            table_data.append([
                diff['column'],
                diff['type_1993'],
                '→',
                diff['type_2003']
            ])
        
        fig2.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Column</b>', '<b>1993</b>', '', '<b>2003</b>'],
                    fill_color='#f0f0f0',
                    align='left',
                    font=dict(size=10, family='Arial Black', color='#1a1a1a'),
                    height=30
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color=[['white', '#f9f9f9'] * len(table_data)],
                    align='left',
                    font=dict(size=9, family='Arial'),
                    height=20
                ),
                columnwidth=[150, 70, 15, 70]
            ),
            row=2, col=2
        )
    else:
        # If no differences, show a message
        fig2.add_trace(
            go.Table(
                header=dict(
                    values=[''],
                    fill_color='#f0f0f0',
                    align='center',
                    font=dict(size=12, family='Arial Black')
                ),
                cells=dict(
                    values=[['✓ No type differences found']],
                    fill_color='white',
                    align='center',
                    font=dict(size=12, color='#2E86AB')
                )
            ),
            row=2, col=2
        )
    
    # Update axes
    fig2.update_xaxes(title_text="Year", row=1, col=1)
    fig2.update_yaxes(
        title_text="Number of Columns",
        row=1, col=1,
        range=[0, max(len(cols_1993), len(cols_2003)) * 1.15]
    )
    
    # Update layout
    fig2.update_layout(
        title=dict(
            text="Schema Comparison: 1993 vs 2003",
            x=0.5,
            font=dict(size=24, family='Arial Black', color='#1a1a1a'),
            xanchor='center',
            y=0.98
        ),
        annotations=[
            dict(
                text="Column Structure and Type Compatibility Analysis",
                x=0.5,
                y=0.95,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12, family='Arial', color='#666', style='italic'),
                xanchor='center'
            )
        ],
        template="plotly_white",
        height=900,
        font=dict(family="Arial", size=11),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )
    
    # Save combined visualization
    save_dual(
        fig2,
        str(viz_dir / "viz_02_schema_column_presence_matrix.plotly.json"),
        str(fig_dir / "fig_02_schema_column_presence_matrix.png"),
        export_png=export_png,
        width=1400,
        height=700
    )
    
    # Save individual components
    viz2_dir = viz_dir / "viz_02_schema_column_presence_matrix"
    viz2_dir.mkdir(exist_ok=True)
    
    # Individual: Column count bar chart
    fig2a = go.Figure()
    fig2a.add_trace(fig2.data[0])  # Bar chart
    fig2a.update_xaxes(title_text="Year")
    fig2a.update_yaxes(title_text="Number of Columns", range=[0, max(len(cols_1993), len(cols_2003)) * 1.15])
    fig2a.update_layout(
        title=dict(text="Column Count Comparison", x=0.5, font=dict(size=18, family='Arial Black')),
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        height=500
    )
    save_plotly_json(fig2a, str(viz2_dir / "column_count.plotly.json"))
    
    # Individual: Schema compatibility pie chart
    fig2b = go.Figure()
    fig2b.add_trace(fig2.data[1])  # Pie chart
    fig2b.update_layout(
        title=dict(text="Schema Compatibility", x=0.5, font=dict(size=18, family='Arial Black')),
        font=dict(family="Arial", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        height=500
    )
    save_plotly_json(fig2b, str(viz2_dir / "schema_compatibility.plotly.json"))
    
    # Individual: Type differences table
    if type_differences:
        fig2c = go.Figure()
        fig2c.add_trace(fig2.data[2])  # Table
        fig2c.update_layout(
            title=dict(text="Type Differences", x=0.5, font=dict(size=18, family='Arial Black')),
            font=dict(family="Arial", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600
        )
        save_plotly_json(fig2c, str(viz2_dir / "type_differences_table.plotly.json"))
    
    print("✓")
    
    # Also save type differences as a separate table for web app
    if type_differences:
        type_diff_table = pd.DataFrame(type_differences)
        type_diff_table = type_diff_table[['column', 'type_1993', 'type_2003']]
        type_diff_path = audit_dir / "tbl_04_type_differences.csv"
        type_diff_table.to_csv(type_diff_path, index=False)
        
        # Also save as JSON for web app
        type_diff_json_path = audit_dir / "tbl_04_type_differences.json"
        save_json(type_diff_table.to_dict('records'), str(type_diff_json_path))
        print(f"  ✓ Saved type differences table: {type_diff_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Ingest Summary:")
    print("=" * 60)
    for year in [1993, 2003]:
        year_rows = row_counts_df[row_counts_df["year"] == year]["rows_raw"].sum()
        print(f"{year}: {year_rows:,} rows, {schemas[year]['column_count']} columns")
    
    total_rows = row_counts_df["rows_raw"].sum()
    print(f"\nTotal: {total_rows:,} rows across both years")
    print(f"Parquet files written to: parquet/raw/")
    print("\n✓ Stage 01 completed successfully!")
    print("=" * 60)
    
    # Close connection
    conn.close()

if __name__ == "__main__":
    main()
