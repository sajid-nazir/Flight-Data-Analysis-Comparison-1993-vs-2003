"""
Report generation utilities

NOTE: The actual report generation implementation is located in:
    - scripts/13_build_report.py
    
Functions available:
    - load_metrics_summary(): Load key metrics from all stages (within-year, cross-year, generalization loss, feature importance)
    - generate_report(): Generate final report markdown with all findings

This script:
    1. Generates final_report.md with all findings
    2. Embeds PNG figures into the markdown
    3. Synthesizes answers to research questions

The report includes:
    - Executive summary
    - Research questions and answers
    - Data overview and cleaning decisions
    - Model performance metrics
    - Feature importance analysis
    - Data drift analysis
    - Conclusions and methodology notes
"""

