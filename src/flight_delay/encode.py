"""
Encoding utilities for categorical features

NOTE: The actual encoding implementation is located in:
    - src/flight_delay/features.py
    
Functions available:
    - fit_target_encoders(): Fit target encoders (mean encoding) on training data
    - apply_target_encoders(): Apply target encoders to DataFrame
    - save_encoders(): Save target encoders to JSON file
    - load_encoders(): Load target encoders from JSON file

Usage examples:
    - scripts/06_features.py: Uses fit_target_encoders() and apply_target_encoders()
    - scripts/08_train_cross_year.py: Uses load_encoders() and apply_target_encoders()
"""

