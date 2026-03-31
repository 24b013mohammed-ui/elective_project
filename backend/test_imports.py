#!/usr/bin/env python3
"""Diagnostic script to check which imports are working"""

import sys

# Test each import
imports_to_test = [
    'torch',
    'numpy',
    'pandas',
    'yfinance',
    'scipy',
    'sklearn',
    'matplotlib',
    'seaborn',
    'config',
    'data_preparation',
    'dataset_preparation',
    'signal_processing',
    'model',
    'training',
    'evaluation',
    'visualization',
    'utils'
]

print("=" * 60)
print("IMPORT DIAGNOSTIC TEST")
print("=" * 60)
print(f"Python: {sys.executable}\n")

for module_name in imports_to_test:
    try:
        __import__(module_name)
        print(f"✓ {module_name:25} - OK")
    except Exception as e:
        print(f"✗ {module_name:25} - ERROR: {str(e)[:50]}")

print("\n" + "=" * 60)
