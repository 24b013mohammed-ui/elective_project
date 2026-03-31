#!/usr/bin/env python3
"""Install packages compatible with Python 3.14"""
import subprocess
import sys

packages = [
    'numpy==2.4.3',
    'pandas>=2.0',
    'scipy>=1.11',
    'scikit-learn>=1.3',
    'matplotlib>=3.8', 
    'seaborn>=0.13',
    'yfinance>=0.2',
    'torch>=2.10',
]

print("Installing packages for Python 3.14...")
print(f"Python: {sys.executable}")
print("=" * 60)

for package in packages:
    print(f"\nInstalling {package}...")
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--upgrade', package, '--prefer-binary'],
        capture_output=False
    )
    if result.returncode != 0:
        print(f"WARNING: Failed to install {package}")

print("\n" + "=" * 60)
print("Installation Complete!")
print("\nVerifying imports...")

try:
    import torch
    print(f"✓ torch {torch.__version__}")
except:
    print("✗ torch not installed")

try:
    import pandas
    print(f"✓ pandas {pandas.__version__}")
except:
    print("✗ pandas not installed")

try:
    import numpy
    print(f"✓ numpy {numpy.__version__}")
except:
    print("✗ numpy not installed")
