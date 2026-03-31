import sys
import os

print("Starting pipeline test...", file=sys.stderr)
print(f"Python: {sys.executable}", file=sys.stderr)
print(f"CWD: {os.getcwd()}", file=sys.stderr)

print("\nImporting modules...", file=sys.stderr)

try:
    import torch
    print(f"✓ torch {torch.__version__}", file=sys.stderr)
except ImportError as e:
    print(f"✗ torch: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import pandas
    print(f"✓ pandas {pandas.__version__}", file=sys.stderr)
except ImportError as e:
    print(f"✗ pandas: {e}", file=sys.stderr)
    sys.exit(1)

print("\nAll imports OK - running main.py...", file=sys.stderr)
exec(open('main.py').read())
