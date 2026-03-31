import sys
print("Python version:", sys.version)
print("Python executable:", sys.executable)

print("\nAttempting imports...")
try:
    import torch
    print("✓ torch imported")
except Exception as e:
    print(f"✗ torch failed: {e}")

try:
    import numpy
    print("✓ numpy imported")
except Exception as e:
    print(f"✗ numpy failed: {e}")

try:
    import pandas
    print("✓ pandas imported")
except Exception as e:
    print(f"✗ pandas failed: {e}")

print("\nTest complete")
