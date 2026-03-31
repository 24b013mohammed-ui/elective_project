import sys
import traceback

try:
    print("=" * 60)
    print("FINANCIAL CNN FORECASTING PIPELINE - DEBUG RUN")
    print("=" * 60)
    print(f"Python: {sys.executable}")
    print()
    
    print("Step 1: Importing PyTorch...")
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    
    print("Step 2: Importing Pandas...")
    import pandas  
    print(f"✓ Pandas {pandas.__version__}")
    
    print("Step 3: Importing config...")
    import config
    print(f"✓ Device: {config.DEVICE}")
    
    print("Step 4: Running main pipeline...")
    exec(open('main.py').read())
    
except Exception as e:
    print()
    print("ERROR OCCURRED:")
    print("=" * 60)
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {str(e)}")
    print()
    print("Full Traceback:")
    print("=" * 60)
    traceback.print_exc()
    print("=" * 60)
    sys.exit(1)
