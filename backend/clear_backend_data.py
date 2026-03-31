import os
import shutil
import glob

def clear_backend_data():
    """Clear all training data and checkpoints"""
    
    items_cleared = []
    
    # 1. Clear checkpoints
    if os.path.exists('checkpoints'):
        for file in glob.glob('checkpoints/*'):
            try:
                os.remove(file)
                items_cleared.append(f"Cleared: {file}")
            except Exception as e:
                print(f"Could not remove {file}: {e}")
    
    # 2. Clear log files
    log_files = [
        'execution_log.txt',
        'install_log.txt',
        'debug_output.txt',
        'test_output.txt',
        'batch_output.txt',
        'pipeline_output.txt',
        'status_full.txt',
        'status_out.txt'
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
                items_cleared.append(f"Cleared: {log_file}")
            except Exception as e:
                print(f"Could not remove {log_file}: {e}")
    
    # 3. Clear output directory
    if os.path.exists('output'):
        for file in glob.glob('output/*'):
            try:
                if os.path.isfile(file):
                    os.remove(file)
                    items_cleared.append(f"Cleared: {file}")
            except Exception as e:
                print(f"Could not remove {file}: {e}")
    
    # 4. Clear __pycache__
    if os.path.exists('__pycache__'):
        try:
            shutil.rmtree('__pycache__')
            items_cleared.append("Cleared: __pycache__ directory")
        except Exception as e:
            print(f"Could not remove __pycache__: {e}")
    
    print("\n" + "="*60)
    print("BACKEND DATA CLEARED")
    print("="*60)
    for item in items_cleared:
        print(item)
    print(f"\nTotal items cleared: {len(items_cleared)}")
    print("\nSystem ready for fresh training!")

if __name__ == '__main__':
    clear_backend_data()
