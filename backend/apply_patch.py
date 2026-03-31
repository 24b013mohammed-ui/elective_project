import shutil
import os

try:
    # Backup original
    shutil.copy('evaluation.py', 'evaluation_backup.py')
    # Replace with patched version
    shutil.copy('evaluation_patched.py', 'evaluation.py')
    print("SUCCESS: evaluation.py patched and backup saved as evaluation_backup.py")
except Exception as e:
    print(f"ERROR: {e}")
