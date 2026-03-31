import os

file_path = 'pipeline_output.txt'

if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-16') as f:
        content = f.read()
        print(content)
else:
    print("File not found")
