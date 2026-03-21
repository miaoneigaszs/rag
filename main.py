from pathlib import Path
import os

file_path = os.path.abspath(__file__)

file_path = "./main.py"

print(file_path)

file_path = str(Path(file_path).resolve())

print(file_path)
print(Path(file_path).name)