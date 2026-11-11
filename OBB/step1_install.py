# Step 1: install and check GPU
# Runtime -> Change runtime type -> A100 GPU before starting.
# In Colab you can also run:
#   !pip install -q ultralytics
#   !nvidia-smi
import subprocess

subprocess.run(['pip', 'install', '-q', 'ultralytics'], check=True)
subprocess.run(['nvidia-smi'])
