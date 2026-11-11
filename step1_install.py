# Cell 1: install and check GPU
# Runtime -> Change runtime type -> T4 GPU before starting.
# In Colab run:
#   !pip install -q ultralytics
#   !nvidia-smi
import subprocess

subprocess.run(['pip', 'install', '-q', 'ultralytics'], check=True)
subprocess.run(['nvidia-smi'])
