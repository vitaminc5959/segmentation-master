# Cell 2 + 3: mount Drive, unzip the dataset, fix paths in data.yaml
import glob
import os
import subprocess

import yaml
from google.colab import drive

drive.mount('/content/drive')

# Finds the zip in MyDrive/topgun even if the exact name differs slightly
ZIP_PATH = glob.glob('/content/drive/MyDrive/topgun/TOPGUN*.zip')[0]
DATA_DIR = '/content/dataset'
YAML_PATH = f'{DATA_DIR}/data.yaml'
print('Using zip:', ZIP_PATH)

os.makedirs(DATA_DIR, exist_ok=True)
subprocess.run(['unzip', '-q', '-o', ZIP_PATH, '-d', DATA_DIR], check=True)
print(os.listdir(DATA_DIR))

# Roboflow's relative paths break in Colab
cfg = yaml.safe_load(open(YAML_PATH))
cfg['path'] = DATA_DIR
cfg['train'] = 'train/images'
cfg['val'] = 'valid/images'
cfg['test'] = 'test/images'
yaml.safe_dump(cfg, open(YAML_PATH, 'w'))
print(cfg)
