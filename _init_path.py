from config import *
import sys
import os

# Check the model folder
if not os.path.exists(model_path):
    os.mkdir(model_path)

# Append the other module
sys.path.append('./lib/')