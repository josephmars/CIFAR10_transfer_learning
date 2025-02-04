import numpy as np
import tensorflow as tf
from tensorflow import keras
import warnings

# Clear memory for training
keras.backend.clear_session()

# Set random seed for reproducibility
np.random.seed(0)

# Remove warnings
warnings.filterwarnings('ignore')

def check_gpu():
    """Check and print available GPUs"""
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) 