from config import check_gpu
from train import train_and_evaluate

if __name__ == "__main__":
    # Check GPU availability
    check_gpu()
    
    # Run training and evaluation
    train_and_evaluate() 