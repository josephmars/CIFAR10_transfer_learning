from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from model import create_cifar10_model

def load_and_preprocess_data():
    """Load and preprocess CIFAR10 dataset"""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def train_and_evaluate():
    """Train and evaluate the model with different dataset sizes"""
    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create model
    cifar10_model = create_cifar10_model()
    
    # Train with different dataset sizes
    for n in [100, 1000, 10000, 50000]:
        cifar10_model.fit(x_train[:n], y_train[:n], 
                         epochs=5, 
                         batch_size=32, 
                         validation_split=0.2)
        
        train_loss, train_acc = cifar10_model.evaluate(x_train[:n], y_train[:n])
        test_loss, test_acc = cifar10_model.evaluate(x_test, y_test)
        
        print(f'Train Accuracy with {n} images: {train_acc}')
        print(f'Test Accuracy with {n} images: {test_acc}') 