from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

def create_cifar10_model():
    """Create and return the modified VGG16 model for CIFAR10"""
    # Load the pre-trained VGG16 model with CIFAR10 input shape
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    
    # Remove the last layer
    vgg16_model.layers.pop()
    
    # Freeze the pre-trained layers
    for layer in vgg16_model.layers:
        layer.trainable = False
    
    # Add new layers
    flatten_layer = Flatten()(vgg16_model.layers[-1].output)
    new_output = Dense(10, activation='softmax')(flatten_layer)
    
    # Create the final model
    cifar10_model = Model(inputs=vgg16_model.input, outputs=new_output)
    
    # Compile the model
    cifar10_model.compile(optimizer=Adam(), 
                         loss='categorical_crossentropy', 
                         metrics=['accuracy'])
    
    return cifar10_model 