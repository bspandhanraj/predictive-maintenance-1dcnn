import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def build_lightweight_1d_cnn(input_shape, num_classes):
    model = models.Sequential([
        # Keep filter sizes and pool sizes small to conserve RAM on ESP32-C3
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    # Load processed data
    X_train = np.load("X_train.npy").astype(np.float32)
    X_test = np.load("X_test.npy").astype(np.float32)
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))
    
    model = build_lightweight_1d_cnn(input_shape, num_classes)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    model.summary()
    
    # Train model
    print("Training the model...")
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the model
    model.save("predictive_maintenance_model.keras")
    print("Model saved as predictive_maintenance_model.keras")