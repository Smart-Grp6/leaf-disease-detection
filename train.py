import mlflow
import mlflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np

# Set the tracking URI for the remote MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Define the CNN model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(225, 225, 3), activation='relu'))  # First convolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2)))  # First max pooling layer
    model.add(Conv2D(64, (3, 3), activation='relu'))  # Second convolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Second max pooling layer
    model.add(Flatten())  # Flatten the output for the dense layers
    model.add(Dense(64, activation='relu'))  # Fully connected layer
    model.add(Dense(3, activation='softmax'))  # Output layer with softmax activation for 3 classes
    return model

# Compile and train the model
def train_model():
    # Load the dataset using image_dataset_from_directory
    train_dataset = image_dataset_from_directory(
        'data/Train',  # Path to the training data directory
        image_size=(225, 225),  # Resize images to 225x225
        batch_size=32,  # Batch size for training
        label_mode='categorical'  # Labels are categorical (one-hot encoded)
    )

    # Create and compile the model
    model = create_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Start an MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("epochs", 5)  # Number of epochs
        mlflow.log_param("batch_size", 32)  # Batch size

        # Train the model
        history = model.fit(train_dataset, epochs=5)

        # Log metrics
        mlflow.log_metric("accuracy", history.history['accuracy'][-1])  # Log the final accuracy

        # Save the model
        mlflow.keras.log_model(model, "leaf_disease_model")  # Log the trained model

        # Display the run ID
        run_id = run.info.run_id
        print(f"Run ID: {run_id}")

        # Save the run_id to a file
        with open("latest_run_id.txt", "w") as f:
            f.write(run_id)

if __name__ == '__main__':
    train_model()