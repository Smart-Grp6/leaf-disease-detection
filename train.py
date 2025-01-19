import mlflow
import mlflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np

# Définir l'URI de suivi pour le serveur MLflow distant
mlflow.set_tracking_uri("http://localhost:5000")

# Définir le modèle CNN
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(225, 225, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    return model

# Compiler et entraîner le modèle
def train_model():
    # Charger le dataset en utilisant image_dataset_from_directory
    train_dataset = image_dataset_from_directory(
        'data/Train',
        image_size=(225, 225),
        batch_size=32,
        label_mode='categorical'
    )

    # Créer et compiler le modèle
    model = create_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Démarrer une run MLflow
    with mlflow.start_run() as run:
        # Loguer les paramètres
        mlflow.log_param("epochs", 5)
        mlflow.log_param("batch_size", 32)

        # Entraîner le modèle
        history = model.fit(train_dataset, epochs=5)

        # Loguer les métriques
        mlflow.log_metric("accuracy", history.history['accuracy'][-1])

        # Sauvegarder le modèle
        mlflow.keras.log_model(model, "leaf_disease_model")

        # Afficher l'ID de la run
        run_id = run.info.run_id
        print(f"Run ID: {run_id}")

        # Save the run_id to a file
        with open("latest_run_id.txt", "w") as f:
            f.write(run_id)

if __name__ == '__main__':
    train_model()