import mlflow
import mlflow.pyfunc
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.utils import load_img, img_to_array
from io import BytesIO
import numpy as np

# Définir l'URI de suivi pour le serveur MLflow distant
mlflow.set_tracking_uri("http://localhost:5000")

# Read the latest run_id from the file
def get_latest_run_id():
    try:
        with open("latest_run_id.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise Exception("latest_run_id.txt not found. Please train the model first.")

# Charger le modèle depuis MLflow
run_id = get_latest_run_id()
model_uri = f"runs:/{run_id}/leaf_disease_model"  # Use the latest run_id
model = mlflow.pyfunc.load_model(model_uri)

# Créer une application Flask
app = Flask(__name__)

# Route pour la page d'accueil
@app.route('/')
def home():
    return render_template('index.html')

# Route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    # Charger et prétraiter l'image
    file = request.files['image']
    try:
        # Convertir FileStorage en BytesIO
        img_bytes = BytesIO(file.read())
        img = load_img(img_bytes, target_size=(225, 225))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        return jsonify({"error": f"Échec du traitement de l'image : {str(e)}"}), 400

    # Faire la prédiction
    try:
        predictions = model.predict(img_array)
        print("Prédictions brutes :", predictions)  # Loguer les prédictions brutes
        predicted_label = np.argmax(predictions)
        class_names = ["Healthy", "Powdery", "Rust"]
        result = class_names[predicted_label]
        print("Label prédit :", result)  # Loguer le label prédit

        # Loguer la prédiction dans MLflow
        with mlflow.start_run():
            mlflow.log_param("input_image", file.filename)
            mlflow.log_metric("predicted_label", int(predicted_label))
            mlflow.log_metric("confidence", float(np.max(predictions)))

        # Retourner le résultat
        return jsonify({'prediction': result, 'confidence': float(np.max(predictions))})
    except Exception as e:
        return jsonify({"error": f"Échec de la prédiction : {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)