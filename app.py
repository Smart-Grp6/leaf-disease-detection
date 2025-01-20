import mlflow
import mlflow.pyfunc
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.utils import load_img, img_to_array
from io import BytesIO
import numpy as np

# Set the tracking URI for the remote MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Read the latest run_id from the file
def get_latest_run_id():
    try:
        with open("latest_run_id.txt", "r") as f:
            return f.read().strip()  # Read and return the run_id from the file
    except FileNotFoundError:
        raise Exception("latest_run_id.txt not found. Please train the model first.")

# Load the model from MLflow
run_id = get_latest_run_id()  # Get the latest run_id
model_uri = f"runs:/{run_id}/leaf_disease_model"  # Construct the model URI using the run_id
model = mlflow.pyfunc.load_model(model_uri)  # Load the model using MLflow

# Create a Flask application
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML template for the home page

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Load and preprocess the image
    file = request.files['image']  # Get the uploaded image file
    try:
        # Convert FileStorage to BytesIO
        img_bytes = BytesIO(file.read())  # Read the image file into a BytesIO object
        img = load_img(img_bytes, target_size=(225, 225))  # Load and resize the image
        img_array = img_to_array(img) / 255.0  # Convert image to array and normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400  # Return error if processing fails

    # Make the prediction
    try:
        predictions = model.predict(img_array)  # Get predictions from the model
        print("Raw predictions:", predictions)  # Log raw predictions for debugging
        predicted_label = np.argmax(predictions)  # Get the index of the highest probability
        class_names = ["Healthy", "Powdery", "Rust"]  # Define class names
        result = class_names[predicted_label]  # Map the predicted label to a class name
        print("Predicted label:", result)  # Log the predicted label for debugging

        # Log the prediction in MLflow
        with mlflow.start_run():
            mlflow.log_param("input_image", file.filename)  # Log the input image filename
            mlflow.log_metric("predicted_label", int(predicted_label))  # Log the predicted label
            mlflow.log_metric("confidence", float(np.max(predictions)))  # Log the confidence score

        # Return the result
        return jsonify({'prediction': result, 'confidence': float(np.max(predictions))})  # Return prediction and confidence
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500  # Return error if prediction fails

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Run the Flask app on all available IPs and port 5000