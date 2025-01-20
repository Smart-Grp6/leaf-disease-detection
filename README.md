# leaf-disease-detection

This project demonstrates how to build, deploy, and monitor a **Leaf Disease Detection** model. The system uses a Convolutional Neural Network (CNN) to classify leaf diseases and integrates tools like **MLflow** for experiment tracking, **Flask** for serving predictions, and **GitHub Actions** for automating monitoring and retraining.

---

## **Features**

- **Model Training**: Train a CNN model using TensorFlow/Keras.
- **Experiment Tracking**: Log experiments, parameters, and metrics using MLflow.
- **REST API**: Serve predictions via a Flask API.
- **Model Monitoring**: Detect model drift and trigger retraining.
- **Automation**: Automate monitoring, retraining and deployment using GitHub Actions.

---

## **Setup**

### **1. Prerequisites**
- Python 3.8 or higher
- Git
- MLflow (for experiment tracking)
- Flask (for serving predictions)

### **2. Install Dependencies**
1. Clone the repository:
   ```bash
   git clone https://github.com/Smart-Grp6/leaf-disease-detection.git
   cd leaf-disease-detection
   
2. Create a virtual environment:
   ```bash
   python -m venv venv

3. Activate the virtual environment:
   ```bash
   .\venv\Scripts\activate

4. Install the required packages:
   ```bash
   pip install -r requirements.txt

Usage
1. Train the Model
Start the MLflow tracking server:
   ```bash
   mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns

2. Train the model:
   ```bash
   python train.py
The trained model and metrics will be logged to MLflow.

3. Serve Predictions
Start the Flask API:
   ```bash
   python app.py

4. Send a POST request to the API with an image file:
   ```bash
   curl -X POST -F "image=@path_to_image.jpg" http://localhost:5000/predict
The API will return the predicted class (e.g., Healthy, Powdery, Rust).

5. Monitor the Model
Run the monitoring script:
   ```bash
   python scripts/monitor.py
The script checks for model drift and triggers retraining if necessary.

4. Automate Retraining
- The GitHub Actions workflow (.github/workflows/retrain.yml) automates retraining and deployment. It runs every Sunday at midnight or can be triggered manually.
- The GitHub Actions workflow (.github/workflows/monitor.yml) automates monitoring. It runs every hour or can be triggered manually.

## **CI/CD Pipeline**
**The CI/CD pipeline automates the following steps:**

- **Checkout the code.**

- **Install dependencies.**

- **Check for drift.**

- **Retrain the model.**

- **Deploy the updated model.**


## **Acknowledgments**
- TensorFlow for the deep learning framework.

- MLflow for experiment tracking.

- Flask for serving predictions.

- GitHub Actions for CI/CD automation.

## **Contact**
For questions or feedback, please contact:

Email: arfalmahdi@gmail.com

GitHub: Smart-Grp6
