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
- Clone the repository to your local machine:
   ```bash
   git clone https://github.com/Smart-Grp6/leaf-disease-detection.git
   cd leaf-disease-detection

- If you forked the repository, use your forked repository's URL:
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/leaf-disease-detection.git
   cd leaf-disease-detection
Replace <YOUR_USERNAME> with your GitHub username.

2. Create a virtual environment:
   ```bash
   python -m venv venv

3. Activate the virtual environment:
   ```bash
   .\venv\Scripts\activate

4. Install the required packages:
   ```bash
   pip install -r requirements.txt

### **3. Usage**
1. Train the Model
   - Start the MLflow tracking server:
      ```bash
      mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns

   - Run the training script:
      ```bash
      python train.py
   The trained model and metrics will be logged to MLflow.

2. Serve Predictions
   Start the Flask API:
      ```bash
      python app.py

3. Send a POST request to the API with an image file:
   ```bash
   curl -X POST -F "image=@path_to_image.jpg" http://localhost:5000/predict
Replace path_to_image.jpg with the path to your image file.
The API will return the predicted class (e.g., Healthy, Powdery, Rust).

4. Monitor the Model
- Update Monitoring Script for Your Repository:
  github_token = os.getenv("GITHUB_TOKEN")  # Set this in your environment
  repo_owner = "<YOUR_USERNAME>"           # Replace with your GitHub username
  repo_name = "leaf-disease-detection"     # Replace if you renamed the repository
  workflow_id = "retrain.yml"
github_token: Generate a GitHub Personal Access Token (PAT) with repo and workflow scopes, and set it as an environment variable:
   ```bash
   export GITHUB_TOKEN="your_token_here"
- Run the monitoring script:
   ```bash
   python scripts/monitor.py
The script checks for model drift and triggers retraining if necessary.

5. Automate Retraining
   1. Retraining Workflow:

      - The retrain.yml workflow retrains the model every Sunday at midnight or can be triggered manually.

      - It deletes the existing MLflow database, retrains the model, and deploys the updated model.


   2. Monitoring Workflow:

      - The monitor.yml workflow checks for model drift every hour or can be triggered manually.

      - If model drift is detected, it triggers the retraining workflow.

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

GitHub: elmahdiarfal
