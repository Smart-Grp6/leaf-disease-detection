import mlflow
import requests
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)  # Configure logging to display INFO-level messages

def trigger_retraining():
    """
    Trigger a GitHub Actions workflow to retrain the model.
    """
    # GitHub Actions API endpoint to trigger the workflow
    github_token = os.getenv("GITHUB_TOKEN")  # Get GitHub token from environment variables
    repo_owner = "elmahdiarfal"  # GitHub username
    repo_name = "leaf-disease-detection"  # Repository name
    workflow_id = "retrain.yml"  # Workflow file name

    # Construct the URL for the GitHub API
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/{workflow_id}/dispatches"
    headers = {
        "Authorization": f"Bearer {github_token}",  # Authenticate using the GitHub token
        "Accept": "application/vnd.github.v3+json"  # Specify the API version
    }
    data = {
        "ref": "main"  # Branch to trigger the workflow on
    }

    # Send POST request to trigger the workflow
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 204:
        logging.info("Retraining workflow triggered successfully!")
    else:
        logging.error(f"Failed to trigger retraining. Status code: {response.status_code}, Response: {response.text}")

def monitor_model():
    """
    Monitor the model's performance and trigger retraining if model drift is detected.
    """
    # Set the MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")  

    # Load production metrics from MLflow
    runs = mlflow.search_runs()  # Search for all runs in MLflow
    if runs.empty:
        logging.warning("No runs found in MLflow.")
        return

    # Get the latest run and its accuracy
    latest_run = runs.iloc[0]  # Get the most recent run
    production_accuracy = latest_run['metrics.accuracy']  # Extract the accuracy metric
    logging.info(f"Latest production accuracy: {production_accuracy}")

    # Compare with the baseline accuracy
    baseline_accuracy = 0.85  # Baseline accuracy threshold
    logging.info(f"Baseline accuracy: {baseline_accuracy}")

    # Check for model drift (10% drop in accuracy)
    if production_accuracy < baseline_accuracy * 0.9:  # If accuracy drops below 90% of baseline
        logging.warning("Model drift detected! Triggering retraining...")
        trigger_retraining()  # Trigger the retraining process
    else:
        logging.info("No model drift detected.")

if __name__ == '__main__':
    monitor_model()  # Run the model monitoring process