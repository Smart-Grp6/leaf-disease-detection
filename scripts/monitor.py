import mlflow
import requests
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def trigger_retraining():
    # GitHub Actions API endpoint to trigger the workflow
    github_token = os.getenv("GITHUB_TOKEN")  # GitHub token for authentication
    repo_owner = "Smart-Grp6"  # GitHub username
    repo_name = "leaf-disease-detection1"  # Repository name
    workflow_id = "retrain.yml"
    
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/{workflow_id}/dispatches"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json"
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
    # l'URI de suivi de MLflow
    mlflow.set_tracking_uri("http://localhost:5000")  

    # Charger les métriques de production depuis MLflow
    runs = mlflow.search_runs()
    if runs.empty:
        logging.warning("No runs found in MLflow.")
        return

    latest_run = runs.iloc[0]
    production_accuracy = latest_run['metrics.accuracy']
    logging.info(f"Latest production accuracy: {production_accuracy}")

    # Comparer avec la précision de référence
    baseline_accuracy = 0.85  # Précision de réference
    logging.info(f"Baseline accuracy: {baseline_accuracy}")

    if production_accuracy < baseline_accuracy * 0.9:  # Une baisse de 10% de la précision
        logging.warning("Model drift detected! Triggering retraining...")
        trigger_retraining()  # Déclencher le processus de réentraînement
    else:
        logging.info("No model drift detected.")

if __name__ == '__main__':
    monitor_model() 