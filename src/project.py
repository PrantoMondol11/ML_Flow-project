from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import dagshub
import os
import shutil

# Connect MLflow to DagsHub
dagshub.init(repo_owner='mondolpranto83', repo_name='ML_Flow-project', mlflow=True)
mlflow.set_experiment("Mlflow")

# Load data
wine = load_wine()
X, y = wine.data, wine.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Model
model = RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Accuracy:", accuracy)

# Log to MLflow @ DagsHub (params, metric, and model as artifacts)
with mlflow.start_run() as run:
    mlflow.autolog()
    # Save model locally in MLflow model format and upload as artifacts (works with DagsHub)
    MODEL_DIR = "model_temp"
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    # save_model writes the MLflow model folder structure locally
    mlflow.sklearn.save_model(model, MODEL_DIR)
    # upload entire folder as artifacts under the run
    mlflow.log_artifacts(MODEL_DIR, artifact_path="model")
    # cleanup local temp folder
    shutil.rmtree(MODEL_DIR)

    print("Run logged successfully. run_id:", run.info.run_id)

print("Done.")
