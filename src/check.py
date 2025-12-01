import mlflow, dagshub, traceback

dagshub.init(repo_owner='mondolpranto83', repo_name='ML_Flow-project', mlflow=True)
mlflow.set_experiment("Mlflow-minimal")

try:
    with mlflow.start_run():
        mlflow.log_param("ping", "ok")
        mlflow.log_metric("accuracy", 0.5)
    print("Minimal log succeeded")
except Exception:
    print("Minimal log failed; traceback:")
    traceback.print_exc()
