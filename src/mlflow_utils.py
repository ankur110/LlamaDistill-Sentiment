# src/mlflow_utils.py
import os
import mlflow
from typing import Dict, Any
import numpy as np
def init_mlflow(mlflow_uri: str = None, experiment_name: str = "deployment-inference"):
    mlflow_uri = mlflow_uri or os.getenv("MLFLOW_URI", f"file:{os.path.abspath('./mlruns')}")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

def log_inference_sample(run_id: str, payload: Dict[str, Any], preds, probs, meta: Dict[str, Any] = None):
    """
    Log a minimal inference summary to MLflow as metrics and params. This starts a short run (nested).
    Don't log raw texts in production.
    """
    try:
        with mlflow.start_run(nested=True):
            mlflow.log_param("inference_run_id", run_id)
            mlflow.log_metric("inference_batch_size", len(preds))
            # log average confidence for positive class
            
            pos_probs = np.array(probs)[:, 1]
            mlflow.log_metric("inference_mean_pos_prob", float(pos_probs.mean()))
            # attach meta keys if present
            if meta:
                for k, v in meta.items():
                    try:
                        mlflow.set_tag(k, str(v))
                    except Exception:
                        pass
    except Exception:
        # keep inference resilient — failures in MLflow logging should not break API
        pass
