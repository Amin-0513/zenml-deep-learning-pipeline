from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from zenml.client import Client
from zenml.integrations.local.orchestrators import LocalOrchestrator
from zenml.artifacts.stores.local_artifact_store import LocalArtifactStore
from zenml.secrets_managers.local_secrets_manager import LocalSecretsManager
from zenml.metadata.metadata_store import SQLiteMetadataStore

from piplines.dl_training_pipline import dl_training_pipeline
from datasetcreation import DatasetCreation

app = FastAPI(title="MLOps Pipeline API")


# -----------------------------
# Request Model
# -----------------------------
class RunRequest(BaseModel):
    username: str


# -----------------------------
# Ensure ZenML stack exists
# -----------------------------
def ensure_stack():
    client = Client()
    try:
        client.active_stack
    except Exception:
        # Create a default local stack
        artifact_store = LocalArtifactStore(uri="./zenml_artifacts")
        orchestrator = LocalOrchestrator()
        metadata_store = SQLiteMetadataStore(uri="./zenml_metadata.db")
        secrets_manager = LocalSecretsManager()

        components = {
            "orchestrator": orchestrator,
            "artifact_store": artifact_store,
            "metadata_store": metadata_store,
            "secrets_manager": secrets_manager
        }

        client.create_stack(name="default_stack", components=components)
        print("Default stack created")

    return client


# -----------------------------
# Root → Redirect to ZenML Dashboard
# -----------------------------
@app.get("/")
def root():
    ensure_stack()
    return RedirectResponse(url="http://127.0.0.1:8237")


# -----------------------------
# Run Pipeline Endpoint
# -----------------------------
@app.post("/run")
def run_pipeline(request: RunRequest):
    try:
        ensure_stack()

        dataset_creation = DatasetCreation()
        dataset_path = dataset_creation.create_dataset()

        dl_training_pipeline(dataset_path, request.username)

        return {
            "message": "MLOps Pipeline API",
            "status": "Pipeline executed successfully",
            "username": request.username
        }

    except Exception as e:
        return {
            "message": "MLOps Pipeline API",
            "status": "Failed",
            "error": str(e)
        }
