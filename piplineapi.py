from fastapi import FastAPI
from pydantic import BaseModel
from piplines.dl_training_pipline import dl_training_pipeline
from datasetcreation import DatasetCreation
app = FastAPI(title="MLOPE PIPLINE API")


class RunRequest(BaseModel):
    username: str

@app.get("/")
def root():
    return {
        "message": "MLOPE PIPLINE API"
    }

@app.post("/run")
def run_pipeline(request: RunRequest):
    try:
        datasetCreation=  DatasetCreation()
        dataset_path = "./brain-tumor-mri-dataset/"

        dl_training_pipeline(
            "datasetpath",
            request.username
        )
        return {
            "message": "MLOPE PIPLINE API",
            "status": "Pipeline executed successfully",
            "username": request.username
        }
    except Exception as e:
        return {
            "message": "MLOPE PIPLINE API",
            "status": "Failed",
            "error": str(e)
        }
