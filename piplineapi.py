from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from piplines.dl_training_pipline import dl_training_pipeline

app = FastAPI(title="MLOPS PIPELINE API")

class RunRequest(BaseModel):
    username: str

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "MLOPS PIPELINE API"}

# ✅ Background task function
def start_pipeline(username: str):
    dataset_path = "./brain-tumor-mri-dataset"
    pipeline = dl_training_pipeline(dataset_path, username)
    pipeline.run()

@app.post("/run")
def run_pipeline(request: RunRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(start_pipeline, request.username)

        return {
            "status": "Pipeline started",
            "username": request.username
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
