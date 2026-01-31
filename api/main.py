from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import os
import shutil
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

from src.automl import run_automl


app = FastAPI(
    title="AutoML Backend",
    description="Upload dataset, train AutoML model, and make predictions",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = "upload"
MODEL_DIR = "artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "user_model.pkl")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    """
    Trains a new AutoML model on a user-uploaded dataset.
    """
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        print(f"[DEBUG] File saved to: {file_path}")
        print(f"[DEBUG] Target column: '{target_column}'")
        
        df = pd.read_csv(file_path)
        print(f"[DEBUG] CSV columns: {list(df.columns)}")
        print(f"[DEBUG] Target column in CSV: {target_column in df.columns}")
        
        result = run_automl(
            csv_path=file_path,
            target_column=target_column,
            model_path=MODEL_PATH
        )
        
        response_details = {
            "problem_type": str(result.get("problem_type", "unknown")),
            "best_model": str(result.get("best_model", "N/A")),
            "best_score": float(result.get("best_score", 0)),
            "test_score": float(result.get("test_score", 0)),
            "best_hyperparameters": result.get("best_hyperparameters", {}),
            "dataset_size": int(result.get("dataset_size", 0)),
            "scoring_metric": str(result.get("scoring_metric", ""))
        }
        
        print(f"[DEBUG] Response details: {response_details}")
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Training failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "message": "Model trained successfully",
        "details": response_details
    }



@app.post("/predict")
def predict(data: dict):
    """
    Uses the trained model to make predictions on new input data.
    """

    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=400,
            detail="Model not found. Train a model first."
        )

    
    model = joblib.load(MODEL_PATH)

    
    df = pd.DataFrame([data])

    
    prediction = model.predict(df)

    return {
        "prediction": prediction.tolist()
    }
