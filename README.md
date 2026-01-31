# AutoML Web Application 

A full-stack AutoML web application where users can upload datasets, automatically train machine learning models, and make predictions through a web interface.

## Features
- Upload CSV datasets
- Automatic classification / regression detection
- Multiple ML models trained
- Hyperparameter tuning using GridSearchCV
- Best model selection
- Model persistence
- Prediction API

## Tech Stack
- Frontend: HTML, CSS, JavaScript
- Backend: FastAPI (Python)
- Machine Learning: scikit-learn, pandas, numpy
- AutoML: Pipelines, GridSearchCV
- Server: Uvicorn

## How to Run

### Backend
```bash
python -m uvicorn api.main:app --reload
