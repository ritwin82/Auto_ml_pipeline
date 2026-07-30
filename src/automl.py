import pandas as pd
import numpy as np
import os
import joblib
import json
import warnings
from pandas.api import types as ptypes

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def detect_problem_type(y):
    """
    Detects whether the problem is classification or regression.
    """
    # Categorical/object/boolean types are classification
    if ptypes.is_categorical_dtype(y) or ptypes.is_object_dtype(y):
        print(f"[AutoML] Detected classification (categorical target)")
        return "classification"

    if ptypes.is_bool_dtype(y):
        print(f"[AutoML] Detected classification (boolean target)")
        return "classification"

    # Work with non-missing values for uniqueness checks
    y_nonnull = y.dropna()
    n_unique = y_nonnull.nunique()
    n_total = len(y_nonnull)
    unique_ratio = (n_unique / n_total) if n_total > 0 else 0

    # Integer-like small-cardinality numeric targets are often classification
    if (ptypes.is_integer_dtype(y) or (ptypes.is_float_dtype(y) and y_nonnull.apply(lambda x: float(x).is_integer()).all())):
        if n_unique <= 10 and unique_ratio < 0.5:
            print(f"[AutoML] Detected classification (integer-like target with {n_unique} unique values)")
            return "classification"

    # Default: numeric -> regression, else classification
    if ptypes.is_numeric_dtype(y):
        print(f"[AutoML] Detected regression (numeric target with {n_unique} unique values, ratio: {unique_ratio:.2f})")
        return "regression"

    print(f"[AutoML] Fallback to classification for target with dtype {y.dtype}")
    return "classification"


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []
    
    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    return ColumnTransformer(transformers)


def get_cv_splitter(n_samples, problem_type, y=None):
    """
    Returns an appropriate cross-validation splitter based on dataset size.
    """
    if problem_type == "classification" and y is not None:
        min_class_count = y.value_counts().min()
        n_splits = min(5, min_class_count, n_samples - 1)
        n_splits = max(2, n_splits)
        print(f"[AutoML] Using StratifiedKFold with {n_splits} splits")
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        n_splits = min(5, n_samples - 1)
        n_splits = max(2, n_splits)
        print(f"[AutoML] Using KFold with {n_splits} splits (dataset size: {n_samples})")
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)


def get_models(problem_type, n_samples):
    """
    Returns models with hyperparameters adjusted for dataset size.
    """
    max_neighbors = max(1, n_samples - 2)
    knn_neighbors = [k for k in [1, 2, 3, 5] if k <= max_neighbors]
    if not knn_neighbors:
        knn_neighbors = [1]
    
    print(f"[AutoML] Dataset size: {n_samples}, KNN neighbors to try: {knn_neighbors}")
    
    if problem_type == "classification":
        models = {
            "LogisticRegression": (
                LogisticRegression(max_iter=2000, solver="liblinear"),
                {"model__C": [0.1, 1, 10]}
            ),
            "DecisionTree": (
                DecisionTreeClassifier(random_state=42),
                {"model__max_depth": [None, 3, 5], "model__min_samples_split": [2]}
            ),
            "RandomForest": (
                RandomForestClassifier(random_state=42, n_estimators=50),
                {"model__max_depth": [None, 3, 5]}
            ),
            "NaiveBayes": (
                GaussianNB(),
                {}
            )
        }
        if max_neighbors >= 1:
            models["KNN"] = (
                KNeighborsClassifier(),
                {"model__n_neighbors": knn_neighbors, "model__weights": ["uniform", "distance"]}
            )
        # Optional: XGBoost and LightGBM (added if available)
        try:
            import xgboost as xgb  # type: ignore
            print("[AutoML] XGBoost available — adding to model candidates")
            models["XGBoost"] = (
                xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42),
                {"model__n_estimators": [50, 100], "model__max_depth": [3, 5], "model__learning_rate": [0.1]}
            )
        except Exception:
            print("[AutoML] XGBoost not available — skipping")

        try:
            import lightgbm as lgb  # type: ignore
            print("[AutoML] LightGBM available — adding to model candidates")
            models["LightGBM"] = (
                lgb.LGBMClassifier(random_state=42, n_jobs=-1),
                {"model__n_estimators": [50, 100], "model__num_leaves": [31, 63], "model__learning_rate": [0.1]}
            )
        except Exception:
            print("[AutoML] LightGBM not available — skipping")
    else:
        models = {
            "LinearRegression": (
                LinearRegression(), 
                {}
            ),
            "DecisionTree": (
                DecisionTreeRegressor(random_state=42),
                {"model__max_depth": [None, 3, 5], "model__min_samples_split": [2]}
            ),
            "RandomForest": (
                RandomForestRegressor(random_state=42, n_estimators=50),
                {"model__max_depth": [None, 3, 5]}
            )
        }
        if max_neighbors >= 1:
            models["KNN"] = (
                KNeighborsRegressor(),
                {"model__n_neighbors": knn_neighbors, "model__weights": ["uniform", "distance"]}
            )
        if n_samples >= 10:
            models["PolynomialRegression"] = (
                Pipeline([("poly", PolynomialFeatures(include_bias=False)), ("lr", LinearRegression())]),
                {"model__poly__degree": [2, 3]}
            )
        # Optional: XGBoost and LightGBM regressors
        try:
            import xgboost as xgb  # type: ignore
            print("[AutoML] XGBoost (regressor) available — adding to model candidates")
            models["XGBoost"] = (
                xgb.XGBRegressor(n_jobs=-1, random_state=42),
                {"model__n_estimators": [50, 100], "model__max_depth": [3, 5], "model__learning_rate": [0.1]}
            )
        except Exception:
            print("[AutoML] XGBoost regressor not available — skipping")

        try:
            import lightgbm as lgb  # type: ignore
            print("[AutoML] LightGBM (regressor) available — adding to model candidates")
            models["LightGBM"] = (
                lgb.LGBMRegressor(random_state=42, n_jobs=-1),
                {"model__n_estimators": [50, 100], "model__num_leaves": [31, 63], "model__learning_rate": [0.1]}
            )
        except Exception:
            print("[AutoML] LightGBM regressor not available — skipping")
    
    return models


def run_automl(csv_path, target_column, model_path="artifacts/user_model.pkl"):
    """
    Main function to run the AutoML pipeline.
    """
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available: {list(df.columns)}")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    n_samples = len(y)
    print(f"[AutoML] Total samples: {n_samples}")
    
    if n_samples < 20:
        print(f"[AutoML] Small dataset detected. Using all data for cross-validation.")
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print(f"[AutoML] Train size: {len(X_train)}, Test size: {len(X_test)}")

    problem_type = detect_problem_type(y)
    preprocessor = build_preprocessor(X)
    cv_splitter = get_cv_splitter(len(X_train), problem_type, y_train if problem_type == "classification" else None)
    
    models = get_models(problem_type, len(X_train))
    
    if problem_type == "classification":
        scoring = "accuracy"
    else:
        scoring = "neg_mean_squared_error"

    best_model = None
    best_score = -np.inf
    best_params = None
    best_model_name = None

    for name, (model, params) in models.items():
        print(f"[AutoML] Training {name}...")
        
        try:
            pipeline = Pipeline([
                ("preprocess", preprocessor),
                ("model", model)
            ])

            if params:
                grid = GridSearchCV(
                    pipeline, params, cv=cv_splitter, scoring=scoring, 
                    n_jobs=-1, error_score='raise'
                )
                grid.fit(X_train, y_train)
                score = grid.best_score_
                current_params = grid.best_params_
                current_model = grid.best_estimator_
            else:
                pipeline.fit(X_train, y_train)
                scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring=scoring)
                score = np.mean(scores)
                current_params = {}
                current_model = pipeline
            
            if np.isnan(score):
                print(f"[AutoML] {name} returned nan score, skipping...")
                continue
                
            print(f"[AutoML] {name} score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_model = current_model
                best_params = current_params
                best_model_name = name
                
        except Exception as e:
            print(f"[AutoML] {name} failed: {str(e)}")
            continue

    if best_model is None:
        raise ValueError("All models failed to train. Please provide a larger dataset.")

    print(f"[AutoML] Best model: {best_model_name} with score: {best_score:.4f}")

    best_model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    
    if problem_type == "classification":
        from sklearn.metrics import accuracy_score
        y_pred = best_model.predict(X_test)
        test_score = accuracy_score(y_test, y_pred)
        display_score = best_score
    else:
        from sklearn.metrics import r2_score
        y_pred = best_model.predict(X_test)
        test_score = r2_score(y_test, y_pred) if len(y_test) > 1 else 0.0
        display_score = best_score
    
    # Ensure values are JSON serializable
    model_info = {
        "problem_type": str(problem_type),
        "best_model": str(best_model_name) if best_model_name else "Unknown",
        "best_score": float(display_score) if np.isfinite(display_score) else None,
        "test_score": float(test_score) if test_score is not None else None,
        "best_hyperparameters": best_params if best_params else {},
        "dataset_size": int(n_samples),
        "scoring_metric": scoring
    }
    
    print(f"[AutoML] Returning model_info: {model_info}")
    
    info_path = os.path.join(os.path.dirname(model_path), "model_info.json")
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=4, default=str)
    
    return model_info
