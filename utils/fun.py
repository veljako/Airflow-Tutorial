from airflow.decorators import task
import json
import joblib
from utils.utils import get_full_pipeline
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import os
import shutil
from datetime import datetime
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

@task(task_id='preprocesiranje')
def preprocesiranje():
    """
        Ovde preprocesiramo train_podatke, i cuvamo preprocessor u models/preprocessor
    """
    print("Preprocesiranje")

    preprocessor = get_full_pipeline()

    train_df = pd.read_parquet('/opt/airflow/data/raw/train.parquet')
    print("uspesno", train_df.shape)


    X_full = train_df.drop(columns=["SALE PRICE"], errors='ignore')
    y_full = np.log1p(train_df["SALE PRICE"].values)

    X_train_preprocessed = preprocessor.fit_transform(X_full, y_full)

    X_df = pd.DataFrame(X_train_preprocessed)
    X_df['target'] = y_full

    X_df.to_parquet('/opt/airflow/data/preprocessed/train_preprocessed.parquet', index=False)
    joblib.dump(preprocessor, '/opt/airflow/models/preprocessor.joblib')

    print("USPESNO IZVRSEN TASK 1")

@task(task_id='treniranje')
def treniranje():
    """
        Treniramo model na preprocesiranim podacima i cuvamo celokupan pipeline (pipeline od prethodnog dana se takodje)
    """
    print("Treniranje")

    # ucitavanje podataka
    df_preprocessed = pd.read_parquet('/opt/airflow/data/preprocessed/train_preprocessed.parquet')
    print("Uspesno ucitan preprocesiran dataset:", df_preprocessed.shape)

    X_train = df_preprocessed.drop(columns=['target'])
    y_train = df_preprocessed['target']

    # ucitavanje modela i pipeline-a
    xgb_model = XGBRegressor(
        learning_rate=0.05,
        max_depth=8,
        n_estimators=600,
        random_state=42,
        n_jobs=-1
    )

    print("Započeto treniranje XGBoost modela...")
    xgb_model.fit(X_train, y_train)
    print("Model uspešno istreniran.")

    preprocessor = joblib.load('/opt/airflow/models/preprocessor.joblib')

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb_model)
    ])

    # prebacivanje starog modela u history i novog u tren_model 
    base_dir = '/opt/airflow/models'
    tren_dir = os.path.join(base_dir, 'tren_model')
    history_dir = os.path.join(base_dir, 'history_models')

    os.makedirs(tren_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    for filename in os.listdir(tren_dir):
        old_file_path = os.path.join(tren_dir, filename)
        if os.path.isfile(old_file_path):
            shutil.move(old_file_path, os.path.join(history_dir, filename))
            print(f"Premešten stari model u history: {filename}")

    today_str = datetime.today().strftime('%Y-%m-%d')
    new_model_name = f"fullpipeline_{today_str}.joblib"
    new_model_path = os.path.join(tren_dir, new_model_name)

    joblib.dump(full_pipeline, new_model_path)

    print(f"USPEŠNO IZVRŠEN TASK 2: Sačuvan novi model u '{new_model_path}'")

@task(task_id='evaluacija')
def evaluacija():
    """
        Zelimo da prikazemo eval trenutnog modela (ne na log skali)
    """

    print("Evaluacija")
    # uvitavanje modela
    tren_dir = '/opt/airflow/models/tren_model'
    model_files = [f for f in os.listdir(tren_dir) if f.endswith('.joblib')]
    
    if not model_files:
        raise FileNotFoundError("Nijedan model nije pronađen u 'tren_model' folderu.")

    latest_model_file = model_files[0]
    model_path = os.path.join(tren_dir, latest_model_file)
    
    pipeline = joblib.load(model_path)
    print(f"Učitan model: {model_path}")

    test_df = pd.read_parquet('/opt/airflow/data/raw/test.parquet')
    X_test = test_df.drop(columns=["SALE PRICE"], errors='ignore')
    y_test_true = test_df["SALE PRICE"].values

    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    rmse = float(root_mean_squared_error(y_test_true, y_pred))
    mae = float(mean_absolute_error(y_test_true, y_pred))
    r2 = float(r2_score(y_test_true, y_pred))

    print(f"RMSE: {rmse:,.2f} | MAE: {mae:,.2f} | R²: {r2:.4f}")

    results_dir = '/opt/airflow/results'
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

    metrics_data = {
        "timestamp": [timestamp],
        "model_file": [latest_model_file],
        "RMSE": [round(rmse, 2)],
        "MAE": [round(mae, 2)],
        "R2": [round(r2, 4)],
        "num_samples": [len(test_df)]
    }

    metrics_df = pd.DataFrame(metrics_data)

    csv_path = os.path.join(results_dir, f"metrics_{timestamp}.csv")
    metrics_df.to_csv(csv_path, index=False)

    print(f"USPEŠNO IZVRŠEN TASK 3: Metrike sačuvane u '{csv_path}'")