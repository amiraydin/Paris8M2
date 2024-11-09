import io
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from tabulate import tabulate
from files_functions import get_files_directory
from datetime import datetime
from api_functions import api

def make_prediction(model_name, x,y, **context):
    model_path = get_files_directory() + model_name
    print(model_path)

    model = pickle.load(open(model_path, "rb"))

    start_of_week = datetime(2016, 12, 13)
    end_of_week = datetime(2016, 12, 13)

    start_time = start_of_week.strftime("%Y-%m-%d")
    end_time = end_of_week.strftime("%Y-%m-%d")

    lat = x
    lon = y

    frame, res = api(start_time, end_time, lat, lon)

    columns_to_convert = [
        "surface_pressure",
        "snowfall",
        "temperature_2m",
        "winddirection_10m",
        "relativehumidity_2m",
        "windgusts_10m",
        "windspeed_10m",
        "precipitation",
        "cloudcover",
    ]

    frame[columns_to_convert] = frame[columns_to_convert].apply(
        pd.to_numeric, errors="coerce"
    )

    frame["time"] = pd.to_datetime(frame["time"], format="%Y-%m-%dT%H:%M")
    frame["sunset"] = pd.to_datetime(frame["sunset"], format="%Y-%m-%dT%H:%M")
    frame["sunrise"] = pd.to_datetime(frame["sunrise"], format="%Y-%m-%dT%H:%M")

    frame["sun_duration"] = (frame["sunset"] - frame["sunrise"]).dt.total_seconds()

    frame = frame[
        [
            "surface_pressure",
            "snowfall",
            "temperature_2m",
            "winddirection_10m",
            "relativehumidity_2m",
            "windgusts_10m",
            "windspeed_10m",
            "precipitation",
            "cloudcover",
            "sun_duration"
        ]
    ]

    X = frame
    print(X.keys())
    prediction = model.predict(X)
    estimation = np.expm1(prediction)
    print(f'The solar production for lat {lat} and long {lon} is : {estimation.mean()} GHI')
    return 0


def model_training(data,**context):
    df_merged =  pd.read_json(io.StringIO(context["ti"].xcom_pull(task_ids=data)))

    print(df_merged.keys())

    df_merged = df_merged.dropna()
    
    X = df_merged.drop(['log_GHI', 'GHI', 'date', 'name', 'address', 'sunrise', 'sunset','time','elevation','timezone','id', 'latitude', 'longitude'], axis=1)
    y = df_merged['log_GHI']
    
    print('\n' + tabulate(X.head(), headers='keys', tablefmt='psql'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_params = {
        'subsample': 0.95,
        'n_estimators': 1500,
        'max_depth': 5,
        'learning_rate': 0.1,
        'gamma': 0,
        'colsample_bytree': 0.5
    }
    
    models = {"XGBoost_tunned_v2": XGBRegressor(**best_params)}

    scores_df = pd.DataFrame()
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_test_original = np.expm1(y_test)
        
        rmse = mean_squared_error(y_test_original, y_pred, squared=False)
        mae = mean_absolute_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        mape = mean_absolute_percentage_error(y_test_original, y_pred)
        scores_df = pd.concat([
            scores_df,
            pd.DataFrame([{
                "model": model_name,
                "R2" : r2,
                "MAE" : mae,
                "RMSE" : rmse,
                "MAPE" : mape
            }])
        ], axis=0)
        print("SCORE DF : \n")
        print("\n" + tabulate(scores_df, headers='keys', tablefmt='psql'))
    model_path = get_files_directory() + 'model.pkl'  
    try:
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        print("Le modèle a été sauvegardé avec succès.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle : {e}")