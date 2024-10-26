##AIRFLOW

import airflow
from airflow import DAG
from airflow.operators.python import PythonOperator

# MANIP DATA

import os
import pandas as pd
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time 

# MODEL TRAINING

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def api(start_date, end_date, latitude, longitude):
    """API call function with a delay between requests."""

    date_string_1 = start_date + "T" + "00:00"
    date_string_2 = end_date + "T" + "00:00"

    date_object1 = datetime.strptime(date_string_1, "%Y-%m-%dT%H:%M")
    date_object2 = datetime.strptime(date_string_2, "%Y-%m-%dT%H:%M")

    date_only1 = date_object1.date()
    date_only2 = date_object2.date()
    date_string_1 = str(date_only1)
    date_string_2 = str(date_only2)

    x = [(latitude, longitude)]
    li = []

    for i in x:
        params = {
            "latitude": i[0],
            "longitude": i[1],
            "start_date": date_string_1,
            "end_date": date_string_2,
            "timezone": "auto",
            "temperature_unit": "fahrenheit",
            "windspeed_unit": "mph",
            "precipitation_unit": "inch",
            "hourly": {
                "precipitation",
                "snowfall",
                "temperature_2m",
                "relativehumidity_2m",
                "surface_pressure",
                "windspeed_10m",
                "winddirection_10m",
                "windgusts_10m",
                "cloudcover",
            },
            "daily": {"sunrise", "sunset"},
        }

        try:
            response = requests.get("https://archive-api.open-meteo.com/v1/era5", params=params, timeout=10)
            response.raise_for_status()  # Raise an error for bad responses (4xx/5xx)



            res = response.json()

            df3 = pd.DataFrame.from_dict(res["daily"], orient="index").T
            df3 = df3.loc[df3.index.repeat(24)].reset_index(drop=True)
            df2 = pd.DataFrame.from_dict(res["hourly"], orient="index").T
            df2 = df2.assign(
                elevation=res["elevation"],
                latitude=res["latitude"],
                longitude=res["longitude"],
                timezone=res["timezone_abbreviation"],
            )
            df2["sunrise"] = df3["sunrise"]
            df2["sunset"] = df3["sunset"]
            li.append(df2)
        except requests.exceptions.Timeout:
            print(f"Request timed out for coordinates: {i}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

    if li:
        frame = pd.concat(li, axis=0, ignore_index=True)
        print("Request OK")
        return frame, res
    else:
        print("Request FAILED")
        return None, None

def get_files_directory():
    current_directory = os.getcwd() + '/files/'
    return current_directory  # Retourne le chemin, qui sera automatiquement stocké dans XCom

def load_and_merge_dataset(**context):
    # Récupérer le chemin depuis XCom
    path = context['ti'].xcom_pull(task_ids='get_files_directory')

    file_path_prod = os.path.join(path, 'Solar_Energy_Production.csv')
    energy_production = pd.read_csv(file_path_prod)

    file_path_sites = os.path.join(path, 'Solar_Photovoltaic_Sites.csv')
    energy_production_sites = pd.read_csv(file_path_sites)

    energy_production = energy_production[['name', 'id', 'address', 'date', 'kWh']]

    energy_production_sites = energy_production_sites[['id', 'latitude', 'longitude']]

    print("Shape production : ", energy_production.shape)
    print("Shape sites : ", energy_production_sites.shape)

    merged_df = pd.merge(energy_production, energy_production_sites,on='id', how='left')

    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df.rename(columns={'kWh': 'GHI'}, inplace=True)
    merged_df = merged_df.sort_values(by=['name', 'date'])

    print("Shape merged : ", merged_df)
    print(merged_df.keys())

    print(merged_df.head(5))

    return merged_df.to_json()

def api_merged_and_clean(**context):

    path = context['ti'].xcom_pull(task_ids='get_files_directory')

    merged_data_json = context['ti'].xcom_pull(task_ids='load_and_merge_dataset')
    merged_df = pd.read_json(merged_data_json)  # Convertir JSON en DataFrame
        
    grouped_conc = merged_df.groupby("id")
    frames = []
    iteration_count = 0  # Initialize a counter

    for id, group in grouped_conc:
        start_time = str(group["date"].iloc[0].date())
        end_time = str(group["date"].iloc[len(group) - 1].date())
        lat = group["latitude"].iloc[0]
        lon = group["longitude"].iloc[0]
        print("---------------------------------------------")
        print(f'Params : \n Start time : {start_time} - End time : {end_time} - Lat : {lat} - Lon : {lon} \n')

        # Make API call
        frame, res = api(start_time, end_time, lat, lon)
        frames.append(frame)

        iteration_count += 1  # Increment the iteration count

        # Every 5 iterations, add a 10-second delay
        if iteration_count % 5 == 0:
            print("Pausing for 30 seconds...")
            time.sleep(30)  # Delay for 10 seconds
    
    frame['time'] = pd.to_datetime(frame['time'], format='%Y-%m-%dT%H:%M')

    frame = frame[
        [ 'time',
          'surface_pressure', 
          'snowfall', 
          'temperature_2m',
          'winddirection_10m',
          'relativehumidity_2m', 
          'windgusts_10m',
          'windspeed_10m', 
          'precipitation', 
          'cloudcover', 
          'elevation', 
          'timezone', 
          'sunrise', 
          'sunset']
        ]
    
    merged_df['time'] = pd.to_datetime(merged_df['date'], format='%Y-%m-%d %H:%M:%S')
    df_merged = pd.merge(merged_df, frame, on='time', how='left')
    df_merged = df_merged.dropna()

    df_merged['log_GHI'] = np.log1p(df_merged['GHI'])

    df_merged['sunrise'] = pd.to_datetime(df_merged['sunrise'], errors='coerce')
    df_merged['sunset'] = pd.to_datetime(df_merged['sunset'], errors='coerce')

    df_merged = df_merged.drop('elevation', axis=1)
    df_merged = df_merged.drop('timezone', axis=1)

    df_merged["sun_duration"] = (df_merged["sunset"] - df_merged["sunrise"]).dt.total_seconds()

    print("Result from API : ", df_merged)
    
    csv_file_path = path + 'df_merged.csv'  # Change le chemin ici
    df_merged.to_csv(csv_file_path, index=False)
    
    return df_merged.to_json()

def model_training(**context):

    path = context['ti'].xcom_pull(task_ids='get_files_directory')

    merged_data_json = context['ti'].xcom_pull(task_ids='api_merged_and_clean')
    df_merged = pd.read_json(merged_data_json)  # Convertir JSON en DataFrame
   

    X = df_merged.drop(['log_GHI', 'GHI', 'date', 'name', 'address', 'sunrise', 'sunset','time'], axis=1)
    y = df_merged['log_GHI']

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
        y_pred = np.exp(y_pred_log)
        y_test_original = np.exp(y_test)
        # Calculating different evaluation metrics
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
    print("SCORE DF :", scores_df)

    model_path = path + 'nithusanModel.pkl'  
    try:
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        print("Le modèle a été sauvegardé avec succès.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle : {e}")

def load_model(**context):
    path = context['ti'].xcom_pull(task_ids='get_files_directory')
    file_path_model = os.path.join(path, 'nithusanModel.pkl')
    try:
        model = pickle.load(open(file_path_model, 'rb'))
        print("Le modèle Nithusan a été chargé avec succès.")
        print(f"Type du modèle : {type(model)}")  # Vérifie le type du modèle
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return  # Sortir si le chargement échoue

# Définir le DAG    
dag = DAG(
    dag_id="projet",
    start_date=airflow.utils.dates.days_ago(14),
    schedule_interval=None,
)
# Définir les tâches
get_files_directory_task = PythonOperator(
    task_id='get_files_directory',
    python_callable=get_files_directory,
    dag=dag
)

load_and_merge_dataset_task = PythonOperator(
    task_id='load_and_merge_dataset',
    python_callable=load_and_merge_dataset,
    provide_context=True,  # Permet de passer le contexte de tâche, nécessaire pour XCom
    dag=dag
)

api_merged_and_clean_task = PythonOperator(
    task_id='api_merged_and_clean',
    python_callable=api_merged_and_clean,
    provide_context=True,  # Permet d'accéder au contexte pour récupérer XCom
    dag=dag
)

model_training_task = PythonOperator(
    task_id='model_training',
    python_callable=model_training,
    provide_context=True,  # Permet d'accéder au contexte pour récupérer XCom
    dag=dag
)

load_model_task = PythonOperator(
    task_id='load_model',
    python_callable=load_model,
    provide_context=True,  # Permet d'accéder au contexte pour récupérer XCom
    dag=dag
)


# Définir les dépendances entre les tâches
get_files_directory_task >> load_and_merge_dataset_task >> api_merged_and_clean_task >> model_training_task >> load_model_task
