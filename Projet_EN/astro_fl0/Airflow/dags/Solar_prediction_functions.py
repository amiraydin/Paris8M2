import os
import io
import pickle
import pandas as pd
import requests
import pandas as pd
from datetime import datetime,timedelta
import time
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from tabulate import tabulate


def get_files_directory():
    current_directory = os.getcwd() + "/files/"
    print(f"Les données sont stockées : {current_directory}")
    return str(current_directory)


def get_dataset(dataset_path, values):
    CSV_path = get_files_directory() + dataset_path
    df = pd.read_csv(CSV_path)
    df = df[values]
    print(df)
    print(df.keys())
    return df.to_json()


def combine_datasets(_dataset1, _dataset2, on, how, **context):
    dataset1 = pd.read_json(io.StringIO(context["ti"].xcom_pull(task_ids=_dataset1)))
    dataset2 = pd.read_json(io.StringIO(context["ti"].xcom_pull(task_ids=_dataset2)))
    merged_dataset = pd.merge(dataset1, dataset2, on=on, how=how)
    print(merged_dataset.head(5))
    return merged_dataset.to_json()


def make_prediction(model_name, x,y, **context):
    model_path = get_files_directory() + model_name
    print(model_path)

    model = pickle.load(open(model_path, "rb"))
    timestamp = datetime.now()

    """start_of_week = timestamp
    end_of_week = timestamp"""

    start_of_week = datetime(2016, 12, 13)
    end_of_week = datetime(2016, 12, 13)

    start_time = start_of_week.strftime("%Y-%m-%d")
    end_time = end_of_week.strftime("%Y-%m-%d")

    """lat = x
    lon = y"""

    lat = 51.08589533
    lon = -113.9835938

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


def preprocessing(data, **context):
    dataset = pd.read_json(io.StringIO(context["ti"].xcom_pull(task_ids=data)))
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset.rename(columns={"kWh": "GHI"}, inplace=True)
    dataset = dataset.sort_values(by=["name", "date"])
    print(type(dataset["id"]))
    return dataset.to_json()


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
            response = requests.get(
                "https://archive-api.open-meteo.com/v1/era5", params=params, timeout=10
            )
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


def get_api_data(data, **context):
    dataset = pd.read_json(io.StringIO(context["ti"].xcom_pull(task_ids=data)))
    print(dataset.head(5))

    grouped_conc = dataset.groupby("id")
    frames = []
    iteration_count = 0

    for id, group in grouped_conc:
        start_time = str(group["date"].iloc[0].date())
        end_time = str(group["date"].iloc[len(group) - 1].date())
        lat = group["latitude"].iloc[0]
        lon = group["longitude"].iloc[0]
        print("---------------------------------------------")
        print(
            f"Params : \n Start time : {start_time} - End time : {end_time} - Lat : {lat} - Lon : {lon} \n"
        )

        frame, res = api(start_time, end_time, lat, lon)
        frames.append(frame)

        iteration_count += 1

        if iteration_count % 5 == 0:
            print("Pausing for 25 seconds...")
            time.sleep(25)

    frame["time"] = pd.to_datetime(frame["time"], format="%Y-%m-%dT%H:%M")
    dataset["time"] = pd.to_datetime(dataset["date"], format="%Y-%m-%d %H:%M:%S")
    frame = frame[
        [
            "time",
            "time",
            "surface_pressure",
            "snowfall",
            "temperature_2m",
            "winddirection_10m",
            "relativehumidity_2m",
            "windgusts_10m",
            "windspeed_10m",
            "precipitation",
            "cloudcover",
            "elevation",
            "timezone",
            "sunrise",
            "sunset",
        ]
    ]

    print(frame.head(5))
    dataset = pd.merge(dataset, frame, on="time", how="left")
    return dataset.to_json()

def process_api_data(data, **context):
    dataset = pd.read_json(io.StringIO(context["ti"].xcom_pull(task_ids=data)))
    print(dataset.keys())
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

    dataset[columns_to_convert] = dataset[columns_to_convert].apply(
        pd.to_numeric, errors="coerce"
    )

    dataset["sunrise"] = pd.to_datetime(dataset["sunrise"], errors="coerce")
    dataset["sunset"] = pd.to_datetime(dataset["sunset"], errors="coerce")

    dataset['log_GHI'] = np.log1p(dataset['GHI'])

    dataset["sun_duration"] = (dataset["sunset"] - dataset["sunrise"]).dt.total_seconds()

    dataset = dataset.dropna()
    
    return dataset.to_json()


def save_dataset(dataset, **context):
    df = pd.read_json(io.StringIO(context["ti"].xcom_pull(task_ids=dataset)))
    print(df.keys())
    df.to_csv(get_files_directory() + "solar_prediction.csv", index=False)
    print("Dataset saved")
    return 0

def model_training(data,**context):

    #path = context['ti'].xcom_pull(task_ids='get_files_directory')

    #merged_data_json = context['ti'].xcom_pull(task_ids=data)
    df_merged =  pd.read_json(io.StringIO(context["ti"].xcom_pull(task_ids=data)))  # Convertir JSON en DataFrame

    print(df_merged.keys())

    df_merged = df_merged.dropna()
    
   
    X = df_merged.drop(['log_GHI', 'GHI', 'date', 'name', 'address', 'sunrise', 'sunset','time','elevation','timezone','id', 'latitude', 'longitude'], axis=1)
    y = df_merged['log_GHI']
    
    print('\n' + tabulate(X.head(), headers='keys', tablefmt='psql'))
    print('\n'+ X.shape())

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
        print("SCORE DF : \n")
        print(tabulate(scores_df, headers='keys', tablefmt='psql'))
    model_path = get_files_directory() + 'model.pkl'  
    try:
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        print("Le modèle a été sauvegardé avec succès.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle : {e}")