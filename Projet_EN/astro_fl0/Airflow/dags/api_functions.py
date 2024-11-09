import requests
from datetime import datetime,timedelta
import time
import pandas as pd
import io
import numpy as np

def api(start_date, end_date, latitude, longitude):
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
            time.sleep(5)

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

    print(li)

    if li:
        frame = pd.concat(li, axis=0, ignore_index=True)
        print("Request OK")
        return frame, res
    else:
        print("Request FAILED")
        return None, None

def get_api_data(data, **context):
    # Charger les données depuis l'API et les grouper par identifiant "id"
    dataset = pd.read_json(io.StringIO(context["ti"].xcom_pull(task_ids=data)))
    print(dataset.keys())
    print(dataset.shape)
    grouped_conc = dataset.groupby("id")
    frames = []
    
    total_ids = len(grouped_conc)  # Nombre total d'ID à traiter
    ids_remaining = total_ids      # Nombre d'ID restants
    stats_data = []  # Liste pour stocker les stats pour chaque ID

    for id, group in grouped_conc:
        start_time = str(group["date"].iloc[0].date())
        end_time = str(group["date"].iloc[-1].date())
        lat = group["latitude"].iloc[0]
        lon = group["longitude"].iloc[0]
        
        max_retries = 10  # Nombre maximal de tentatives
        retry_count = 0
        success = False
        
        while not success and retry_count < max_retries:
            print("---------------------------------------------")
            print(f"Traitement de l'id: {id} - IDs restants: {ids_remaining}/{total_ids}")
            print(
                f"Tentative {retry_count + 1} pour les paramètres : \n Start time : {start_time} - End time : {end_time} - Lat : {lat} - Lon : {lon} \n"
            )
            
            frame, res = api(start_time, end_time, lat, lon)
            
            if frame is not None and not frame.empty and res:
                print("Requête réussie")
                frames.append(frame)
                stats_data.append({"id": id, "status": "succès", "tentatives": retry_count + 1})
                success = True
            else:
                print("Échec de la requête - Pause de 60 secondes avant de réessayer...")
                time.sleep(60)
                retry_count += 1
        
        # Lever une exception si toutes les tentatives échouent pour cet id
        if not success:
            stats_data.append({"id": id, "status": "échec", "tentatives": max_retries})
            raise Exception(f"Requête échouée après {max_retries} tentatives pour l'id: {id}")
        
        # Décrémenter le compteur d'IDs restants après chaque traitement
        ids_remaining -= 1

    # Traitement final sur les données après la boucle
    if frames:
        combined_frame = pd.concat(frames, ignore_index=True)
        # Créer un DataFrame pour les statistiques de traitement
        stats_frame = pd.DataFrame(stats_data)
        # Retourner les deux datasets sous forme de dictionnaire JSON
        return {
            "data": combined_frame.to_json(),
            "stats": stats_frame.to_json()
        }

def process_api_data(data, **context):
    dataset = pd.read_json(io.StringIO(context["ti"].xcom_pull(task_ids=data))).dataset
    combined_frame = pd.read_json(io.StringIO(context["ti"].xcom_pull(task_ids=data))).data
    print(dataset.keys())

    combined_frame["time"] = pd.to_datetime(combined_frame["time"], format="%Y-%m-%dT%H:%M")
    dataset["time"] = pd.to_datetime(dataset["date"], format="%Y-%m-%d %H:%M:%S")
        
    selected_columns = [
        "time", "surface_pressure", "snowfall", "temperature_2m", "winddirection_10m",
        "relativehumidity_2m", "windgusts_10m", "windspeed_10m", "precipitation", 
        "cloudcover", "elevation", "timezone", "sunrise", "sunset"
    ]
    combined_frame = combined_frame[selected_columns]

    print(combined_frame.keys())
    print(combined_frame.shape)

    dataset = pd.merge(dataset, combined_frame, on="time", how="left")

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