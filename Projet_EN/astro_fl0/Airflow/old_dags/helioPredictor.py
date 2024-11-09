import json
import pathlib
import airflow
import Solar_prediction_functions as solar
import requests
import requests.exceptions as requests_exceptions
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from xgboost import XGBRegressor
from datetime import datetime

dag = DAG(
 dag_id="helioPredictor",
 start_date=airflow.utils.dates.days_ago(0),
 schedule_interval="0 0 */3 * *"
)

get_dataset = PythonOperator(
    task_id='get_dataset',
    python_callable=solar.get_dataset,
    op_kwargs={"dataset_path": "solar_prediction.csv","values": ['name', 'id', 'address', 'date', 'GHI', 'latitude', 'longitude', 'time',
       'surface_pressure', 'snowfall', 'temperature_2m', 'winddirection_10m',
       'relativehumidity_2m', 'windgusts_10m', 'windspeed_10m',
       'precipitation', 'cloudcover', 'elevation', 'timezone', 'sunrise',
       'sunset', 'log_GHI', 'sun_duration']},
    dag=dag,
)

get_api_data = PythonOperator(
    task_id='get_api_data',
    python_callable=solar.get_api_data_v2,
    op_kwargs={'data': 'get_dataset'},
    dag=dag,
)

process_api_data = PythonOperator(
    task_id='process_api_data',
    python_callable=solar.process_api_data,
    op_kwargs={'data': 'get_api_data'},
    dag=dag,
)

save_dataset = PythonOperator(
    task_id='save_dataset',
    python_callable=solar.save_dataset,
    op_kwargs={'dataset': 'process_api_data'},
    dag=dag,
)


get_dataset >> get_api_data >> process_api_data >> save_dataset  